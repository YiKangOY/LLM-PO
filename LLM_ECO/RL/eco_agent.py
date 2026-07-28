#!/usr/bin/env python3
"""
ECO A3C agent mirroring the reference agents in `dse/algo/a3c/agent`.
It wraps the discrete ECO environment and reuses the shared buffer,
preference, and network utilities.
"""

import os
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from dse.algo.a3c.agent.agent import Agent
from dse.algo.a3c.buffer import Buffer
from dse.algo.a3c.preference import Preference
from dse.algo.a3c.functions import (
    make_a3c_vec_envs,
    array_to_tensor,
    tensor_to_array,
    space_dim,
)
from dse.algo.a3c.model import ECOActorCriticNetwork
from utils.utils import remove_suffix, if_exist


class ECOAgent(Agent):
    """
    Drop-in agent following the reference BOOM/Rocket agents but targeting the
    ECO design space and environment wrapper.
    """

    def __init__(self, configs, env_cls):
        super(ECOAgent, self).__init__(configs)
        self.gpu_id = int(self.configs["algo"].get("gpu-id", 0))
        use_cuda = self.configs["algo"].get("use-cuda", False) and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(self.gpu_id)
            self.device = torch.device("cuda:{}".format(self.gpu_id))
        else:
            self.device = torch.device("cpu")
        self.envs = make_a3c_vec_envs(self.configs, env_cls)
        self.observation_dim = space_dim(self.envs.observation_space)
        self.action_dim = space_dim(self.envs.action_space)
        self.model = ECOActorCriticNetwork(
            self.observation_dim,
            self.action_dim,
            self.envs.reward_space
        )
        self.model.to(self.device)
        self._model = copy.deepcopy(self.model)
        self.training = self.set_mode()
        self.preference = Preference(
            self.configs["algo"]["test"]["ppa-preference"],
            self.envs.reward_space
        )
        self.buffer = Buffer(
            self.observation_dim,
            self.envs.reward_space,
            self.sample_size
        )
        self.temperature = self.configs["algo"]["train"]["temperature"]
        self.lr = self.learning_rate
        self.save_interval = int(self.configs["algo"]["train"].get("save-interval", 100))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.mse = nn.MSELoss()
        self.set_random_state(self.configs["algo"].get("random-seed", 0))

    def set_random_state(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def set_mode(self):
        if self.mode == "test":
            self.model.eval()
            self._model.eval()
            self.load(self.configs["algo"]["test"]["rl-model"])
        return True if self.mode == "train" else False

    def get_action(self, state, preference):
        state = array_to_tensor(state, device=self.device)
        preference = array_to_tensor(preference, device=self.device)
        policy, value = self.model(state, preference)
        if self.training:
            policy = F.softmax(policy / self.temperature, dim=-1)
        else:
            policy = F.softmax(policy, dim=-1)
        dist = Categorical(policy)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        _policy = tensor_to_array(policy)
        return tensor_to_array(action), _policy, tensor_to_array(log_prob)

    def anneal(self):
        self.temperature = 0.01 + 0.99 * self.temperature

    def forward_transition(self, preference):
        buffer = self.buffer.batch_pool
        state = array_to_tensor(buffer["state"], device=self.device)
        next_state = array_to_tensor(buffer["next-state"], device=self.device)
        preference = array_to_tensor(preference, device=self.device)
        _, value = self._model(state, preference)
        policy, _ = self.model(state, preference)
        _, next_value = self._model(next_state, preference)
        # Keep batch dimension when reward_space == 1; explicitly reshape to
        # (batch, reward_space) so downstream slicing always works.
        value = tensor_to_array(value).reshape(-1, self.envs.reward_space)
        next_value = tensor_to_array(next_value).reshape(-1, self.envs.reward_space)
        return value, next_value, policy

    def calc_discounted_reward(self, value, next_value):
        buffer = self.buffer.batch_pool
        # Defensive reshape: guard against accidental scalar inputs when
        # reward_space is 1 so indexing below never fails.
        value = np.asarray(value).reshape(-1, self.envs.reward_space)
        next_value = np.asarray(next_value).reshape(-1, self.envs.reward_space)

        def _calc_discounted_reward(reward, done, value, next_value):
            discounted_reward = np.empty([self.num_step, self.envs.reward_space])
            gae = np.zeros(self.envs.reward_space)
            for t in range(self.num_step - 1, -1, -1):
                delta = reward[t] + self.gamma * next_value[t] * (1 - done[t]) - value[t]
                gae = delta + self.gamma * self.lam * (1 - done[t]) * gae
                discounted_reward[t] = gae + value[t]
            return discounted_reward

        total_discounted_reward = []
        for idx in range(self.sample_size):
            n_step = self.num_parallel * self.num_step
            for worker in range(self.num_parallel):
                start = worker * self.num_step + idx * n_step
                end = (worker + 1) * self.num_step + idx * n_step
                discounted_reward = _calc_discounted_reward(
                    buffer["reward"][start:end],
                    buffer["done"][start:end],
                    value[start:end],
                    next_value[start:end],
                )
                total_discounted_reward.append(discounted_reward)
        return np.concatenate(total_discounted_reward).reshape(-1, self.envs.reward_space)

    def envelope_operator(self, preference, discounted_reward, value, episode):
        ofs = self.num_parallel * self.num_step

        def apply_envelope_operator(discounted_reward, preference):
            prod = np.inner(discounted_reward, preference)
            mask = prod.transpose().reshape(self.sample_size, -1, ofs).argmax(axis=1)
            mask = mask.reshape(-1) * ofs + np.array(list(range(ofs)) * self.sample_size)
            return discounted_reward[mask]

        if episode > self.start_envelope:
            discounted_reward = apply_envelope_operator(discounted_reward, preference)
        adv = discounted_reward - value
        return adv, discounted_reward

    def optimize_actor_critic(self, preference, reward, adv):
        buffer = self.buffer.batch_pool
        with torch.no_grad():
            state = array_to_tensor(buffer["state"], device=self.device)
            action = array_to_tensor(buffer["action"], device=self.device, fmt=int)
            reward = array_to_tensor(reward, device=self.device)
        preference = array_to_tensor(preference, device=self.device)
        adv = array_to_tensor(adv, device=self.device)

        adv_w = torch.bmm(adv.unsqueeze(1), preference.unsqueeze(2)).squeeze()
        # Robust normalization: avoid NaNs when only one element or zero variance.
        adv_w_mean = adv_w.mean()
        adv_w_std = adv_w.std(unbiased=False)
        if torch.isfinite(adv_w_std) and adv_w_std > 1e-8:
            adv_w = (adv_w - adv_w_mean) / adv_w_std
        else:
            adv_w = adv_w - adv_w_mean

        policy, value = self.model(state, preference)
        optimal_action = Categorical(F.softmax(policy, dim=-1))

        value_w = torch.bmm(value.unsqueeze(1), preference.unsqueeze(2)).squeeze()
        reward_w = torch.bmm(reward.unsqueeze(1), preference.unsqueeze(2)).squeeze()

        self.actor_loss = (-optimal_action.log_prob(action) * adv_w).mean()
        self.entropy = optimal_action.entropy().mean()

        critic_loss_1 = self.mse(value_w, reward_w)
        critic_loss_2 = self.mse(value.view(-1), reward.view(-1))
        self.critic_loss = 0.5 * (self.beta * critic_loss_1 + (1 - self.beta) * critic_loss_2)

        self.loss = self.actor_loss + self.critic_loss - self.alpha * self.entropy

        self.optimizer.zero_grad()
        self.loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()

    def optimize_ppo(self, preference, reward, adv):
        buffer = self.buffer.batch_pool
        state = array_to_tensor(buffer["state"], device=self.device)
        action = array_to_tensor(buffer["action"], device=self.device, fmt=int)
        old_log_prob = array_to_tensor(buffer["log-prob"], device=self.device)
        preference = array_to_tensor(preference, device=self.device)
        reward = array_to_tensor(reward, device=self.device)
        adv = array_to_tensor(adv, device=self.device)

        adv_w = torch.bmm(adv.unsqueeze(1), preference.unsqueeze(2)).squeeze()
        adv_w_mean = adv_w.mean()
        adv_w_std = adv_w.std(unbiased=False)
        if torch.isfinite(adv_w_std) and adv_w_std > 1e-8:
            adv_w = (adv_w - adv_w_mean) / adv_w_std
        else:
            adv_w = adv_w - adv_w_mean
        returns_w = torch.bmm(reward.unsqueeze(1), preference.unsqueeze(2)).squeeze()

        policy_logits, value = self.model(state, preference)
        policy = F.softmax(policy_logits / (self.temperature if self.training else 1), dim=-1)

        batch_size = state.size(0)
        batch_inds = np.arange(batch_size)

        self.actor_loss = torch.tensor(0.0, device=self.device)
        self.critic_loss = torch.tensor(0.0, device=self.device)
        self.entropy = torch.tensor(0.0, device=self.device)
        self.loss = torch.tensor(0.0, device=self.device)

        for _ in range(self.ppo_epoch):
            np.random.shuffle(batch_inds)
            for start in range(0, batch_size, self.ppo_batch_size):
                end = start + self.ppo_batch_size
                idx = batch_inds[start:end]

                dist = Categorical(policy[idx])
                log_prob = dist.log_prob(action[idx])
                ratio = torch.exp(log_prob - old_log_prob[idx])

                surr1 = ratio * adv_w[idx]
                surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * adv_w[idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                value_w = torch.bmm(value[idx].unsqueeze(1), preference[idx].unsqueeze(2)).squeeze()
                critic_loss = self.mse(value_w, returns_w[idx])
                entropy = dist.entropy().mean()

                loss = actor_loss + self.ppo_value_coef * critic_loss - self.alpha * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()

                self.actor_loss = actor_loss.detach()
                self.critic_loss = critic_loss.detach()
                self.entropy = entropy.detach()
                self.loss = loss.detach()

    def schedule_lr(self, episode):
        self.lr = self.learning_rate - (episode / self.max_episode) * self.learning_rate
        for params in self.optimizer.param_groups:
            params["lr"] = self.lr

    def save(self, episode):
        interval = max(1, self.save_interval)
        if episode % interval == 0:
            base_name = remove_suffix(os.path.basename(self.configs["log-path"]), ".log")
            model_path = os.path.join(
                self.configs["model-path"],
                "{}_episode_{}.pt".format(base_name, episode),
            )
            torch.save(self.model.state_dict(), model_path)

    def load(self, path):
        if_exist(path, strict=True)
        print("Loading model from {}".format(path), flush=True)
        if self.device.type == "cpu":
            self.model.load_state_dict(torch.load(path, map_location="cpu"))
        else:
            self.model.load_state_dict(torch.load(path))
        self._model = copy.deepcopy(self.model)

    def sync_critic(self, episode):
        if episode % self.update_critic_episode == 0:
            self._model.load_state_dict(self.model.state_dict())
