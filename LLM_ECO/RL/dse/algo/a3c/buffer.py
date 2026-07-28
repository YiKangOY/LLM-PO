# Author: anonymous


import torch
import numpy as np


class Buffer(object):
    def __init__(self, observation_space, reward_space, sample_size):
        super(Buffer, self).__init__()
        self.total_state = []
        self.total_action = []
        self.total_log_prob = []
        self.total_next_state = []
        self.total_reward = []
        self.total_done = []
        self.observation_space = observation_space
        self.reward_space = reward_space
        self.sample_size = sample_size
        self.advantage = None
        self.returns = None

    def reset(self):
        self.total_state = []
        self.total_action = []
        self.total_log_prob = []
        self.total_next_state = []
        self.total_reward = []
        self.total_done = []
        self.advantage = None
        self.returns = None

    def insert(self, state, action, next_state, reward, done, log_prob=None):
        """
            Each `state` is with a shape: step x parallel x dims_of_state
        """
        self.total_state.append(state)
        self.total_action.append(action)
        if log_prob is not None:
            self.total_log_prob.append(log_prob)
        self.total_next_state.append(next_state)
        self.total_reward.append(reward)
        self.total_done.append(done)

    def attach_postprocess(self, advantage=None, returns=None):
        """
            Store flattened advantage/return for minibatch
            sampling after rollout collection.
        """
        self.advantage = advantage
        self.returns = returns

    def generate_batch_with_n_step(self):
        def _generate_batch_with_n_step():
            """
                Each `total_state` is with a shape: (parallel x step x sample) x dims_of_state
                E.g.:
                    total_state:
                        | worker 1's state @step 1 |
                        | ------------------------ |
                        | worker 1's state @step 2 |
                        | ------------------------ |
                        | worker 1's state @step 3 |
                        |           ...            |
            """
            total_state = np.stack(self.total_state).transpose(
                [1, 0, 2]
            ).reshape(-1, self.observation_space)
            total_state = np.tile(total_state, (self.sample_size, 1))
            total_action = np.stack(self.total_action).transpose().reshape([-1])
            total_action = np.tile(total_action, self.sample_size)
            total_log_prob = None
            if len(self.total_log_prob) == len(self.total_action) and \
                    len(self.total_log_prob) > 0:
                total_log_prob = np.stack(
                    self.total_log_prob
                ).transpose().reshape([-1])
                total_log_prob = np.tile(total_log_prob, self.sample_size)
            total_next_state = np.stack(self.total_next_state).transpose(
                [1, 0, 2]
            ).reshape(-1, self.observation_space)
            total_next_state = np.tile(total_next_state, (self.sample_size, 1))
            total_reward = np.stack(self.total_reward).transpose(
                [1, 0, 2]
            ).reshape([-1, self.reward_space])
            total_reward = np.tile(total_reward, (self.sample_size, 1))
            total_done = np.stack(self.total_done).transpose().reshape([-1])
            total_done = np.tile(total_done, self.sample_size)
            batch = {
                "state": total_state,
                "action": total_action,
                "next-state": total_next_state,
                "reward": total_reward,
                "done": total_done
            }
            if total_log_prob is not None:
                batch["log-prob"] = total_log_prob
            if self.advantage is not None:
                batch["advantage"] = self.advantage
            if self.returns is not None:
                batch["return"] = self.returns
            return batch
        self.batch_pool = _generate_batch_with_n_step()

    def get_flat_batch(self):
        """
            Return the flattened rollout dictionary for minibatch sampling.
        """
        return self.batch_pool
