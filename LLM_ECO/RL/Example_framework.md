# Repository Guidelines

## Project Structure & Module Organization
- Core RL framework lives in `algo/a3c`: policy logic (`a3c.py`), models (`model.py`), rollout storage (`buffer.py`), action/value helpers (`functions.py`), and preference handling (`preference.py`). Extend these when altering algorithm behavior.
- Environments and design spaces are under `env/boom` and `env/rocket`; shared abstractions in `env/base_design_space.py`. Keep env APIs stable so agents remain drop-in across spaces.
- Other utilities exist outside this directory; skim them only as needed. Keep changes here cohesive and framework-focused.

## Build, Test, and Development Commands
- Install deps: `pip3 install -r requirements.txt`.
- Set environment for imports: `export PYTHONPATH=$(pwd)`.
- Run a quick smoke by executing a minimal training script or interactive check that imports `algo.a3c` and steps an env (match existing patterns if a runner script lives in your workspace).

## Coding Style & Naming Conventions
- Python 3 with 4-space indents; follow the PEP8-ish style already in the repo.
- Prefer explicit names (e.g., `boom_action_prob`), CamelCase for classes, snake_case for functions/vars. Keep argument names consistent across agent/model/env boundaries.
- Factor shared logic into helpers (e.g., `functions.py` for math utilities, `utils` modules if available) rather than duplicating code.
- Type hints are optional; when adding complex signatures, include concise docstrings or inline comments explaining non-obvious behaviors.

## Testing Guidelines
- No formal suite; validate with a short training dry run on the smallest config available and watch logs for divergence, NaNs, or stalled rewards.
- When adjusting envs, manually step through `env/boom` or `env/rocket` to confirm action/observation shapes align with `model.py` expectations.
- Keep buffers and preference logic backward-compatible; if shapes change, add lightweight assertions in code paths that process rollouts.

## RL Workflow & Call Flow
- Entry: `a3c(env, configs)` instantiates the env-specific agent (`boom.py` or `rocket.py`) and dispatches to `train_a3c` or `test_a3c` based on `algo.mode`.
- Train loop: `train_a3c` resets `VecEnv`, rolls out `num_step` per worker via `get_action` (policy/value from model, env-driven candidate masking, temperature softmax), steps envs, and inserts transitions into `Buffer`. Preferences renew per worker on `done`.
- Update step: `train_a3c_impl` samples preference weights, `buffer.generate_batch_with_n_step()`, runs `forward_transition` to compute values/policies, then `calc_discounted_reward` (GAE) and `envelope_operator` for multi-objective weighting. Loss/backprop lives in `optimize_actor_critic`; learning rate annealed in `schedule_lr`; checkpoints via `save`; critic weights synchronized with `_model` in `sync_critic`.
- Metrics: `Status.update` logs PPA metrics, losses, learning rate, and BOOM action distributions to TensorBoard; follow this structure for new signals.
- Test loop: `test_a3c` mirrors rollout without updates; loads model in `set_mode` and still renews preferences on terminated workers.

## Commit & Pull Request Guidelines
- Commit messages typically use a bracketed tag prefix (e.g., `[MISC]: tune buffer`); keep subjects imperative and scoped to one concern.
- PRs should summarize behavioral impact, affected configs/paths, and reproduction commands (dry run + env stepping). Include log snippets if learning dynamics change.

## Security & Configuration Tips
- Avoid hard-coded absolute paths in any new configs or loaders; keep paths relative to the repo.
- Do not check in licensed datasets or generated artifacts; point configs to local `data/` or similar when applicable.
