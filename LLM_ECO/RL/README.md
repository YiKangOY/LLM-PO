# RL-Based ECO Optimization (A3C/PPO)

A reinforcement-learning stack for ECO optimization that mirrors the reference A3C framework (see `dse` example). The Stable-Baselines3 pipeline has been removed in favor of a custom actor-critic loop and discrete action space tailored to ECO commands.

## Layout
```
RL/
├── eco_a3c.py              # Train/test loop (A3C or PPO)
├── eco_agent.py            # ECO actor-critic agent (mirrors BOOM/Rocket agents)
├── eco_env.py              # A3C-friendly wrapper around ECOEnvironment
├── eco_action_space.py     # Discrete enumeration of ECO actions (24-d vectors)
├── eco_a3c_config.py       # Default config helper for the custom loop
├── rl_environment.py       # Core environment logic (metrics, rewards, execution)
├── state_encoder.py        # Reports/history → observation vector
├── action_decoder.py       # Action dict → TCL command
├── reward_calculator.py    # PPA-based reward computation
├── train_eco_a3c.py        # CLI entry for the custom A3C/PPO loop
├── train_eco_a3c_ppo.py    # CLI entry for the PPO-only loop
└── requirements.txt
```

## Quick Start
```bash
pip install -r requirements.txt

# Minimal training smoke (simulated execution, CPU)
python train_eco_a3c.py --mode train --episodes 5

# Full train command on GPU 0 (A3C)
python train_eco_a3c.py --mode train --episodes 5 --algo a3c --use-cuda --gpu-id 0 --design ECO_Vex

# Switch to PPO-style updates
python train_eco_a3c.py --mode train --episodes 50 --algo ppo --continue --use-cuda --gpu-id 0 --design mempool_tile_wrap

# Log per-episode runtime breakdowns in the JSONL training log
python train_eco_a3c.py --mode train --episodes 5 --algo a3c --log-runtime-breakdown

# PPO-only entrypoint (same flow, no --algo flag needed)
python train_eco_a3c_ppo.py --mode train --episodes 5
# Artifacts are PPO-tagged: logs use eco_a3c_ppo.log, checkpoints use eco_a3c_ppo_episode_*.pt,
# and run dirs are RL/run_dir/run_ppo_{env_id}

# Select a design-specific override set
python train_eco_a3c.py --mode train --episodes 5 --design ECO_Vex

# Test mode (loads algo.test.rl-model from eco_a3c_config.py)
python train_eco_a3c.py --mode test --design ECO_Vex
```

Programmatic entry:
```python
from eco_a3c import a3c
from eco_a3c_config import default_eco_a3c_config

cfg = default_eco_a3c_config()
cfg["algo"]["mode"] = "train"  # or "test"
a3c(cfg)
```

## Training Config (key knobs)
`eco_a3c_config.py` returns a dict in the same shape as the reference A3C code:
- `algo.algo`: `"a3c"` or `"ppo"`; switches update style.
- `algo.max-episode`: number of training episodes.
- `algo.num-parallel`: parallel envs (SubprocVecEnv).
- `algo.train.num-step`: steps per episode (mirrors the env iteration cap from `rl_config`).
- `algo.train.learning-rate`: optimizer LR (annealed each episode).
- `algo.train.temperature`: action sampling temperature (annealed upward).
- `algo.train.clip-grad-norm`: gradient norm clip.
- `algo.train.eval-frequency`: run eval every N episodes (0 disables); logs to `*_metrics_eval.csv`
  and `*_eval_<timestamp>.log`, artifacts under `base_path/RL/run_dir/run_eval_{episode}`.
- `algo.train.log-runtime-breakdown`: when enabled by `--log-runtime-breakdown`, writes
  per-episode `eda_tool_runtime_s`, `rl_training_inference_runtime_s`, and `other_runtime_s`
  into the JSONL train/eval logs.
- `algo.test.ppa-preference`: reward weighting for scalarization (vector length matches reward dims).
- Paths: `model-path` and `log-path` control output locations and checkpoint naming.
- Design overrides: `design_configs.py` maps `--design` to per-design `rl_config` overrides
  (action spaces, normalization ranges, base_path, iteration limits).

## Core Flow (mirrors reference A3C)
- `ECOActionSpace`: Enumerates discrete ECO commands; each index maps to the 24-d vector expected by `ECOEnvironment`’s decoder.
- `ECOEnv` (BasicEnv analogue): Wraps `ECOEnvironment`, exposes discrete actions, vector rewards, and terminal observation handling for SubprocVecEnv.
- `ECOAgent`: Custom actor-critic with preference sampling, buffer flattening, temperature annealing, and optional PPO clipping.
- `eco_a3c.train_a3c/test_a3c`: Manual rollout/update loop identical in spirit to `dse/algo/a3c/a3c.py` (preference generation, envelope operator, critic sync, LR scheduling).

## Notes
- Gymnasium: Uses `gymnasium` throughout; no `gym` fallback is required.
- Real execution: Configure `rl_environment` hooks (`execute_tcl_command`, report loaders) as before; the wrapping does not change metrics/reward logic.
- Models/Logs: Defaults point to `models/` and `logs/`; adjust via `default_eco_a3c_config`.
