# LLM-ECO

LLM-based ECO optimization with Agent and RL (A3C/PPO) baselines.

## Citation

```bibtex
@inproceedings{ouyang2026retrieve,
  title     = {Retrieve, Schedule, Reflect: LLM Agents for Chip QoR Optimization},
  author    = {Yikang Ouyang and Dongsheng Zuo and Yang Luo and Yuzhe Ma},
  booktitle = {ACM/IEEE International Symposium on Machine Learning for CAD},
  year      = {2026}
}
```

## Dependencies

### Common
- Python 3 (scripts are `#!/usr/bin/env python3`)
- Design data under `data/` (base paths are configured per module)
- If running real ECO commands, `pt_shell` must be available on `PATH`

### Agent (LLM_ECO/Agent)
Core Agent uses LangChain/LangGraph:
- `langchain-openai`
- `langchain-core`
- `langgraph`

RAG example dependencies (from `LLM_ECO/Agent/rag_example.py`):
```
pip install langchain langchain-community langchain-openai langgraph pypdf typing-extensions
```

Environment variables:
- `OPENAI_API_KEY` (required for OpenAI-backed models)
- `OPENAI_BASE_URL` (optional, custom endpoint)

### RL (LLM_ECO/RL)
Imports indicate these runtime dependencies:
- `torch`
- `numpy`
- `gymnasium`

Install the complete RL environment with:
```
pip install -r LLM_ECO/RL/requirements.txt
```

## Design-specific configuration

### Agent (LLM_ECO/Agent)
Edit `LLM_ECO/Agent/configs.py`:
- Add or update an entry in `DESIGN_CONFIG_OVERRIDES` with at least:
  - `base_path`: points at your design data under `data/`
  - `max_iterations_per_trace`
  - `objectives`
- Add matching entries in `design_runtime_budget` and
  `design_max_iterations_per_trace` so budgets and per-trace limits are set.

Example:
```
DESIGN_CONFIG_OVERRIDES["my_design"] = {
    "base_path": "data/my_design",
    "max_iterations_per_trace": 10,
    "objectives": DEFAULT_OBJECTIVES,
}
design_runtime_budget["my_design"] = 999999
design_max_iterations_per_trace["my_design"] = 10
```

### Agent variants
Each variant has its own config file (e.g. `LLM_ECO/Agent_noRAG/config_noRAG.py`)
with the same `DESIGN_CONFIG_OVERRIDES` and budget dictionaries. Update those
files if you run a variant.

### RL (LLM_ECO/RL)
Edit `LLM_ECO/RL/design_configs.py` and add a new entry to
`DESIGN_CONFIG_OVERRIDES`:
- `environment.base_path` and `environment.design_name` are required
- `environment.model_path`, `rl.max_iterations_per_episode`,
  `action_spaces`, and `normalization_ranges` are optional but commonly set
- Use `_build_normalization_ranges(runtime_budget, max_iterations)` if you
  want to keep the normalization ranges consistent with runtime budgets

RL also uses runtime budgets from `LLM_ECO/Agent/configs.py`
(`design_runtime_budget`), so add your design there if you want a non-default
budget.

## Running

### Agent (LLM_ECO/Agent)
Main agent flow:
```
cd LLM_ECO/Agent
python eco_ppa_agent.py
```

RAG wiring check:
```
cd LLM_ECO/Agent
python rag_example.py
```

PT server run example:
```
cd LLM_ECO/Agent
python eco_ppa_agent_ptserver.py --design aes_256 --iterations 10 --rounds 10
```

Configuration:
- Base paths and design overrides live in `LLM_ECO/Agent/configs.py`

### Agent variants (no RAG / no reflection / no both)
Each variant has its own directory and PT-server entrypoint.

No RAG:
```
cd LLM_ECO/Agent_noRAG
python eco_ppa_agent_norag_ptserver.py
```

No reflection:
```
cd LLM_ECO/Agent_noReflection
python eco_ppa_agent_noreflect_ptserver.py
```

No RAG + no reflection:
```
cd LLM_ECO/Agent_noBoth
python eco_ppa_agent_noboth_ptserver.py
```

Each variant has its own config file in the same folder (e.g. `config_noRAG.py`).

### RL (LLM_ECO/RL)
Minimal training smoke:
```
cd LLM_ECO/RL
python train_eco_a3c.py --mode train --episodes 5
```


Test with a design override:
```
cd LLM_ECO/RL
python train_eco_a3c.py --mode test --design ECO_Vex
```

Configuration:
- Base path, PT server toggles, and defaults live in `LLM_ECO/RL/rl_config.py`
- Design overrides live in `LLM_ECO/RL/design_configs.py`
