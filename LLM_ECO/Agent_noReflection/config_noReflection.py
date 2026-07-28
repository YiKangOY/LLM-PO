import os

BUFFER_LIST = "{BUFx10_ASAP7_75t_L BUFx12_ASAP7_75t_L BUFx12f_ASAP7_75t_L BUFx16f_ASAP7_75t_L BUFx2_ASAP7_75t_L BUFx3_ASAP7_75t_L BUFx4_ASAP7_75t_L BUFx4f_ASAP7_75t_L BUFx5_ASAP7_75t_L BUFx6f_ASAP7_75t_L BUFx8_ASAP7_75t_L CKINVDCx10_ASAP7_75t_L CKINVDCx11_ASAP7_75t_L CKINVDCx12_ASAP7_75t_L CKINVDCx14_ASAP7_75t_L CKINVDCx16_ASAP7_75t_L CKINVDCx20_ASAP7_75t_L CKINVDCx5p33_ASAP7_75t_L CKINVDCx6p67_ASAP7_75t_L CKINVDCx8_ASAP7_75t_L CKINVDCx9p33_ASAP7_75t_L BUFx10_ASAP7_75t_SL BUFx12_ASAP7_75t_SL BUFx12f_ASAP7_75t_SL BUFx16f_ASAP7_75t_SL BUFx2_ASAP7_75t_SL BUFx3_ASAP7_75t_SL BUFx4_ASAP7_75t_SL BUFx4f_ASAP7_75t_SL BUFx5_ASAP7_75t_SL BUFx6f_ASAP7_75t_SL BUFx8_ASAP7_75t_SL CKINVDCx10_ASAP7_75t_SL CKINVDCx11_ASAP7_75t_SL CKINVDCx12_ASAP7_75t_SL CKINVDCx14_ASAP7_75t_SL CKINVDCx16_ASAP7_75t_SL CKINVDCx20_ASAP7_75t_SL CKINVDCx5p33_ASAP7_75t_SL CKINVDCx6p67_ASAP7_75t_SL CKINVDCx8_ASAP7_75t_SL CKINVDCx9p33_ASAP7_75t_SL BUFx10_ASAP7_75t_R BUFx12_ASAP7_75t_R BUFx12f_ASAP7_75t_R BUFx16f_ASAP7_75t_R BUFx2_ASAP7_75t_R BUFx3_ASAP7_75t_R BUFx4_ASAP7_75t_R BUFx4f_ASAP7_75t_R BUFx5_ASAP7_75t_R BUFx6f_ASAP7_75t_R BUFx8_ASAP7_75t_R CKINVDCx10_ASAP7_75t_R CKINVDCx11_ASAP7_75t_R CKINVDCx12_ASAP7_75t_R CKINVDCx14_ASAP7_75t_R CKINVDCx16_ASAP7_75t_R CKINVDCx20_ASAP7_75t_R CKINVDCx5p33_ASAP7_75t_R CKINVDCx6p67_ASAP7_75t_R CKINVDCx8_ASAP7_75t_R CKINVDCx9p33_ASAP7_75t_R}"
base_path = "data/hidden2"
AGENT_DIR_NAME = "Agent"
USE_OPENAI_REASONING = False

def get_agent_dir():
    return os.path.join(base_path, AGENT_DIR_NAME)

def get_agent_logs_dir():
    return os.path.join(get_agent_dir(), "logs")

def get_agent_run_dir():
    return os.path.join(get_agent_dir(), "run_dir")
DEFAULT_OBJECTIVES = "Optimize timing to fix TNS and WNS violations, with minimal power and area overheads."
DESIGN_CONFIG_OVERRIDES = {
    "NV_NVDLA_partition_m": {
        "base_path": "data/NVDLA_partition_m",
        "max_iterations_per_trace": 10,
        "objectives": DEFAULT_OBJECTIVES,
    },
    "NV_NVDLA_partition_p": {
        "base_path": "data/NVDLA_partition_p",
        "max_iterations_per_trace": 10,
        "objectives": DEFAULT_OBJECTIVES,
    },
    "ariane136": {
        "base_path": "data/ariane136",
        "max_iterations_per_trace": 10,
        "objectives": DEFAULT_OBJECTIVES,
    },
    "mempool_tile_wrap": {
        "base_path": "data/mempool_tile_wrap",
        "max_iterations_per_trace": 10,
        "objectives": DEFAULT_OBJECTIVES,
    },
    "aes_256": {
        "base_path": "data/aes_256",
        "max_iterations_per_trace": 10,
        "objectives": DEFAULT_OBJECTIVES,
    },
    "hidden1": {
        "base_path": "data/hidden1",
        "max_iterations_per_trace": 10,
        "objectives": DEFAULT_OBJECTIVES,
    },
    "hidden2": {
        "base_path": "data/hidden2",
        "max_iterations_per_trace": 10,
        "objectives": DEFAULT_OBJECTIVES,
    },
    "hidden3": {
        "base_path": "data/hidden3",
        "max_iterations_per_trace": 10,
        "objectives": DEFAULT_OBJECTIVES,
    },
    "hidden5": {
        "base_path": "data/hidden5",
        "max_iterations_per_trace": 10,
        "objectives": DEFAULT_OBJECTIVES,
    },
    "NV_NVDLA_partition_m_test": {
        "base_path": "data/NVDLA_partition_m_test",
        "max_iterations_per_trace": 1,
        "objectives": DEFAULT_OBJECTIVES,
    },
}
design_runtime_budget = {
    "NV_NVDLA_partition_m":999999,
    "NV_NVDLA_partition_p":999999,
    "aes_256": 999999,
    "ariane136": 999999,
    "mempool_tile_wrap": 999999,
    "hidden1": 999999,
    "hidden2": 999999,
    "hidden3": 999999,
    "hidden5": 999999,
}
design_max_iterations_per_trace = {
    "NV_NVDLA_partition_m":10,
    "NV_NVDLA_partition_p":10,
    "ariane136":10,
    "mempool_tile_wrap":10,
    "aes_256":10,
    "hidden1":10,
    "hidden2":10,
    "hidden3":10,
    "hidden5":10,
    "NV_NVDLA_partition_m_test":1,
}

TARGET_STRATEGY_DOC_PATH = "LLM_ECO/docs/opt_target_strategies.txt"
COMMAND_STRATEGY_DOC_PATH = "LLM_ECO/docs/opt_command_strategies.txt"

# Default number of comparison pairs for short-term trace reviews per round
number_of_pairs_default = 5

PT_SERVER_CONFIG = {
    "use_pt_server": True,
    "host": "127.0.0.1",
    "base_port": 12900,
    "port_stride": 1,
    "start_timeout_s": 1200.0,
    "command_timeout_s": 1200.0,
}
