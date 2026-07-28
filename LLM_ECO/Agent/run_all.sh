# nohup python eco_ppa_agent_ptserver.py --design NV_NVDLA_partition_m  --iterations 10 --rounds 10 > data/NV_NVDLA_partition_m/NV_NVDLA_partition_m.log 2>&1 &
# echo "Started NV_NVDLA_partition_m -> data/NV_NVDLA_partition_m/NV_NVDLA_partition_m.log"

nohup python eco_ppa_agent_ptserver.py --design NV_NVDLA_partition_p  --iterations 10 --rounds 10 > data/NV_NVDLA_partition_p/NV_NVDLA_partition_p.log 2>&1 &
echo "Started NV_NVDLA_partition_p -> data/NV_NVDLA_partition_p/NV_NVDLA_partition_p.log"
nohup python eco_ppa_agent_ptserver.py --design ariane136  --iterations 10 --rounds 10 > data/ariane136/ariane136.log 2>&1 &
echo "Started ariane136 -> data/ariane136/ariane136.log"
nohup python eco_ppa_agent_ptserver.py --design mempool_tile_wrap  --iterations 10 --rounds 10 > data/mempool_tile_wrap/mempool_tile_wrap.log 2>&1 &
echo "Started mempool_tile_wrap -> data/mempool_tile_wrap/mempool_tile_wrap.log"
nohup python eco_ppa_agent_ptserver.py --design aes_256  --iterations 10 --rounds 10 > data/aes_256/aes_256.log 2>&1 &
echo "Started aes_256 -> data/aes_256/aes_256.log"
sleep 600
# nohup python eco_ppa_agent_ptserver.py --design hidden1  --iterations 10 --rounds 10 > data/hidden1/hidden1.log 2>&1 &
# echo "Started hidden1 -> data/hidden1/hidden1.log"

# nohup python eco_ppa_agent_ptserver.py --design hidden2  --iterations 10 --rounds 10 > data/hidden2/hidden2.log 2>&1 &
# echo "Started hidden2 -> data/hidden2/hidden2.log"
# nohup python eco_ppa_agent_ptserver.py --design hidden3  --iterations 10 --rounds 10 > data/hidden3/hidden3.log 2>&1 &
# echo "Started hidden3 -> data/hidden3/hidden3.log"
# nohup python eco_ppa_agent_ptserver.py --design hidden5  --iterations 10 --rounds 10 > data/hidden5/hidden5.log 2>&1 &
# echo "Started hidden5 -> data/hidden5/hidden5.log"
