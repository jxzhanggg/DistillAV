
cd ..



source ~/bashrc_home.sh
conda activate av2vec



export OPENBLAS_NUM_THREADS=1
export NCCL_DEBUG_SUBSYS=P2P,SHM,NET
export NCCL_IB_GID_INDEX=3
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_NET_GDR_LEVEL=PXB
export NCCL_P2P_LEVEL=NVL
export NCCL_LL_THRESHOLD=0
export OMP_NUM_THREADS=1


export HYDRA_FULL_ERROR=1


export MASTER_ADDR="localhost"
export MASTER_PORT='23456'
export WORLD_SIZE=1
export RANK=0
export HYDRA_FULL_ERROR=1

expdir=$1

python hydra_train.py --config-dir conf/pretrain \
    --config-name noise_base_lrs3_iter5 \
    common.tensorboard_logdir=$expdir \
    task.data=/train8/asrprg/jxzhang46/avsr_data_en/LRS3/433h_data \
    task.label_dir=/train8/asrprg/jxzhang46/avsr_data_en/LRS3/dump_iter4_auth \
    hydra.run.dir=$expdir \
    task.noise_wav=/train8/asrprg/jxzhang46/musan/musan/tsv/all \
    task.noise_prob=0.1 \
    task.noise_snr=5 \
    dataset.max_tokens=2200 \
    dataset.num_workers=6 \
    model.masking_type="feature" \
    optimization.max_update=200000 \
    checkpoint.save_interval_updates=20000 \
    checkpoint.keep_interval_updates=2 \
    lr_scheduler._name="tri_stage" \
    +lr_scheduler.phase_ratio=[0.05,0.85,0.1] \
    common.log_interval=50 \
    distributed_training.distributed_world_size=8 \
    optimization.update_freq=[2] \
    model.label_rate=25 \
    common.user_dir=`pwd` 