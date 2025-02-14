

cd ../../


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

fullexpdir=$1
input=$2


expdir=$(dirname "$fullexpdir")
expdir=$(dirname "$expdir")



if [ $input = 'a' ]; then
    modalities=["audio"]
    subexp="finetuned/audio_debug"

elif [ $input = 'v' ]; then
    modalities=["video"]
    subexp="finetuned/video_debug"

elif [ $input = "av" ]; then
    modalities=["audio","video"]
    subexp="finetuned/audiovideo_debug"
else
    echo  "modality must be a, v or av"
    exit
fi

echo $modalities

python hydra_train.py \
--config-dir conf/finetune \
    --config-name base_lrs3_30h \
    common.tensorboard_logdir=$expdir/$subexp \
    task.data=/train8/asrprg/jxzhang46/avsr_data_en/LRS3/30h_data \
    task.label_dir=/train8/asrprg/jxzhang46/avsr_data_en/LRS3/30h_data \
    task.tokenizer_bpe_model=/train8/asrprg/jxzhang46/avsr_data_en/LRS3/spm1000/spm_unigram1000.model \
    dataset.num_workers=3 \
    hydra.run.dir=$expdir/$subexp \
    model.w2v_path=$fullexpdir \
    task.modalities=$modalities \
    common.log_interval=10 \
    distributed_training.distributed_world_size=8 \
    optimization.update_freq=[1] \
    common.user_dir=`pwd`