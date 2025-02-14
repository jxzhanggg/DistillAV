

export CUDA_VISIBLE_DEVICES=0
source ~/bashrc_home.sh
conda activate av2vec


modal="audio"

ckpt="/train8/asrprg/jxzhang46/exp/av2vec_mlm/9_wavlm_conformer_alignmtl/finetuned/${modal}/checkpoints/checkpoint_last.pt"

results_path=$(dirname $ckpt)
results_path=$(dirname $results_path)/decode

python infer_s2s.py \
    --config-dir ./conf/ --config-name s2s_decode.yaml \
    dataset.gen_subset=test \
    common_eval.path=$ckpt \
    common_eval.results_path=$results_path \
    override.modalities=[\'$modal\'] \
    override.data="/train8/asrprg/jxzhang46/avsr_data_en/LRS3/30h_data" \
    override.label_dir="/train8/asrprg/jxzhang46/avsr_data_en/LRS3/30h_data" \
    common.user_dir=`pwd` \
