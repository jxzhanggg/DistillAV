
# LRS3 base
ckpt="$PWD/../../checkpoints/lrs3_base_pt.pt"
# # LRS3 large
ckpt="$PWD/../../checkpoints/lrs3_large_pt.pt"
# # Vox base
ckpt="$PWD/../../checkpoints/vox_base_pt.pt"
# # Vox large
ckpt="$PWD/../../checkpoints/vox_large_pt.pt"

bash run_finetune_fn.sh $ckpt a
bash run_finetune_fn.sh $ckpt v