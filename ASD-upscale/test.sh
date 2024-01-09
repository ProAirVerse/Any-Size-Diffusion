CKPT_PATH=/home/zhengqingping/git_repo/ASD-upscale/models/stablesr_000117.ckpt
VQGANCKPT_PATH=/home/zhengqingping/git_repo/ASD-upscale/models/vqgan_cfw_00011.ckpt
INPUT_PATH=/home/zhengqingping/git_repo/datasets/Set5/LR_bicubic/X2/babyx2.png

# CUDA_VISIBLE_DEVICES=7 python scripts/sr_val_ddpm_text_T_vqganfin_old.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt $CKPT_PATH --vqgan_ckpt $VQGANCKPT_PATH --init-img INPUT_PATH --outdir OUT_DIR --ddpm_steps 200 --dec_w 0.5 --colorfix_type adain

CUDA_VISIBLE_DEVICES=5,7 scripts/upscale_accelerate.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt $CKPT_PATH --vqgan_ckpt $VQGANCKPT_PATH --init-img INPUT_PATH --outdir OUT_DIR --ddpm_steps 200 --dec_w 0.5 --colorfix_type adain
