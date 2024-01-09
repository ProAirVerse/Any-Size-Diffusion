import os, sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import subprocess
import argparse
import glob
import random

parser = argparse.ArgumentParser("Launcher for ASD upscaling")

parser.add_argument("upscale", type=int, default=2)
parser.add_argument("input_dir", type=str, default="", help="input directory")
parser.add_argument("output_dir", type=str, default="", help="output directory")
parser.add_argument("--gpus", type=int, nargs="+", default=[0])
parser.add_argument("--ddpm_steps", type=int, default=100)
parser.add_argument("--overlap", type=int, default=32)
parser.add_argument("--offset_stride", type=int, default=32)
parser.add_argument("--encoder_tile_size", type=int, default=1680)
parser.add_argument("--decoder_tile_size", type=int, default=196)
parser.add_argument("--script", type=str, default="scripts/upscale_accelerate.py")

def call(cmd,  gpu_id, generate_cmd_file=False):
    gpu_cmd = "CUDA_VISIBLE_DEVICES={} ".format(gpu_id % 8) + cmd
    if generate_cmd_file:
        with open(f"launch_{gpu_id}.sh", "w") as f:
            f.write(gpu_cmd)
    subprocess.check_call(gpu_cmd, shell=True)

def split_array(array, n):
    random.shuffle(array)
    split_size = len(array) // n
    # leftovers = len(array) % n

    result = []
    start = 0
    for i in range(n):
        size = split_size  # 根据剩余元素决定分片大小
        if i == (n-1):
            result.append(array[start:])
        else:
            result.append(array[start:start + size])
            start += size

    return result

if __name__ == "__main__":
    args = parser.parse_args()
    if len(args.gpus) == 1:

        cmd = """CUDA_VISIBLE_DEVICES={} python {} \
            --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt ./stablesr_000117.ckpt \
            --vqgan_ckpt ./vqgan_cfw_00011.ckpt --init-img {} \
            --outdir {} --ddpm_steps {} --dec_w 0.5 --colorfix_type adain --upscale {} --offset_stride {} --encoder_tile_size {} --decoder_tile_size {}""".format(args.gpus[0],
                                                                                                  args.script,
                                                                                                  args.input_dir,
                                                                                                  args.output_dir,
                                                                                                  args.ddpm_steps,
                                                                                                  args.upscale,
                                                                                                  args.offset_stride,
                                                                                                  args.encoder_tile_size,
                                                                                                  args.decoder_tile_size)

        cmd += " --noise_reference --random_offset"
        subprocess.call(cmd, shell=True)
    else:
        ## first split input path ##
        input_dir = args.input_dir
        image_list = sorted(glob.glob(os.path.join(input_dir, "*")))
        ngpus = len(args.gpus)
        image_partitions = split_array(image_list, ngpus)
        cmds = []
        for sub_list in image_partitions:
            cmd = "python {} --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt ./stablesr_000117.ckpt".format(args.script)
            cmd += " --vqgan_ckpt ./vqgan_cfw_00011.ckpt --outdir {} --ddpm_steps {} --dec_w 0.5 --colorfix_type adain --upscale {} --offset_stride {} --encoder_tile_size {} --decoder_tile_size {}".format(
                                                                                                                                            args.output_dir,
                                                                                                                                            args.ddpm_steps,
                                                                                                                                            args.upscale,
                                                                                                                                            args.offset_stride,
                                                                                                                                            args.encoder_tile_size,
                                                                                                                                            args.decoder_tile_size)

            cmd += " --noise_reference --random_offset"
            cmd += """ --image_list {}""".format(""" """.join(sub_list))
            cmds.append(cmd)
            print("sub list length is {}".format(len(sub_list)))
        print("call using image list, image list total length is {}".format(len(image_list)))
        with ThreadPoolExecutor() as executor:
            executor.map(call, cmds, args.gpus)
