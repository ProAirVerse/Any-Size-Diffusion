"""make variations of input image"""

import argparse, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import trange
from itertools import islice
from einops import repeat
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
import copy
import torch.nn.functional as F
from util_image import ImageSpliterTh
from pathlib import Path
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization
import os
# os.environ['CURL_CA_BUNDLE'] = ""
from utils.tiled_vae.vae_hook import VAEHook, VQEncoderHook, VQDecoderHook


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]   #[250,]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def read_image(im_path):
    im = np.array(Image.open(im_path).convert("RGB"))
    im = im.astype(np.float32)/255.0
    im = im[None].transpose(0,3,1,2)
    im = (torch.from_numpy(im) - 0.5) / 0.5

    return im.cuda()

def hook_vae(vae, opt):
    """
    modify the forward function of vae encoder and vae decoder
    """
    encoder = vae.encoder
    decoder = vae.decoder
    kwargs = {
        'fast_decoder': True,
        'fast_encoder': True
    }
    if not hasattr(encoder, 'original_forward'): setattr(encoder, 'original_forward', encoder.forward)
    if not hasattr(decoder, 'original_forward'): setattr(decoder, 'original_forward', decoder.forward)

    encoder.forward = VAEHook(encoder, opt.encoder_tile_size, is_decoder=False, **kwargs)
    decoder.forward = VAEHook(decoder, opt.decoder_tile_size, is_decoder=True, **kwargs)

def hook_vqmodel(vqmodel, opt):
    encoder = vqmodel.encoder
    decoder = vqmodel.decoder
    kwargs = {
        'fast_decoder': False,
        'fast_encoder': True
    }
    if not hasattr(encoder, 'original_forward'): setattr(encoder, 'original_forward', encoder.forward)
    if not hasattr(decoder, 'original_forward'): setattr(decoder, 'original_forward', decoder.forward)

    encoder.forward = VQEncoderHook(encoder, opt.encoder_tile_size, is_decoder=False, **kwargs)
    decoder.forward = VQDecoderHook(decoder, opt.decoder_tile_size, is_decoder=True, **kwargs)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_list",
        type=str,
        nargs="+",
        help="paths to input image list",
        default=None)
    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image",
        default="inputs/user_upload"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/user_upload"
    )
    parser.add_argument(
        "--ddpm_steps",
        type=int,
        default=1000,
        help="number of ddpm sampling steps",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./stablesr_000117.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--vqgan_ckpt",
        type=str,
        default="./vqgan_cfw_00011.ckpt",
        help="path to checkpoint of VQGAN model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--dec_w",
        type=float,
        default=0.5,
        help="weight for combining VQGAN and Diffusion",
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=32,
        help="tile overlap size (in latent)",
    )
    parser.add_argument(
        "--upscale",
        type=float,
        default=4.0,
        help="upsample scale",
    )
    parser.add_argument(
        "--colorfix_type",
        type=str,
        default="nofix",
        help="Color fix type to adjust the color of HR result according to LR input: adain (used in paper); wavelet; nofix",
    )
    parser.add_argument(
        "--encoder_tile_size",
        type=int,
        default=1680,
        help="encoder tile size"
    )
    parser.add_argument(
        "--decoder_tile_size",
        type=int,
        default=196,
        help="decoder tile size"
    )
    parser.add_argument(
        "--vqgantile_stride",
        type=int,
        default=1000,
        help="the stride for tile operation before VQGAN decoder (in pixel)",
    )
    parser.add_argument(
        "--vqgantile_size",
        type=int,
        default=1280,
        help="the size for tile operation before VQGAN decoder (in pixel)",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
        help="input size",
    )
    parser.add_argument("--accelerate", action="store_true", default=False,
                        help="use accelerate tech")
    parser.add_argument("--noise_reference", "-nr", action="store_true", default=False)
    parser.add_argument("--offset_stride", type=int, default=32)
    parser.add_argument("--random_reset_step", type=int, default=4)
    parser.add_argument("--random_offset", action="store_true", default=False)
    parser.add_argument("--complete_offset", action="store_true", default=False)
    parser.add_argument("--noise_ema", type=float, default=None) ## noise EMA during noise prediction
    opt, others = parser.parse_known_args()
    seed_everything(opt.seed)

    print('>>>>>>>>>>color correction>>>>>>>>>>>')
    if opt.colorfix_type == 'adain':
        print('Use adain color correction')
    elif opt.colorfix_type == 'wavelet':
        print('Use wavelet color correction')
    else:
        print('No color correction')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    model.configs = config

    vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
    vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)
    vq_model = vq_model.to(device)
    vq_model.decoder.fusion_w = opt.dec_w

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    if opt.accelerate:
        print("====> using accelerate!!!")
    if opt.noise_ema is not None:
        print("====> using noise ema!!! noise ema is {}".format(opt.noise_ema))

    if opt.image_list is not None:
        print("using image list as inputs")
        images_path = copy.deepcopy(opt.image_list)
        output_list = [x.rsplit(".", 1)[0] for x in os.listdir(outpath)]
        for image_path in opt.image_list:
            image_name = os.path.basename(image_path)
            image_name = image_name.rsplit(".", 1)[0] ## remove effect of .png & .jpg
            if image_name in output_list:
                print("skip {} because exists".format(image_name))
                images_path.remove(image_path)
        print(f"Found {len(images_path)} inputs.")

    else:
        images_path_ori = sorted(glob.glob(os.path.join(opt.init_img, "*")))
        images_path = copy.deepcopy(images_path_ori)
        output_list = [x.rsplit(".", 1)[0] for x in os.listdir(outpath)]
        for item in images_path_ori:
            img_name = item.split('/')[-1]
            img_name = img_name.rsplit(".", 1)[0]
            if img_name in output_list:
                images_path.remove(item)
        print(f"Found {len(images_path)} inputs.")

    model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
    model.num_timesteps = 1000

    sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)

    use_timesteps = set(space_timesteps(1000, [opt.ddpm_steps]))
    last_alpha_cumprod = 1.0
    new_betas = []
    timestep_map = []
    for i, alpha_cumprod in enumerate(model.alphas_cumprod):
        ## for example, ddpm steps=200
        ## then, take [0, 4, 8, ...1000] and the corresponding betas
        if i in use_timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
            timestep_map.append(i)
    new_betas = [beta.data.cpu().numpy() for beta in new_betas]
    model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
    model.num_timesteps = 1000
    model.ori_timesteps = list(use_timesteps)
    model.ori_timesteps.sort()
    model = model.to(device)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    niqe_list = []
    ## hook tiled-vae encode
    hook_vae(model.first_stage_model, opt)
    hook_vqmodel(vq_model, opt)
    with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
        tic = time.time()
        all_samples = list()
        for n in trange(len(images_path), desc="Sampling"):
            if (n + 1) % opt.n_samples == 1 or opt.n_samples == 1:
                cur_image = read_image(images_path[n])
                size_min = min(cur_image.size(-1), cur_image.size(-2))
                upsample_scale = max(opt.input_size/size_min, opt.upscale)
                cur_image = F.interpolate(
                            cur_image,
                            size=(int(cur_image.size(-2)*upsample_scale),
                                  int(cur_image.size(-1)*upsample_scale)),
                            mode='bicubic',
                            )
                cur_image = cur_image.clamp(-1, 1)
                im_lq_bs = [cur_image, ]  # 1 x c x h x w, [-1, 1]
                im_path_bs = [images_path[n], ]
            else:
                cur_image = read_image(images_path[n])
                size_min = min(cur_image.size(-1), cur_image.size(-2))
                upsample_scale = max(opt.input_size/size_min, opt.upscale)
                ## first upscale to large image
                cur_image = F.interpolate(
                            cur_image,
                            size=(int(cur_image.size(-2)*upsample_scale),
                                  int(cur_image.size(-1)*upsample_scale)),
                            mode='bicubic',
                            )
                cur_image = cur_image.clamp(-1, 1)
                im_lq_bs.append(cur_image) # 1 x c x h x w, [-1, 1]
                im_path_bs.append(images_path[n]) # 1 x c x h x w, [-1, 1]

            if (n + 1) % opt.n_samples == 0 or (n+1) == len(images_path):
                im_lq_bs = torch.cat(im_lq_bs, dim=0)
                ori_h, ori_w = im_lq_bs.shape[2:]
                ref_patch = None
                if not (ori_h % 32 == 0 and ori_w % 32 == 0):
                    flag_pad = True
                    pad_h = ((ori_h // 32) + 1) * 32 - ori_h
                    pad_w = ((ori_w // 32) + 1) * 32 - ori_w
                    im_lq_bs = F.pad(im_lq_bs, pad=(0, pad_w, 0, pad_h), mode='reflect')
                else:
                    flag_pad = False


                ## upscale
                init_latent = model.get_first_stage_encoding(model.encode_first_stage(im_lq_bs))  # move to latent space
                print("init latent shape: {} | device {}".format(init_latent.shape, init_latent.device))
                text_init = ['']*opt.n_samples
                semantic_c = model.cond_stage_model(text_init)
                noise = torch.randn_like(init_latent)
                # If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
                t = repeat(torch.tensor([999]), '1 -> b', b=im_lq_bs.size(0))
                t = t.to(device).long()
                x_T = model.q_sample_respace(x_start=init_latent, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
                samples, _ = model.sample_canvas_accelerate(cond=semantic_c, struct_cond=init_latent,
                                                            batch_size=im_lq_bs.size(0),
                                                            timesteps=opt.ddpm_steps,
                                                            time_replace=opt.ddpm_steps, x_T=x_T,
                                                            return_intermediates=True,
                                                            tile_size=int(opt.input_size / 8),
                                                            ## 64 in latent space
                                                            tile_overlap=opt.tile_overlap,
                                                            batch_size_sample=opt.n_samples,
                                                            offset_stride=opt.offset_stride,
                                                            complete_offset=opt.complete_offset,
                                                            use_noise_reference=opt.noise_reference,
                                                            noise_ema=opt.noise_ema,
                                                            random_offset=opt.random_offset,
                                                            random_reset_step=opt.random_reset_step)

                x_samples = model.decode_first_stage(samples)
                if opt.colorfix_type == 'adain':
                    x_samples = adaptive_instance_normalization(x_samples, im_lq_bs)
                elif opt.colorfix_type == 'wavelet':
                    x_samples = wavelet_reconstruction(x_samples, im_lq_bs)
                im_sr = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                if upsample_scale > opt.upscale: ## first up-sample, then down-sample
                    im_sr = F.interpolate(
                                im_sr,
                                size=(int(im_lq_bs.size(-2)*opt.upscale/upsample_scale),
                                      int(im_lq_bs.size(-1)*opt.upscale/upsample_scale)),
                                mode='bicubic',
                                )
                    im_sr = torch.clamp(im_sr, min=0.0, max=1.0)

                im_sr = im_sr.cpu().numpy().transpose(0,2,3,1)*255   # b x h x w x c

                if flag_pad:
                    im_sr = im_sr[:, :ori_h, :ori_w, ]

                for jj in range(im_lq_bs.shape[0]):
                    img_name = str(Path(im_path_bs[jj]).name)
                    basename = os.path.splitext(os.path.basename(img_name))[0]
                    outpath = str(Path(opt.outdir)) + '/' + basename + '.png'
                    Image.fromarray(im_sr[jj, ].astype(np.uint8)).save(outpath)

        toc = time.time()
        print("total time is {:.2f} s".format(toc - tic))

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
