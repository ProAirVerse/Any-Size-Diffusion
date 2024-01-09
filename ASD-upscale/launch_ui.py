import gradio as gr
import base64
from io import BytesIO
import gc
import torch
import math
import os
import safetensors
from utils.ui.share_btn import community_icon_html, loading_icon_html, share_js
import numpy as np

# from ldm.util import instantiate_from_config
os.environ['CURL_CA_BUNDLE'] = ""  ## added to prevent SSLError certificate.
offload_base = os.getenv("OFFLOAD_BASE", "true").lower() == "true"

# Generate how many images by default
default_num_images = int(os.getenv("DEFAULT_NUM_IMAGES", "1"))
if default_num_images < 1:
    default_num_images = 1

# Create public link
share = os.getenv("SHARE", "false").lower() == "true"
server_name = os.getenv("SERVER_NAME", "10.155.193.184")
server_port = int(os.getenv("PORT", 7863))


# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

# NOTE: we do not have word list filtering in this gradio demo


def upscale(images, scale_factor=2, model=None, vq_model=None, overlap=64, upscale_method="implicit"):
    opt.upscale = float(scale_factor)
    if upscale_method == "implicit":
        print("use implicit overlap")
        opt.accelerate = True
        opt.random_offset = True
        opt.noise_reference = True
        opt.offset_stride = overlap
        upscaled_images = main(opt, images, model, vq_model)
    else:
        print("use explicit overlap")
        opt.tile_overlap = overlap
        opt.accelerate = False
        opt.random_offset = False
        upscaled_images = main(opt, images, model, vq_model)
    return upscaled_images

def infer(prompt, negative, scale, samples=1, steps=50, seed=-1,
          height=1024, width=1024, method="ASD", overlap=32, upscale_method="implicit"):
    if method == "ARAD":
        print("direct generate w/o upscale")
        g = torch.Generator(device="cuda")
        target_width, target_height = width, height
        if seed != -1:
            g.manual_seed(seed)
        else:
            g.seed()

        images_b64_list = []

        images = pipe(prompt=prompt, negative_prompt=negative, guidance_scale=scale, num_inference_steps=steps,
                      generator=g,
                      height=height, width=width).images

        gc.collect()
        torch.cuda.empty_cache()
    else:
        ## reduce size
        height, width, target_height, target_width, scale_factor = get_size_and_upscale_factor(height, width)
        prompt, negative = [prompt] * samples, [negative] * samples
        g = torch.Generator(device="cuda")
        if seed != -1:
            g.manual_seed(seed)
        else:
            g.seed()

        images_b64_list = []

        images = pipe(prompt=prompt, negative_prompt=negative, guidance_scale=scale, num_inference_steps=steps,
                       generator=g,
                      height=height, width=width).images

        gc.collect()
        torch.cuda.empty_cache()
        if scale_factor > 1:
            print("using upscale")
            images = upscale(images, scale_factor=scale_factor, model=upscale_model, vq_model=vq_model,
                                    overlap=overlap, upscale_method=upscale_method)
            gc.collect()
            torch.cuda.empty_cache()
    for image in images:
        image = image.resize((target_width, target_height))
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        image_b64 = (f"data:image/jpeg;base64,{img_str}")
        images_b64_list.append(image_b64)

    return images_b64_list


css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        #advanced-btn {
            font-size: .7rem !important;
            line-height: 19px;
            margin-top: 12px;
            margin-bottom: 12px;
            padding: 2px 8px;
            border-radius: 14px !important;
        }
        #advanced-options {
            display: none;
            margin-bottom: 20px;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        #share-btn-container {
            display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
            margin-top: 10px;
            margin-left: auto;
        }
        #share-btn {
            all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;right:0;
        }
        #share-btn * {
            all: unset;
        }
        #share-btn-container div:nth-child(-n+2){
            width: auto !important;
            min-height: 0px !important;
        }
        #share-btn-container .wrap {
            display: none !important;
        }

        .gr-form{
            flex: 1 1 50%; border-top-right-radius: 0; border-bottom-right-radius: 0;
        }
        #prompt-container{
            gap: 0;
        }
        #image-input, #prompt-text-input, #negative-prompt-text-input{padding: .45rem 0.625rem}
        #component-16{border-top-width: 1px!important;margin-top: 1em}
        .image_duplication{position: absolute; width: 100px; left: 50px}
"""

block = gr.Blocks(css=css)

with block:
    gr.HTML(
        """
            <div style="text-align: center; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <svg
                  width="0.65em"
                  height="0.65em"
                  viewBox="0 0 115 115"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect width="23" height="23" fill="white"></rect>
                  <rect y="69" width="23" height="23" fill="white"></rect>
                  <rect x="23" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="46" width="23" height="23" fill="white"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" width="23" height="23" fill="black"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="92" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="115" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="46" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="46" y="46" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="115" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="23" y="46" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="23" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="black"></rect>
                </svg>
                <h1 style="font-weight: 900; margin-bottom: 7px;margin-top:5px">
                  Any-Size-Diffusion Demo
                </h1>
              </div>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
            with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
                with gr.Column():
                    image = gr.Image(
                        label="Upload your image",
                        show_label=False,
                        max_lines=1,
                        placeholder="Upload your image",
                        elem_id="image-input",
                    ).style(
                        border=(True, False, True, True),
                        rounded=(True, False, False, True),
                        container=False,
                    )
                    text = gr.Textbox(
                        label="Enter your prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter your prompt",
                        elem_id="prompt-text-input",
                    ).style(
                        border=(True, False, True, True),
                        rounded=(True, False, False, True),
                        container=False,
                    )
                    negative = gr.Textbox(
                        label="Enter your negative prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter a negative prompt",
                        elem_id="negative-prompt-text-input",
                    ).style(
                        border=(True, False, True, True),
                        rounded=(True, False, False, True),
                        container=False,
                    )
                btn = gr.Button("Generate image").style(
                    margin=False,
                    rounded=(False, True, True, False),
                    full_width=False,
                )

        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(grid=[2], height="auto")

        with gr.Group(elem_id="container-advanced-btns"):
            # advanced_button = gr.Button("Advanced options", elem_id="advanced-btn")
            with gr.Group(elem_id="share-btn-container"):
                community_icon = gr.HTML(community_icon_html)
                loading_icon = gr.HTML(loading_icon_html)
                share_button = gr.Button("Share to community", elem_id="share-btn")

        with gr.Accordion("Advanced settings", open=True):
            #    gr.Markdown("Advanced settings are temporarily unavailable")
            samples = gr.Slider(label="Images", minimum=1, maximum=max(4, default_num_images), value=default_num_images,
                                step=1)
            steps = gr.Slider(label="Steps", minimum=1, maximum=250, value=50, step=1)
            width = gr.Slider(label="Width", minimum=128, maximum=8192, value=512, step=8)
            height = gr.Slider(label="Height", minimum=128, maximum=8192, value=512, step=8)

            guidance_scale = gr.Slider(
                label="Guidance Scale", minimum=0, maximum=50, value=7, step=0.1
            )
            method = gr.Dropdown(["ASD", "ARAD"], label="Method", value="ASD")
            overlap = gr.Slider(label="overlap", minimum=0, maximum=64, value=32, step=1)
            upscale_method = gr.Dropdown(["explicit", "implicit"], label="upscale-overlap-tile", value="implicit")
            seed = gr.Slider(
                label="Seed",
                minimum=-1,
                maximum=2147483647,
                step=1,
                randomize=True,
            )

        # ex = gr.Examples(examples=examples, fn=infer, inputs=[text, negative, guidance_scale],
        #                  outputs=[gallery, community_icon, loading_icon, share_button], cache_examples=False)
        # ex.dataset.headers = [""]
        negative.submit(infer, inputs=[image, text, negative, guidance_scale, samples, steps, seed, height, width],
                        outputs=[gallery], postprocess=False)
        text.submit(infer,
                    inputs=[image, text, negative, guidance_scale, samples, steps,  seed, height, width, method, overlap, upscale_method],
                    outputs=[gallery], postprocess=False)
        btn.click(infer, inputs=[image, text, negative, guidance_scale, samples, steps, seed, height, width, method, overlap, upscale_method],
                  outputs=[gallery], postprocess=False)

        share_button.click(
            None,
            [],
            [],
            _js=share_js,
        )

block.queue().launch(share=True, server_name=server_name, server_port=server_port)