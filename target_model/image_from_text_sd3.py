import argparse
import os
from PIL import Image
import torch
from diffusers import StableDiffusion3Pipeline


def ascii_from_image(image: Image.Image, size: int = 128) -> str:
    gray_pixels = image.resize((size, int(0.55 * size))).convert('L').getdata()
    chars = list('.,;/IOX')
    chars = [chars[i * len(chars) // 256] for i in gray_pixels]
    chars = [chars[i * size: (i + 1) * size] for i in range(size // 2)]
    return '\n'.join(''.join(row) for row in chars)


def save_image(image: Image.Image, path: str):
    if os.path.isdir(path):
        path = os.path.join(path, 'generated.png')
    elif not path.endswith('.png'):
        path += '.png'
    image.save(path)
    return image


pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def generate_image(
    is_mega: bool,
    text: str,
    seed: int,
    grid_size: int,
    top_k: int,
    image_path: str,
    models_root: str,
    fp16: bool,
):
    image = pipe(
        text,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images[0]
    save_image(image, image_path)

def gen_img_from_text(ori_sent, ori_img_path, seed=-1):

    generate_image(
        is_mega=False,
        text=ori_sent,
        seed=seed,
        grid_size=1,
        top_k=256,
        image_path=ori_img_path,
        models_root='pretrained',
        fp16=False,
    )

    return