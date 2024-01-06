import os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import importlib 

config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")  

ckpt = "models/ldm/text2img-large/model.ckpt"    
print(f"Loading model from {ckpt}")
pl_sd = torch.load(ckpt, map_location="cpu")
sd = pl_sd["state_dict"]

target = "ldm.models.diffusion.ddpm.LatentDiffusion"
module, cls = target.rsplit(".", 1)
obj = getattr(importlib.import_module(module, package=None), cls)
model = obj(**config.model.get("params", dict()))

m, u = model.load_state_dict(sd, strict=False)
model.cuda()
model.eval()


device = torch.device("cuda")
model = model.to(device)
sampler = DDIMSampler(model)

prompt = 'a painting of a virus monster playing guitar'

outpath = 'outputs/txt2img-samples'
os.makedirs(outpath, exist_ok=True)

sample_path = os.path.join(outpath, "samples")
os.makedirs(sample_path, exist_ok=True)

base_count = len(os.listdir(sample_path))

all_samples=list()
with torch.no_grad():
    with model.ema_scope():
        uc = model.get_learned_conditioning(4 * [""])
        for n in trange(1, desc="Sampling"):
            c = model.get_learned_conditioning(4 * [prompt])
            shape = [4, 32, 32]
            samples_ddim, _ = sampler.sample(S=200, 
                                             conditioning=c,
                                             batch_size=4,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=5.0,
                                             unconditional_conditioning=uc,
                                             eta=0.0)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                         min=0.0, max=1.0)

            for x_sample in x_samples_ddim:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 
                                            'c h w -> h w c')
                Image.fromarray(x_sample.astype(np.uint8)).save(
                        os.path.join(sample_path, f"{base_count:04}.png"))
                base_count += 1
            all_samples.append(x_samples_ddim)

print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")
