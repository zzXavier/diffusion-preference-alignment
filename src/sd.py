"""Shared SD1.5 loading, deterministic sampling, LoRA save/reload and measured output utilities."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from diffusers import DDPMScheduler, StableDiffusionPipeline
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file

BASE="stable-diffusion-v1-5/stable-diffusion-v1-5"
REV="451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
SMALL="hf-internal-testing/tiny-stable-diffusion-pipe"
SMALL_REV="3ee6c9f225f088ad5d35b624b6514b091e6a4849"


def load_pipeline(device,smoke,cache,base_path=None):
    if smoke and base_path is None:
        from tools.fetch_smoke import fetch
        base_path=cache/"tiny-sd"
        if not (base_path/"model_index.json").is_file() or not (base_path/"unet/diffusion_pytorch_model.bin").is_file():
            fetch(base_path)
    if not smoke and base_path is not None:
        manifest=json.loads((Path(__file__).resolve().parents[1]/"configs/sd15-source.json").read_text())
        for entry in manifest["files"]:
            path=Path(base_path)/entry["rfilename"]
            if not path.is_file() or path.stat().st_size!=entry["size"]:
                raise ValueError(f"Local SD1.5 snapshot differs at {entry['rfilename']}")
            if entry.get("lfs"):
                with path.open("rb") as f:
                    digest=hashlib.file_digest(f,"sha256").hexdigest()
                if digest!=entry["lfs"]["sha256"]:
                    raise ValueError(f"Local SD1.5 weight hash differs at {entry['rfilename']}")
            else:
                body=path.read_bytes()
                git_blob=hashlib.sha1(b"blob "+str(len(body)).encode()+bytes([0])+body).hexdigest()
                if git_blob!=entry["blobId"]:
                    raise ValueError(f"Local SD1.5 config hash differs at {entry['rfilename']}")
    dtype=torch.float32 if device=="cpu" or smoke else torch.float16
    kw=dict(torch_dtype=dtype,cache_dir=str(cache/"hub"),safety_checker=None,
            feature_extractor=None,requires_safety_checker=False)
    if not smoke and (not base_path or (Path(base_path)/"unet/diffusion_pytorch_model.fp16.safetensors").is_file()):
        kw.update(variant="fp16",use_safetensors=True)
    pipe=StableDiffusionPipeline.from_pretrained(
        str(base_path) if base_path else SMALL if smoke else BASE,
        revision=None if base_path else SMALL_REV if smoke else REV,**kw).to(device)
    pipe.set_progress_bar_config(disable=True)
    for module in (pipe.unet,pipe.vae,pipe.text_encoder):
        module.requires_grad_(False).eval()
    return pipe


def add_adapter(unet,rank=4):
    unet.add_adapter(LoraConfig(r=rank,lora_alpha=rank,init_lora_weights="gaussian",
                                target_modules=["to_q","to_k","to_v","to_out.0"]))
    for p in unet.parameters():
        if p.requires_grad:
            p.data=p.data.float()
    return [p for p in unet.parameters() if p.requires_grad]


@torch.no_grad()
def encode(pipe,prompts):
    tokens=pipe.tokenizer(prompts,padding="max_length",max_length=pipe.tokenizer.model_max_length,
                          truncation=True,return_tensors="pt")
    return pipe.text_encoder(tokens.input_ids.to(pipe.device))[0]


@torch.no_grad()
def latents(pipe,paths,size):
    images=[]
    for f in paths:
        with Image.open(f) as im:
            im=im.convert("RGB").resize((size,size),Image.Resampling.BICUBIC)
            images.append(np.array(im,dtype=np.float32)/127.5-1)
    pixel=torch.tensor(np.stack(images)).permute(0,3,1,2).to(pipe.device,dtype=pipe.vae.dtype)
    return pipe.vae.encode(pixel).latent_dist.mode()*pipe.vae.config.scaling_factor


@contextmanager
def no_adapter(unet):
    unet.disable_adapters()
    try:
        yield
    finally:
        unet.enable_adapters()


def trainable_state(unet):
    return {k:v.detach().float().cpu().contiguous().clone() for k,v in get_peft_model_state_dict(unet).items()}


def save_reload(unet,initial,dest):
    state=trainable_state(unet)
    update=float(sum((v-initial[k]).square().sum().item() for k,v in state.items())**.5)
    if not np.isfinite(update) or update<=0:
        raise RuntimeError("No finite LoRA update")
    save_file(state,str(dest))
    for p in unet.parameters():
        if p.requires_grad:
            p.data.zero_()
    set_peft_model_state_dict(unet,load_file(str(dest)))
    recovered=trainable_state(unet)
    if any(not torch.equal(v,recovered[k]) for k,v in state.items()):
        raise RuntimeError("LoRA reload changed parameters")
    return update


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def sync(device):
    if device.type=="cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate(pipe,prompt,seed,steps,size,dest):
    sync(pipe.device)
    if pipe.device.type=="cuda":
        torch.cuda.reset_peak_memory_stats(pipe.device)
    start=time.perf_counter()
    out=pipe(prompt,num_inference_steps=steps,height=size,width=size,guidance_scale=1.,
             generator=torch.Generator(device=pipe.device).manual_seed(seed)).images[0]
    sync(pipe.device)
    elapsed=time.perf_counter()-start
    peak=torch.cuda.max_memory_allocated(pipe.device) if pipe.device.type=="cuda" else None
    dest.parent.mkdir(parents=True,exist_ok=True)
    out.save(dest)
    return {"seconds":elapsed,"peak_cuda_bytes":peak}


def attachment_manifest(path):
    items=json.loads((path/"pairs.json").read_text(encoding="utf8"))
    for r in items:
        for file,sha in zip(r["images"],r["sha256"]):
            if hashlib.sha256((path/file).read_bytes()).hexdigest()!=sha:
                raise ValueError(f"Image checksum mismatch: {file}")
    return items
