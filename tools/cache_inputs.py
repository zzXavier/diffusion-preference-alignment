"""Cache prompt embeddings and posterior modes on CPU while the SD1.5 GPU is busy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from diffusers import AutoencoderKL
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import CLIPTextModel, CLIPTokenizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.sd import REV,attachment_manifest,encode,latents


def metadata():
    config=json.loads((ROOT/"configs/experiment.json").read_text())
    return dict(base_revision=REV,split_sha256=hashlib.sha256((ROOT/"configs/pairs-lock.json").read_bytes()).hexdigest(),
                size=str(config["resolution"]),precision="float32",device="cpu")


def prepare(base,pairs,file):
    cfg=metadata()
    records=attachment_manifest(pairs)
    torch.set_num_threads(2)
    tokenizer=CLIPTokenizer.from_pretrained(base,subfolder="tokenizer",local_files_only=True)
    model=CLIPTextModel.from_pretrained(base,subfolder="text_encoder",variant="fp16",
                                        local_files_only=True,torch_dtype=torch.float32).eval()
    wrapper=SimpleNamespace(device=torch.device("cpu"),text_encoder=model,tokenizer=tokenizer)
    tensors={}
    for r in records:
        tensors["t"+str(r["id"])]=encode(wrapper,[r["prompt"]]).contiguous()
    del wrapper,model,tokenizer
    vae=AutoencoderKL.from_pretrained(base,subfolder="vae",variant="fp16",local_files_only=True,
                                      torch_dtype=torch.float32).eval()
    wrapper=SimpleNamespace(device=torch.device("cpu"),vae=vae)
    for i,r in enumerate(records,1):
        tensors["x"+str(r["id"])]=latents(wrapper,[pairs/f for f in r["images"]],int(cfg["size"])).contiguous()
        print("Encoded pair",i,"of",len(records),flush=True)
    file.parent.mkdir(parents=True,exist_ok=True)
    tmp=file.with_suffix(".tmp")
    save_file(tensors,str(tmp),metadata=cfg)
    tmp.replace(file)


def load(file,records,pipe):
    result={}
    with safe_open(str(file),framework="pt",device="cpu") as h:
        if h.metadata()!=metadata():
            raise ValueError("Cached inputs do not match the base, split or preprocessing protocol")
        for r in records:
            i=str(r["id"])
            result[r["id"]]=(h.get_tensor("x"+i).to(pipe.device,pipe.unet.dtype),
                             h.get_tensor("t"+i).to(pipe.device,pipe.unet.dtype))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--base",type=Path,required=True)
    p.add_argument("--pairs",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    prepare(a.base,a.pairs,a.out)
