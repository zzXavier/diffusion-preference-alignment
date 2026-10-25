"""Download Hugging Face's pinned small pipeline and verify LFS digests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import time

import requests

REPO="hf-internal-testing/tiny-stable-diffusion-pipe"
SHA="3ee6c9f225f088ad5d35b624b6514b091e6a4849"


def fetch(root):
    tree=requests.get(f"https://huggingface.co/api/models/{REPO}/tree/{SHA}",params={"recursive":"true"},timeout=35)
    tree.raise_for_status()
    files=[r for r in tree.json() if r["type"]=="file" and r["path"].split("/")[0] in
           {"unet","vae","text_encoder","tokenizer","scheduler","model_index.json"} and "flax" not in r["path"]]
    def download(e):
        path=root/e["path"]
        if path.exists() and path.stat().st_size==e["size"]:
            return
        path.parent.mkdir(parents=True,exist_ok=True)
        for attempt in range(4):
            try:
                r=requests.get(f"https://huggingface.co/{REPO}/resolve/{SHA}/{e['path']}",timeout=75)
                r.raise_for_status()
                if len(r.content)!=e["size"] or (e.get("lfs") and sha256(r.content).hexdigest()!=e["lfs"]["oid"]):
                    raise ValueError(e["path"]+" digest mismatch")
                path.write_bytes(r.content)
                return
            except requests.RequestException:
                if attempt==3:
                    raise
                time.sleep(attempt+1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download,files))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    fetch(p.parse_args().output)
