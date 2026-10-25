"""Get a deterministic, small Pick-a-Pic subset without fetching entire parquet shards."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time

from PIL import Image
import requests

DATASET="pickapic-anonymous/pickapic_v1"
REVISION="8e69b58d729339cb7771133c70ee18b527fd4ae3"
OFFSETS={"train":[0], "validation":[0]}


def get_rows(job):
    split,offset=job
    for retry in range(3):
        try:
            r=requests.get("https://datasets-server.huggingface.co/rows",params=dict(
                dataset=DATASET,config="default",split=("validation_unique" if split=="train" else "test_unique"),offset=offset,length=(64 if split=="train" else 32)),timeout=45)
            r.raise_for_status()
            if r.headers.get("x-revision",REVISION)!=REVISION:
                raise RuntimeError("Dataset viewer no longer matches pinned dataset revision")
            return split,[v["row"] for v in r.json()["rows"]]
        except requests.RequestException:
            if retry==2:
                raise
            time.sleep(2**retry)


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/"pairs.json").exists():
        return json.loads((out/"pairs.json").read_text(encoding="utf-8"))
    manifests=[]
    seen_prompts=set()
    seen_images=set()
    jobs=[(s,i) for s in OFFSETS for i in OFFSETS[s]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for split,rows in pool.map(get_rows,jobs):
            for r in rows:
                if not r["has_label"] or {r["label_0"],r["label_1"]}!={0,1}:
                    continue
                prompt=r["caption"].strip()
                if not prompt or prompt.casefold() in seen_prompts:
                    continue
                ids=[r["image_0_uid"],r["image_1_uid"]]
                if len(set(ids))!=2 or seen_images.intersection(ids):
                    continue
                paths=[]
                digests=[]
                for i in range(2):
                    raw=base64.b64decode(r[f"jpg_{i}"],validate=True)
                    with Image.open(BytesIO(raw)) as im:
                        im.verify()
                    file=f"{int(r['ranking_id'])}-{i}.jpg"
                    (out/file).write_bytes(raw)
                    paths.append(file)
                    digests.append(sha256(raw).hexdigest())
                seen_images.update(ids)
                seen_prompts.add(prompt.casefold())
                manifests.append(dict(split=split,id=int(r["ranking_id"]),prompt=prompt,
                                      images=paths,sha256=digests,winner=int(r["label_1"])))
                if sum(v["split"]==split for v in manifests)>=(24 if split=="train" else 12):
                    break
    if sum(x["split"]=="train" for x in manifests)<8 or sum(x["split"]=="validation" for x in manifests)<4:
        raise RuntimeError("Insufficient labeled disjoint pairs")
    (out/"pairs.json").write_text(json.dumps(manifests,ensure_ascii=False,indent=2),encoding="utf8")
    return manifests


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("target",type=Path)
    rows=prepare(p.parse_args().target)
    print(len(rows),"distinct labeled prompts")
