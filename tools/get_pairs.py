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


def pinned_parquet_rows(split,count):
    import tempfile
    import pyarrow.parquet as pq
    name=("validation_unique-00000-of-00001-c7e520171372df72.parquet" if split=="train"
          else "test_unique-00000-of-00001-c163d5839d8aed9f.parquet")
    url=f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/{name}"
    with tempfile.TemporaryDirectory() as directory:
        target=Path(directory)/name
        with requests.get(url,stream=True,timeout=90) as response:
            response.raise_for_status()
            with target.open("wb") as output:
                for data in response.iter_content(1024*1024):
                    output.write(data)
        with pq.ParquetFile(target) as reader:
            cols=["ranking_id","caption","jpg_0","jpg_1","has_label","label_0","label_1","image_0_uid","image_1_uid"]
            return next(reader.iter_batches(batch_size=count,columns=cols)).to_pylist()


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
                return split,pinned_parquet_rows(split,64 if split=="train" else 32)
            time.sleep(2**retry)


def verify_selected(rows):
    projection=[{key:r[key] for key in ("id","winner","split","sha256")} |
                {"prompt_sha256":sha256(r["prompt"].encode()).hexdigest()} for r in rows]
    expected=json.loads((Path(__file__).resolve().parents[1]/"configs/pairs-lock.json").read_text())
    if projection!=expected:
        raise ValueError("Public data selection differs from the pinned ID and content hashes")
    return rows


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/"pairs.json").exists():
        return verify_selected(json.loads((out/"pairs.json").read_text(encoding="utf-8")))
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
                    value=r[f"jpg_{i}"]
                    raw=base64.b64decode(value,validate=True) if isinstance(value,str) else bytes(value)
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
    verify_selected(manifests)
    (out/"pairs.json").write_text(json.dumps(manifests,ensure_ascii=False,indent=2),encoding="utf8")
    return manifests


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("target",type=Path)
    rows=prepare(p.parse_args().target)
    print(len(rows),"distinct labeled prompts")
