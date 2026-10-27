"""Train matched LoRA SFT / DPO arms and evaluate preference margins on real hold-out pairs."""
import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from diffusers import DDPMScheduler
from peft import set_peft_model_state_dict
from src.alignment import dpo_loss,pair_errors
from src.sd import add_adapter,attachment_manifest,encode,generate,latents,load_pipeline,no_adapter
from src.sd import BASE,REV,SMALL,SMALL_REV,save_reload,seed_all,trainable_state
from tools.get_pairs import prepare

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/"configs/experiment.json").read_text())


def evaluate(out):
    observations=json.loads((out/"heldout.json").read_text())
    summary=[]
    for method in ("base","sft","dpo"):
        values=[r for r in observations if r["method"]==method]
        count=len(values)
        if not count:
            raise ValueError(f"missing observations for {method}")
        correct=sum(r["predicted_winner"]==r["human_winner"] for r in values)
        summary.append(dict(method=method,pairs=count,correct=correct,agreement=correct/count,
                            mean_winner_margin=sum(r["winner_margin"] for r in values)/count))
    with (out/"summary.csv").open("w",newline="") as fh:
        writer=csv.DictWriter(fh,summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    base_margins={r["pair"]:r["winner_margin"] for r in observations if r["method"]=="base"}
    implicit=[{"method":r["method"],"pair":r["pair"],"human_winner":r["human_winner"],
               "implicit_reward_margin_for_human_winner":CFG["beta"]*(r["winner_margin"]-base_margins[r["pair"]])}
              for r in observations if r["method"]!="base"]
    (out/"implicit-rewards.json").write_text(json.dumps(implicit,indent=2))
    mismatches=[r for r in implicit if r["implicit_reward_margin_for_human_winner"]<0]
    (out/"disagreements.json").write_text(json.dumps(mismatches,indent=2))
    canvas=Image.new("RGB",(480,180),"white")
    draw=ImageDraw.Draw(canvas)
    for i,m in enumerate(("base","sft","dpo")):
        draw.text((i*160+5,4),m,fill="black")
        with Image.open(out/(m+"-sample.png")) as image:
            canvas.paste(image.resize((152,152)),(i*160+3,24))
    canvas.save(out/"comparison.png")
    print((out/"summary.csv").read_text())


def run(args):
    smoke=args.mode=="demo"
    out=args.out or ROOT/("demo-output" if smoke else "results")
    if args.mode=="evaluate":
        return evaluate(out)
    out.mkdir(parents=True,exist_ok=True)
    pairs=args.pairs or args.cache/"pairs"
    prepare(pairs)
    records=attachment_manifest(pairs)
    train=[r for r in records if r["split"]=="train"][:4 if smoke else None]
    test=[r for r in records if r["split"]=="validation"][:2 if smoke else None]
    if not train or not test:
        raise RuntimeError("Need disjoint training and evaluation pairs")
    torch.set_num_threads(2)
    if not smoke:
        if args.base is None:
            from huggingface_hub import snapshot_download
            manifest=json.loads((ROOT/"configs/sd15-source.json").read_text())
            args.base=Path(snapshot_download(manifest["repo"],revision=manifest["revision"],
                           allow_patterns=[e["rfilename"] for e in manifest["files"]],cache_dir=args.cache/"hub"))
        from tools.cache_inputs import prepare as prepare_inputs
        args.input_cache=args.input_cache or args.cache/"p05-inputs.safetensors"
        if not args.input_cache.is_file():
            prepare_inputs(args.base,pairs,args.input_cache)
    pipe=load_pipeline(args.device,smoke,args.cache,args.base)
    seed_all(CFG["seed"])
    size=32 if smoke else CFG["resolution"]
    sched=DDPMScheduler.from_config(pipe.scheduler.config)
    if args.input_cache:
        if smoke:
            raise ValueError("The optional fp32 input cache is only for formal SD1.5")
        from tools.cache_inputs import load
        cache=load(args.input_cache,train+test,pipe)
    else:
        cache={}
        for r in train+test:
            cache[r["id"]]=(latents(pipe,[pairs/f for f in r["images"]],size),encode(pipe,[r["prompt"]]))
    params=add_adapter(pipe.unet,CFG["rank"])
    start=trainable_state(pipe.unet)
    formal_steps=3 if smoke else CFG["updates"]
    output_rows=[]
    status={}
    for method in ("base","sft","dpo"):
        set_peft_model_state_dict(pipe.unet,start)
        seed_all(CFG["seed"]+7)
        opt=torch.optim.AdamW(params,lr=CFG["learning_rate"])
        if method!="base":
            losses=[]
            for step in range(formal_steps):
                r=train[step%len(train)]
                clean,text=cache[r["id"]]
                t=torch.randint(1,sched.config.num_train_timesteps,(1,),device=pipe.device)
                noise=torch.randn_like(clean[:1])
                if method=="dpo":
                    with torch.no_grad(),no_adapter(pipe.unet):
                        ref=pair_errors(pipe.unet,sched,clean,text,noise,t,pipe.device)
                estimates=pair_errors(pipe.unet,sched,clean,text,noise,t,pipe.device)
                loss=estimates[r["winner"]] if method=="sft" else dpo_loss(estimates,ref,r["winner"],CFG["beta"])
                opt.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{method} loss diverged at step {step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params,1.)
                opt.step()
                losses.append(float(loss.detach()))
                if (step+1)%8==0 or step+1==formal_steps:
                    print(f"{method} {step+1}/{formal_steps}, loss={losses[-1]:.6f}",flush=True)
            update=save_reload(pipe.unet,start,out/f"{method}-adapter.safetensors")
            status[method]={"updates":formal_steps,"delta_l2":update,"train_loss":losses}
        print(f"{method}: scoring {len(test)} held-out pairs",flush=True)
        with torch.inference_mode():
            for r in test:
                clean,text=cache[r["id"]]
                torch.manual_seed(CFG["seed"]+r["id"])
                margins=[]
                for _ in range(2 if smoke else CFG["eval_seeds"]):
                    t=torch.randint(1,sched.config.num_train_timesteps,(1,),device=pipe.device)
                    noise=torch.randn_like(clean[:1])
                    loss=pair_errors(pipe.unet,sched,clean,text,noise,t,pipe.device)
                    margins.append(float(loss[1]-loss[0]))
                margin=sum(margins)/len(margins)
                output_rows.append(dict(method=method,pair=r["id"],human_winner=r["winner"],
                                    predicted_winner=0 if margin>=0 else 1,
                                    winner_margin=margin if r["winner"]==0 else -margin))
        generate(pipe,test[0]["prompt"],CFG["seed"],3 if smoke else 12,size,out/(method+"-sample.png"))
    (out/"heldout.json").write_text(json.dumps(output_rows,indent=2))
    (out/"run.json").write_text(json.dumps({"profile":args.mode,"base":SMALL if smoke else BASE,"revision":SMALL_REV if smoke else REV,
          "device":str(pipe.device),"input_protocol":"cached-fp32-cpu" if args.input_cache else "on-device","train_ids":[r["id"] for r in train],"test_ids":[r["id"] for r in test],
          "runs":status,"torch":str(torch.__version__),"hardware":torch.cuda.get_device_name() if pipe.device.type=="cuda" else "cpu"},indent=2))
    evaluate(out)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["demo","reproduce","evaluate"])
    p.add_argument("--device",choices=["cuda","cpu"],default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache",type=Path,default=ROOT/".cache")
    p.add_argument("--base",type=Path)
    p.add_argument("--pairs",type=Path)
    p.add_argument("--input-cache",type=Path)
    p.add_argument("--out",type=Path)
    run(p.parse_args())
