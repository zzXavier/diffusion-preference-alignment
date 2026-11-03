# Diffusion preference alignment

This reproduction trains matched LoRA adapters with supervised denoising on the human-preferred image and Diffusion-DPO on both candidates. Start with `python run.py demo --device cpu` after installing `requirements.txt`. The smoke profile loads Hugging Face's tiny test pipeline to verify training, saving and reloading. The formal profile uses pinned SD1.5.

## Reproduction

```shell
python -m unittest discover -s tests
python run.py demo --device cpu
python run.py reproduce --device cuda
python run.py evaluate
```

Use `--pairs PATH` to reuse the public Pick-a-Pic subset, `--base PATH` for a local Diffusers snapshot and `--out PATH` for a fresh run. The same training IDs, seed, updates and input resolutions are used for SFT and DPO. A frozen base branch computes DPO's matched-noise reference margin. Evaluation labels are untouched human votes. For each held-out prompt the code saves a paired denoising-margin preference prediction, records the disagreements and samples the same prompt and seed from all three arms. Denoising margins are a model-based proxy for image preference; this protocol does not infer human judgments on new generated outputs.

`src/alignment.py` defines the losses; `src/sd.py` provides SD loading and verified LoRA save/reload; `configs/` fixes the formal budget; `results/` is reserved for formal measured outputs. See `HANDOFF.md` for the actual run status.
