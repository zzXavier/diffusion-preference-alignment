"""Matched SFT / Diffusion-DPO objectives. Adapted from Diffusers' Apache-2.0 example."""
import torch
import torch.nn.functional as F


def pair_errors(unet,scheduler,clean,context,noise,t,device):
    target=noise.expand_as(clean)
    times=t.reshape(1).repeat(len(clean)).to(device)
    noisy=scheduler.add_noise(clean,target,times)
    pred=unet(noisy,times,context.expand(len(clean),-1,-1)).sample
    if scheduler.config.prediction_type=="v_prediction":
        target=scheduler.get_velocity(clean,target,times)
    elif scheduler.config.prediction_type!="epsilon":
        raise ValueError(f"Unsupported prediction type {scheduler.config.prediction_type}")
    return (pred.float()-target.float()).square().flatten(1).mean(1)


def dpo_loss(errors,reference,winner,beta):
    loser=1-winner
    implicit_margin=(reference[winner]-reference[loser])-(errors[winner]-errors[loser])
    return -F.logsigmoid(beta*implicit_margin)
