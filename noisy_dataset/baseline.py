"""Defines the baseline Stable Diffusion image-to-image model. If used as a command, applies"""
import torch
import torch.nn as nn
import diffusers
import argparse
from util import device, evaluate
from noisy_dataset import NoisyDataset


class ImageToImageBaseline(nn.Module):
    def __init__(self, sd_pipeline: diffusers.StableDiffusionImg2ImgPipeline, strength: float, nsteps: int = 50,
                 prompt=None, negative_prompt=None, cfg: float = 0):
        super().__init__()
        self.sd = sd_pipeline
        self.strength = strength
        self.nsteps = nsteps

        if prompt is not None or negative_prompt is not None:
            assert cfg > 0
        else:
            assert cfg == 0  # this leads to unconditional sampling
        self.cfg = cfg
        self.prompt = prompt if prompt is not None else ""
        self.negative_prompt = negative_prompt

    def forward(self, xs):
        # xs : B x C x H x W
        (out, ), _ = self.sd(
            prompt=self.prompt,
            image=xs,
            strength=self.strength,
            num_inference_steps=self.nsteps,
            guidance_scale=self.cfg,
            negative_prompt=self.negative_prompt,
            return_dict=False,
            output_type="pt"  # return a Tensor
        )
        # Given batch size 1, the pipeline automatically removes this dimension
        if len(out.shape) == 3:
            return out.unsqueeze(0)
        return out

