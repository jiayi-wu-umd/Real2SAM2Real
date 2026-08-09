#!/usr/bin/env python3
"""
Sweep inference across multiple checkpoints on Sam3Render dataset.

Loads the base model ONCE, then iterates over assigned checkpoints swapping
only LoRA + pose_patch_embedding weights.

Output structure:
    <output_root>/checkpoint-XXXX/<scene>/<camera>/generated.mp4

Usage (prefer the launcher: bash example_inference.sh):
    python real2sam2real/infer.py \
        --checkpoints_root checkpoints \
        --checkpoint_steps infer \
        --data_root data/test/scenes \
        --output_root output \
        --gpu 0
"""

import argparse
import gc
import inspect
import json
import math
import os
import sys
import types

import cv2
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from safetensors.torch import load_file
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
_repo_root = os.path.dirname(os.path.dirname(current_file_path))
# Locate the cloned VideoX-Fun checkout. Override with VIDEOX_FUN_ROOT if you
# keep it elsewhere; defaults to <repo_root>/VideoX-Fun (created by setup.sh).
_videox_fun_root = os.environ.get(
    "VIDEOX_FUN_ROOT", os.path.join(_repo_root, "VideoX-Fun")
)
project_roots = [
    os.path.dirname(current_file_path),  # real2sam2real/ (sibling modules)
    _videox_fun_root,                    # videox_fun package
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from videox_fun.models import (
    AutoencoderKLWan,
    CLIPModel,
    Wan2_2Transformer3DModel_Animate,
    WanT5EncoderModel,
)
from videox_fun.pipeline import Wan2_2AnimatePipeline
from videox_fun.utils.utils import (
    calculate_dimensions,
    get_image,
    get_video_to_video_latent,
    save_videos_grid,
)


def filter_kwargs(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self", "cls"}
    return {k: v for k, v in kwargs.items() if k in valid_params}


def discover_samples(data_root):
    """Find all dirs containing source_image.png + object_animation_normal.mp4."""
    samples = []
    for root, dirs, files in os.walk(data_root):
        if "source_image.png" in files and "object_animation_normal.mp4" in files:
            rel_path = os.path.relpath(root, data_root)
            parts = rel_path.split(os.sep)
            if len(parts) < 2:
                continue
            scene = parts[0]
            camera = parts[1]
            prompt = ""
            caption_path = os.path.join(root, "caption.json")
            if os.path.exists(caption_path):
                with open(caption_path, "r") as f:
                    prompt = json.load(f).get("text", "")
            samples.append({
                "scene": scene,
                "camera": camera,
                "ref_image": os.path.join(root, "source_image.png"),
                "normal_video": os.path.join(root, "object_animation_normal.mp4"),
                "prompt": prompt,
            })
    samples.sort(key=lambda x: (x["scene"], x["camera"]))
    return samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="models/Diffusion_Transformer/Wan2.2-Animate-14B")
    parser.add_argument("--config_path", type=str, default="config/wan2.2/wan_civitai_animate.yaml")
    parser.add_argument("--checkpoints_root", type=str, required=True)
    parser.add_argument("--checkpoint_steps", type=str, required=True,
                        help="Comma-separated list of checkpoint steps to run")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--video_sample_size", type=int, default=960)
    parser.add_argument("--video_sample_n_frames", type=int, default=81)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--target_name", type=str, default="q,k,v,ffn.0,ffn.2")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard_idx", type=int, default=0,
                        help="This shard's index (0..num_shards-1)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards to split samples across")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16

    # Checkpoint names, used verbatim to build "checkpoint-<name>" dirs.
    # Accepts numeric steps (e.g. 3000) or names (e.g. "infer").
    checkpoint_steps = [s.strip() for s in args.checkpoint_steps.split(",")]
    print(f"GPU {args.gpu}: will run checkpoints {checkpoint_steps}")

    samples = discover_samples(args.data_root)
    total = len(samples)
    samples = [s for i, s in enumerate(samples) if i % args.num_shards == args.shard_idx]
    print(f"Found {total} samples, shard {args.shard_idx}/{args.num_shards} -> {len(samples)} samples")

    config = OmegaConf.load(args.config_path)

    # ---- Load frozen components ONCE ----
    print("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.base_model, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(device, dtype=weight_dtype).eval()

    print("Loading text encoder...")
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.base_model, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.base_model, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")),
    )

    print("Loading CLIP image encoder...")
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(args.base_model, config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder")),
    ).eval()

    # ---- Load transformer + inject LoRA skeleton ----
    print("Loading Wan2.2-Animate-14B transformer...")
    sub_path = config["transformer_additional_kwargs"].get("transformer_low_noise_model_subpath", "transformer")
    transformer3d = Wan2_2Transformer3DModel_Animate.from_pretrained(
        os.path.join(args.base_model, sub_path),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
    ).to(weight_dtype)

    def _after_patch_embedding_no_face(self, x, pose_latents, face_pixel_values):
        pose_latents = [self.pose_patch_embedding(u.unsqueeze(0)) for u in pose_latents]
        for x_, pose_latents_ in zip(x, pose_latents):
            x_[:, :, 1:] += pose_latents_
        dummy = torch.zeros(len(x), 1, 1, self.dim, device=x[0].device, dtype=x[0].dtype)
        return x, dummy

    def _after_transformer_block_no_face(self, block_idx, x, motion_vec, motion_masks=None):
        return x

    transformer3d.after_patch_embedding = types.MethodType(_after_patch_embedding_no_face, transformer3d)
    transformer3d.after_transformer_block = types.MethodType(_after_transformer_block_no_face, transformer3d)

    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        target_modules=args.target_name.split(","),
    )
    transformer3d = inject_adapter_in_model(lora_config, transformer3d)
    transformer3d = transformer3d.to(device).eval()

    # ---- Build pipeline ----
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config["scheduler_kwargs"]))
    )
    pipeline = Wan2_2AnimatePipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        transformer=transformer3d, transformer_2=None,
        scheduler=scheduler, clip_image_encoder=clip_image_encoder,
    )
    pipeline = pipeline.to(device)
    print("Base model loaded + LoRA skeleton injected\n")

    # ---- Sweep checkpoints ----
    for step in checkpoint_steps:
        ckpt_dir = os.path.join(args.checkpoints_root, f"checkpoint-{step}")
        if not os.path.isdir(ckpt_dir):
            print(f"[step {step}] checkpoint dir not found, skipping: {ckpt_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"[step {step}] Loading weights from {ckpt_dir}")

        lora_path = os.path.join(ckpt_dir, "lora_diffusion_pytorch_model.safetensors")
        lora_state = load_file(lora_path, device=str(device))
        set_peft_model_state_dict(transformer3d, lora_state)

        pose_path = os.path.join(ckpt_dir, "pose_patch_embedding.safetensors")
        pose_state = load_file(pose_path, device=str(device))
        transformer3d.pose_patch_embedding.load_state_dict(pose_state)
        del lora_state, pose_state

        ckpt_output_dir = os.path.join(args.output_root, f"checkpoint-{step}")

        for idx, sample in enumerate(samples):
            sample_output_dir = os.path.join(ckpt_output_dir, sample["scene"], sample["camera"])
            mp4_path = os.path.join(sample_output_dir, "generated.mp4")

            if os.path.exists(mp4_path):
                print(f"  [{idx+1}/{len(samples)}] Skip (exists): {sample['scene']}/{sample['camera']}")
                continue

            os.makedirs(sample_output_dir, exist_ok=True)
            print(f"  [{idx+1}/{len(samples)}] {sample['scene']}/{sample['camera']}")

            cap = cv2.VideoCapture(sample["normal_video"])
            vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            width, height = calculate_dimensions(
                args.video_sample_size * args.video_sample_size, vid_w / vid_h
            )
            video_length = int(
                (args.video_sample_n_frames - 1) // vae.config.temporal_compression_ratio
                * vae.config.temporal_compression_ratio
            ) + 1

            pose_video, _, _, _ = get_video_to_video_latent(
                sample["normal_video"], video_length=video_length,
                sample_size=[height, width], fps=args.fps, ref_image=None,
            )
            ref_image = get_image(sample["ref_image"])
            dummy_face = torch.zeros(1, 3, video_length, 512, 512)

            generator = torch.Generator(device=device).manual_seed(args.seed)

            try:
                with torch.no_grad():
                    result = pipeline(
                        sample["prompt"],
                        segment_frame_length=77,
                        negative_prompt="bad detailed",
                        height=height, width=width,
                        generator=generator,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.num_inference_steps,
                        pose_video=pose_video,
                        face_video=dummy_face,
                        ref_image=ref_image,
                        bg_video=None, mask_video=None,
                        replace_flag=False,
                    ).videos

                save_videos_grid(result, mp4_path, fps=args.fps)
                gif_path = os.path.join(sample_output_dir, "generated.gif")
                save_videos_grid(result, gif_path)
                print(f"    Saved: {mp4_path}")
            except Exception as e:
                print(f"    ERROR: {e}")

            torch.cuda.empty_cache()

        print(f"[step {step}] Done: {len(samples)} samples")

    print(f"\nAll done. GPU {args.gpu} finished {len(checkpoint_steps)} checkpoints.")


if __name__ == "__main__":
    main()
