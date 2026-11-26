"""Load the native video DiT, VAE, and text encoders from a Base AV bundle."""
from pathlib import Path

import torch
from accelerate import init_empty_weights
from omegaconf import OmegaConf
from safetensors import safe_open
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from .dit import DiffusionTransformer3D
from .text_embedders import Kandinsky5TextEmbedder
from .vae import AutoencoderKLHunyuanVideo


def load_dit(checkpoint: Path) -> DiffusionTransformer3D:
    config = OmegaConf.load(checkpoint / "config.json")
    if not (
        config.get("visual_cond")
        and config.get("visual_token_type_num_embeddings") == 2
        and config.out_visual_dim == config.in_visual_dim
        and config.attention_type == "regular"
    ):
        raise ValueError("Expected the Base AV transformer with two visual token types")
    architecture = {
        key: config[key]
        for key in (
            "in_visual_dim", "in_text_dim", "in_text_dim2", "model_dim", "time_dim",
            "ff_dim", "axes_dims", "patch_size", "num_text_blocks", "num_visual_blocks",
        )
    }
    with init_empty_weights(include_buffers=False):
        model = DiffusionTransformer3D(**architecture)
    state = {}
    with safe_open(checkpoint / "diffusion_pytorch_model.safetensors", framework="pt", device="cpu") as f:
        for name in f.keys():
            target = name.replace("visual_transformer_blocks.", "visual_blocks.", 1)
            if target in state:
                raise ValueError(f"Duplicate checkpoint parameter {target}")
            state[target] = f.get_tensor(name).to(torch.bfloat16)
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()


def place_model(model, blocks, device, mesh):
    if mesh is not None:
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, output_dtype=torch.bfloat16)
        for block in blocks:
            fully_shard(block, mesh=mesh, mp_policy=policy)
        fully_shard(model, mesh=mesh, mp_policy=policy)
    model.to(device)


def load_models(bundle: Path, device, mesh):
    dit = load_dit(bundle / "transformer")
    place_model(dit, dit.visual_blocks, device, mesh)

    encoder_dir = bundle / "text_encoder"
    text = Kandinsky5TextEmbedder(
        OmegaConf.load(encoder_dir / "config.json"),
        encoder_dir,
        bundle / "tokenizer",
        bundle / "text_encoder_2",
        device,
        clip_tokenizer_dir=bundle / "tokenizer_2",
    )
    place_model(text.qwen, text.qwen.layers, device, mesh)
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        bundle / "vae", torch_dtype=torch.float16, local_files_only=True
    ).eval().to(device)
    return dit, vae, text
