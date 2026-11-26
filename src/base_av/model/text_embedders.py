import json
from collections import defaultdict
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer

from .qwen.qwen import Qwen2_5_VLTextModel


# The prefix and token crop are part of the original video encoder.
VIDEO_TEMPLATE = "\n".join((
    "<|im_start|>system\nYou are a promt engineer. Describe the video in detail.",
    "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
    "Describe the location of the video, main characters or objects and their action.",
    "Describe the dynamism of the video and presented actions.",
    "Name the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.",
    "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
    "Pay attention to the order of key actions shown in the scene.<|im_end|>",
    "<|im_start|>user\n{}<|im_end|>",
))
CROP_START = 129


class Kandinsky5TextEmbedder:
    def __init__(self, config, weights, tokenizer, clip_dir, device, clip_tokenizer_dir=None):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_tokenizer_dir or clip_dir, local_files_only=True)
        self.clip = CLIPTextModel.from_pretrained(clip_dir, local_files_only=True).eval().to(device)
        # Older public Qwen configs omit the padding ID and per-layer attention types.
        config.setdefault('pad_token_id', self.tokenizer.pad_token_id)
        config.setdefault('layer_types', ['full_attention'] * config.num_hidden_layers)
        with init_empty_weights(include_buffers=False):
            self.qwen = Qwen2_5_VLTextModel(config)
        weights = Path(weights)
        if weights.is_dir():
            index = json.loads((weights / 'model.safetensors.index.json').read_text())['weight_map']
            prefix = 'model.language_model.' if 'model.language_model.embed_tokens.weight' in index else 'model.'
            by_shard = defaultdict(list)
            for name in self.qwen.state_dict():
                shard = index[prefix + name]
                if Path(shard).name != shard:
                    raise ValueError(f'Expected a shard filename, got {shard}')
                by_shard[shard].append(name)
            text_state = {}
            for shard, names in by_shard.items():
                with safe_open(weights / shard, framework='pt', device='cpu') as f:
                    for name in names:
                        text_state[name] = f.get_tensor(prefix + name)
        else:
            state = torch.load(weights, mmap=True, map_location='cpu', weights_only=True)
            prefix = 'model.language_model.'
            text_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        self.qwen.load_state_dict(text_state, strict=True, assign=True)
        self.qwen.eval()

    def encode(self, caption):
        tokens = self.tokenizer(VIDEO_TEMPLATE.format(caption), truncation=True,
                                max_length=1024 + CROP_START, return_tensors='pt')
        clip_tokens = self.clip_tokenizer(caption, max_length=77, padding='max_length',
                                          truncation=True, return_tensors='pt').to(self.device)
        return (self.qwen(tokens.input_ids.to(self.device))[CROP_START:],
                self.clip(**clip_tokens).pooler_output)
