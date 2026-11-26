import json
import torch
from omegaconf import OmegaConf

from torch.distributed._composable import checkpoint
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from transformers import AutoProcessor

from .qwen import Qwen2_5_VLForConditionalGeneration
from .qwen_nn_vision import Qwen2_5_VLVisionBlock
from .qwen_nn_vlm import Qwen2_5_VLDecoderLayer
from .parallelize_qwen import parallelize_qwen


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    return model


def crop_multimodal_tokens(inputs, text_max_len, SOI_TOK_ID=151652, PAD_TOK_ID=151643):
    device = inputs['input_ids'].device
    new_input_ids = []
    new_attention_mask = []

    # cut text and image fragments
    for i in range(inputs['input_ids'].shape[0]):
        inp = inputs['input_ids'][i:i+1]
        msk = inputs['attention_mask'][i:i+1]
        # find text
        img_start = torch.where(inp == SOI_TOK_ID)[1].item()
        txt_end = min(img_start, text_max_len)
        txt = inp[:, :txt_end]
        txt_msk = msk[:, :txt_end]
        # find image
        img_end = torch.where(inp == PAD_TOK_ID)[1]
        img_end = img_end[0].item() if img_end.shape[0] > 0 else inp.shape[1]
        img = inp[:, img_start:img_end]
        img_msk = msk[:, img_start:img_end]

        new_input_ids.append((txt, img))
        new_attention_mask.append((txt_msk, img_msk))

    # add padding
    max_batch_len = max([x[0].shape[1] + x[1].shape[1] for x in new_input_ids])
    for i in range(len(new_input_ids)):
        pad_size = max_batch_len - (new_input_ids[i][0].shape[1] + new_input_ids[i][1].shape[1])
        new_input_ids[i] = torch.cat([
            *new_input_ids[i],
            torch.full((1, pad_size), PAD_TOK_ID, device=device, dtype=inputs["input_ids"].dtype)
        ], dim=1)
        new_attention_mask[i] = torch.cat([
            *new_attention_mask[i],
            torch.zeros((1, pad_size), device=device, dtype=inputs["attention_mask"].dtype)
        ], dim=1)

    # concat tensor
    new_inputs = inputs.copy()
    new_inputs.update({
        "input_ids": torch.cat(new_input_ids, dim=0),
        "attention_mask": torch.cat(new_attention_mask, dim=0)
    })
    return new_inputs


class Qwen2_5_VLTextEmbedder:

    PROMPT_TEMPLATE = {
        "template": {
            "video": (
                "<|im_start|>system\nYou are a promt engineer. Describe the video in detail.",
                "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
                "Describe the location of the video, main characters or objects and their action.",
                "Describe the dynamism of the video and presented actions.",
                "Name the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.",
                "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
                "Pay attention to the order of key actions shown in the scene.<|im_end|>",
                "<|im_start|>user\n{}<|im_end|>"
            ),
            "image2video": (
                "<|im_start|>system\nYou are a promt engineer. Your task is to create a highly detailed and effective video description based on a provided input image.",
                "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
                "Describe main characters actions.",
                "Describe the dynamism of the video and presented actions.",
                "Name the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.",
                "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
                "Pay attention to the order of key actions shown in the scene.<|im_end|>",
                "<|im_start|>user\n{}<|im_end|>"
            ),
            "image": (
                "<|im_start|>system\nYou are a promt engineer. Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>",
                "<|im_start|>user\n{}<|im_end|>"
            ),
            "image_instruct": (
                "<|im_start|>system\nYou are a promt engineer. Based on the provided source image (first image) and target image (second image), create an interesting text prompt that can be used together with the source image to create the target image:<|im_end|>",
                "<|im_start|>user\n{}"
            )
        },
        "crop_start": {
            "video": 129,
            "image": 41,
            "image_instruct": 55,
            "image2video": 132
        },
    }

    def __init__(self, conf, device, device_mesh=None, shard_vision=False):
        self.model = Qwen2_5_VLForConditionalGeneration(conf)
        state_dict = torch.load(conf.model_checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
        self.model.load_state_dict(state_dict)

        if device_mesh is None:
            inv_freq = self.model.model.visual.rotary_pos_emb.inv_freq.clone()
            self.model = self.model.to(dtype=torch.bfloat16, device=device)
            self.model.model.visual.rotary_pos_emb.inv_freq = inv_freq.to(device)
        else:
            if 'model' in device_mesh:
                mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.bfloat16)
                for module in self.model.modules():
                    # NOTE: when Qwen accepts mixed inputs (with and without vision part) while being sharded,
                    # it blocks in FSDP. Do not use shard_vision for instruct.
                    # NOTE: I2V is ok with this, since batch is designed to always be homogeneous.
                    if (
                        isinstance(module, Qwen2_5_VLDecoderLayer) or
                        (shard_vision and isinstance(module, Qwen2_5_VLVisionBlock))
                    ):
                        checkpoint(module)
                        fully_shard(module, mesh=device_mesh["model"], mp_policy=mp_policy)
                fully_shard(self.model, mesh=device_mesh["model"], mp_policy=mp_policy)
            self.model.to(device=device)

            if 'data' in device_mesh:
                self.model = parallelize_qwen(self.model, device_mesh["data"])

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(conf.processor_checkpoint_path, local_files_only=True)
        self.max_length = conf.max_length
        self.mode = getattr(conf, 'mode', 't2v')
        self.device = device

    def __call__(self, texts, images=None, videos=None, type_of_content="video", mode=None):
        mode = self.mode if mode in (None, "") else mode

        if mode == "i2v" and images is not None:
            embed, cu_seqlens = self.encode_i2v_mixed(
                texts, images=images, videos=videos,
                type_of_content=type_of_content
            )
        else:
            embed, cu_seqlens = self.encode(
                texts, images=images, videos=videos,
                type_of_content=type_of_content
            )

        return embed, cu_seqlens

    def encode(self, texts, images=None, videos=None, type_of_content="video"):
        prompt_template = "\n".join(self.PROMPT_TEMPLATE["template"][type_of_content])
        crop_start = self.PROMPT_TEMPLATE["crop_start"][type_of_content]
        full_texts = list(map(lambda x: prompt_template.format(x), texts))

        max_length = self.max_length + crop_start
        if type_of_content == 'image_instruct':
            for i in range(len(images)):
                image_tokens = ''.join(['<|vision_start|><|image_pad|><|vision_end|>'] * len(images[i]))
                full_texts[i] = full_texts[i] + image_tokens + "<|im_end|>"

        inputs = self.processor(
            text=full_texts, images=images, videos=videos,
            truncation=True, return_tensors="pt", padding=True,
            max_length=(
                max_length if type_of_content != 'image_instruct' else
                None
            )
        ).to(self.device)

        with torch.no_grad():
            embeds = self.model(**inputs, output_hidden_states=True)

        cu_seqlens = embeds["cu_seqlens"]
        embeds = embeds["hidden_states"][-1]
        new_embed = torch.cat([emb[crop_start:] for emb in embeds.split(cu_seqlens.diff().tolist(), dim=0)], dim=0)
        new_cu_seqlens = torch.cumsum(cu_seqlens.diff() - crop_start, dim=0)
        new_cu_seqlens = torch.cat([torch.zeros(1, device=new_cu_seqlens.device), new_cu_seqlens], dim=0).to(dtype=torch.int32)
        return new_embed, new_cu_seqlens

    def encode_i2v(self, texts, images=None, videos=None, type_of_content="image2video"):
        prompt_template = "\n".join(self.PROMPT_TEMPLATE["template"][type_of_content])
        crop_start = self.PROMPT_TEMPLATE["crop_start"][type_of_content]
        # create a placeholder for image tokens
        if images is not None and len(images) > 0:
            if len(images) != len(texts):
                raise RuntimeError(
                    f"length of texts{len(texts)} and images({len(images)}) should be equal"
                )
            for i in range(len(texts)):
                texts[i] = texts[i] + '<|vision_start|><|image_pad|><|vision_end|>'

        full_texts = list(map(lambda x: prompt_template.format(x), texts))
        max_text_length = self.max_length + crop_start

        inputs = self.processor(
            text=full_texts, images=images, videos=videos,
            truncation=True, return_tensors="pt", padding=True,
            max_length=None
        ).to(self.device)
        # crop text, preserve visual tokens
        inputs = crop_multimodal_tokens(inputs, max_text_length)

        # embed text and vision data
        with torch.no_grad():
            embeds = self.model(**inputs, output_hidden_states=True)

        # remove system prompt part of the sequence
        cu_seqlens = embeds["cu_seqlens"]
        embeds = embeds["hidden_states"][-1]
        new_embed = torch.cat([emb[crop_start:] for emb in embeds.split(cu_seqlens.diff().tolist(), dim=0)], dim=0)
        new_cu_seqlens = torch.cumsum(cu_seqlens.diff() - crop_start, dim=0)
        new_cu_seqlens = torch.cat([torch.zeros(1, device=new_cu_seqlens.device), new_cu_seqlens], dim=0).to(dtype=torch.int32)
        return new_embed, new_cu_seqlens

    def encode_i2v_mixed(self, text, images=None, videos=None, type_of_content="video"):
        i2v_ids, i2v_text, i2v_images = {}, [], []
        if images is None:
            t2v_ids = {i: i for i in range(len(text))}
            t2v_text = text
        else:
            t2v_ids, t2v_text = {}, []
            for i in range(len(images)):
                if images[i] is None:
                    t2v_ids[i] = len(t2v_ids)
                    t2v_text.append(text[i])
                else:
                    i2v_ids[i] = len(i2v_ids)
                    i2v_text.append(text[i])
                    i2v_images.append(images[i])

        t2v_embed, t2v_cu_seqlens = None, None
        if len(t2v_ids) > 0:
            t2v_embed, t2v_cu_seqlens = self.encode(t2v_text, type_of_content=type_of_content)

        i2v_embed, i2v_cu_seqlens = None, None
        if len(i2v_ids) > 0:
            i2v_embed, i2v_cu_seqlens = self.encode_i2v(i2v_text, images=i2v_images, type_of_content=type_of_content)

        # reconstruct full sequence of samples in the correct order
        if len(t2v_ids) > 0 and len(i2v_ids) > 0:
            embed, cu_seqlens = [], [0]
            for i in range(len(text)):
                if i in t2v_ids:
                    list_id = t2v_ids[i]
                    start = t2v_cu_seqlens[list_id]
                    end = t2v_cu_seqlens[list_id+1]
                    cur_embed = t2v_embed[start:end]
                else:
                    list_id = i2v_ids[i]
                    start = i2v_cu_seqlens[list_id]
                    end = i2v_cu_seqlens[list_id+1]
                    cur_embed = i2v_embed[start:end]

                cur_seqlen = end - start
                embed.append(cur_embed)
                cu_seqlens.append(cu_seqlens[-1] + cur_seqlen)

            embed = torch.cat(embed, dim=0)
            cu_seqlens = torch.tensor(cu_seqlens, device=self.device, dtype=torch.int32)
        else:
            embed = i2v_embed if i2v_embed is not None else t2v_embed
            cu_seqlens = i2v_cu_seqlens if i2v_cu_seqlens is not None else t2v_cu_seqlens

        return embed, cu_seqlens

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self
