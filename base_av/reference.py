from itertools import islice

import av
import torch
from PIL import Image
from torchvision.transforms import functional as F


RESOLUTIONS = {
    256: [(256, 256), (256, 384), (384, 256)],
    512: [(512, 512), (512, 768), (768, 512)],
}


def resize_video(video, visual_size):
    """Resize and center-crop TCHW frames to the closest training aspect ratio."""
    h, w = video.shape[-2:]
    height, width = min(RESOLUTIONS[visual_size], key=lambda hw: abs(hw[0] / hw[1] - h / w))
    scale = min(h / height, w / width)
    resized = F.resize(video, (int(h / scale), int(w / scale)))
    top = (resized.shape[-2] - height) // 2
    left = (resized.shape[-1] - width) // 2
    return F.crop(resized, top, left, height, width)


def encode_reference(path, mode, prefix_frames, visual_size, vae, device, generator):
    if mode == 'i2v':
        with Image.open(path) as img:
            frames = F.pil_to_tensor(img.convert('RGB')).unsqueeze(0)
    else:
        with av.open(str(path)) as container:
            frames = [F.pil_to_tensor(frame.to_image()) for frame in
                      islice(container.decode(video=0), prefix_frames)]
        if len(frames) != prefix_frames:
            raise ValueError(f'{path} contains {len(frames)} frames, need {prefix_frames}')
        frames = torch.stack(frames)
    pixels = resize_video(frames, visual_size) / 127.5 - 1
    pixels = pixels.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=vae.dtype)
    latent = vae.encode(pixels).latent_dist.sample(generator=generator)
    return (latent[0] * vae.config.scaling_factor).permute(1, 2, 3, 0).float()
