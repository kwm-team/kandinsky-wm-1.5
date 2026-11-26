"""Flow-matching inference with clean reference tokens at the end of the sequence."""
import torch


def sample_video(dit, vae, text, negative, reference, mode, latent_frames, seed,
                 num_steps, scheduler_scale, guidance_weight, temporal_policy=None):
    device = reference.device
    count = len(reference)
    start = count if mode == 'v2v' else 0
    if latent_frames <= start or (mode == 'i2v' and count != 1):
        raise ValueError('Reference length is incompatible with output duration')
    conditional_kwargs, negative_kwargs = {}, {}
    if temporal_policy is not None:
        from .model.temporal import PackedRouting, TemporalCondition

        if mode != "i2v" or latent_frames != 31 or count != 1:
            raise ValueError("Temporal text requires I2V with 31 generated latents")
        if not isinstance(text, TemporalCondition) or not isinstance(negative, TemporalCondition):
            raise ValueError("Temporal text requires grouped positive and negative conditions")
        if reference.shape[1] % 2 or reference.shape[2] % 2:
            raise ValueError("Temporal text requires even latent spatial dimensions")
        patches = (reference.shape[1] // 2) * (reference.shape[2] // 2)
        conditional_kwargs["text_routing"] = PackedRouting(text.group_lengths, patches, device, temporal_policy)
        negative_kwargs["text_routing"] = PackedRouting(negative.group_lengths, patches, device, temporal_policy)
        text = (text.text, text.pooled)
        negative = (negative.text, negative.pooled)
    generator = torch.Generator(device=device).manual_seed(seed)
    generated = torch.randn((latent_frames - start, *reference.shape[1:]), device=device,
                            generator=generator)
    # I2V appends an independent first-frame reference at position zero.
    # V2V moves the observed prefix to the tail, retaining its original time positions.
    positions = torch.cat((torch.arange(start, latent_frames), torch.arange(count))).to(device)
    types = torch.cat((torch.zeros(len(generated)), torch.ones(count))).long().to(device)
    clean_mask = types[:, None, None, None].expand(-1, *generated.shape[1:3], 1)
    zero_cond = reference.new_zeros((len(types), *reference.shape[1:]))
    times = torch.linspace(1, 0, num_steps, device=device)
    times = scheduler_scale * times / (1 + (scheduler_scale - 1) * times)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for now, delta in zip(times[:-1], times.diff()):
            x = torch.cat((generated, reference))
            model_input = torch.cat((x, zero_cond, clean_mask), dim=-1)
            t = now.reshape(1) * 1000
            cond = dit(model_input, *text, t, positions, types, **conditional_kwargs)[:len(generated)]
            uncond = dit(model_input, *negative, t, positions, types, **negative_kwargs)[:len(generated)]
            generated += delta * (uncond + guidance_weight * (cond - uncond))

        latent = torch.cat((reference, generated)) if mode == 'v2v' else generated
        latent = latent.permute(3, 0, 1, 2).unsqueeze(0) / vae.config.scaling_factor
        pixels = vae.decode(latent).sample[0]
    return ((pixels.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
