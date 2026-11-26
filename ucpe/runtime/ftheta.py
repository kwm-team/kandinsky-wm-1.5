"""NVIDIA f-theta calibration at token centers (no image rectification).

Calibration rows: width, height, cx, cy, backward polynomial coefficients 0..4.
Optional trailing values: resize_x, resize_y, crop_x, crop_y. With these values,
cx/cy and the polynomial are in original sensor pixels; width/height are encoded.
All pixel quantities describe the encoded image; coefficients map pixel radius
to ray angle in radians. A spatially varying ray-equivalent K is used by UCPE,
not a claim that the distorted image has a global pinhole projection.
"""
import torch


def ftheta_grid(calibration, height, width, frame_height, frame_width):
    c = torch.as_tensor(calibration)
    if c.shape[-1] not in (9, 13) or not torch.isfinite(c).all():
        raise ValueError("FTHETA requires 9 or 13 finite calibration values")
    if not ((c[..., 0] == frame_width) & (c[..., 1] == frame_height)).all():
        raise ValueError("FTHETA calibration must match encoded image dimensions")
    y, x = torch.meshgrid(
        (torch.arange(height, device=c.device, dtype=c.dtype) + .5) * frame_height / height,
        (torch.arange(width, device=c.device, dtype=c.dtype) + .5) * frame_width / width,
        indexing="ij",
    )
    sx = c[..., 9, None, None] if c.shape[-1] == 13 else 1.
    sy = c[..., 10, None, None] if c.shape[-1] == 13 else 1.
    ox = c[..., 11, None, None] if c.shape[-1] == 13 else 0.
    oy = c[..., 12, None, None] if c.shape[-1] == 13 else 0.
    if torch.as_tensor(sx).min() <= 0 or torch.as_tensor(sy).min() <= 0:
        raise ValueError("Invalid FTHETA resize scales")
    dx = (x + ox) / sx - c[..., 2, None, None]
    dy = (y + oy) / sy - c[..., 3, None, None]
    radius = torch.sqrt(dx.square() + dy.square())
    theta = torch.zeros_like(radius)
    for i in range(4, -1, -1):
        theta = theta * radius + c[..., 4 + i, None, None]
    derivative = torch.zeros_like(radius)
    for i in range(4, 0, -1):
        derivative = derivative * radius + i * c[..., 4 + i, None, None]
    if (derivative <= 0).any():
        raise ValueError("Nonmonotonic FTHETA calibration")
    if (theta < -1e-6).any() or (theta >= torch.pi / 2).any():
        raise ValueError("FTHETA grid contains invalid or rear-facing rays")
    if (c[..., 4].abs() > 1e-6).any() or (c[..., 5] <= 0).any():
        raise ValueError("FTHETA needs zero on-axis angle and positive central slope")
    tangent_scale = torch.where(radius > 1e-6, theta.tan() / radius.clamp_min(1e-6),
                                c[..., 5, None, None])
    rays = torch.stack((dx * tangent_scale, dy * tangent_scale, torch.ones_like(dx)), -1)
    rays = torch.nn.functional.normalize(rays, dim=-1)
    K = torch.zeros((*radius.shape, 3, 3), device=c.device, dtype=c.dtype)
    K[..., 0, 0] = sx / tangent_scale
    K[..., 1, 1] = sy / tangent_scale
    K[..., 0, 2] = c[..., 2, None, None] * sx - ox
    K[..., 1, 2] = c[..., 3, None, None] * sy - oy
    K[..., 2, 2] = 1
    return rays, K
