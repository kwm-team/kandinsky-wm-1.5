# MIT License
#
# Copyright (c) Authors of
# "Cameras as Relative Positional Encoding" https://arxiv.org/pdf/2507.10496
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# How to use PRoPE attention for self-attention:
#
# 1. Easiest way (fast):
#    attn = PropeDotProductAttention(...)
#    o = attn(q, k, v, viewmats, Ks)
#
# 2. More flexible way (fast):
#    attn = PropeDotProductAttention(...)
#    attn._precompute_and_cache_apply_fns(viewmats, Ks)
#    q = attn._apply_to_q(q)
#    k = attn._apply_to_kv(k)
#    v = attn._apply_to_kv(v)
#    o = F.scaled_dot_product_attention(q, k, v, **kwargs)
#    o = attn._apply_to_o(o)
#
# 3. The most flexible way (but slower because repeated computation of RoPE coefficients):
#    o = prope_dot_product_attention(q, k, v, ...)
#
# How to use PRoPE attention for cross-attention:
#
#    attn_src = PropeDotProductAttention(...)
#    attn_tgt = PropeDotProductAttention(...)
#    attn_src._precompute_and_cache_apply_fns(viewmats_src, Ks_src)
#    attn_tgt._precompute_and_cache_apply_fns(viewmats_tgt, Ks_tgt)
#    q_src = attn_src._apply_to_q(q_src)
#    k_tgt = attn_tgt._apply_to_kv(k_tgt)
#    v_tgt = attn_tgt._apply_to_kv(v_tgt)
#    o_src = F.scaled_dot_product_attention(q_src, k_tgt, v_tgt, **kwargs)
#    o_src = attn_src._apply_to_o(o_src)

from functools import partial
from typing import Callable, Optional, Tuple, List

import torch
import torch.nn.functional as F


class PropeDotProductAttention(torch.nn.Module):
    """PRoPE attention with precomputed RoPE coefficients."""

    coeffs_x_0: torch.Tensor
    coeffs_x_1: torch.Tensor
    coeffs_y_0: torch.Tensor
    coeffs_y_1: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        patches_x: int,
        patches_y: int,
        image_width: int,
        image_height: int,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
        precompute_coeffs: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.image_width = image_width
        self.image_height = image_height

        if precompute_coeffs:
            coeffs_x: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
                torch.tile(torch.arange(patches_x), (patches_y,)),
                freq_base=freq_base,
                freq_scale=freq_scale,
                feat_dim=head_dim // 4,
            )
            coeffs_y: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
                torch.repeat_interleave(torch.arange(patches_y), patches_x),
                freq_base=freq_base,
                freq_scale=freq_scale,
                feat_dim=head_dim // 4,
            )
            # Do not save coeffs to checkpoint as `cameras` might change during testing.
            self.register_buffer("coeffs_x_0", coeffs_x[0], persistent=False)
            self.register_buffer("coeffs_x_1", coeffs_x[1], persistent=False)
            self.register_buffer("coeffs_y_0", coeffs_y[0], persistent=False)
            self.register_buffer("coeffs_y_1", coeffs_y[1], persistent=False)
        else:
            self.coeffs_x_0 = None
            self.coeffs_x_1 = None
            self.coeffs_y_0 = None
            self.coeffs_y_1 = None

    # override load_state_dict to not load coeffs if they exist (for backward compatibility)
    def load_state_dict(self, state_dict, strict=True):
        # remove coeffs from state_dict
        state_dict.pop("coeffs_x_0", None)
        state_dict.pop("coeffs_x_1", None)
        state_dict.pop("coeffs_y_0", None)
        state_dict.pop("coeffs_y_1", None)
        super().load_state_dict(state_dict, strict)

    def forward(
        self,
        q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
        Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
        **kwargs,
    ) -> torch.Tensor:
        return prope_dot_product_attention(
            q,
            k,
            v,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=self.patches_x,
            patches_y=self.patches_y,
            image_width=self.image_width,
            image_height=self.image_height,
            coeffs_x=(self.coeffs_x_0, self.coeffs_x_1),
            coeffs_y=(self.coeffs_y_0, self.coeffs_y_1),
            **kwargs,
        )

    def _precompute_and_cache_apply_fns(
        self, viewmats: torch.Tensor, Ks: Optional[torch.Tensor],
        coeffs_x: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        coeffs_y: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        (batch, cameras, _, _) = viewmats.shape
        assert viewmats.shape == (batch, cameras, 4, 4)
        assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
        # self.cameras = cameras

        self.apply_fn_q, self.apply_fn_kv, self.apply_fn_o = _prepare_apply_fns(
            head_dim=self.head_dim,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=self.patches_x,
            patches_y=self.patches_y,
            image_width=self.image_width,
            image_height=self.image_height,
            coeffs_x=(self.coeffs_x_0, self.coeffs_x_1) if coeffs_x is None else coeffs_x,
            coeffs_y=(self.coeffs_y_0, self.coeffs_y_1) if coeffs_y is None else coeffs_y,
        )

    def _apply_to_q(self, q: torch.Tensor) -> torch.Tensor:
        (batch, num_heads, seqlen, head_dim) = q.shape
        # assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert q.shape == (batch, num_heads, seqlen, head_dim)
        assert self.apply_fn_q is not None
        return self.apply_fn_q(q)

    def _apply_to_kv(self, kv: torch.Tensor) -> torch.Tensor:
        (batch, num_heads, seqlen, head_dim) = kv.shape
        # assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert kv.shape == (batch, num_heads, seqlen, head_dim)
        assert self.apply_fn_kv is not None
        return self.apply_fn_kv(kv)

    def _apply_to_o(self, o: torch.Tensor) -> torch.Tensor:
        (batch, num_heads, seqlen, head_dim) = o.shape
        # assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert o.shape == (batch, num_heads, seqlen, head_dim)
        assert self.apply_fn_o is not None
        return self.apply_fn_o(o)


def prope_dot_product_attention(
    q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    *,
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    coeffs_x: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    coeffs_y: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    """Similar to torch.nn.functional.scaled_dot_product_attention, but applies PRoPE-style
    positional encoding.

    Currently, we assume that the sequence length is equal to:

        cameras * patches_x * patches_y

    And token ordering allows the `(seqlen,)` axis to be reshaped into
    `(cameras, patches_x, patches_y)`.
    """
    # We're going to assume self-attention: all inputs are the same shape.
    (batch, num_heads, seqlen, head_dim) = q.shape
    cameras = viewmats.shape[1]
    assert q.shape == k.shape == v.shape
    assert viewmats.shape == (batch, cameras, 4, 4)
    assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
    assert seqlen == cameras * patches_x * patches_y

    apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns(
        head_dim=head_dim,
        viewmats=viewmats,
        Ks=Ks,
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
        coeffs_x=coeffs_x,
        coeffs_y=coeffs_y,
    )

    out = F.scaled_dot_product_attention(
        query=apply_fn_q(q),
        key=apply_fn_kv(k),
        value=apply_fn_kv(v),
        **kwargs,
    )
    out = apply_fn_o(out)
    assert out.shape == (batch, num_heads, seqlen, head_dim)
    return out


def _invert_pose(viewmats: torch.Tensor, pose_dim: int) -> torch.Tensor:
    if pose_dim == 4:
        return _invert_SE3(viewmats)
    if pose_dim == 5:
        return _invert_SE23(viewmats)
    if pose_dim == 6:
        return _invert_SE6(viewmats)
    raise ValueError(f"Unsupported pose_dim={pose_dim}")


def _prepare_apply_fns(
    head_dim: int,  # Q/K/V will have this last dimension
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    coeffs_x: Optional[torch.Tensor] = None,
    coeffs_y: Optional[torch.Tensor] = None,
    rope_scale: Optional[torch.Tensor] = None,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Prepare transforms for PRoPE-style positional encoding."""
    device = viewmats.device
    (batch, cameras, _, _) = viewmats.shape
    dtype = viewmats.dtype
    pose_dim = int(viewmats.shape[-1])
    if viewmats.shape[-2:] not in ((4, 4), (5, 5), (6, 6)):
        raise ValueError(
            "viewmats must contain 4x4 SE(3), 5x5 SE_2(3), or 6x6 (SGal(3)+w) matrices, "
            f"got {viewmats.shape[-2:]}"
        )

    # F-theta rows are evaluated at token centers, yielding per-ray projections.
    per_ray = Ks is not None and Ks.shape[-1] in (9, 13)
    if per_ray:
        from ..ftheta import ftheta_grid
        _, Ks = ftheta_grid(Ks, patches_y, patches_x, image_height, image_width)
        Ks = Ks.reshape(batch, cameras * patches_y * patches_x, 3, 3)
        viewmats = viewmats.repeat_interleave(patches_y * patches_x, dim=1)
    # Normalize camera intrinsics.
    if Ks is not None:
        Ks_norm = torch.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_width
        Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_height
        Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_width - 0.5
        Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_height - 0.5
        Ks_norm[..., 2, 2] = 1.0
        del Ks

        # Compute the camera projection matrices we use in PRoPE.
        # - K is an `image<-camera` transform.
        # - viewmats is a `camera<-world` transform.
        # - P = lift(K) @ viewmats is an `image<-world` transform.
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm, pose_dim=pose_dim), viewmats)
        P_T = P.transpose(-1, -2)
        inv_pose = _invert_pose(viewmats, pose_dim)
        P_inv = torch.einsum(
            "...ij,...jk->...ik",
            inv_pose,
            _lift_K(_invert_K(Ks_norm), pose_dim=pose_dim),
        )

    else:
        # GTA formula. P is `camera<-world` transform.
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = _invert_pose(viewmats, pose_dim)

    projection_count = cameras * patches_y * patches_x if per_ray else cameras
    assert P.shape == P_inv.shape == (batch, projection_count, pose_dim, pose_dim)
    proj_block, rope_x_block, rope_y_block = _ucpe_block_sizes(head_dim, pose_dim)

    # Precompute cos/sin terms for RoPE. We use tiles/repeats for 'row-major'
    # broadcasting.
    if coeffs_x is None:
        coeffs_x = _rope_precompute_coeffs(
            torch.tile(torch.arange(patches_x, device=device), (patches_y * cameras,)),
            freq_base=100.0,
            freq_scale=1.0,
            feat_dim=rope_x_block,
            dtype=dtype,
            angle_scale=rope_scale,
        )
    if coeffs_y is None:
        coeffs_y = _rope_precompute_coeffs(
            torch.tile(
                torch.repeat_interleave(
                    torch.arange(patches_y, device=device), patches_x
                ),
                (cameras,),
            ),
            freq_base=100.0,
            freq_scale=1.0,
            feat_dim=rope_y_block,
            dtype=dtype,
            angle_scale=rope_scale,
        )

    # Block-diagonal transforms to the inputs and outputs of the attention operator.
    transforms_q = [
        (partial(_apply_tiled_projmat, matrix=P_T), proj_block),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), rope_x_block),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), rope_y_block),
    ]
    transforms_kv = [
        (partial(_apply_tiled_projmat, matrix=P_inv), proj_block),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), rope_x_block),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), rope_y_block),
    ]
    transforms_o = [
        (partial(_apply_tiled_projmat, matrix=P), proj_block),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x, inverse=True), rope_x_block),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y, inverse=True), rope_y_block),
    ]

    apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    return apply_fn_q, apply_fn_kv, apply_fn_o


def _ucpe_block_sizes(head_dim: int, pose_dim: int) -> Tuple[int, int, int]:
    if pose_dim == 4:
        if head_dim % 4 != 0:
            raise ValueError(f"SE(3) UCPE requires head_dim divisible by 4, got {head_dim}")
        return head_dim // 2, head_dim // 4, head_dim // 4
    if pose_dim not in (5, 6):
        raise ValueError(f"Unsupported UCPE pose_dim={pose_dim}")
    for proj_block in range(head_dim - (head_dim % pose_dim), pose_dim - 1, -pose_dim):
        rem = head_dim - proj_block
        if rem > 0 and rem % 2 == 0 and (rem // 2) % 2 == 0:
            return proj_block, rem // 2, rem // 2
    raise ValueError(
        f"pose_dim={pose_dim} UCPE requires a head_dim with a {pose_dim}-divisible projection "
        f"block and even RoPE blocks, got head_dim={head_dim}"
    )


def _apply_tiled_projmat(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    matrix: torch.Tensor,  # (batch, cameras, D, D) or (batch, seqlen, D, D)
) -> torch.Tensor:
    """Apply projection matrix to features."""
    # - seqlen => (cameras, patches_x * patches_y)
    # - feat_dim => (feat_dim // D, D)
    (batch, num_heads, seqlen, feat_dim) = feats.shape
    D = matrix.shape[-1]
    assert feat_dim % D == 0, f"feat_dim={feat_dim} must be divisible by D={D}"

    if matrix.shape[1] == seqlen:
        # Per-ray projection: matrix shape [B, seqlen, D, D]
        feats_ = feats.view(batch, num_heads, seqlen, feat_dim // D, D)
        out = torch.einsum("btij,bntpj->bntpi", matrix, feats_)
        return out.reshape(feats.shape)

    # Per-camera projection (original implementation)
    cameras = matrix.shape[1]
    assert seqlen > cameras and seqlen % cameras == 0
    assert matrix.shape == (batch, cameras, D, D)
    assert feat_dim % D == 0
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
    ).reshape(feats.shape)


def _rope_precompute_coeffs(
    positions: torch.Tensor,  # (seqlen,)
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
    dtype: torch.dtype = torch.float32,
    angle_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE coefficients."""
    assert len(positions.shape) == 1
    assert feat_dim % 2 == 0
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base
        ** (
            -torch.arange(num_freqs, device=positions.device)[None, None, None, :]
            / num_freqs
        )
    )
    angles = positions[None, None, :, None] * freqs
    if angle_scale is not None:
        angle_scale = _scale_to_rope_u(angle_scale, device=positions.device)
        angles = angles * angle_scale[:, None, None, None]
    # Shape should be: `(batch, num_heads, seqlen, num_freqs)`; we're
    # broadcasting across `batch` and `num_heads`.
    assert angles.shape == (
        1 if angle_scale is None else angle_scale.shape[0],
        1,
        positions.shape[0],
        num_freqs,
    )
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


def _scale_to_rope_u(
    scale: torch.Tensor,
    device: torch.device,
    s_ref: float = 5.0,
    lo: float = 1e-2,
    hi: float = 740.0,
) -> torch.Tensor:
    scale = torch.as_tensor(scale, device=device, dtype=torch.float32).flatten()
    scale = scale.clamp(min=0.0)
    winsor = torch.where(
        scale < lo, torch.zeros_like(scale), scale.clamp(lo, hi)
    )
    return torch.log1p(winsor / s_ref)


def _rope_apply_coeffs(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    coeffs: Tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    """Apply RoPE coefficients to features. We adopt a 'split' ordering
    convention. (in contrast to 'interleaved')"""
    cos, sin = coeffs
    # We allow (cos, sin) to be either with shape (1, 1, seqlen, feat_dim // 2),
    # or (1, 1, seqlen_per_image, feat_dim // 2) and we repeat it to
    # match the shape of feats.
    if cos.shape[2] != feats.shape[2]:
        n_repeats = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, n_repeats, 1)
        sin = sin.repeat(1, 1, n_repeats, 1)
    assert len(feats.shape) == len(cos.shape) == len(sin.shape) == 4
    assert cos.shape[-1] == sin.shape[-1] == feats.shape[-1] // 2
    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2 :]
    return torch.cat(
        (
            [cos * x_in + sin * y_in, -sin * x_in + cos * y_in]
            if not inverse
            else [cos * x_in - sin * y_in, sin * x_in + cos * y_in]
        ),
        dim=-1,
    )


def _apply_block_diagonal(
    feats: torch.Tensor,  # (..., dim)
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply a block-diagonal function to an input array.

    Each function is specified as a tuple with form:

        ((Tensor) -> Tensor, int)

    Where the integer is the size of the input to the function.
    """
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat(
        [f(x_block) for f, x_block in zip(funcs, x_blocks)],
        dim=-1,
    )
    assert out.shape == feats.shape, "Input/output shapes should match."
    return out


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _invert_SE23(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 5x5 SE_2(3) matrix with layout [[R, v, t], [0, 1, 0], [0, 0, 1]]."""
    assert transforms.shape[-2:] == (5, 5)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., :3, 4] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 4])
    out[..., 3, 3] = 1.0
    out[..., 4, 4] = 1.0
    return out


def _invert_SE6(transforms: torch.Tensor) -> torch.Tensor:
    """Invert the bespoke 6x6 [[R, v, w, p], [0, 1, 0, tau], [0, 0, 1, 0], [0, 0, 0, 1]].

    Matches ``camera_control.se6_inverse``: R->R^T; v->-R^T v; w->-R^T w;
    p->-R^T (p - tau*v); tau->-tau.
    """
    assert transforms.shape[-2:] == (6, 6)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    velocity = transforms[..., :3, 3]
    tau = transforms[..., 3, 5]
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, velocity)
    out[..., :3, 4] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 4])
    trans_corr = transforms[..., :3, 5] - tau.unsqueeze(-1) * velocity
    out[..., :3, 5] = -torch.einsum("...ij,...j->...i", Rinv, trans_corr)
    out[..., 3, 3] = 1.0
    out[..., 4, 4] = 1.0
    out[..., 5, 5] = 1.0
    out[..., 3, 5] = -tau
    return out


def _lift_K(Ks: torch.Tensor, pose_dim: int = 4) -> torch.Tensor:
    """Lift 3x3 intrinsics to homogeneous 4x4 / SE_2(3) 5x5 / bespoke 6x6 matrices.

    Intrinsics touch only the rotation block; the extra velocity/angular/time
    diagonal entries are identity so ``lift(K) @ pose`` scales just the top rows.
    """
    assert Ks.shape[-2:] == (3, 3)
    if pose_dim not in (4, 5, 6):
        raise ValueError(f"pose_dim must be 4, 5 or 6, got {pose_dim}")
    out = torch.zeros(Ks.shape[:-2] + (pose_dim, pose_dim), device=Ks.device, dtype=Ks.dtype)
    out[..., :3, :3] = Ks
    for d in range(3, pose_dim):
        out[..., d, d] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 intrinsics matrices. Assumes no skew."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out
