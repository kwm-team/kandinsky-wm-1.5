"""Raw-frame Plücker raymap construction for SANA-WM's fine camera branch.

Given per-raw-frame poses (`viewmats`, camera<-world) and pinhole intrinsics
(`Ks`, raw-pixel scale), produce the (latent_t, latent_h, latent_w, 6·R)
tensor consumed by ``PluckerEmbeddings`` and the per-block ``plucker_proj``.

Convention (matches ``kandinsky/camera_control.py``):
- ``viewmats`` is ``camera<-world`` (a.k.a. w2c); we invert to get c2w.
- ``Ks`` is a pinhole intrinsics matrix at raw frame resolution ``(frame_h,
  frame_w)``. Off-diagonal skew is ignored (Kandi data is rectified).

Ray geometry per cell:
- ``u_raw = (j + 0.5) * frame_w / latent_w`` (center of the (h, w) latent cell
  in raw pixel coords), and the analogous ``v_raw``.
- ``d_cam = ((u_raw - cx) / fx, (v_raw - cy) / fy, 1)``, then normalize.
- ``d_world = R_c2w @ d_cam``  (still unit-norm; rotation is isometric).
- ``o = -R_c2w @ t_w2c``  (world-space camera center).
- ``ρ = (d_world, o × d_world)`` — the 6-channel Plücker raymap.

Packed across the ``R = raw_frames_per_latent`` raw frames inside one VAE
temporal stride to a ``6 · R``-channel tensor per latent cell.
"""

from __future__ import annotations

import torch


def _c2w_from_viewmats(viewmats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(R_c2w, o_world)`` from a (..., 4, 4) ``camera<-world`` batch.

    ``o_world`` is the world-space camera center; it is what we cross
    ``d_world`` with to get the Plücker moment.
    """
    R_w2c = viewmats[..., :3, :3]
    t_w2c = viewmats[..., :3, 3]
    R_c2w = R_w2c.transpose(-1, -2)
    o_world = -torch.einsum("...ij,...j->...i", R_c2w, t_w2c)
    return R_c2w, o_world


def build_plucker_raymaps_from_viewmats(
    viewmats: torch.Tensor,  # (F_raw, 4, 4), camera<-world
    Ks: torch.Tensor,         # (F_raw, 3, 3), pinhole intrinsics at raw scale
    frame_w: int,
    frame_h: int,
    latent_h: int,
    latent_w: int,
    raw_frames_per_latent: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build raw-rate Plücker raymaps packed by VAE temporal stride.

    Args:
        viewmats: (F_raw, 4, 4) — ``F_raw = latent_t * R``. Each entry is a
            ``camera<-world`` SE(3) transform for one raw frame.
        Ks: (F_raw, 3, 3) — pinhole intrinsics. Off-diagonals ignored.
        frame_w, frame_h: pixel size of the raw frame (used to scale latent
            grid into raw-pixel coordinates).
        latent_h, latent_w: spatial size of the latent grid (one ray per
            latent cell, sampled at cell center in raw pixel coords).
        raw_frames_per_latent: ``R``. ``F_raw`` must equal ``latent_t * R``.
        dtype: optional output dtype; defaults to viewmats.dtype.

    Returns:
        Tensor of shape ``(latent_t, latent_h, latent_w, 6 * R)``: the per-raw
        frame Plücker raymaps concatenated along the channel axis, in raw-frame
        order (frame 0 then 1 then ... then R-1 inside each latent token).
    """
    assert viewmats.ndim == 3 and viewmats.shape[-2:] == (4, 4), viewmats.shape
    ftheta = Ks.ndim == 2 and Ks.shape[-1] in (9, 13)
    assert ftheta or (Ks.ndim == 3 and Ks.shape[-2:] == (3, 3)), Ks.shape
    F_raw = viewmats.shape[0]
    assert F_raw == Ks.shape[0], (F_raw, Ks.shape[0])
    R = int(raw_frames_per_latent)
    assert F_raw % R == 0, f"F_raw={F_raw} not divisible by R={R}"
    latent_t = F_raw // R

    device = viewmats.device
    work_dtype = torch.float32  # numerical stability for geometry
    out_dtype = dtype if dtype is not None else viewmats.dtype

    vm = viewmats.to(work_dtype)
    Ks_ = Ks.to(work_dtype)
    ftheta_rays = None
    if ftheta:
        from ..ftheta import ftheta_grid
        ftheta_rays, _ = ftheta_grid(Ks_, latent_h, latent_w, frame_h, frame_w)
        # The pinhole calculation below is discarded only for FTHETA samples.
        Ks_ = torch.eye(3, device=device, dtype=work_dtype).expand(F_raw, 3, 3)

    # Latent-cell centers in raw pixel coordinates.
    h_idx = torch.arange(latent_h, device=device, dtype=work_dtype)
    w_idx = torch.arange(latent_w, device=device, dtype=work_dtype)
    u_raw = (w_idx + 0.5) * (float(frame_w) / float(latent_w))  # (W,)
    v_raw = (h_idx + 0.5) * (float(frame_h) / float(latent_h))  # (H,)

    grid_v, grid_u = torch.meshgrid(v_raw, u_raw, indexing="ij")  # (H, W)
    # (H, W, 3): pixel-homogeneous coords. We'll broadcast against (F, 1, 1, 3)
    # below.
    pix = torch.stack([grid_u, grid_v, torch.ones_like(grid_u)], dim=-1)

    fx = Ks_[:, 0, 0].view(F_raw, 1, 1)
    fy = Ks_[:, 1, 1].view(F_raw, 1, 1)
    cx = Ks_[:, 0, 2].view(F_raw, 1, 1)
    cy = Ks_[:, 1, 2].view(F_raw, 1, 1)

    # Per-frame camera-frame ray direction at each cell.
    x = (pix[..., 0].unsqueeze(0) - cx) / fx  # (F, H, W)
    y = (pix[..., 1].unsqueeze(0) - cy) / fy  # (F, H, W)
    z = torch.ones_like(x)
    d_cam = torch.stack([x, y, z], dim=-1)    # (F, H, W, 3)
    d_cam = d_cam / d_cam.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    if ftheta_rays is not None:
        d_cam = ftheta_rays

    R_c2w, o_world = _c2w_from_viewmats(vm)   # (F, 3, 3), (F, 3)

    # d_world = R_c2w @ d_cam  (per pixel)
    d_world = torch.einsum("fij,fhwj->fhwi", R_c2w, d_cam)
    # The cross product moment m = o × d_world. Broadcast o over (H, W).
    o_bcast = o_world[:, None, None, :].expand_as(d_world)
    m = torch.cross(o_bcast, d_world, dim=-1)

    plucker_raw = torch.cat([d_world, m], dim=-1)  # (F_raw, H, W, 6)

    # Pack R raw frames per latent token along the channel axis:
    # (latent_t, R, H, W, 6) -> (latent_t, H, W, R, 6) -> (latent_t, H, W, 6*R)
    plucker = plucker_raw.view(latent_t, R, latent_h, latent_w, 6)
    plucker = plucker.permute(0, 2, 3, 1, 4).contiguous().view(
        latent_t, latent_h, latent_w, 6 * R
    )
    return plucker.to(out_dtype)


def verify_raymaps_correct(
    *,
    frame_w: int = 256,
    frame_h: int = 256,
    latent_h: int = 16,
    latent_w: int = 16,
    raw_frames_per_latent: int = 4,
    latent_t: int = 3,
    device: torch.device | str = "cpu",
    atol: float = 1e-4,
) -> list[tuple[str, bool, str]]:
    """Numerical sanity tests for ``build_plucker_raymaps_from_viewmats``.

    Returns a list of ``(name, passed, info)`` tuples. Designed to be called
    from the smoke / unit-test driver without side effects.

    Tests:
      1. Identity pose + centered pinhole: at the center cell, ``d_world``
         is the camera +z axis, ``o`` is the origin, moment is zero.
      2. ``||d_world||`` = 1 within `atol` for all cells.
      3. Plücker invariant ``d · m`` = 0 within `atol`.
      4. Pure translation t_c2w = (a, b, c) with R=I: ``o`` matches, and
         ``m = o × d`` decomposes correctly.
      5. Pure rotation 90° about world-Y (c2w): ``d_cam = (0,0,1)`` maps to
         ``d_world = (1, 0, 0)`` (camera now looks along +X world).
      6. NaN/Inf-free for random valid inputs.
    """
    results: list[tuple[str, bool, str]] = []
    device = torch.device(device)
    R = raw_frames_per_latent
    F_raw = latent_t * R

    # ---- common pinhole intrinsics ----
    fx = fy = float(frame_w)
    cx = frame_w / 2.0
    cy = frame_h / 2.0
    K_one = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device
    )

    # ---- Test 1: identity poses + centered pinhole, 1x1 grid ----
    # A 1x1 latent grid puts the single cell center at (frame_w/2, frame_h/2),
    # exactly the principal point — so the camera-frame ray is exactly +z
    # without any half-cell normalization wobble.
    viewmats = torch.eye(4, device=device, dtype=torch.float32).expand(F_raw, 4, 4).clone()
    Ks = K_one.expand(F_raw, 3, 3).clone()
    rays_one = build_plucker_raymaps_from_viewmats(
        viewmats, Ks, frame_w, frame_h, 1, 1, R
    )
    p_one = rays_one.reshape(latent_t, 1, 1, R, 6)
    d_one = p_one[0, 0, 0, 0, :3]
    m_one = p_one[0, 0, 0, 0, 3:]
    cond1 = (
        abs(d_one[0].item() - 0.0) < atol
        and abs(d_one[1].item() - 0.0) < atol
        and abs(d_one[2].item() - 1.0) < atol
        and m_one.abs().max().item() < atol
    )
    results.append(("identity_center_ray_is_z_axis", cond1,
                    f"d={d_one.tolist()} m={m_one.tolist()}"))

    # Also build the full grid for the remaining tests.
    rays = build_plucker_raymaps_from_viewmats(
        viewmats, Ks, frame_w, frame_h, latent_h, latent_w, R
    )

    # ---- Test 2: ||d_world|| = 1 ----
    d = rays.reshape(latent_t, latent_h, latent_w, R, 6)[..., :3]
    norms = d.norm(dim=-1)
    cond2 = (norms - 1.0).abs().max().item() < atol
    results.append(("d_world_unit_norm", cond2,
                    f"max|‖d‖-1|={float((norms-1).abs().max()):.2e}"))

    # ---- Test 3: Plücker invariant d · m = 0 ----
    p = rays.reshape(latent_t, latent_h, latent_w, R, 6)
    dot = (p[..., :3] * p[..., 3:]).sum(dim=-1)
    cond3 = dot.abs().max().item() < atol
    results.append(("plucker_invariant_d_dot_m_zero", cond3,
                    f"max|d·m|={dot.abs().max().item():.2e}"))

    # ---- Test 4: pure translation, R_c2w = I ----
    t_c2w = torch.tensor([1.0, -2.0, 3.0], device=device)
    c2w = torch.eye(4, device=device).expand(F_raw, 4, 4).clone()
    c2w[:, :3, 3] = t_c2w
    # viewmats (w2c) is the inverse: with R=I, t_w2c = -t_c2w.
    vm = torch.eye(4, device=device).expand(F_raw, 4, 4).clone()
    vm[:, :3, 3] = -t_c2w
    rays_t = build_plucker_raymaps_from_viewmats(
        vm, Ks, frame_w, frame_h, latent_h, latent_w, R
    )
    pt = rays_t.reshape(latent_t, latent_h, latent_w, R, 6)
    d_t = pt[..., :3]
    m_t = pt[..., 3:]
    # Recompute expected moment m = t_c2w × d_world for a few sampled pixels.
    sample = d_t[0, latent_h // 4, latent_w // 4, 0]
    m_expected = torch.cross(t_c2w, sample, dim=-1)
    cond4 = torch.allclose(m_t[0, latent_h // 4, latent_w // 4, 0], m_expected, atol=atol)
    results.append(("pure_translation_moment_equals_o_cross_d", cond4,
                    f"m_actual={m_t[0, latent_h//4, latent_w//4, 0].tolist()} "
                    f"m_expected={m_expected.tolist()}"))

    # ---- Test 5: 90° rotation about Y so camera +z maps to world +x ----
    # c2w rotation: R_y(90°) such that R @ (0,0,1)^T = (1,0,0)^T. That's
    #    [[ 0, 0, 1],
    #     [ 0, 1, 0],
    #     [-1, 0, 0]]
    R_c2w = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], device=device
    )
    R_w2c = R_c2w.T
    c2w_r = torch.eye(4, device=device).expand(F_raw, 4, 4).clone()
    c2w_r[:, :3, :3] = R_c2w
    vm_r = torch.eye(4, device=device).expand(F_raw, 4, 4).clone()
    vm_r[:, :3, :3] = R_w2c
    vm_r[:, :3, 3] = 0.0  # origin at world 0 ⇒ no translation flip needed
    # Use the 1x1 grid again so the single cell aligns exactly with the
    # principal point — then d_cam = +z exactly, and the rotation maps it
    # bit-for-bit to world +x.
    rays_r = build_plucker_raymaps_from_viewmats(
        vm_r, Ks, frame_w, frame_h, 1, 1, R
    )
    pr = rays_r.reshape(latent_t, 1, 1, R, 6)
    d_center_r = pr[0, 0, 0, 0, :3]
    cond5 = (
        abs(d_center_r[0].item() - 1.0) < atol
        and abs(d_center_r[1].item() - 0.0) < atol
        and abs(d_center_r[2].item() - 0.0) < atol
    )
    results.append(("pure_rotation_camera_z_maps_to_world_x", cond5,
                    f"d={d_center_r.tolist()}"))

    # ---- Test 6.5: time-varying poses produce per-frame-correct slices ----
    # Catches reshape/permute order bugs in the (F_raw, ...) -> (latent_t, R, ...)
    # packing. Each raw frame f has a distinct camera translation (no rotation).
    # We then verify, at a handful of explicit (f, h, w) triples, that the
    # tensor slot ``rays[t = f//R, h, w, r = f%R, :]`` exactly equals the ρ we
    # reconstruct analytically from viewmats[f] and the cell-center pixel.
    o_per_frame = (
        torch.linspace(0.5, 5.0, F_raw, device=device).unsqueeze(-1)
        * torch.tensor([[1.0, 0.7, -0.3]], device=device)
    )  # (F, 3), distinct per frame
    vm_tv = torch.eye(4, device=device).expand(F_raw, 4, 4).clone()
    # R_c2w=I ⇒ o_world = -R_c2w @ t_w2c = -t_w2c, so t_w2c = -o_per_frame.
    vm_tv[:, :3, 3] = -o_per_frame
    rays_tv = build_plucker_raymaps_from_viewmats(
        vm_tv, Ks, frame_w, frame_h, latent_h, latent_w, R
    )
    p_tv = rays_tv.reshape(latent_t, latent_h, latent_w, R, 6)
    # (a) per-frame slices at one cell are pairwise distinct (catches "all R
    # slots got the same raymap" bugs).
    cell = p_tv[0, 0, 0, :, :]  # (R, 6)
    pairwise = (cell[:, None] - cell[None, :]).abs().sum(-1)
    pairwise.fill_diagonal_(float("inf"))
    cond_distinct = pairwise.min().item() > atol

    # (b) Per-frame correctness via explicit reconstruction at sample cells.
    # We sample three (f, h, w) triples spanning early/mid/late frames.
    triples = [(0, 0, 0), (F_raw // 2, latent_h // 2, latent_w // 2), (F_raw - 1, latent_h - 1, latent_w - 1)]
    max_err = 0.0
    for f, h_idx, w_idx in triples:
        u_raw = (w_idx + 0.5) * (frame_w / latent_w)
        v_raw = (h_idx + 0.5) * (frame_h / latent_h)
        d_cam = torch.tensor(
            [(u_raw - cx) / fx, (v_raw - cy) / fy, 1.0],
            device=device, dtype=torch.float32,
        )
        d_cam = d_cam / d_cam.norm()
        # R_c2w = I in this test
        d_world_exp = d_cam
        o_exp = o_per_frame[f]
        m_exp = torch.cross(o_exp, d_world_exp, dim=-1)
        rho_exp = torch.cat([d_world_exp, m_exp])
        rho_act = p_tv[f // R, h_idx, w_idx, f % R, :]
        max_err = max(max_err, float((rho_act - rho_exp).abs().max()))
    cond_recover = max_err < atol
    results.append((
        "time_varying_frames_produce_distinct_slices",
        cond_distinct and cond_recover,
        f"min_pairwise_Δ={pairwise.min().item():.2e} max_per_cell_err={max_err:.2e}",
    ))

    # ---- Test 6: NaN/Inf-free for random valid inputs ----
    torch.manual_seed(0)
    rand_R = torch.randn(F_raw, 3, 3, device=device)
    U, _, Vh = torch.linalg.svd(rand_R)
    rand_R = U @ Vh  # nearest orthogonal (each frame a valid rotation)
    rand_t = torch.randn(F_raw, 3, device=device) * 2.0
    vm_rand = torch.eye(4, device=device).expand(F_raw, 4, 4).clone()
    vm_rand[:, :3, :3] = rand_R
    vm_rand[:, :3, 3] = rand_t
    rays_rand = build_plucker_raymaps_from_viewmats(
        vm_rand, Ks, frame_w, frame_h, latent_h, latent_w, R
    )
    cond6 = bool(torch.isfinite(rays_rand).all().item())
    results.append(("random_input_finite", cond6,
                    f"any_nan={torch.isnan(rays_rand).any().item()} "
                    f"any_inf={torch.isinf(rays_rand).any().item()}"))

    return results
