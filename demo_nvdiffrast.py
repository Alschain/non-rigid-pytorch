"""Multi-view differentiable rendering non-rigid deformation demo using nvdiffrast.

Demonstrates how to combine DeformableMesh (ED-Graph) with NvdiffrastMeshUlitRenderer
to deform a source mesh to match a target mesh using ONLY 2D multi-view observations
(antialiased silhouette, depth, and normal maps) without 3D point-to-point correspondences.
"""

import os
import time
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from util.mesh import DeformableMesh
from util.mesh_util import compute_normals, compute_edge_length, knn_points
from util.obj_io import load_obj, save_obj
from renderer.ulit_renderer import NvdiffrastMeshUlitRenderer


def look_at_opencv(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build 4x4 world-to-camera extrinsic matrix in OpenCV convention (X right, Y down, Z forward)."""
    z_axis = target - eye
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(z_axis, up)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    E = np.eye(4, dtype=np.float32)
    E[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=0)
    E[:3, 3] = -E[:3, :3] @ eye
    return E


def build_ring_cameras(num_views: int = 8, radius: float = 1.15, render_size: int = 256):
    """Create multi-view OpenCV cameras orbiting the object at alternating elevations."""
    fx = fy = render_size * 1.25
    cx = cy = render_size * 0.5
    K = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    extrinsics = []
    for i in range(num_views):
        azim = 2.0 * np.pi * i / num_views + 0.35
        elev = np.deg2rad(32.0 if i % 2 == 0 else -24.0)
        eye = np.array(
            [
                radius * np.cos(elev) * np.cos(azim),
                radius * np.cos(elev) * np.sin(azim),
                radius * np.sin(elev),
            ],
            dtype=np.float32,
        )
        extrinsics.append(look_at_opencv(eye, target, up))
    return K, extrinsics


def compute_3d_metrics(src_vs: torch.Tensor, tgt_vs: torch.Tensor):
    with torch.no_grad():
        d_s2t, _, _ = knn_points(src_vs.unsqueeze(0), tgt_vs.unsqueeze(0), K=1)
        d_t2s, _, _ = knn_points(tgt_vs.unsqueeze(0), src_vs.unsqueeze(0), K=1)
        s2t = torch.sqrt(d_s2t[0, :, 0].clamp_min(0.0))
        t2s = torch.sqrt(d_t2s[0, :, 0].clamp_min(0.0))
        return {
            "s2t_mean_mm": s2t.mean().item() * 1000.0,
            "chamfer_mm": 0.5 * (s2t.mean().item() + t2s.mean().item()) * 1000.0,
        }


def _normal_to_rgb_u8(normal: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    n = normal[0].detach().cpu().numpy()
    m = mask[0].detach().cpu().numpy()
    rgb = np.clip(n * 0.5 + 0.5, 0.0, 1.0)
    bg = np.array([0.95, 0.96, 0.98], dtype=np.float32)
    img = rgb * m + bg * (1.0 - m)
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def _depth_to_rgb_u8(depth: torch.Tensor, mask: torch.Tensor, zmin: float = 0.75, zmax: float = 1.65) -> np.ndarray:
    d = depth[0, ..., 0].detach().cpu().numpy()
    m = mask[0, ..., 0].detach().cpu().numpy()
    norm_d = 1.0 - np.clip((d - zmin) / max(zmax - zmin, 1e-6), 0.0, 1.0)
    # Cool blue-cyan depth shading
    rgb = np.stack([0.12 + 0.55 * norm_d, 0.30 + 0.58 * norm_d, 0.55 + 0.42 * norm_d], axis=-1)
    bg = np.array([0.95, 0.96, 0.98], dtype=np.float32)
    img = rgb * m[..., None] + bg * (1.0 - m[..., None])
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def _load_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def save_nvdiffrast_comparison(src_mesh, tgt_mesh, def_mesh, out_path: str, device: str):
    """Render side-by-side nvdiffrast normal + depth maps of Source, Target, and Deformed."""
    vis_size = 384
    renderer = NvdiffrastMeshUlitRenderer(
        {"render_size": [vis_size, vis_size], "device": device, "rastype": "cuda"}
    )
    fx = fy = vis_size * 1.45
    cx = cy = vis_size * 0.5
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    E = look_at_opencv(
        eye=np.array([0.56, -0.94, 0.90], dtype=np.float32),
        target=np.array([0.01, 0.0, 0.01], dtype=np.float32),
        up=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    renderer.set_camera_intrinsic(K)
    renderer.set_camera_extrinsic(E)

    panels_top = []
    panels_bot = []
    titles = ["Source (nvdiffrast)", "Target (2D Supervision)", "Deformed (2D-Only Optim)"]

    with torch.no_grad():
        for mesh, title in zip([src_mesh, tgt_mesh, def_mesh], titles):
            out = renderer(mesh, antialias=True, render_normal=True)
            n_img = _normal_to_rgb_u8(out["normal"], out["mask"])
            d_img = _depth_to_rgb_u8(out["depth"], out["mask"], zmin=0.95, zmax=1.85)
            panels_top.append((n_img, title, "Normal Map"))
            panels_bot.append((d_img, None, "Depth Map"))

    def _label(arr, top_txt, bot_txt):
        pil = Image.fromarray(arr)
        draw = ImageDraw.Draw(pil)
        w, h = pil.size
        f_top = _load_font(17, bold=True)
        f_bot = _load_font(14, bold=False)
        if top_txt:
            bb = draw.textbbox((0, 0), top_txt, font=f_top)
            draw.text(((w - (bb[2] - bb[0])) // 2, 12), top_txt, fill=(30, 41, 59), font=f_top)
        if bot_txt:
            bb = draw.textbbox((0, 0), bot_txt, font=f_bot)
            draw.text(((w - (bb[2] - bb[0])) // 2, h - 26), bot_txt, fill=(71, 85, 105), font=f_bot)
        return np.asarray(pil)

    sep_v = np.full((vis_size, 3, 3), 215, dtype=np.uint8)
    row1 = np.concatenate(
        [_label(p, t, b) for p, t, b in [panels_top[0]]]
        + [sep_v, _label(*panels_top[1]), sep_v, _label(*panels_top[2])],
        axis=1,
    )
    row2 = np.concatenate(
        [_label(*panels_bot[0]), sep_v, _label(*panels_bot[1]), sep_v, _label(*panels_bot[2])],
        axis=1,
    )
    sep_h = np.full((3, row1.shape[1], 3), 215, dtype=np.uint8)
    grid = np.concatenate([row1, sep_h, row2], axis=0)

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    Image.fromarray(grid).save(out_path, optimize=True)
    print(f"Saved nvdiffrast visualization: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-view differentiable rendering non-rigid deformation demo (nvdiffrast)"
    )
    parser.add_argument("--src", type=str, default="./assets/data/source.obj")
    parser.add_argument("--target", type=str, default="./assets/data/target.obj")
    parser.add_argument("--out_dir", type=str, default="./debug_out/nvdiffrast_demo")
    parser.add_argument("--save_vis", type=str, default="./assets/demo_nvdiffrast.png")
    parser.add_argument("--render_size", type=int, default=256)
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--max_iter", type=int, default=600)
    parser.add_argument("--log_iter", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--node_num", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "nvdiffrast CUDA renderer requires a CUDA GPU."
    device = "cuda"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    src_mesh = load_obj(args.src, device=device)
    tgt_mesh = load_obj(args.target, device=device)

    # Constant white vertex colors so renderer's antialiased color output acts as a differentiable silhouette
    src_mesh.vcs = torch.ones_like(src_mesh.vs)
    tgt_mesh.vcs = torch.ones_like(tgt_mesh.vs)

    init_metrics = compute_3d_metrics(src_mesh.vs, tgt_mesh.vs)
    print(
        f"[Before] 3D Src->Tgt mean: {init_metrics['s2t_mean_mm']:.2f} mm, "
        f"Chamfer: {init_metrics['chamfer_mm']:.2f} mm"
    )

    renderer = NvdiffrastMeshUlitRenderer(
        {"render_size": [args.render_size, args.render_size], "device": device, "rastype": "cuda"}
    )
    K, extrinsics = build_ring_cameras(num_views=args.num_views, render_size=args.render_size)
    renderer.set_camera_intrinsic(K)

    # Pre-render 2D multi-view target observations (silhouette, depth, normal)
    target_views = []
    with torch.no_grad():
        for E in extrinsics:
            renderer.set_camera_extrinsic(E)
            out = renderer(tgt_mesh, antialias=True, render_normal=True)
            target_views.append(
                {
                    "extrinsic": E,
                    "sil": out["color"][..., :1].detach(),
                    "mask": out["mask"].detach(),
                    "depth": out["depth"].detach(),
                    "normal": out["normal"].detach(),
                }
            )

    # Build Embedded Deformation Graph
    deformable_mesh = DeformableMesh(src_mesh)
    deformable_mesh.build_ednodes_on_vertices(args.node_num, ed_k=6, v_k=4)
    deformable_mesh.generate_trainable_params()
    deformable_mesh.set_params_trainable()

    ori_vn, ori_fn = compute_normals(src_mesh.vs.detach(), src_mesh.f_v_idx)
    ori_link_nc = ori_vn[src_mesh.edges[:, 0]] - ori_vn[src_mesh.edges[:, 1]]
    ori_link_fc = ori_fn[src_mesh.connected_faces[:, 0]] - ori_fn[src_mesh.connected_faces[:, 1]]
    ori_edge_len = compute_edge_length(src_mesh.vs.detach(), src_mesh.edges)

    optimizer = torch.optim.AdamW([deformable_mesh.Rs, deformable_mesh.Ts], lr=args.lr, weight_decay=1e-4)
    work_mesh = src_mesh.clone()

    t0 = time.time()
    for it in range(args.max_iter + 1):
        if it in (args.max_iter // 2, (3 * args.max_iter) // 4):
            for g in optimizer.param_groups:
                g["lr"] *= 0.5

        deformed = deformable_mesh.apply_deformation()
        work_mesh.vs = deformed["deformed_vertices"]

        loss_sil = 0.0
        loss_depth = 0.0
        loss_norm = 0.0
        for tv in target_views:
            renderer.set_camera_extrinsic(tv["extrinsic"])
            pred = renderer(work_mesh, antialias=True, render_normal=True)
            pred_sil = pred["color"][..., :1]

            loss_sil = loss_sil + torch.mean((pred_sil - tv["sil"]) ** 2)
            # Supervise depth & normal on overlapping foreground pixels
            overlap = pred["mask"] * tv["mask"]
            denom = overlap.sum().clamp_min(1.0)
            loss_depth = loss_depth + ((pred["depth"] - tv["depth"]).abs() * overlap).sum() / denom
            loss_norm = loss_norm + (((pred["normal"] - tv["normal"]) ** 2) * overlap).sum() / denom

        loss_sil = 20.0 * (loss_sil / len(target_views))
        loss_depth = 10.0 * (loss_depth / len(target_views))
        loss_norm = 2.0 * (loss_norm / len(target_views))

        # ED-Graph rigidity & smoothness regularizers
        Rs = deformable_mesh.Rs
        rot = (
            (Rs[:, 0] * Rs[:, 1]).sum(dim=1) ** 2
            + (Rs[:, 0] * Rs[:, 2]).sum(dim=1) ** 2
            + (Rs[:, 1] * Rs[:, 2]).sum(dim=1) ** 2
            + ((Rs[:, 0] * Rs[:, 0]).sum(dim=1) - 1.0) ** 2
            + ((Rs[:, 1] * Rs[:, 1]).sum(dim=1) - 1.0) ** 2
            + ((Rs[:, 2] * Rs[:, 2]).sum(dim=1) - 1.0) ** 2
        )
        loss_rot = 5.0 * rot.sum()

        reg_dist = ((deformed["deformed_EDNeighbors"] - deformed["deformed_EDNodes"].unsqueeze(1)) ** 2).sum(dim=-1)
        loss_reg = (2.0 / (1.5 ** (it // 150))) * reg_dist.sum()

        vn, fn = compute_normals(work_mesh.vs, work_mesh.f_v_idx)
        loss_nc = (0.005 / (1.5 ** (it // 200))) * (
            ((vn[src_mesh.edges[:, 0]] - vn[src_mesh.edges[:, 1]] - ori_link_nc) ** 2).sum()
            + ((fn[src_mesh.connected_faces[:, 0]] - fn[src_mesh.connected_faces[:, 1]] - ori_link_fc) ** 2).sum()
        )
        edge_len = compute_edge_length(work_mesh.vs, src_mesh.edges)
        loss_el = (5.0 / (1.5 ** (it // 200))) * ((edge_len - ori_edge_len) ** 2).sum()

        total_loss = loss_sil + loss_depth + loss_norm + loss_rot + loss_reg + loss_nc + loss_el

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if it % args.log_iter == 0:
            print(
                f"iter {it:4d} | total {total_loss.item():.5f} | "
                f"sil {loss_sil.item():.5f} | depth {loss_depth.item():.5f} | "
                f"norm {loss_norm.item():.5f} | reg {loss_reg.item():.5f}"
            )

    elapsed = time.time() - t0
    with torch.no_grad():
        final_mesh = src_mesh.clone()
        final_mesh.vs = deformable_mesh.apply_deformation()["deformed_vertices"].detach().clone()
        final_mesh.update_normal()

    out_obj = os.path.join(args.out_dir, "deformed_nvdiffrast.obj")
    save_obj(out_obj, final_mesh)

    final_metrics = compute_3d_metrics(final_mesh.vs, tgt_mesh.vs)
    print(
        f"[After ] 3D Src->Tgt mean: {final_metrics['s2t_mean_mm']:.2f} mm, "
        f"Chamfer: {final_metrics['chamfer_mm']:.2f} mm (2D supervision only, {elapsed:.2f}s)"
    )

    if args.save_vis:
        save_nvdiffrast_comparison(src_mesh, tgt_mesh, final_mesh, args.save_vis, device=device)


if __name__ == "__main__":
    main()
