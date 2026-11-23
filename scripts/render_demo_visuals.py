"""Render publication-quality visuals and optimization GIF for the non-rigid-pytorch demo.

Uses a self-contained perspective z-buffer rasterizer with smooth per-pixel Phong
lighting and 2x supersampling antialiasing (no OpenGL/nvdiffrast compilation needed).
"""

import os
import glob
import argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio


def load_obj_np(path: str):
    """Load vertices (N, 3) and triangle faces (F, 3) from a Wavefront .obj file."""
    verts = []
    faces = []
    edges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            tag = parts[0].lower()
            if tag == "v":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif tag == "f":
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                for i in range(len(idx) - 2):
                    faces.append([idx[0], idx[i + 1], idx[i + 2]])
            elif tag == "l":
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                for i in range(len(idx) - 1):
                    edges.append([idx[i], idx[i + 1]])
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)
    edges = np.asarray(edges, dtype=np.int32) if edges else np.zeros((0, 2), dtype=np.int32)
    return verts, faces, edges


def compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn_len = np.linalg.norm(fn, axis=-1, keepdims=True) + 1e-12
    fn = fn / fn_len
    vns = np.zeros_like(verts)
    np.add.at(vns, faces[:, 0], fn)
    np.add.at(vns, faces[:, 1], fn)
    np.add.at(vns, faces[:, 2], fn)
    vns = vns / (np.linalg.norm(vns, axis=-1, keepdims=True) + 1e-12)
    return vns


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray):
    """Build OpenCV-style world-to-camera extrinsic matrix (X right, Y down, Z forward)."""
    z_axis = target - eye
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(z_axis, up)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)  # points downward in image space
    R = np.stack([x_axis, y_axis, z_axis], axis=0)
    t = -R @ eye
    return R.astype(np.float32), t.astype(np.float32)


def project_points(pts: np.ndarray, R: np.ndarray, t: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    cam = (pts @ R.T) + t[None, :]
    z = cam[:, 2]
    u = fx * (cam[:, 0] / np.maximum(z, 1e-6)) + cx
    v = fy * (cam[:, 1] / np.maximum(z, 1e-6)) + cy
    return cam, u, v, z


def rasterize_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    size: int = 520,
    ssaa: int = 2,
    base_color=(0.30, 0.56, 0.86),
    vert_colors: np.ndarray = None,
    bg_top=(0.97, 0.98, 0.99),
    bg_bot=(0.91, 0.93, 0.96),
    fov_deg: float = 38.0,
    ed_nodes: np.ndarray = None,
    ed_edges: np.ndarray = None,
    ghost_verts: np.ndarray = None,
    ghost_faces: np.ndarray = None,
    ghost_color=(0.16, 0.72, 0.56),
    ghost_alpha: float = 0.28,
):
    """Render mesh with exact z-buffer, smooth Phong lighting, optional ghost overlay and ED-Graph."""
    W = H = size * ssaa
    fx = fy = 0.5 * W / np.tan(np.deg2rad(fov_deg) * 0.5)
    cx = cy = 0.5 * W

    def _raster_pass(vs, fs, v_cols, color_rgb):
        vns = compute_vertex_normals(vs, fs)
        vns_cam = vns @ R.T
        _, u, v, z = project_points(vs, R, t, fx, fy, cx, cy)

        zbuf = np.full((H, W), np.inf, dtype=np.float32)
        nbuf = np.zeros((H, W, 3), dtype=np.float32)
        cbuf = np.zeros((H, W, 3), dtype=np.float32)
        mask = np.zeros((H, W), dtype=bool)

        if v_cols is None:
            v_cols_arr = np.broadcast_to(np.asarray(color_rgb, dtype=np.float32), (len(vs), 3))
        else:
            v_cols_arr = v_cols.astype(np.float32)

        f0, f1, f2 = fs[:, 0], fs[:, 1], fs[:, 2]
        u0, u1, u2 = u[f0], u[f1], u[f2]
        v0, v1, v2 = v[f0], v[f1], v[f2]
        z0, z1, z2 = z[f0], z[f1], z[f2]

        # Signed 2x area in screen space (OpenCV Y-down -> front-facing has area < 0 or > 0 depending on winding)
        area = (u1 - u0) * (v2 - v0) - (v1 - v0) * (u2 - u0)
        # Perspective backface culling: triangle faces camera when dot(face_normal, view_ray_from_cam) < 0
        cam_pts = (vs @ R.T) + t[None, :]
        fn_cam = np.cross(cam_pts[f1] - cam_pts[f0], cam_pts[f2] - cam_pts[f0])
        view_dot = np.sum(fn_cam * cam_pts[f0], axis=-1)
        front = (view_dot < 0.0) & (np.abs(area) > 1e-4) & (z0 > 0.05) & (z1 > 0.05) & (z2 > 0.05)
        valid_ids = np.nonzero(front)[0]

        for fid in valid_ids:
            a = area[fid]
            x_min = max(int(np.floor(min(u0[fid], u1[fid], u2[fid]))), 0)
            x_max = min(int(np.ceil(max(u0[fid], u1[fid], u2[fid]))), W - 1)
            y_min = max(int(np.floor(min(v0[fid], v1[fid], v2[fid]))), 0)
            y_max = min(int(np.ceil(max(v0[fid], v1[fid], v2[fid]))), H - 1)
            if x_min > x_max or y_min > y_max:
                continue

            xs = np.arange(x_min, x_max + 1, dtype=np.float32)[None, :] + 0.5
            ys = np.arange(y_min, y_max + 1, dtype=np.float32)[:, None] + 0.5

            inv_a = 1.0 / a
            w0 = ((u1[fid] - xs) * (v2[fid] - ys) - (v1[fid] - ys) * (u2[fid] - xs)) * inv_a
            w1 = ((u2[fid] - xs) * (v0[fid] - ys) - (v2[fid] - ys) * (u0[fid] - xs)) * inv_a
            w2 = 1.0 - w0 - w1

            inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
            if not np.any(inside):
                continue

            # Perspective-correct depth interpolation
            iz0, iz1, iz2 = 1.0 / z0[fid], 1.0 / z1[fid], 1.0 / z2[fid]
            inv_z = w0 * iz0 + w1 * iz1 + w2 * iz2
            zz = 1.0 / np.maximum(inv_z, 1e-8)

            z_patch = zbuf[y_min : y_max + 1, x_min : x_max + 1]
            upd = inside & (zz < z_patch)
            if not np.any(upd):
                continue

            z_patch[upd] = zz[upd]
            mask[y_min : y_max + 1, x_min : x_max + 1][upd] = True

            # Interpolate normal and vertex color
            i0, i1, i2 = f0[fid], f1[fid], f2[fid]
            w0u, w1u, w2u = w0[upd, None], w1[upd, None], w2[upd, None]
            n_interp = w0u * vns_cam[i0] + w1u * vns_cam[i1] + w2u * vns_cam[i2]
            c_interp = w0u * v_cols_arr[i0] + w1u * v_cols_arr[i1] + w2u * v_cols_arr[i2]

            nbuf[y_min : y_max + 1, x_min : x_max + 1][upd] = n_interp
            cbuf[y_min : y_max + 1, x_min : x_max + 1][upd] = c_interp

        return zbuf, nbuf, cbuf, mask

    def _shade(nbuf, cbuf, mask):
        n = nbuf[mask]
        n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
        # Flip normals pointing away from camera just in case
        n = np.where(n[:, 2:3] > 0, -n, n)
        c = cbuf[mask]

        # Camera-space 3-point studio lighting (camera looks along +Z, view dir is (0, 0, -1))
        view_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        key_dir = np.array([0.45, -0.55, -0.70], dtype=np.float32)
        key_dir /= np.linalg.norm(key_dir)
        fill_dir = np.array([-0.65, 0.25, -0.71], dtype=np.float32)
        fill_dir /= np.linalg.norm(fill_dir)
        rim_dir = np.array([0.0, -0.60, 0.80], dtype=np.float32)
        rim_dir /= np.linalg.norm(rim_dir)

        diff_key = np.clip(n @ key_dir, 0.0, 1.0)[:, None]
        diff_fill = np.clip(n @ fill_dir, 0.0, 1.0)[:, None]
        # Blinn-Phong specular
        half_vec = key_dir + view_dir
        half_vec /= np.linalg.norm(half_vec)
        spec = (np.clip(n @ half_vec, 0.0, 1.0) ** 36.0)[:, None]
        # Fresnel rim highlight
        ndotv = np.clip(n @ view_dir, 0.0, 1.0)[:, None]
        rim = ((1.0 - ndotv) ** 2.6) * (0.35 + 0.65 * np.clip(n @ rim_dir, 0.0, 1.0)[:, None])

        ambient = 0.24
        lit = (
            c * (ambient + 0.68 * diff_key + 0.22 * diff_fill)
            + 0.22 * spec * np.array([[1.0, 0.98, 0.95]], dtype=np.float32)
            + 0.14 * rim * np.array([[0.90, 0.95, 1.0]], dtype=np.float32)
        )
        return np.clip(lit, 0.0, 1.0)

    # Vertical gradient studio background
    grad = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None, None]
    bg_col = (1.0 - grad) * np.asarray(bg_top, dtype=np.float32)[None, None, :] + grad * np.asarray(
        bg_bot, dtype=np.float32
    )[None, None, :]
    img = np.broadcast_to(bg_col, (H, W, 3)).copy()

    # Optional ghost target mesh underlay/overlay
    if ghost_verts is not None and ghost_faces is not None:
        gz, gn, gc, gmask = _raster_pass(ghost_verts, ghost_faces, None, ghost_color)
        if np.any(gmask):
            g_shaded = _shade(gn, gc, gmask)
            img[gmask] = (1.0 - ghost_alpha) * img[gmask] + ghost_alpha * g_shaded

    zbuf, nbuf, cbuf, mask = _raster_pass(verts, faces, vert_colors, base_color)
    if np.any(mask):
        img[mask] = _shade(nbuf, cbuf, mask)

    # If ghost is in front of main mesh, blend a subtle hint so alignment is visible
    if ghost_verts is not None and ghost_faces is not None:
        in_front = gmask & mask & (gz < zbuf - 1e-3)
        if np.any(in_front):
            g_shaded_front = _shade(gn, gc, in_front)
            img[in_front] = (1.0 - ghost_alpha * 0.75) * img[in_front] + (ghost_alpha * 0.75) * g_shaded_front

    img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)

    # Optional depth-aware ED-Graph overlay
    if ed_nodes is not None and len(ed_nodes) > 0:
        _, nu, nv, nz = project_points(ed_nodes, R, t, fx, fy, cx, cy)
        n_vis = np.zeros(len(ed_nodes), dtype=bool)
        for i in range(len(ed_nodes)):
            px, py = int(round(nu[i])), int(round(nv[i]))
            if 0 <= px < W and 0 <= py < H:
                if nz[i] <= zbuf[py, px] + 0.022:
                    n_vis[i] = True

        # Draw ED-Graph edges
        if ed_edges is not None and len(ed_edges) > 0:
            edge_overlay = img_u8.copy()
            for a, b in ed_edges:
                if n_vis[a] and n_vis[b]:
                    p1 = (int(round(nu[a])), int(round(nv[a])))
                    p2 = (int(round(nu[b])), int(round(nv[b])))
                    cv2.line(edge_overlay, p1, p2, (235, 110, 45), thickness=max(1, ssaa), lineType=cv2.LINE_AA)
            img_u8 = cv2.addWeighted(edge_overlay, 0.72, img_u8, 0.28, 0)

        # Draw ED-Graph nodes (visible nodes crisp amber/orange with dark border)
        for i in range(len(ed_nodes)):
            if n_vis[i]:
                p = (int(round(nu[i])), int(round(nv[i])))
                cv2.circle(img_u8, p, radius=int(3.2 * ssaa), color=(60, 28, 12), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(img_u8, p, radius=int(2.2 * ssaa), color=(255, 145, 35), thickness=-1, lineType=cv2.LINE_AA)

    if ssaa > 1:
        img_u8 = cv2.resize(img_u8, (size, size), interpolation=cv2.INTER_AREA)
    return img_u8


def compute_s2t_distances_mm(src_vs: np.ndarray, tgt_vs: np.ndarray) -> np.ndarray:
    """Exact nearest-neighbor distance from src_vs to tgt_vs in millimeters."""
    # Chunked numpy distance
    dists = np.empty(len(src_vs), dtype=np.float32)
    chunk = 512
    for s in range(0, len(src_vs), chunk):
        e = min(s + chunk, len(src_vs))
        diff = src_vs[s:e, None, :] - tgt_vs[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        dists[s:e] = np.sqrt(np.min(d2, axis=-1))
    return dists * 1000.0


def distances_to_rgb(dist_mm: np.ndarray, vmax: float = 40.0) -> np.ndarray:
    """Map error distances (mm) to a clean Turbo-style perceptual colormap in [0, 1]."""
    t = np.clip(dist_mm / max(vmax, 1e-6), 0.0, 1.0)
    t_u8 = (t * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(t_u8[:, None], cv2.COLORMAP_TURBO)[:, 0, :]
    rgb = bgr[:, ::-1].astype(np.float32) / 255.0
    return rgb


def get_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def add_panel_label(img_rgb: np.ndarray, title: str, subtitle: str = None) -> np.ndarray:
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    w, h = pil.size
    font_title = get_font(20, bold=True)
    font_sub = get_font(15, bold=False)

    # Top title bar
    bbox = draw.textbbox((0, 0), title, font=font_title)
    tw = bbox[2] - bbox[0]
    draw.text(((w - tw) // 2, 16), title, fill=(30, 41, 59), font=font_title)

    if subtitle:
        bbox_s = draw.textbbox((0, 0), subtitle, font=font_sub)
        sw = bbox_s[2] - bbox_s[0]
        draw.text(((w - sw) // 2, h - 32), subtitle, fill=(71, 85, 105), font=font_sub)

    return np.asarray(pil)


def build_colorbar(height: int, width: int = 86, vmax: float = 40.0) -> np.ndarray:
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)

    bar_top = 64
    bar_bot = height - 64
    bar_left = 14
    bar_w = 20
    bar_h = bar_bot - bar_top

    vals = np.linspace(vmax, 0.0, bar_h, dtype=np.float32)
    colors = (distances_to_rgb(vals, vmax=vmax) * 255.0).astype(np.uint8)
    for y in range(bar_h):
        c = tuple(int(v) for v in colors[y])
        draw.line([(bar_left, bar_top + y), (bar_left + bar_w, bar_top + y)], fill=c)
    draw.rectangle([bar_left, bar_top, bar_left + bar_w, bar_bot], outline=(100, 116, 139), width=1)

    font = get_font(13, bold=False)
    font_hdr = get_font(14, bold=True)
    draw.text((8, 34), "Error", fill=(30, 41, 59), font=font_hdr)
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = int(bar_top + (1.0 - frac) * bar_h)
        val = frac * vmax
        draw.line([(bar_left + bar_w, y), (bar_left + bar_w + 4, y)], fill=(71, 85, 105), width=1)
        draw.text((bar_left + bar_w + 7, y - 7), f"{val:.0f}mm", fill=(51, 65, 85), font=font)

    return np.asarray(pil)


def main():
    parser = argparse.ArgumentParser(description="Render demo figures and GIF for README")
    parser.add_argument("--src", type=str, default="./assets/data/source.obj")
    parser.add_argument("--target", type=str, default="./assets/data/target.obj")
    parser.add_argument("--debug_dir", type=str, default="./debug_out/nrd_demo")
    parser.add_argument("--out_dir", type=str, default="./assets")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    src_vs, src_fs, _ = load_obj_np(args.src)
    tgt_vs, tgt_fs, _ = load_obj_np(args.target)
    def_vs, def_fs, _ = load_obj_np(os.path.join(args.debug_dir, "deformed_final.obj"))
    ed_nodes, _, ed_edges = load_obj_np(os.path.join(args.debug_dir, "ed_graph.obj"))

    # Studio 3/4 elevated perspective camera showing both in-plane stretch and out-of-plane saddle wave
    eye = np.array([0.56, -0.94, 0.90], dtype=np.float32)
    target = np.array([0.01, 0.0, 0.01], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    R, t = look_at(eye, target, up)

    panel_size = 480

    # 1) Overview figure: 4 panels
    print("Rendering demo_overview.png ...")
    p1 = rasterize_mesh(src_vs, src_fs, R, t, size=panel_size, base_color=(0.30, 0.56, 0.86))
    p1 = add_panel_label(p1, "1. Source Mesh", f"{len(src_vs):,} verts / {len(src_fs):,} faces")

    p2 = rasterize_mesh(
        src_vs, src_fs, R, t, size=panel_size,
        base_color=(0.68, 0.77, 0.88),
        ed_nodes=ed_nodes, ed_edges=ed_edges,
    )
    p2 = add_panel_label(p2, "2. Embedded Deformation Graph", f"{len(ed_nodes)} ED nodes (k-NN graph)")

    p3 = rasterize_mesh(tgt_vs, tgt_fs, R, t, size=panel_size, base_color=(0.18, 0.68, 0.55))
    p3 = add_panel_label(p3, "3. Target Mesh", f"{len(tgt_vs):,} verts / {len(tgt_fs):,} faces")

    p4 = rasterize_mesh(def_vs, def_fs, R, t, size=panel_size, base_color=(0.30, 0.56, 0.86))
    p4 = add_panel_label(p4, "4. Deformed Result", "Topology preserved from source")

    sep = np.full((panel_size, 3, 3), 215, dtype=np.uint8)
    overview = np.concatenate([p1, sep, p2, sep, p3, sep, p4], axis=1)
    overview_path = os.path.join(args.out_dir, "demo_overview.png")
    Image.fromarray(overview).save(overview_path, optimize=True)
    print(f"Saved {overview_path}")

    # 2) Surface distance error map (Before vs After)
    print("Rendering demo_error_map.png ...")
    d_before = compute_s2t_distances_mm(src_vs, tgt_vs)
    d_after = compute_s2t_distances_mm(def_vs, tgt_vs)
    vmax = float(np.percentile(d_before, 98))
    vmax = max(round(vmax / 5.0) * 5.0, 20.0)

    c_before = distances_to_rgb(d_before, vmax=vmax)
    c_after = distances_to_rgb(d_after, vmax=vmax)

    e1 = rasterize_mesh(src_vs, src_fs, R, t, size=panel_size, vert_colors=c_before)
    e1 = add_panel_label(
        e1,
        "Before Registration",
        f"Mean: {d_before.mean():.2f} mm  |  Max: {d_before.max():.2f} mm",
    )

    e2 = rasterize_mesh(def_vs, def_fs, R, t, size=panel_size, vert_colors=c_after)
    e2 = add_panel_label(
        e2,
        "After Non-Rigid Deformation",
        f"Mean: {d_after.mean():.2f} mm  |  Max: {d_after.max():.2f} mm",
    )

    cbar = build_colorbar(panel_size, width=96, vmax=vmax)
    error_fig = np.concatenate([e1, sep, e2, sep, cbar], axis=1)
    error_path = os.path.join(args.out_dir, "demo_error_map.png")
    Image.fromarray(error_fig).save(error_path, optimize=True)
    print(f"Saved {error_path}")

    # 3) Optimization progress GIF (overlaying translucent green target + deforming blue mesh)
    print("Rendering demo_progress.gif ...")
    iter_files = []
    for fpath in glob.glob(os.path.join(args.debug_dir, "*.obj")):
        stem = os.path.splitext(os.path.basename(fpath))[0]
        if stem.isdigit():
            iter_files.append((int(stem), fpath))
    iter_files.sort(key=lambda x: x[0])

    gif_size = 420
    frames = []
    for it_num, fpath in iter_files:
        vs_i, fs_i, _ = load_obj_np(fpath)
        d_i = compute_s2t_distances_mm(vs_i, tgt_vs)
        frame = rasterize_mesh(
            vs_i,
            fs_i,
            R,
            t,
            size=gif_size,
            ssaa=2,
            base_color=(0.30, 0.56, 0.86),
            ghost_verts=tgt_vs,
            ghost_faces=tgt_fs,
            ghost_color=(0.16, 0.72, 0.56),
            ghost_alpha=0.30,
        )
        frame = add_panel_label(
            frame,
            f"Iteration {it_num:4d}",
            f"Mean Src->Tgt Error: {d_i.mean():.2f} mm  (Green: Target)",
        )
        frames.append(frame)

    # Pause slightly on first and final frames for clear visual reading
    if frames:
        gif_frames = [frames[0], frames[0]] + frames + [frames[-1]] * 6
        gif_path = os.path.join(args.out_dir, "demo_progress.gif")
        imageio.mimsave(gif_path, gif_frames, duration=0.22, loop=0)
        print(f"Saved {gif_path} ({len(iter_files)} checkpoints)")


if __name__ == "__main__":
    main()
