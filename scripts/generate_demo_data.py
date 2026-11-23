"""Generate a clean synthetic non-rigid deformation mesh pair (source.obj & target.obj).

Creates two watertight, manifold 3D torus meshes in meter scale with different
resolutions/topologies so the demo exercises genuine topology-agnostic non-rigid
surface registration (no 1-to-1 vertex correspondence).
"""

import os
import argparse
import numpy as np


def _build_torus_surface(n_u: int, n_v: int, deform: bool = False):
    """Build a watertight manifold torus (optionally with smooth non-rigid warp)."""
    u = np.linspace(0.0, 2.0 * np.pi, n_u, endpoint=False)
    v = np.linspace(0.0, 2.0 * np.pi, n_v, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")  # (n_u, n_v)

    R = 0.28  # major radius (meters)
    r = 0.10  # minor radius (meters)

    if deform:
        # Non-rigid local thickness modulation + smooth twist
        r_local = r * (
            1.0
            + 0.18 * np.sin(uu - 0.5)
            + 0.10 * np.cos(2.0 * uu + 0.4) * np.cos(vv)
        )
        twist = 0.35 * np.sin(uu)
        v_eff = vv + twist
    else:
        r_local = np.full_like(uu, r)
        v_eff = vv

    x = (R + r_local * np.cos(v_eff)) * np.cos(uu)
    y = (R + r_local * np.cos(v_eff)) * np.sin(uu)
    z = r_local * np.sin(v_eff)

    if deform:
        # Global non-rigid bend, asymmetric stretch, and saddle wave (in meters)
        x_new = x * 1.12 + 0.025 * np.sin(2.0 * uu)
        y_new = y * 0.88 + 0.030 * np.cos(uu + 0.3)
        z_new = (
            z
            + 0.055 * np.sin(2.0 * uu)
            + 0.030 * np.exp(-((uu - 0.9) ** 2) / 0.45)
        )
        x, y, z = x_new, y_new, z_new

    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)

    # Triangle faces with consistent outward winding
    faces = []
    for i in range(n_u):
        i_next = (i + 1) % n_u
        for j in range(n_v):
            j_next = (j + 1) % n_v
            v00 = i * n_v + j
            v10 = i_next * n_v + j
            v11 = i_next * n_v + j_next
            v01 = i * n_v + j_next
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    faces = np.asarray(faces, dtype=np.int32)

    # Compute smooth vertex normals
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn_norm = np.linalg.norm(fn, axis=-1, keepdims=True) + 1e-12
    fn = fn / fn_norm

    vns = np.zeros_like(verts)
    np.add.at(vns, faces[:, 0], fn)
    np.add.at(vns, faces[:, 1], fn)
    np.add.at(vns, faces[:, 2], fn)
    vns = vns / (np.linalg.norm(vns, axis=-1, keepdims=True) + 1e-12)

    return verts, vns, faces


def _write_obj(path: str, verts: np.ndarray, vns: np.ndarray, faces: np.ndarray):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Procedural non-rigid deformation demo mesh (units: meters)\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for vn in vns:
            f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")
        for tri in faces:
            a, b, c = tri[0] + 1, tri[1] + 1, tri[2] + 1
            f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate demo meshes for non-rigid-pytorch")
    parser.add_argument("--out_dir", type=str, default="./assets/data")
    args = parser.parse_args()

    src_vs, src_vns, src_fs = _build_torus_surface(n_u=72, n_v=36, deform=False)
    tgt_vs, tgt_vns, tgt_fs = _build_torus_surface(n_u=84, n_v=42, deform=True)

    src_path = os.path.join(args.out_dir, "source.obj")
    tgt_path = os.path.join(args.out_dir, "target.obj")

    _write_obj(src_path, src_vs, src_vns, src_fs)
    _write_obj(tgt_path, tgt_vs, tgt_vns, tgt_fs)

    print(f"Saved source mesh: {src_path} ({len(src_vs)} verts, {len(src_fs)} faces)")
    print(f"Saved target mesh: {tgt_path} ({len(tgt_vs)} verts, {len(tgt_fs)} faces)")


if __name__ == "__main__":
    main()
