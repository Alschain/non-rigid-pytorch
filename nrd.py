import os
import time
import argparse
import torch

from core.deform import NonRigidDeformer
from util.mesh import DeformableMesh
from util.mesh_util import knn_points
from util.obj_io import load_obj, save_obj, save_ednode


def compute_registration_metrics(src_vs: torch.Tensor, tgt_vs: torch.Tensor):
    """Compute source->target and symmetric Chamfer distances (in meters)."""
    with torch.no_grad():
        d_s2t, _, _ = knn_points(src_vs.unsqueeze(0), tgt_vs.unsqueeze(0), K=1)
        d_t2s, _, _ = knn_points(tgt_vs.unsqueeze(0), src_vs.unsqueeze(0), K=1)
        s2t = torch.sqrt(d_s2t[0, :, 0].clamp_min(0.0))
        t2s = torch.sqrt(d_t2s[0, :, 0].clamp_min(0.0))
        return {
            "s2t_mean_mm": s2t.mean().item() * 1000.0,
            "s2t_max_mm": s2t.max().item() * 1000.0,
            "chamfer_mm": 0.5 * (s2t.mean().item() + t2s.mean().item()) * 1000.0,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Non-Rigid Embedded Deformation demo (PyTorch)"
    )
    parser.add_argument(
        "--src",
        type=str,
        default="./assets/data/source.obj",
        help="Path to source mesh (.obj, units in meters)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="./assets/data/target.obj",
        help="Path to target mesh (.obj, units in meters)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./debug_out/nrd_demo",
        help="Output directory for intermediate and final meshes",
    )
    parser.add_argument("--max_iter", type=int, default=1500)
    parser.add_argument("--log_iter", type=int, default=100)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--dist_thres", type=float, default=0.15)
    parser.add_argument("--node_num", type=int, default=250)
    parser.add_argument("--ed_k", type=int, default=6)
    parser.add_argument("--v_k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    deformer_args = {
        # global
        "max_iter": args.max_iter,
        "log_iter": args.log_iter,
        "debug_dir": args.out_dir,
        "dist_thres": args.dist_thres,
        "dist_thres_lower": 0.0001,
        "lr": args.lr,
        "weight_decay": 3e-4,
        "adjust_lr_iter": max(args.max_iter // 2, 1),
        # losses
        "point2point": True,
        "point2plane": True,
        "E_rot": True,
        "E_reg": True,
        "E_corr": False,
        "E_nc": True,
        "E_fixable": False,
        "E_edgelen": True,
        "E_reg_init": 0.01,
        "E_reg_relax": 150,
        "E_nc_init": 0.02,
        "E_nc_relax": 200,
        "E_el_init": 20.0,
        "E_el_relax": 200,
        "E_fix_init": 100.0,
        "E_fix_relax": 300,
    }

    print(f"Using device: {args.device}")
    nrd = NonRigidDeformer(deformer_args)

    # Note: vertices are expected in meters
    print(f"Loading source mesh: {args.src}")
    src_mesh = load_obj(args.src, device=args.device)

    print(f"Loading target mesh: {args.target}")
    target_mesh = load_obj(args.target, device=args.device)

    init_metrics = compute_registration_metrics(src_mesh.vs, target_mesh.vs)
    print(
        f"[Before] Src->Tgt mean: {init_metrics['s2t_mean_mm']:.2f} mm, "
        f"max: {init_metrics['s2t_max_mm']:.2f} mm, "
        f"Chamfer: {init_metrics['chamfer_mm']:.2f} mm"
    )

    print("Generating deformable mesh and ED-Graph...")
    deformable_mesh = DeformableMesh(src_mesh)
    deformable_mesh.build_ednodes_on_vertices(
        args.node_num, ed_k=args.ed_k, v_k=args.v_k
    )
    deformable_mesh.generate_trainable_params()
    deformable_mesh.set_params_trainable()

    ed_graph_path = os.path.join(args.out_dir, "ed_graph.obj")
    save_ednode(ed_graph_path, deformable_mesh)

    t0 = time.time()
    result_mesh = nrd.run_optim(deformable_mesh, target_mesh)
    elapsed = time.time() - t0

    final_path = os.path.join(args.out_dir, "deformed_final.obj")
    save_obj(final_path, result_mesh)

    final_metrics = compute_registration_metrics(result_mesh.vs, target_mesh.vs)
    print(
        f"[After ] Src->Tgt mean: {final_metrics['s2t_mean_mm']:.2f} mm, "
        f"max: {final_metrics['s2t_max_mm']:.2f} mm, "
        f"Chamfer: {final_metrics['chamfer_mm']:.2f} mm"
    )
    print(f"Done in {elapsed:.2f}s. Final mesh saved to: {final_path}")


if __name__ == "__main__":
    main()
