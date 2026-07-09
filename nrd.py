from core.deform import NonRigidDeformer
from util.mesh import Mesh, DeformableMesh
from util.obj_io import *

from pytorch3d.ops import knn_points

deformer_args = {
    # global
    'max_iter': 5000,
    'log_iter': 200,
    'debug_dir': './debug_out/nrd_test',
    'dist_thres': 0.02,
    'dist_thres_lower': 0.0001,
    'lr':3e-5,
    'weight_decay':3e-4,
    'adjust_lr_iter': 1000,

    # losses
    'point2point': False,
    'point2plane': True,
    'E_rot': True,
    'E_reg': True,
    'E_corr': False,
    'E_nc': True,
    'E_fixable': True,
    'E_edgelen': True,

    'E_reg_init': 0.01,
    'E_reg_relax': 200,
    'E_nc_init': 0.1,
    'E_nc_relax': 1000,
    'E_el_init': 5000,
    'E_el_relax': 1000,
    'E_fix_init': 1000,
    'E_fix_relax': 1000,

}


nrd = NonRigidDeformer(deformer_args)

src_mesh_path = r''
target_mesh_path = r''

print("Loading source mesh...")
src_mesh = load_obj(src_mesh_path)

# Note: here for verts in meter
print("Loading target mesh...")
target_mesh = load_obj(target_mesh_path)


dist2, index, _ = knn_points(src_mesh.vs.unsqueeze(0), target_mesh.vs.unsqueeze(0), K=1)

dist = torch.sqrt(dist2[0,:,0])


print("Generating deformable mesh...")
deformable_mesh = DeformableMesh(src_mesh)
deformable_mesh.build_ednodes_on_vertices(2000, ed_k=8, dists=dist, dist_thres=0.05, inside_node_num=10000)
deformable_mesh.generate_trainable_params()
deformable_mesh.set_params_trainable()

nrd.run_optim(deformable_mesh, target_mesh)