import os
import cv2
import torch

import numpy as np

from util.mesh import Mesh


def load_obj(obj_path, tex_path=None, device='cuda'):
    with open(obj_path) as f:
        lines = f.readlines()

    # verts attributes
    verts, uvs, norms, vcolors  = [], [], [], []
    for line in lines:
        elems = line.split()
        if len(elems) == 0:
            continue
        
        prefix = elems[0].lower()
        if prefix == 'v':
            verts.append([float(v) for v in elems[1:4]])
            if len(elems) > 4:
                vcolors.append([float(v) for v in elems[4:]])
        elif prefix == 'vt':
            val = [float(v) for v in elems[1:]]
            uvs.append([val[0], 1.0 - val[1]])
        elif prefix == 'vn':
            norms.append([float(v) for v in elems[1:]])

    uv = False if len(uvs) == 0 else True
    nor = False if len(norms) == 0 else True

    # load faces
    f_v_idx, f_vt_idx, f_vn_idx = [], [], []

    if uv and nor:
        for line in lines:
            if len(line.split()) == 0:
                continue
            prefix = line.split()[0].lower()
            if prefix == 'f':
                vs = line.split()[1:]
                nv = len(vs)
                vv = vs[0].split('/')
                v0 = int(vv[0]) - 1
                t0 = int(vv[1]) - 1 if vv[1] != "" else -1
                n0 = int(vv[2]) - 1 if vv[2] != "" else -1
                for i in range(nv - 2):
                    vv = vs[i + 1].split('/')
                    v1 = int(vv[0]) - 1
                    t1 = int(vv[1]) - 1 if vv[1] != "" else -1
                    n1 = int(vv[2]) - 1 if vv[2] != "" else -1
                    vv = vs[i + 2].split('/')
                    v2 = int(vv[0]) - 1
                    t2 = int(vv[1]) - 1 if vv[1] != "" else -1
                    n2 = int(vv[2]) - 1 if vv[2] != "" else -1
                    f_v_idx.append([v0, v1, v2])
                    f_vt_idx.append([t0, t1, t2])
                    f_vn_idx.append([n0, n1, n2])
        assert len(f_v_idx) == len(f_vt_idx) and len(f_v_idx) == len (f_vn_idx)
    elif uv:
        for line in lines:
            if len(line.split()) == 0:
                continue
            prefix = line.split()[0].lower()
            if prefix == 'f':
                vs = line.split()[1:]
                nv = len(vs)
                vv = vs[0].split('/')
                v0 = int(vv[0]) - 1
                t0 = int(vv[1]) - 1 if vv[1] != "" else -1
                for i in range(nv - 2):
                    vv = vs[i + 1].split('/')
                    v1 = int(vv[0]) - 1
                    t1 = int(vv[1]) - 1 if vv[1] != "" else -1
                    vv = vs[i + 2].split('/')
                    v2 = int(vv[0]) - 1
                    t2 = int(vv[1]) - 1 if vv[1] != "" else -1
                    f_v_idx.append([v0, v1, v2])
                    f_vt_idx.append([t0, t1, t2])
        assert len(f_v_idx) == len(f_vt_idx)
    elif nor:
        for line in lines:
            if len(line.split()) == 0:
                continue
            prefix = line.split()[0].lower()
            if prefix == 'f':
                vs = line.split()[1:]
                nv = len(vs)
                vv = vs[0].split('/')
                v0 = int(vv[0]) - 1
                n0_str = vv[2] if len(vv) >= 3 else (vv[1] if len(vv) == 2 else "")
                n0 = int(n0_str) - 1 if n0_str != "" else -1
                for i in range(nv - 2):
                    vv = vs[i + 1].split('/')
                    v1 = int(vv[0]) - 1
                    n1_str = vv[2] if len(vv) >= 3 else (vv[1] if len(vv) == 2 else "")
                    n1 = int(n1_str) - 1 if n1_str != "" else -1
                    vv = vs[i + 2].split('/')
                    v2 = int(vv[0]) - 1
                    n2_str = vv[2] if len(vv) >= 3 else (vv[1] if len(vv) == 2 else "")
                    n2 = int(n2_str) - 1 if n2_str != "" else -1
                    f_v_idx.append([v0, v1, v2])
                    f_vn_idx.append([n0, n1, n2])
        assert len(f_v_idx) == len (f_vn_idx)
    else:
        for line in lines:
            if len(line.split()) == 0:
                continue
            prefix = line.split()[0].lower()
            if prefix == 'f':
                vs = line.split()[1:]
                nv = len(vs)
                vv = vs[0].split('/')
                v0 = int(vv[0]) - 1
                for i in range(nv - 2):
                    vv = vs[i + 1].split('/')
                    v1 = int(vv[0]) - 1
                    vv = vs[i + 2].split('/')
                    v2 = int(vv[0]) - 1
                    f_v_idx.append([v0, v1, v2])


    verts = torch.tensor(verts, dtype=torch.float32, device=device)
    uvs = torch.tensor(uvs, dtype=torch.float32, device=device) if len(uvs) > 0 else None
    norms = torch.tensor(norms, dtype=torch.float32, device=device) if len(norms) > 0 else None
    vcolors = torch.tensor(vcolors, dtype=torch.float32, device=device) if len(vcolors) > 0 else None
    
    f_v_idx = torch.tensor(f_v_idx, dtype=torch.int32, device=device).contiguous()
    f_vt_idx = torch.tensor(f_vt_idx, dtype=torch.int32, device=device).contiguous() if len(f_vt_idx) != 0 else None
    f_vn_idx = torch.tensor(f_vn_idx, dtype=torch.int32, device=device).contiguous() if len(f_vn_idx) != 0 else None

    texture = torch.tensor(cv2.imread(tex_path, -1)[...,:3], dtype=torch.float32, device=device).contiguous() / 255. if tex_path is not None else None

    return Mesh(vs=verts, vns=norms, vts=uvs, vcs=vcolors, f_v_idx=f_v_idx, f_vt_idx=f_vt_idx, f_vn_idx=f_vn_idx, tex=texture)

def save_obj(obj_path, mesh, verbose=True):
    parent_dir = os.path.dirname(obj_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    obj_name = os.path.basename(obj_path).split('.')[0]
    if verbose:
        print("Writing mesh: ", obj_path)
    with open(obj_path, "w") as f:
        if mesh.tex is not None:
            f.write(f"mtllib {obj_name}.mtl\n")
            f.write(f"usemtl {obj_name}\n")

        vs = mesh.vs.detach().cpu().numpy() if mesh.vs is not None else None
        vns = mesh.vns.detach().cpu().numpy() if mesh.vns is not None else None
        vts = mesh.vts.detach().cpu().numpy() if mesh.vts is not None else None

        f_v_idx = mesh.f_v_idx.detach().cpu().numpy() if mesh.f_v_idx is not None else None
        f_vn_idx = mesh.f_vn_idx.detach().cpu().numpy() if mesh.f_vn_idx is not None else None
        f_vt_idx = mesh.f_vt_idx.detach().cpu().numpy() if mesh.f_vt_idx is not None else None

        if verbose:
            print("    writing %d vertices" % len(vs))
        for v in vs:
            f.write('v {} {} {} \n'.format(v[0], v[1], v[2]))
       
        if vts is not None:
            if verbose:
                print("    writing %d texcoords" % len(vts))
            assert(len(f_v_idx) == len(f_vt_idx))
            for v in vts:
                f.write('vt {} {} \n'.format(v[0], 1.0 - v[1]))

        if vns is not None:
            if verbose:
                print("    writing %d normals" % len(vns))
            assert(len(f_v_idx) == len(f_vn_idx))
            for v in vns:
                f.write('vn {} {} {}\n'.format(v[0], v[1], v[2]))


        if f_v_idx is not None:
            # Write faces
            if verbose:
                print("    writing %d faces" % len(f_v_idx))
            for i in range(len(f_v_idx)):
                f.write("f ")
                if vts is not None and vns is not None:
                    for j in range(3):
                        f.write(' %s/%s/%s' % (str(f_v_idx[i][j]+1), str(f_vt_idx[i][j]+1), str(f_vn_idx[i][j]+1)))
                elif vts is not None and vns is None:
                    for j in range(3):
                        f.write(' %s/%s' % (str(f_v_idx[i][j]+1), str(f_vt_idx[i][j]+1)))
                elif vns is not None and vts is None:
                    for j in range(3):
                        f.write(' %s//%s' % (str(f_v_idx[i][j]+1), str(f_vn_idx[i][j]+1)))
                else:
                    for j in range(3):
                        f.write(' %s' % (str(f_v_idx[i][j]+1)))

                
                f.write("\n")

    if mesh.tex is not None:
        tex_file = os.path.join(parent_dir, obj_name + '.png')
        tex = mesh.tex.detach().cpu().numpy()
        if np.max(tex) <= 1:
            tex = tex * 255
        cv2.imwrite(tex_file, tex.astype(np.uint8))


        mtl_file = os.path.join(parent_dir, obj_name + '.mtl')
        if verbose:
            print("Writing material: ", mtl_file)
        with open(mtl_file, 'w') as f:
            f.write(f'newmtl {obj_name}\n')
            f.write('Kd 1 1 1\n')
            f.write('Ks 0 0 0\n')
            f.write('Ka 0 0 0\n')
            f.write(f'map_Kd {obj_name}.png')
    return

def save_ednode(obj_path, deformable_mesh):
    parent_dir = os.path.dirname(obj_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(obj_path, 'w') as f:
        ednodes = deformable_mesh.EDNodes.detach().cpu().numpy() if deformable_mesh.EDNodes is not None else None

        print("    writing %d ednodes" % len(ednodes))
        for ednode in ednodes:
            f.write('v {} {} {} \n'.format(ednode[0], ednode[1], ednode[2]))

        lines = deformable_mesh.e_e_relation.detach().cpu().numpy() if deformable_mesh.e_e_relation is not None else None

        if lines is not None:
            for id, line in enumerate(lines):
                for eachline in line:
                    f.write('l {} {}\n'.format(id+1, eachline+1))

    return


def save_ednode_and_verts(obj_path, deformable_mesh):
    parent_dir = os.path.dirname(obj_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(obj_path, 'w') as f:
        ednodes = deformable_mesh.EDNodes.detach().cpu().numpy() if deformable_mesh.EDNodes is not None else None
        ednodes_num = ednodes.shape[0]


        print("    writing %d ednodes" % ednodes_num)
        for ednode in ednodes:
            f.write('v {} {} {} \n'.format(ednode[0], ednode[1], ednode[2]))

        vs = deformable_mesh.mesh.vs.detach().cpu().numpy() if deformable_mesh.mesh.vs is not None else None

        print("    writing %d vertices" % (len(vs)))
        for v in vs:
            f.write('v {} {} {} \n'.format(v[0], v[1], v[2]))


        lines = deformable_mesh.v_e_relation.detach().cpu().numpy() if deformable_mesh.v_e_relation is not None else None
        if lines is not None:
            for id, line in enumerate(lines):
                for eachline in line:
                    f.write('l {} {}\n'.format(id+1+ednodes_num, eachline+1))

    return

def write_ply(ply_path, verts, vert_colors=None):
    parent_dir = os.path.dirname(ply_path)
    os.makedirs(parent_dir, exist_ok=True)
    ply_color_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    end_header
    '''
    verts = verts.reshape(-1, 3)
    if vert_colors is not None:
        assert verts.shape == vert_colors.shape, "vertex and color should be in same size!"
        colors = vert_colors.reshape(-1, 3)
        verts = np.hstack([verts, colors])
        with open(ply_path, 'wb') as f:
            f.write((ply_color_header % dict(vert_num=len(verts))).encode('utf-8'))
            np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
    else:
        with open(ply_path, 'wb') as f:
            f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
            np.savetxt(f, verts, fmt='%f %f %f')
    return