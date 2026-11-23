import torch
import numpy as np
from torch import nn
import nvdiffrast.torch as dr



class NvdiffrastMeshUlitRenderer(nn.Module):
    def __init__(self, args):
        super(NvdiffrastMeshUlitRenderer, self).__init__()
        self.args = args
        self.render_size = self.args['render_size']
        self.device = args['device']
        self.glctx = None

        self.extrinsic = torch.eye(4).to(self.device)
        self.intrinsic = torch.eye(4).to(self.device)
        self.ndc_proj = None


    def forward(self, mesh, antialias=True, render_normal=False):
        out = {}

        if self.ndc_proj is None:
            self.calc_ndc_projection()
        if self.glctx is None:
            self.glctx = dr.RasterizeGLContext() if 'opengl' == self.args['rastype'] else dr.RasterizeCudaContext()

        if mesh.vs.shape[-1] == 3:
            vertex = torch.cat([mesh.vs, torch.ones([mesh.vs.shape[0], 1], device=self.device, dtype=mesh.vs.dtype)], dim=-1).unsqueeze(0)
        else:
            vertex = mesh.vs.unsqueeze(0) if mesh.vs.ndim == 2 else mesh.vs
        vertex_ndc = torch.bmm(self.ndc_proj, vertex.permute(0,2,1)).permute(0,2,1)

        f_v_idx = mesh.f_v_idx.int().contiguous()
        rast_out, rast_db = dr.rasterize(self.glctx, vertex_ndc.contiguous(), f_v_idx, resolution=self.render_size)

        ndc_depth = rast_out[...,2:3]
        mask = (rast_out[..., 3] > 0).float().unsqueeze(-1)
        

        img = None
        if mesh.vts is not None and mesh.tex is not None:
            if mesh.f_vt_idx is not None:
                interp_out, inpterp_out_db = dr.interpolate(mesh.vts.contiguous(), rast_out, mesh.f_vt_idx.int().contiguous(), rast_db, diff_attrs='all')
            else:
                interp_out, inpterp_out_db = dr.interpolate(mesh.vts.contiguous(), rast_out, f_v_idx, rast_db, diff_attrs='all')
                
            img = dr.texture(mesh.tex.unsqueeze(0), interp_out, inpterp_out_db)
            img = img * mask
            if antialias:
                img = dr.antialias(img.contiguous(), rast_out, vertex_ndc.contiguous(), f_v_idx)
        elif mesh.vcs is not None:
            interp_out, inter_out_db = dr.interpolate(mesh.vcs.contiguous(), rast_out, f_v_idx, rast_db, diff_attrs='all')
            img = interp_out * mask
            if antialias:
                img = dr.antialias(img.contiguous(), rast_out, vertex_ndc.contiguous(), f_v_idx)

        if render_normal:
            mesh.update_normal()

            interp_out, inter_out_db = dr.interpolate(mesh.vns.contiguous(), rast_out, f_v_idx, rast_db, diff_attrs='all')
            out['normal'] = interp_out * mask

        depth = self.calc_ndc_to_depth(ndc_depth)
        depth = mask * depth
        
        out['mask'] = mask
        out['depth'] = depth
        out['color'] = img

        return out

    def set_camera_intrinsic(self, intrinsic):
        if isinstance(intrinsic, np.ndarray):
            intrinsic = torch.FloatTensor(intrinsic)

        if intrinsic.numel() == 9:
            K = torch.eye(4, dtype=intrinsic.dtype)
            K[:3, :3] = intrinsic.reshape(3, 3)
            intrinsic = K
        else:
            intrinsic = intrinsic.reshape(4, 4)

        self.intrinsic = intrinsic.to(self.device)
        self.ndc_proj = None

    def set_camera_extrinsic(self, extrinsic):
        if isinstance(extrinsic, np.ndarray):
            extrinsic = torch.FloatTensor(extrinsic)

        self.extrinsic = extrinsic.reshape(4,4).to(self.device)
        self.ndc_proj = None

    def calc_ndc_projection(self):
        P = self.calc_projection(
            self.intrinsic[0][0].item(), self.intrinsic[1][1].item(),
            self.intrinsic[0][2].item(), self.intrinsic[1][2].item(),
            self.render_size[1], self.render_size[0]
            ).to(self.device)
        self.ndc_proj = torch.matmul(P, self.extrinsic.to(self.device)).unsqueeze(0).to(self.device)
        return
      
    def calc_ndc_to_depth(self, ndc_z, near=0.1, far=20.0):
        return (2 * near * far) / (far + near - ndc_z * (far - near))
    
    def calc_projection(self, fx, fy, cx, cy, w, h, near=0.1, far=20.0):
        P = torch.FloatTensor([
            [2*fx/w, 0, (2*cx-w)/w, 0],
            [0, 2*fy/h, (2*cy-h)/h, 0],
            [0, 0, (far+near)/(far-near), -(2*far*near)/(far-near)],
            [0, 0, 1, 0]])
        
        return P