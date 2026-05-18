import torch

from util.mesh import Mesh, DeformableMesh
from util.obj_io import *
from util.mesh_util import *

from pytorch3d.ops import knn_points

class NonRigidDeformer:
    def __init__(self, args):
        default_args = {
            # global
            'max_iter': 2000,
            'log_iter': 50,
            'debug_dir': './debut_out',
            'dist_thres': 0.05,
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
            'E_edgelen': False,

            'E_reg_init': 0.01,
            'E_reg_relax': 200,
            'E_nc_init': 0.1,
            'E_nc_relax': 1000,
            'E_el_init': 5000,
            'E_el_relax': 500,
            'E_fix_init': 1000,
            'E_fix_relax': 500,
        }
        self.args = args
        for key, value in default_args.items():
            if key not in self.args.keys():
                self.args[key] = value

        print('NRD args')
        for key, value in self.args.items():
            print('{0}: \t{1}'.format(key, value))
        os.makedirs(self.args['debug_dir'], exist_ok=True)

    def run_optim(self, deformed_mesh:DeformableMesh, target_mesh:Mesh, src_mask:torch.tensor=None, target_scores:torch.tensor=None, eps:float=1e-8):
        
        deformed_mesh.set_params_trainable()
        optimizer = torch.optim.AdamW(
            [deformed_mesh.Rs, deformed_mesh.Ts],
            lr=self.args['lr'],
            weight_decay=self.args['weight_decay'])
        
        if self.args['E_nc']:
            if deformed_mesh.mesh.edges is None:
                deformed_mesh.mesh.edges = find_edges(deformed_mesh.mesh.f_v_idx, remove_duplicates=True)
                deformed_mesh.mesh.connected_faces = find_connected_faces(deformed_mesh.mesh.f_v_idx)
                    
            ori_vertex_normals, ori_face_normals = compute_normals(deformed_mesh.mesh.vs.detach(), deformed_mesh.mesh.f_v_idx)
            ori_link_nc = (ori_vertex_normals[deformed_mesh.mesh.edges[:,0]] - ori_vertex_normals[deformed_mesh.mesh.edges[:,1]])
            ori_link_face_nc = (ori_face_normals[deformed_mesh.mesh.connected_faces[:,0]] - ori_face_normals[deformed_mesh.mesh.connected_faces[:,1]])

        if self.args['E_edgelen']:
            if deformed_mesh.mesh.edges is None:
                deformed_mesh.mesh.edges = find_edges(deformed_mesh.mesh.f_v_idx, remove_duplicates=True)
                deformed_mesh.mesh.connected_faces = find_connected_faces(deformed_mesh.mesh.f_v_idx)
                    
            ori_edge_len = compute_edge_length(deformed_mesh.mesh.vs.clone().detach(), deformed_mesh.mesh.edges)



        for it in range(self.args['max_iter']+1):
            if (it + 1) % self.args['adjust_lr_iter'] == 0:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.5

            losses = {}

            ##########################
            # Step 1, apply deformation
            ##########################
            deformed = deformed_mesh.apply_deformation()
            
            ##########################
            # Step 2, calc losses
            ##########################

            # point to point loss
            dist, index, _ = knn_points(deformed['deformed_vertices'].unsqueeze(0), target_mesh.vs.unsqueeze(0), K=1)
            dist = torch.sqrt(dist)
            mask = torch.logical_and(dist[0,:,0] < self.args['dist_thres'], dist[0,:,0] > self.args['dist_thres_lower'])
            if src_mask is not None:
                mask = torch.logical_and(mask, src_mask)
            index = index[0,:,0]

            if self.args['point2point']:
                nearest_points = torch.index_select(target_mesh.vs, 0, index)
                p2p_dists = \
                    torch.masked_select(
                        torch.sum((deformed['deformed_vertices'] - nearest_points)**2, -1), mask
                        )
                
                if target_scores is not None:
                    nearest_scores = torch.index_select(target_scores, 0, index)
                    p2p_dists = torch.masked_select(nearest_scores, mask) * p2p_dists

                if it == 0:
                    if p2p_dists.sum() > eps:
                        p2p_init = 10 / p2p_dists.sum().item()
                    else:
                        p2p_init = 10000

                losses['E_p2p'] = p2p_init * p2p_dists.sum()
            if self.args['point2plane']:
                nearest_points = torch.index_select(target_mesh.vs, 0, index)
                if target_mesh.vns is None:
                    target_mesh.update_normal()
                nearest_normals = torch.index_select(target_mesh.vns, 0, index)

                p2p_dists = \
                    torch.masked_select(
                    torch.sum(((deformed['deformed_vertices'] - nearest_points) * nearest_normals)**2, dim=-1), mask
                    )
                if target_scores is not None:
                    nearest_scores = torch.index_select(target_scores, 0, index)
                    p2p_dists = torch.masked_select(nearest_scores, mask) * p2p_dists
                
                if it == 0:
                    if p2p_dists.sum() > eps:
                        p2p_init = 10 / p2p_dists.sum().item()
                    else:
                        p2p_init = 10000

                losses['E_p2p'] = p2p_init * p2p_dists.sum()

            if self.args['E_fixable']:
                fixable_mask = torch.logical_or(dist[0,:,0] > self.args['dist_thres'], dist[0,:,0] < self.args['dist_thres_lower'])
                fixable_dists = \
                    torch.masked_select(
                        torch.sum((deformed['deformed_vertices'] - deformed_mesh.mesh.vs)**2, -1), fixable_mask
                        )
                losses['E_fixable'] = (self.args['E_fix_init'] / 1.5**((it+1)//self.args['E_fix_relax'])) * fixable_dists.sum()


            if self.args['E_rot']:
                rot = (torch.sum(deformed_mesh.Rs[:,0,:] * deformed_mesh.Rs[:,1,:], dim=1))**2\
                    + (torch.sum(deformed_mesh.Rs[:,0,:] * deformed_mesh.Rs[:,2,:], dim=1))**2\
                    + (torch.sum(deformed_mesh.Rs[:,1,:] * deformed_mesh.Rs[:,2,:], dim=1))**2\
                    + (torch.sum(deformed_mesh.Rs[:,0,:] * deformed_mesh.Rs[:,0,:], dim=1) - 1)**2\
                    + (torch.sum(deformed_mesh.Rs[:,1,:] * deformed_mesh.Rs[:,1,:], dim=1) - 1)**2\
                    + (torch.sum(deformed_mesh.Rs[:,2,:] * deformed_mesh.Rs[:,2,:], dim=1) - 1)**2
                
                losses['E_rot'] = 10 * rot.sum()
            
            if self.args['E_reg']:

                reg_dist = torch.sum(
                    (deformed['deformed_EDNeighbors'] - deformed['deformed_EDNodes'].unsqueeze(1))**2,
                    dim=-1)
                
                losses['E_reg'] = 1000 * (self.args['E_reg_init'] / 1.5**((it+1)//self.args['E_reg_relax'])) * reg_dist.sum()

            if self.args['E_corr'] and deformed['deformed_corrs'] is not None:
                corr_dist =\
                    torch.sum(
                    (deformed['deformed_corrs'] - deformed_mesh.target_corrs)**2,
                    dim=-1)
                
                losses['E_corr'] = corr_dist.sum()

            if self.args['E_nc']:
                vertex_normals, face_normals = compute_normals(deformed['deformed_vertices'], deformed_mesh.mesh.f_v_idx)
                losses['E_nc'] = (self.args['E_nc_init'] / 1.5**((it+1)//self.args['E_nc_relax']))*\
                    ((((vertex_normals[deformed_mesh.mesh.edges[:,0]] - vertex_normals[deformed_mesh.mesh.edges[:,1]]) - ori_link_nc) ** 2).sum()+\
                    (((face_normals[deformed_mesh.mesh.connected_faces[:,0]] - face_normals[deformed_mesh.mesh.connected_faces[:,1]]) - ori_link_face_nc)**2).sum())
            if self.args['E_edgelen']:
                edges_len = compute_edge_length(deformed['deformed_vertices'], deformed_mesh.mesh.edges)

                losses['E_edgelen'] = (self.args['E_el_init'] / 1.5**((it+1)//self.args['E_el_relax'])) * ((edges_len - ori_edge_len)**2).sum()


            total_loss = 0.
            for key, value in losses.items():
                total_loss += value
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (it % self.args['log_iter']) == 0:
                contentline = 'iteration \t{0}, \ttotal \t{1:.6f}, '.format(it, total_loss.item())
                for key, value in losses.items():
                    contentline = contentline + '\t{0} \t{1:.6f}, '.format(key, value.item())
                print(contentline)

                debug_mesh = deformed_mesh.mesh.clone()
                debug_mesh.vs = deformed['deformed_vertices'].detach().clone()
                debug_mesh.update_normal()
                save_obj(os.path.join(self.args['debug_dir'], f'{it}.obj'), debug_mesh, verbose=False)