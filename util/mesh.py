import torch
import torch.nn as nn
import numpy as np

from util.mesh_util import *

from queue import Queue
from collections import namedtuple

_Neighbor = namedtuple("Neighbor", "id dist")

class Mesh:
    def __init__(self, vs=None, vns=None, vts=None, vcs=None, f_v_idx=None, f_vt_idx=None, f_vn_idx=None, tex=None, clone=False):
        '''
        vs          :   vertex positions
        vns         :   vertex normals
        vts         :   uv coordinates
        vcs         :   vertex colors
        f_v_idx     :   triangle vertex position index
        f_vt_idx    :   triangle uv coordinates index
        f_vn_idx    :   triangle vertex normal index
        tex         :   texture map
        '''
        self.vs = vs
        self.vns = vns
        self.vts = vts
        self.vcs = vcs
        
        self.f_ns = None
        self.f_v_idx = f_v_idx
        self.f_vt_idx = f_vt_idx if f_vt_idx is not None else f_v_idx
        self.f_vn_idx = f_vn_idx if f_vn_idx is not None else f_v_idx

        self.tex = tex

        if clone == False:
            self.edges = find_edges(self.f_v_idx, remove_duplicates=True) if self.f_v_idx is not None else None
            self.connected_faces = find_connected_faces(self.f_v_idx) if self.f_v_idx is not None else None


    def clone(self):
        out = Mesh(clone=True)
        if self.vs is not None:
            out.vs = self.vs.clone()
        if self.vns is not None:
            out.vns = self.vns.clone()
        if self.vts is not None:
            out.vts = self.vts.clone()
        if self.vcs is not None:
            out.vcs = self.vcs.clone()


        if self.f_ns is not None:
            out.f_ns = self.f_ns.clone()        
        if self.f_v_idx is not None:
            out.f_v_idx = self.f_v_idx.clone()
        if self.f_vt_idx is not None:
            out.f_vt_idx = self.f_vt_idx.clone()
        if self.f_vn_idx is not None:
            out.f_vn_idx = self.f_vn_idx.clone()
        
        if self.tex is not None:
            out.tex = self.tex.clone()

        if self.edges is not None:
            out.edges = self.edges.clone()
        if self.connected_faces is not None:
            out.connected_faces = self.connected_faces.clone()
        return out

    def update_normal(self):
        self.vns, self.f_ns = compute_normals(self.vs, self.f_v_idx)

    def apply_transform(self, global_s=torch.ones(1), global_rm=torch.eye(3), global_t=torch.zeros(3)):
        self.vs = torch.matmul(global_rm.to((self.vs.device)), global_s.to(self.vs.device) * self.vs.permute(1,0)).permute(1,0) + global_t.to(self.vs.device)
        return
    
    def build_adjacent_list(self):
        if self.f_v_idx is None:
            return None
        
        vs = self.vs.cpu().numpy()
        f_v_idx = self.f_v_idx.cpu().numpy()
        adj_set = [set() for i in range(len(vs))]
        for fid in range(f_v_idx.shape[0]):
            face = f_v_idx[fid]
            adj_set[face[0]].add(face[1])
            adj_set[face[0]].add(face[2])
    
            adj_set[face[1]].add(face[0])
            adj_set[face[1]].add(face[2])
            
            adj_set[face[2]].add(face[0])
            adj_set[face[2]].add(face[1])

        return [list(s) for s in adj_set]

    def get_refenrece_vs_mask(self):
        references_vs_index = self.f_v_idx.clone().flatten()
        remain_verts_mask = torch.zeros(self.vs.shape[0]).type(torch.BoolTensor).to(self.vs.device)
        remain_verts_mask[references_vs_index.type(torch.LongTensor)] = 1
        return remain_verts_mask

    def delete_faces(self, tri_idx:torch.LongTensor):
        remain_tris_mask = torch.ones(self.f_v_idx.shape[0]).type(torch.BoolTensor).to(self.f_v_idx.device)
        remain_tris_mask[tri_idx] = 0
        remain_tris_mask = remain_tris_mask.unsqueeze(-1).expand(-1, 3)
        self.f_v_idx = torch.masked_select(self.f_v_idx, remain_tris_mask).reshape(-1, 3).contiguous()

        remain_verts_mask = self.get_refenrece_vs_mask()
        new_index = torch.cumsum(remain_verts_mask.type(torch.LongTensor), 0).to(self.vs.device) - 1
        remain_verts_mask = remain_verts_mask.unsqueeze(-1).expand(-1, 3)

        self.vs = torch.masked_select(self.vs, remain_verts_mask).reshape(-1, 3).contiguous()
        if self.vns is not None:
            self.vns = torch.masked_select(self.vns, remain_verts_mask).reshape(-1, 3).contiguous()
        if self.vcs is not None:
            self.vcs = torch.masked_select(self.vcs, remain_verts_mask).reshape(-1, 3).contiguous()
        
        # reindex
        new_f_v_idx_num, _ = self.f_v_idx.shape
        self.f_v_idx = torch.index_select(new_index, 0, self.f_v_idx.reshape(new_f_v_idx_num * 3)).reshape(new_f_v_idx_num, 3).contiguous()

        if self.f_ns is not None:
            self.f_ns = torch.masked_select(self.f_ns, remain_tris_mask).reshape(-1, 3).contiguous()
        if self.f_vt_idx is not None:
            self.f_vt_idx = torch.masked_select(self.f_vt_idx, remain_tris_mask).reshape(-1, 3).contiguous()
        if self.f_vn_idx is not None:
            new_f_vn_idx_num, _ = self.f_vn_idx.shape
            self.f_vn_idx = torch.index_select(new_index, 0, self.f_vn_idx.reshape(new_f_vn_idx_num * 3)).reshape(new_f_vn_idx_num, 3)
            self.f_vn_idx = torch.masked_select(self.f_vn_idx, remain_tris_mask).reshape(-1, 3).contiguous()

        self.edges = find_edges(self.f_v_idx, remove_duplicates=True) if self.f_v_idx is not None else None
        self.connected_faces = find_connected_faces(self.f_v_idx) if self.f_v_idx is not None else None

        return new_index

class DeformableMesh:
    def __init__(self, mesh:Mesh, params=12):
        self.mesh = mesh
        self.params = params

        self.EDNodes = None
        self.e_e_relation = None
        self.e_e_weight = None

        self.v_e_relation = None
        self.v_e_weight = None

        self.Rs = None
        self.Ts = None

        self.corr = None
        self.corr_target = None
        self.corr_e_relation = None
        self.corr_e_weight = None

    
    def build_ednodes_on_vertices(self, node_num, ed_k=4, v_k=4, method='ecu', eps=1e-7, dists=None, dist_thres=None, inside_node_num=None):
        # Step 1, FPS for getting nodes
        if node_num > self.mesh.vs.shape[0]:
            node_num = self.mesh.vs.shape[0]

        if dists is None or dist_thres is None or inside_node_num is None:
            _, self.EDNodes = UniformSamplePoints(self.mesh.vs, node_num)
        else:
            assert dists.shape[0] == self.mesh.vs.shape[0], "dist not match vertices"
            inside_mask = torch.nonzero(dists <= dist_thres)[...,0]
            outside_mask = torch.nonzero(dists > dist_thres)[...,0]
            sampled_groups = []
            if inside_mask.numel() > 0:
                inside_verts = torch.index_select(self.mesh.vs, 0, inside_mask)
                inside_k = min(inside_verts.shape[0], inside_node_num)
                if inside_k > 0:
                    _, inside_nodes = UniformSamplePoints(inside_verts, inside_k)
                    sampled_groups.append(inside_nodes)
            if outside_mask.numel() > 0:
                outside_verts = torch.index_select(self.mesh.vs, 0, outside_mask)
                outside_k = min(outside_verts.shape[0], node_num)
                if outside_k > 0:
                    _, outside_nodes = UniformSamplePoints(outside_verts, outside_k)
                    sampled_groups.append(outside_nodes)

            self.EDNodes = torch.cat(sampled_groups, dim=0)

        # Step 2, construct EDNodes - EDNodes relation
        print("[DeformableMesh] Perform Node to Node skinning...")
        self.compute_n2n_skinning(ed_k, method)

        # Step 3, construct vertices - EDNodes relation
        print("[DeformableMesh] Perform Vertex to Node skinning...")
        self.compute_v2n_skinning(v_k, method)

        return
    
    def compute_n2n_skinning(self, neigh_k, method, eps=1e-7):
        # ecu knn
        ed_dist, ed_index, _ = knn_points(self.EDNodes.unsqueeze(0), self.EDNodes.unsqueeze(0), K=neigh_k+2)
        self.e_e_relation = ed_index[0,:,1:-1]

        if method != 'geo':
            self.e_e_weight = (1 - ed_dist[0,:,1:-1] / ed_dist[0,:,-1:])**2
            self.e_e_weight = self.e_e_weight / (torch.sum(self.e_e_weight, dim=1).unsqueeze(-1) + eps) 
            return

        nearest_vert_dist, nearest_vert, _ = knn_points(self.EDNodes.unsqueeze(0), self.mesh.vs.unsqueeze(0), K=1)
        nearest_vert_dist, nearest_vert = nearest_vert_dist[0,...,0].cpu().numpy(), nearest_vert[0,...,0].cpu().numpy()
        nodes_nearby_vert = [[] for i in range(self.mesh.vs.shape[0])]
        for nid in range(len(nearest_vert)):
            nearest_vid = nearest_vert[nid]
            nodes_nearby_vert[nearest_vid].append(nid)

        n2n_idx = self.e_e_relation.cpu().numpy()
        n2n_dist = ed_dist[0,:,1:-1].cpu().numpy() * 3
        nodes = self.EDNodes.cpu().numpy()
        vs = self.mesh.vs.cpu().numpy()
        adj_list = self.mesh.build_adjacent_list()

        for i in range(nodes.shape[0]):
            neigh_nodes = []
            neigh_nodes_dist = []

            visited = np.zeros(vs.shape[0], dtype=bool)
            frontier = Queue()
            curr_vert = nearest_vert[i]
            curr_dist = np.linalg.norm(nodes[i] - vs[curr_vert])
            frontier.put((curr_vert, curr_dist))
            visited[curr_vert] = True

            while not frontier.empty() and len(neigh_nodes) < neigh_k:
                curr_vert, curr_dist = frontier.get()

                # find nodes nearby current vertex
                for nid in nodes_nearby_vert[curr_vert]:
                    if nid == i:
                        continue

                    neigh_nodes.append(nid)
                    neigh_nodes_dist.append(curr_dist + np.linalg.norm(nodes[nid] - vs[curr_vert]))

                # add adjacent vertices to frontier
                for adj_v in adj_list[curr_vert]:
                    if visited[adj_v]:
                        continue
                    frontier.put((adj_v, curr_dist + np.linalg.norm(vs[curr_vert] - vs[adj_v])))
                    visited[adj_v] = True

            # update skinning dist
            n_cnt = neigh_k if len(neigh_nodes) >= neigh_k else len(neigh_nodes)
            for n in range(n_cnt):
                n2n_idx[i][n] = neigh_nodes[n]
                n2n_dist[i][n] = neigh_nodes_dist[n]

        self.e_e_relation = torch.tensor(n2n_idx, dtype=torch.int32).to(self.EDNodes.device)
        n2n_dist = torch.tensor(n2n_dist, dtype=torch.float32).to(self.EDNodes.device)
        n2n_dist_mean = torch.mean(n2n_dist, dim=1).unsqueeze(-1)
        self.e_e_weight = torch.exp(-2. * n2n_dist / n2n_dist_mean)
        self.e_e_weight = self.e_e_weight / (torch.sum(self.e_e_weight, dim=1).unsqueeze(-1))

    def compute_v2n_skinning(self, neigh_k, method, eps=1e-7):
        v_dist, v_index, _ = knn_points(self.mesh.vs.unsqueeze(0), self.EDNodes.unsqueeze(0), K=neigh_k+1)
        self.v_e_relation = v_index[0,:,:-1]
        
        if method != 'geo':
            self.v_e_weight = (1 - v_dist[0,:,:-1] / v_dist[0,:,-1:])**2
            self.v_e_weight = self.v_e_weight / (torch.sum(self.v_e_weight, dim=1).unsqueeze(-1) + eps)
            return
        
        nearest_vert_dist, nearest_vert, _ = knn_points(self.EDNodes.unsqueeze(0), self.mesh.vs.unsqueeze(0), K=1)
        nearest_vert_dist, nearest_vert = nearest_vert_dist[0,...,0].cpu().numpy(), nearest_vert[0,...,0].cpu().numpy()
        nodes_nearby_vert = [[] for i in range(self.mesh.vs.shape[0])]
        for nid in range(len(nearest_vert)):
            nearest_vid = nearest_vert[nid]
            nodes_nearby_vert[nearest_vid].append(nid)

        v2n_idx = self.v_e_relation.cpu().numpy()
        v2n_dist = v_dist[0,:,0:-1].cpu().numpy() * 3
        nodes = self.EDNodes.cpu().numpy()
        vs = self.mesh.vs.cpu().numpy()
        adj_list = self.mesh.build_adjacent_list()

        for i in range(vs.shape[0]):
            neigh_nodes = []
            neigh_nodes_dist = []

            visited = np.zeros(vs.shape[0], dtype=bool)
            frontier = Queue()
            curr_vert = i
            curr_dist = 0.
            frontier.put((curr_vert, curr_dist))
            visited[curr_vert] = True

            while not frontier.empty() and len(neigh_nodes) < neigh_k:
                curr_vert, curr_dist = frontier.get()

                # find nodes nearby current vertex
                for nid in nodes_nearby_vert[curr_vert]:
                    if nid == i:
                        continue

                    neigh_nodes.append(nid)
                    neigh_nodes_dist.append(curr_dist + np.linalg.norm(nodes[nid] - vs[curr_vert]))

                # add adjacent vertices to frontier
                for adj_v in adj_list[curr_vert]:
                    if visited[adj_v]:
                        continue
                    frontier.put((adj_v, curr_dist + np.linalg.norm(vs[curr_vert] - vs[adj_v])))
                    visited[adj_v] = True

            # update skinning dist
            n_cnt = neigh_k if len(neigh_nodes) >= neigh_k else len(neigh_nodes)
            for n in range(n_cnt):
                v2n_idx[i][n] = neigh_nodes[n]
                v2n_dist[i][n] = neigh_nodes_dist[n]

        self.v_e_relation = torch.tensor(v2n_idx, dtype=torch.int32).to(self.EDNodes.device)
        v2n_dist = torch.tensor(v2n_dist, dtype=torch.float32).to(self.EDNodes.device)
        v2n_dist_mean = torch.mean(v2n_dist, dim=1).unsqueeze(-1)
        self.v_e_weight = torch.exp(-2. * v2n_dist / v2n_dist_mean)
        self.v_e_weight = self.v_e_weight / (torch.sum(self.v_e_weight, dim=1).unsqueeze(-1))

    
    def set_correspondence(self, corrs, target_corrs, v_k=4, eps=1e-7):
        self.corr = corrs.to(self.EDNodes.device)
        self.target_corrs = target_corrs.to(self.EDNodes.device)
        v_dist, v_index, _ = knn_points(corrs.unsqueeze(0), self.EDNodes.unsqueeze(0), K=v_k+1)
        self.corr_e_relation = v_index[0,:,:-1]
        self.corr_e_weight = (1 - v_dist[0,:,:-1] / v_dist[0,:,-1:])**2
        self.corr_e_weight = self.corr_e_weight / (torch.sum(self.corr_e_weight, dim=1).unsqueeze(-1) + eps)

        return
    

    def generate_trainable_params(self):
        if self.EDNodes is None:
            print("Generation failed, build ednodes first.")
            return
        if self.params == 12:
            Rs = torch.eye(3).unsqueeze(0).repeat(self.EDNodes.shape[0], 1, 1)
            Ts = torch.zeros(3).unsqueeze(0).repeat(self.EDNodes.shape[0], 1)
            self.Rs = nn.Parameter(Rs.to(self.EDNodes.device), requires_grad=True)
            self.Ts = nn.Parameter(Ts.to(self.EDNodes.device), requires_grad=True)        
        
        return
    
    def set_params_untrainable(self):
        if self.params == 12:
            self.Rs.requires_grad_(False)
            self.Ts.requires_grad_(False)
        return
    
    def set_params_trainable(self):
        if self.params == 12:
            self.Rs.requires_grad_(True)
            self.Ts.requires_grad_(True)
        return
    
    def apply_deformation(self):
        out = {}
        if self.params == 12:
            # deformed_EDNodes = torch.bmm(self.Rs, self.EDNodes.unsqueeze(-1)).squeeze(-1) + self.Ts
            deformed_EDNodes = self.EDNodes + self.Ts

            ed_num, ed_k = self.e_e_relation.shape
            ed_Rs = torch.index_select(self.Rs, dim=0, index=self.e_e_relation.reshape(ed_num*ed_k)) # ed_num * ed_k, 3, 3
            ed_Ts = torch.index_select(self.Ts, dim=0, index=self.e_e_relation.reshape(ed_num*ed_k)) # ed_num * ed_k, 3
            ed_EDNodes = torch.index_select(self.EDNodes, dim=0, index=self.e_e_relation.reshape(ed_num*ed_k)) # ed_num * ed_k, 3


            deformed_EDNeighbors = torch.bmm(
                ed_Rs,
                self.EDNodes.unsqueeze(1).unsqueeze(-1).expand(-1,ed_k,-1,-1).reshape(ed_num*ed_k,3,1) - ed_EDNodes.unsqueeze(-1)
            ).squeeze(-1) + ed_EDNodes + ed_Ts
            
            # deformed_EDNeighbors = torch.sum(
            #     deformed_EDNeighbors.reshape(ed_num, ed_k, 3) * self.e_e_weight.unsqueeze(-1),
            #     dim=1)

            out['deformed_EDNodes'] = deformed_EDNodes
            out['deformed_EDNeighbors'] = deformed_EDNeighbors.reshape(ed_num, ed_k, 3)

            v_num, v_k = self.v_e_relation.shape
            vertices_Rs = torch.index_select(self.Rs, dim=0, index=self.v_e_relation.reshape(v_num*v_k)) # v_num * v_k, 3, 3
            vertices_Ts = torch.index_select(self.Ts, dim=0, index=self.v_e_relation.reshape(v_num*v_k)) # v_num * v_k, 3
            vertices_EDNodes = torch.index_select(self.EDNodes, dim=0, index=self.v_e_relation.reshape(v_num*v_k)) # v_num * v_k, 3

            deformed_vertices = torch.bmm(
                vertices_Rs,
                self.mesh.vs.unsqueeze(1).unsqueeze(-1).expand(-1,v_k,-1,-1).reshape(v_num*v_k,3,1) - vertices_EDNodes.unsqueeze(-1)
                ).squeeze(-1) + vertices_EDNodes + vertices_Ts

            deformed_vertices = torch.sum(
                deformed_vertices.reshape(v_num, v_k, 3) * self.v_e_weight.unsqueeze(-1),
                dim=1)
            
            out['deformed_vertices'] = deformed_vertices


            if self.corr is not None:
                corr_num, corr_k = self.corr_e_relation.shape
                corr_Rs = torch.index_select(self.Rs, dim=0, index=self.corr_e_relation.reshape(corr_num*corr_k)) # corr_num * corr_k, 3, 3
                corr_Ts = torch.index_select(self.Ts, dim=0, index=self.corr_e_relation.reshape(corr_num*corr_k)) # corr_num * corr_k, 3
                corr_EDNodes = torch.index_select(self.EDNodes, dim=0, index=self.corr_e_relation.reshape(corr_num*corr_k)) # corr_num * corr_k, 3


                deformed_corrs = torch.bmm(
                    corr_Rs,
                    self.corr.unsqueeze(1).unsqueeze(-1).expand(-1,corr_k,-1,-1).reshape(corr_num*corr_k,3,1) - corr_EDNodes.unsqueeze(-1)
                    ).squeeze(-1) + corr_EDNodes + corr_Ts
                deformed_corrs = torch.sum(
                    deformed_corrs.reshape(corr_num, corr_k, 3) * self.corr_e_weight.unsqueeze(-1),
                    dim=1)
                
                out['deformed_corrs'] = deformed_corrs

        return out