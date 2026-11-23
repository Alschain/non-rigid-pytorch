import torch
import torch.nn.functional as F

try:
    from pytorch3d.ops import knn_points as _pytorch3d_knn_points
except ImportError:
    _pytorch3d_knn_points = None


def knn_points(p1, p2, K=1, chunk_size=4096):
    """K-nearest neighbors returning (dists_sq, idx, None), compatible with pytorch3d.ops.knn_points.

    Uses PyTorch3D's CUDA kernel when installed, otherwise falls back to chunked pure-PyTorch search.
    """
    if _pytorch3d_knn_points is not None:
        return _pytorch3d_knn_points(p1, p2, K=K)

    B, N1, _ = p1.shape
    _, N2, _ = p2.shape
    k = min(K, N2)

    dists_list = []
    idx_list = []
    for b in range(B):
        pts1 = p1[b]
        pts2 = p2[b]
        b_dists = []
        b_idx = []
        for start in range(0, N1, chunk_size):
            end = min(start + chunk_size, N1)
            d = torch.cdist(pts1[start:end], pts2, p=2.0)
            vals, inds = torch.topk(d, k=k, dim=-1, largest=False, sorted=True)
            b_dists.append(vals.pow(2))
            b_idx.append(inds)
        dists_list.append(torch.cat(b_dists, dim=0))
        idx_list.append(torch.cat(b_idx, dim=0))

    dists = torch.stack(dists_list, dim=0)
    idx = torch.stack(idx_list, dim=0)
    return dists, idx, None


def compute_normals(vertices, triangles):
    # Compute the face normals
    a = vertices[triangles.long()][:, 0, :]
    b = vertices[triangles.long()][:, 1, :]
    c = vertices[triangles.long()][:, 2, :]
    face_normals = torch.nn.functional.normalize(torch.cross(b - a, c - a, dim=-1), p=2, dim=-1) 

    # Compute the vertex normals
    vertex_normals = torch.zeros_like(vertices)
    vertex_normals = vertex_normals.index_add(0, triangles.long()[:, 0], face_normals)
    vertex_normals = vertex_normals.index_add(0, triangles.long()[:, 1], face_normals)
    vertex_normals = vertex_normals.index_add(0, triangles.long()[:, 2], face_normals)
    vertex_normals = torch.nn.functional.normalize(vertex_normals, p=2, dim=-1)

    return vertex_normals, face_normals


def find_edges(indices, remove_duplicates=True):
    # Extract the three edges (in terms of vertex indices) for each face 
    # edges_0 = [f0_e0, ..., fN_e0]
    # edges_1 = [f0_e1, ..., fN_e1]
    # edges_2 = [f0_e2, ..., fN_e2]
    edges_0 = torch.index_select(indices, 1, torch.tensor([0,1], device=indices.device))
    edges_1 = torch.index_select(indices, 1, torch.tensor([1,2], device=indices.device))
    edges_2 = torch.index_select(indices, 1, torch.tensor([2,0], device=indices.device))

    # Merge the into one tensor so that the three edges of one face appear sequentially
    # edges = [f0_e0, f0_e1, f0_e2, ..., fN_e0, fN_e1, fN_e2]
    edges = torch.cat([edges_0, edges_1, edges_2], dim=1).view(indices.shape[0] * 3, -1)

    if remove_duplicates:
        edges, _ = torch.sort(edges, dim=1)
        edges = torch.unique(edges, dim=0)

    return edges.long()

def find_connected_faces(indices):
    edges = find_edges(indices, remove_duplicates=False)

    # Make sure that two edges that share the same vertices have the vertex ids appear in the same order
    edges, _ = torch.sort(edges, dim=1)

    # Now find edges that share the same vertices and make sure there are only manifold edges
    _, inverse_indices, counts = torch.unique(edges, dim=0, sorted=False, return_inverse=True, return_counts=True)
    assert counts.max() == 2

    # We now create a tensor that contains corresponding faces.
    # If the faces with ids fi and fj share the same edge, the tensor contains them as
    # [..., [fi, fj], ...]
    face_ids = torch.arange(indices.shape[0])               
    face_ids = torch.repeat_interleave(face_ids, 3, dim=0) # Tensor with the face id for each edge

    face_correspondences = torch.zeros((counts.shape[0], 2), dtype=torch.int64)
    face_correspondences_indices = torch.zeros(counts.shape[0], dtype=torch.int64)

    # ei = edge index
    for ei, ei_unique in enumerate(list(inverse_indices.cpu().numpy())):
        face_correspondences[ei_unique, face_correspondences_indices[ei_unique]] = face_ids[ei] 
        face_correspondences_indices[ei_unique] += 1

    return face_correspondences[counts.cpu() == 2].long().to(device=indices.device)

def compute_laplacian_uniform(vertices, edges):
    """
    Computes the laplacian in packed form.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph
    Returns:
        Sparse FloatTensor of shape (V, V) where V = sum(V_n)
    """

    # This code is adapted from from PyTorch3D 
    # (https://github.com/facebookresearch/pytorch3d/blob/88f5d790886b26efb9f370fb9e1ea2fa17079d19/pytorch3d/structures/meshes.py#L1128)

    verts_packed = vertices # (sum(V_n), 3)
    edges_packed = edges    # (sum(E_n), 2)
    V = vertices.shape[0]

    e0, e1 = edges_packed.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (sum(E_n), 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (sum(E_n), 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*sum(E_n))

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=vertices.device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=vertices.device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=vertices.device)
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))

    return L

def compute_triangle_area(vertices, triangles):
    
    p0 = vertices[triangles.long()][:, 0, :]
    p1 = vertices[triangles.long()][:, 1, :]
    p2 = vertices[triangles.long()][:, 2, :]


    a = torch.sqrt(torch.sum((p0-p1)**2, dim=-1))
    b = torch.sqrt(torch.sum((p0-p2)**2, dim=-1))
    c = torch.sqrt(torch.sum((p1-p2)**2, dim=-1))

    p = (a + b + c) / 2

    s = torch.sqrt(p*(p-a)*(p-b)*(p-c))

    return s

def compute_edge_length(vertices, edges):
    
    p0 = vertices[edges][:, 0, :]
    p1 = vertices[edges][:, 1, :]


    L = torch.sqrt(torch.sum((p0-p1)**2, dim=-1))

    return L


def RandomSamplePoints(vertices, num):
    '''
    method:
        random sample num points from given vertices
    params:
        vertices: N x 3, N vertices
        num:      number of sampled points
    return:
        sample_indices: num x 1, indices of sampled points in original array
        sample_points:  num x 3, sampled points
    '''
    total_num = vertices.shape[0]
    sample_indices = torch.randperm(total_num)[:num]
    sample_indices = sorted(sample_indices)[::-1]
    sample_points = vertices[sample_indices]

    return sample_indices, sample_points

def UniformSamplePoints(vertices, num, method='fps'):
    '''
    method:
        uniform sample num points from given vertices
    params:
        vertices: N x 3, N vertices
        num:      number of sampled points
    return:
        sample_indices: num x 1, indices of sampled points in original array
        sample_points:  num x 3, sampled points
    '''
    total_num = vertices.shape[0]
    assert total_num >= num
    sample_indices = None
    sample_points = None

    if num == 0:
        sample_indices = torch.empty(0, dtype=torch.long, device=vertices.device)
        return sample_indices, vertices[:0]
    if num == total_num:
        sample_indices = torch.arange(total_num - 1, -1, -1, dtype=torch.long, device=vertices.device)
        return sample_indices, vertices[sample_indices]

    if method == 'fps':
        startid = int(torch.randint(0, total_num, (1,)).item())
        selected = torch.empty(num, dtype=torch.long, device=vertices.device)
        selected[0] = startid
        distmatrix = torch.sum((vertices[startid] - vertices) ** 2, dim=1)
        for i in range(1, num):
            distmatrix[selected[i - 1]] = -1.0
            select_index = int(torch.argmax(distmatrix).item())
            selected[i] = select_index
            curdist = torch.sum((vertices[select_index] - vertices) ** 2, dim=1)
            distmatrix = torch.minimum(distmatrix, curdist)
        sample_indices, _ = torch.sort(selected, descending=True)
        sample_points = vertices[sample_indices]
    else:
        pass
    return sample_indices, sample_points

def merge_mesh(meshes):
    mesh_count = len(meshes)
    if mesh_count == 0:
        return None
    if mesh_count == 1:
        return meshes[0]


    outmesh = meshes[0].clone()

    for mesh_id in range(1, mesh_count):
        vert_num = outmesh.vs.shape[0]
        print(vert_num)
        outmesh.vs = torch.cat([outmesh.vs, meshes[mesh_id].vs], dim=0)
        outmesh.f_v_idx = torch.cat([outmesh.f_v_idx, meshes[mesh_id].f_v_idx + vert_num], dim=0)

        if meshes[0].vts is not None:
            vt_num = outmesh.vts.shape[0]
            meshes[mesh_id].vts[:, 0] = meshes[mesh_id].vts[:, 0] + mesh_id
            outmesh.vts = torch.cat([outmesh.vts, meshes[mesh_id].vts], dim=0)
            outmesh.f_vt_idx = torch.cat([outmesh.f_vt_idx, meshes[mesh_id].f_vt_idx + vt_num], dim=0)

        if meshes[0].vns is not None:
            vn_num = outmesh.vns.shape[0]
            outmesh.vn = torch.cat([outmesh.vn, meshes[mesh_id].vn], dim=0)
            outmesh.f_vn_idx = torch.cat([outmesh.f_vn_idx, meshes[mesh_id].f_vn_idx + vn_num], dim=0)

        if meshes[0].vcs is not None:
            vc_num = outmesh.vcs.shape[0]
            outmesh.vcs = torch.cat([outmesh.vcs, meshes[mesh_id].vcs + vc_num], dim=0)
            outmesh.f_vc_idx = torch.cat([outmesh.f_vc_idx, meshes[mesh_id].f_vc_idx + vc_num], dim=0)

        if meshes[0].tex is not None:
            h, w, c = meshes[0].tex.shape
            meshes[mesh_id].tex = F.interpolate(
                meshes[mesh_id].tex.permute(2,0,1).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=True
                ).squeeze(0).permute(1,2,0)
            outmesh.tex = torch.cat([outmesh.tex, meshes[mesh_id].tex], dim=1)

    outmesh.vts[:, 0] /= mesh_count

    outmesh.edges = find_edges(outmesh.f_v_idx, remove_duplicates=True) if outmesh.f_v_idx is not None else None
    outmesh.connected_faces = find_connected_faces(outmesh.f_v_idx) if outmesh.f_v_idx is not None else None

    return outmesh