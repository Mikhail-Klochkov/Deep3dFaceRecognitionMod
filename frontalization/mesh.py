import numpy as np, torch, copy

from utils_mesh import MeshOperator


class Mesh():

    def __init__(self, vertices:np.ndarray, triangles:np.ndarray, textures:np.ndarray):
        is_valid, data = self._is_valid_mesh(vertices, triangles, textures)
        assert is_valid, 'Invalid data for mesh construction! Check shapes and dtypes!'
        self._vertices = data['vertices']
        self._triangles = data['triangles']
        self._textures = data['textures']


    @property
    def vertices(self):
        return self._vertices


    @property
    def triangles(self):
        return self._triangles


    @property
    def textures(self):
        return self._textures


    @property
    def tensor_mesh_objects(self):
        data_torch = {}
        data_torch['vertices'] = torch.from_numpy(self._vertices)
        data_torch['triangles'] = torch.from_numpy(self._triangles)
        data_torch['textures'] = torch.from_numpy(self._textures)
        return data_torch


    @property
    def mesh_open3d(self):
        return MeshOperator.get_mesh_by(self._vertices, self._triangles, self._textures)


    def _is_valid_mesh(self, vertices:np.ndarray, triangles:np.ndarray, textures:np.ndarray):
        n_pts_ver, dim_ver = vertices.shape
        n_pts_tex, dim_tex = textures.shape
        triangles_dtype = triangles.dtype

        vertices_copy = copy.deepcopy(vertices)
        triangles_copy = copy.deepcopy(triangles)
        textures_copy = copy.deepcopy(textures)

        if triangles_dtype == np.float64:
            triangles_copy = triangles_copy.astype(np.int32)
        is_correct = (n_pts_ver == n_pts_tex) and (dim_tex == dim_ver) and (triangles_copy.dtype in (np.int32, np.int64))
        return is_correct, {'vertices': vertices_copy, 'triangles': triangles_copy, 'textures': textures_copy}