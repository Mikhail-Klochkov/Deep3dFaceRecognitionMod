import numpy as np, torch, pathlib

from typing import Union, Callable

from frontalization.basel_face_model_2009 import BaselFaceModel2009


class GeometryFeaturesGradient():


    def __init__(self, A:np.ndarray, b:np.ndarray):
        self.A = torch.from_numpy(A.astype(np.float32))
        self.b = torch.from_numpy(b.astype(np.float32))
        self.A.requires_grad = False
        self.b.requires_grad = False
        # before 6, 4
        self.width_lower_geom_id = 6
        self.width_upper_geom_id = 5
        self.height_geom_id = 4
        self.width_lips_id = 8
        self.height_lips_id = 9


    def geom_feature(self, alpha:torch.Tensor, id_geom:int):
        return (self.A @ alpha + self.b)[id_geom]


    def wrapper_width_height_width_lips(self, alpha_init:torch.Tensor):
        alpha_init = self.check_alpha_init(alpha_init)
        def rel_width_height_width_lips(alpha:torch.Tensor):
            width_lips_new = self.geom_feature(alpha, self.width_lips_id)
            width_lips_old = self.geom_feature(alpha_init, self.width_lips_id)
            height_lips_new = self.geom_feature(alpha, self.height_lips_id)
            height_lips_old = self.geom_feature(alpha_init, self.height_lips_id)
            return (height_lips_new/width_lips_new) - (height_lips_old/width_lips_old)

        return rel_width_height_width_lips


    def wrapper_width_head_width_lips(self, alpha_init:torch.Tensor):
        alpha_init = self.check_alpha_init(alpha_init)
        def rel_width_head_width_lips(alpha:torch.Tensor):
            width_lips_new = self.geom_feature(alpha, self.width_lips_id)
            width_lips_old = self.geom_feature(alpha_init, self.width_lips_id)
            height_lips_new = self.geom_feature(alpha, self.height_lips_id)
            height_lips_old = self.geom_feature(alpha_init, self.height_lips_id)
            return (height_lips_new/width_lips_new) - (height_lips_old/width_lips_old)

        return rel_width_head_width_lips


    def wrapper_height_lips_width_lips(self, alpha_init: torch.Tensor):
        alpha_init = self.check_alpha_init(alpha_init)
        def rel_height_lips_width_lips(alpha: torch.Tensor):
            width_lips_new = self.geom_feature(alpha, self.width_lips_id)
            width_lips_old = self.geom_feature(alpha_init, self.width_lips_id)
            width_head_new = self.geom_feature(alpha, self.width_lower_geom_id)
            width_head_old = self.geom_feature(alpha_init, self.width_lower_geom_id)
            return (width_head_new / width_lips_new) - (width_head_old / width_lips_old)

        return rel_height_lips_width_lips


    def wrapper_width_upper_width_lower(self, alpha_init:torch.Tensor):
        def rel_width_upper_width_lower(alpha:torch.Tensor):
            width_lower_new = self.geom_feature(alpha, self.width_lower_geom_id)
            width_upper_new = self.geom_feature(alpha, self.width_upper_geom_id)
            width_lower_old = self.geom_feature(alpha_init, self.width_lower_geom_id)
            width_upper_old = self.geom_feature(alpha_init, self.width_upper_geom_id)
            return (width_upper_new / width_lower_new) - (width_upper_old / width_lower_old)

        return rel_width_upper_width_lower


    def wrapper_height_width(self, alpha_init:torch.Tensor):
        alpha_init = self.check_alpha_init(alpha_init)
        def rel_height_width(alpha:torch.Tensor):
            width_new = self.geom_feature(alpha, self.width_lower_geom_id)
            height_new = self.geom_feature(alpha, self.height_geom_id)
            width_old = self.geom_feature(alpha_init, self.width_lower_geom_id)
            height_old = self.geom_feature(alpha_init, self.height_geom_id)
            return (height_new/width_new) - (height_old/width_old)

        return rel_height_width


    def check_alpha_init(self, alpha_init:torch.Tensor):
        assert alpha_init.dtype == torch.float32, f'Incorrect torch dtype: {alpha_init.dtype}'
        assert not alpha_init.requires_grad, f'Alpha init should be not required_gradients'
        return alpha_init


    def get_new_alpha(self, alpha_init:torch.Tensor, grad:torch.Tensor, t:Union[float, torch.Tensor]):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32)

        return grad * t + alpha_init



class GeometryDirCalcSpaceParam():


    def __init__(self, bfm, path_geometry_feats, num_id_coefs=80):
        self.geom_feats = GeometryFeatures(path_geometry_feats, bfm, num_id_coefs)
        A, b = self.geom_feats.A, self.geom_feats.b
        self.geom_feats_g = GeometryFeaturesGradient(A, b)
        self.avail_dir_consts = set()
        for nameattr, _ in self.geom_feats_g.__dict__.items():
            if nameattr.startswith('wrapper'):
                self.avail_dir_consts.add(nameattr)


    def get_dir_space_params(self, nameattr:str, id_coefs_init:torch.Tensor):
        assert nameattr not in self.avail_dir_consts, f'Incorrect nameattr: {nameattr}.'
        if not hasattr(self.geom_feats_g, nameattr):
            assert False, f'Incorrect nameattr {nameattr} for class object {self.geom_feats_g}'
        callable_hyper_pl = getattr(self.geom_feats_g, nameattr)(id_coefs_init)
        if not isinstance(callable_hyper_pl, Callable):
            assert False, f'nameattr "{nameattr}" is not callable'
        id_coefs_init_cp = id_coefs_init.clone()
        id_coefs_init_cp.requires_grad = True
        hyper_plane_constr = callable_hyper_pl(id_coefs_init_cp)
        hyper_plane_constr.backward()
        # this is direction
        return id_coefs_init_cp.grad


    def get_new_id_coefs(self, id_coefs:torch.Tensor, dir:torch.Tensor, t:float):
        return self.geom_feats_g.get_new_alpha(alpha_init=id_coefs, grad=dir, t=t)



class GeometryFeatures():


    def __init__(self, path_geom_descrs: pathlib.Path, bfm: BaselFaceModel2009, num_id_coefs:int=50):
        self.bfm = bfm
        self.geom_data = self.read_geom_feats(path_geom_descrs)
        if num_id_coefs > 80:
            self.num_basis = 80
        else:
            self.num_basis = num_id_coefs
        self.A, self.b = self.construct_linear_geoms()


    def get_face_id_params_grad(self, id_geom:int, id_coefs_init:np.ndarray, t:float):
        return self.A[id_geom-1, :] * t + id_coefs_init


    def get_face_id_params(self, id_geom:int, id_coefs_init:np.ndarray, t:float):
        direction = np.zeros(self.A.shape[0])
        direction[id_geom-1] = 1.
        return id_coefs_init - t * (self.A.T @ self.A) @ self.A.T @ direction


    def get_geoms_vector_by_id_coefs(self, id_coefs:np.ndarray):
        if id_coefs.shape[0] != self.num_basis:
            assert False, f'Incorrect shape if id_coefs {id_coefs}.'
        return self.A @ id_coefs + self.b


    def construct_linear_geoms(self, scale:bool=False):
        A = np.zeros((len(self.geom_data), self.num_basis))
        b = np.zeros(len(self.geom_data))
        for idx_geom, data_per_g in enumerate(self.geom_data):
            b_per_g = 0.
            a = np.zeros(self.num_basis)
            for pt_info in data_per_g:
                n_kpt, axis, coefs = pt_info['n_kpt'], pt_info['axis'], pt_info['coefs']
                idx_vrtx, axis = self.get_index(n_kpt, axis)
                b_per_g += coefs * self.bfm.mean_shape[idx_vrtx, axis]
                for id_comp in range(self.num_basis):
                    if scale:
                        a[id_comp] += coefs * (self.bfm.shape_basis[idx_vrtx, axis, id_comp] * 1e5)
                    else:
                        a[id_comp] += coefs * (self.bfm.shape_basis[idx_vrtx, axis, id_comp])

            b[idx_geom] = b_per_g
            A[idx_geom] = a

        return A, b


    def get_index(self, n_kpt:int, axis:str, format:str='matrix'):
        assert axis in ('x', 'y', 'z'), f'Incorrect axis: {axis}.'
        assert format in ("vector", 'matrix'), f'Incorrect output format param.'
        idx_vrtx = self.bfm.keypoints_indeces[n_kpt]
        if format == 'vector':
            idx_sh_vec = idx_vrtx * 3
            if axis == 'x':
                idx_sh_vec += 0
            elif axis == 'y':
                idx_sh_vec += 1
            else:
                idx_sh_vec += 3
            return idx_sh_vec
        else:
            if axis == 'x':
                return idx_vrtx, 0
            elif axis == 'y':
                return idx_vrtx, 1
            else:
                return idx_vrtx, 2


    def read_geom_feats(self, path_geom_description: pathlib.Path):
        assert path_geom_description.is_file(), f'The file {path_geom_description} not found!'
        data = []
        with path_geom_description.open() as reader:
            # n_pt, axis, coefs | ... \n
            for idx, line in enumerate(reader):
                line = line.rstrip('\n')
                points = line.split('|')
                data_per_g = []
                for point in points:
                    n_kpt, axis, coefs = point.split(',')
                    coefs = float(coefs)
                    n_kpt = int(n_kpt)
                    if axis not in ("x", "y", "z") and n_kpt > 68 and n_kpt < 0:
                        assert False, f'Incorrect format line: {line}.'

                    data_per_g.append({'n_kpt': n_kpt, 'axis':axis, 'coefs': coefs})
                axises = set([record['axis'] for record in data_per_g])
                if len(axises) > 1:
                    assert False, f'Incorrect format line: {line}.'

                data.append(data_per_g)

        return data


    def get_geoms_vector_by_face_shape(self, vertices:np.ndarray):
        f_geoms = np.zeros(len(self.geom_data))
        for idx_g, data_per_g in enumerate(self.geom_data):
            f = 0.
            for idx, pt_info in enumerate(data_per_g):
                n_kpt, axis, coefs = pt_info['n_kpt'], pt_info['axis'], pt_info['coefs']
                idx_vrtx, axis = self.get_index(n_kpt, axis)
                f += coefs * vertices[idx_vrtx, axis]

            f_geoms[idx_g] = f

        return f_geoms