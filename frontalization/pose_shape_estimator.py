import numpy as np, torch, torch.nn as nn, pathlib, logging, cv2 as cv, open3d as op3d, copy

from typing import Union
from pathlib import Path

import open3d.visualization
from torch.optim import SGD, Adam
from collections import Counter
from scipy.spatial import KDTree

from basel_face_model_2009 import BaselFaceModel2009
from frontalizer import CameraCalibrator
from landmarks_predictor import LandmarksPredictorDlib, ImagesGenerator
from generator_bfm_model import GeneratorWrapperBaselFaceModel
from utils_mesh import MeshOperator, create_sphere_mesh
from utils_mesh import load_68_keypts
from mesh import Mesh
#from renderer import Render


path_bfm_model = Path('../BFM/01_MorphableModel.mat')
path_landmarks_bfm_indeces = Path('../BFM/landmarks68_BFM.anl')
path_dir_dlib_shape_preds = Path('../checkpoints/lm_model')


logging.basicConfig(
     level=logging.INFO, format= '{%(pathname)s:%(lineno)d} %(levelname)s - %(message)s', datefmt='%H:%M:%S')



class LandmarksProjectorBfmModel(nn.Module):


    _default_config = {'shape_dim': 50,
                       'pose_representation': 'matrix',
                       'dtype' : torch.float32,
                       'device': 'cpu',
                       'fit_shape': False,
                       'image_shape': (640, 640),
                       'shape_init': 'random',
                       'scaled': True}


    _required_parameters = {'dtype', 'pose_representation', 'device', 'image_shape'}


    def __init__(self, path_bfm_model=None, path_landmarks_indeces=None, **kwargs):
        super().__init__()
        #assert 'image_shape' in kwargs, 'Incorrect config, need "image_shape" parameter!'
        self.config = self._default_config
        for key, value in kwargs.items():
            if key in self._default_config:
                if isinstance(value, type(self._default_config[key])):
                    self.config[key] = value
                else:
                    assert False, f'Incorrect type of value: "{value}"  type: "{type(value)}"'

        for required_param in self._required_parameters:
            if required_param not in self.config:
                assert False, f'Incorrect config, not found: {required_param} param.'

        self._dtype = self.config['dtype']
        self._device = self.config['device']
        self._pose_representation = self.config['pose_representation']
        # need for extracted mean face and basis
        assert path_landmarks_indeces and path_bfm_model, 'Undefined path bfm model and path landmarks indeces.'
        self._bfm = BaselFaceModel2009(path_bfm_model=path_bfm_model,
                                       path_landmark_indeces=path_landmarks_indeces,
                                       create_mean_face_mesh=False)
        # for symmetry mapping pixels
        self._bfm.build_vert_symmetry_pt_correspondence(complex=True)
        self._keypts_indeces = self._bfm.keypoints_indeces
        self._mean_shape = torch.tensor(self._bfm.mean_shape, dtype=self._dtype, device=self._device)
        if self.config['fit_shape']:
            shape_basis = self._bfm.shape_basis[:, :, :self.config['shape_dim']]
            shape_basis = shape_basis.reshape((-1, self.config['shape_dim']))
            self._shape_basis = torch.tensor(shape_basis, dtype=self._dtype, device=self._device)

    @property
    def dtype(self):
        return self._dtype


    @property
    def device(self):
        return self._device


    def init_model_parameters(self, R:np.ndarray, image_shape:tuple, t:np.ndarray=None, p:np.ndarray=None):
        assert hasattr(self, '_pose_representation'), f'Undefined pose_representation: "pose_representation"'
        # we need define representation (R, t)
        # Rodrigez projection procedure
        if self._pose_representation == 'matrix':
            assert R.shape == (3, 3) and (t.shape == (3, 1) or t.shape == (3,)),\
                'Incorrect shapes of pose parameters for initialization matrix pose_representation.'
        elif self._pose_representation == 'rodriguez':
            assert R.shape == (3, 1) or R.shape == (3,), \
                "Incorrect shapes of pose parameters for initialization Rodriguez pose_representation"
            R = R.squeeze()
        else:
            assert False, f'Undefined pose_representation="{self._pose_representation}"'
        self._t = nn.Parameter(torch.tensor(t, requires_grad=True, dtype=self._dtype, device=self._device))

        if self._pose_representation == 'matrix':
            self._R = nn.Parameter(torch.tensor(R, requires_grad=True, dtype=self._dtype, device=self._device))

        # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        elif self._pose_representation == 'rodriguez':
            R_tensor = torch.tensor(R, dtype=self._dtype, device=self._device)
            angle = torch.norm(R_tensor)
            angle.requires_grad = True
            if torch.allclose(angle, torch.tensor(0.0)):
                assert False, f'Incorrect division on zero. Vector norm: {angle}.'
            direction = R_tensor / angle
            direction.requires_grad = True
            self._R = {'direction': direction, 'angle': angle}
        else:
            assert False, f'Undefined pose_representation: {self._pose_representation}.'

        # define p (should be vector)
        if p is not None:
            shape_p = p.shape
            if any([dim_axis == 1 for dim_axis in shape_p]):
                p = p.squeeze()
            self._p = torch.tensor(p, dtype=self._dtype, device=self._device, requires_grad=True).reshape((1, -1))
        else:
            # or just without fitted shape (BFM model)
            if self.config['fit_shape']:
                # gaussian with mean=0, std=1. for all independent variables
                num_params_p = self.config['shape_dim']
                assert isinstance(num_params_p, int), f'Number shape parameters: {num_params_p}.'
                # correct zero for mean shape first initialization
                if self.config['shape_init'] == 'mean_shape':
                    self._p = torch.zeros(num_params_p, device=self._device, dtype=self._dtype)
                elif self.config['shape_init'] == 'random':
                    self._p = torch.normal(mean=torch.zeros(num_params_p), std=torch.ones(num_params_p))
                else:
                    assert False, 'Incorrect shape_init: "{}"'.format(self.config['shape_init'])

                self._p = self._p.to(device=self._device, dtype=self._dtype).reshape((1, -1))
                self._p.requires_grad = True
                # as parameter
                self._p = nn.Parameter(self._p)
                # TODO: DELETE that
                self._R.requires_grad = False
                self._t.requires_grad = False

        # self._p would be (1, d) array
        # init initrinsic camera matrix based on image shape
        self.config['image_shape'] = image_shape
        self._init_intrinsic_camera_matrix()
        logging.info('Initialize model LandmarkProjector parameters!')
        return self


    def _init_intrinsic_camera_matrix(self):
        if 'image_shape' not in self.config:
            assert False, 'Incorrect config. Not found "image_shape".'
        shape_image = self.config['image_shape']
        intrinsic_camera_matrix = CameraCalibrator.default_camera_matrix(shape_image, focal_lenght=shape_image[1])
        self._intrinsic_camera_matrix = torch.tensor(intrinsic_camera_matrix, dtype=self._dtype, device=self._device)


    # need only keypoints output should be (1, 68, 3)
    def forward_shape(self, all_points:bool=False, scale:bool=False):
        face_shape = self._mean_shape
        if self.config['fit_shape'] and hasattr(self, '_shape_basis'):
            # TODO: scale self._shape_basis
            if scale:
                id_part = torch.einsum('ij,aj->ai', self._shape_basis * 1e5 , self._p)
            else:
                id_part = torch.einsum('ij,aj->ai', self._shape_basis , self._p)

            face_shape = id_part + face_shape.reshape((1, -1))
        face_shape = face_shape.reshape((1, -1, 3))
        if all_points:
            return face_shape
        else:
            return face_shape[:, self._keypts_indeces]


    def forward(self, all_points:bool=False):
        # face shape should be (1, 68, 3)
        face_shape = self.forward_shape(all_points=all_points)
        # need projected this part on image
        if self._pose_representation == 'rodriguez':
            assert False, 'NotImplemented for rodriguez rotation.'
        assert self._pose_representation == 'matrix', 'Implement with matrix representation.'
        face_rotated = self._R @ face_shape.transpose(2, 1) + self._t
        face_proj = self._intrinsic_camera_matrix @ face_rotated
        face_proj = face_proj.transpose(2, 1)[0]
        # need divide z
        face_proj_2d = face_proj[..., :2] / face_proj[:, 2:]
        if self.config['scaled']:
            face_proj_2d[:, 0] /= self.config['image_shape'][1]
            face_proj_2d[:, 1] /= self.config['image_shape'][0]
            return face_proj_2d
        return face_proj_2d


    def get_transformed_pts(self, all_points=False):
        _3d_face_shape = self.forward_shape(all_points=all_points)
        _3d_face_transformed = self._R @ _3d_face_shape.transpose(2, 1) + self._t
        _3d_face_transformed = _3d_face_transformed.transpose(2, 1).squeeze().detach().numpy()
        return _3d_face_transformed


    def _get_normalize_cross_poduct(self, pts_1, pts_2, pts_3):
        cross_product = np.cross(pts_3 - pts_2, pts_1 - pts_3)
        cross_product /= np.linalg.norm(cross_product, axis=1)[:, np.newaxis]
        return cross_product


    def _get_map_from_vertex_pt_to_tris(self):
        triangles = self._bfm._triangles
        vrtx_idx_to_tris_inds = {}
        for idx_tri, tri in enumerate(triangles):
            for idx_pt in tri:
                if idx_pt in vrtx_idx_to_tris_inds:
                    vrtx_idx_to_tris_inds[idx_pt] += [idx_tri]
                else:
                    vrtx_idx_to_tris_inds[idx_pt] = [idx_tri]

        return vrtx_idx_to_tris_inds


    # TODO:
    def get_vertex_normals_custom(self):
        _3d_face_transformed = self.get_transformed_pts(all_points=True)
        triangles = self._bfm.triangles
        idxs_1, idxs_2, idxs_3 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        pts_1, pts_2, pts_3 = _3d_face_transformed[idxs_1], _3d_face_transformed[idxs_2], _3d_face_transformed[idxs_3]
        triangle_normals = self._get_normalize_cross_poduct(pts_1, pts_2, pts_3)
        vrtx_idx_to_tris_inds = self._get_map_from_vertex_pt_to_tris()
        vrtx_idx_to_normal = np.empty((_3d_face_transformed.shape[0], 3), dtype=np.float32)
        for idx_pt, tris_inds in vrtx_idx_to_tris_inds.items():
            normals = triangle_normals[tris_inds]
            normal = np.sum(normals, axis=0)
            normal /= np.linalg.norm(normal)
            vrtx_idx_to_normal[idx_pt, :] = normal

        return vrtx_idx_to_normal


    # TODO: after rotation mesh
    def extract_visible_3d_pts_custom(self, unit_camera_vector:np.ndarray=np.array([0, 0, -1])):
        vertex_normals = self.get_vertex_normals_custom()
        dot_product_norms_camera_vector = np.dot(vertex_normals, unit_camera_vector[:, np.newaxis])
        positive_product_indeces = np.where((dot_product_norms_camera_vector > 0).astype(np.bool))[0]
        return positive_product_indeces


    # TODO:
    def get_vertex_normals_open3d(self):
        _3d_face_transformed = self.get_transformed_pts(all_points=True)
        triangles = self._bfm.triangles
        face_mesh_rotated = MeshOperator.get_mesh_by(vertices=_3d_face_transformed, triangles=triangles)
        face_mesh_rotated.compute_vertex_normals(normalized=True)


    # TODO: Bad working
    def extract_visible_pts(self, strategy='all'):
        logging.info(f'Start computing visible points!')
        _3d_face_transformed = self.get_transformed_pts(all_points=True)
        # take triangles
        triangles = self._bfm._triangles
        idxs_1, idxs_2, idxs_3 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        pts_1, pts_2, pts_3 = _3d_face_transformed[idxs_1], _3d_face_transformed[idxs_2], _3d_face_transformed[idxs_3]
        triangle_normals = self._get_normalize_cross_poduct(pts_1, pts_2, pts_3)
        camera_unit_vect = np.array([0, 0, -1], np.float32)
        scalar_products = np.dot(triangle_normals, camera_unit_vect[:, np.newaxis])
        logging.info(f'Computed triangle normals!')
        idx_vert_to_idx_tris = self._get_connected_triangles(triangles)
        visible_pts = set()
        counter_vis_pts = 0
        for idx_pt, idxs_tris in idx_vert_to_idx_tris.items():
                if strategy == 'one':
                    # (n_normals_connected_pt, scalar_product)
                    if np.any(scalar_products[idxs_tris] > 0):
                        visible_pts.add(idx_pt)
                        counter_vis_pts +=1

                elif strategy == 'all':
                    if np.all(scalar_products[idxs_tris] > 0):
                        visible_pts.add(idx_pt)
                        counter_vis_pts +=1

                else:
                    raise NotImplemented(f'Not implemented strategy: {strategy} for searching visible points.')

                if idx_pt % 1000:
                    logging.info(f'Check: {idx_pt} points searching visible points.')


        return visible_pts


    def get_right_symmetry_idx_by_left(self, glob_l_idx:int):
        return self._bfm.get_right_glob_idx_by_left(glob_l_idx)


    def get_left_symmetry_idx_by_right(self, glob_r_idx: int):
        return self._bfm.get_left_glob_idx_by_right(glob_r_idx)


    def get_left_face_glob_idxs(self):
        if not hasattr(self._bfm, 'left_idx_to_right_idx_sym') or not hasattr(self._bfm, 'right_idx_to_left_idx_sym'):
            assert False, 'Not found dicts for symmetric mappings.'

        return self._bfm.left_idx_to_right_idx_sym.keys()


    def get_right_face_glob_idxs(self):
        if not hasattr(self._bfm, 'left_idx_to_right_idx_sym') or not hasattr(self._bfm, 'right_idx_to_left_idx_sym'):
            assert False, 'Not found dicts for symmetric mappings.'

        return self._bfm.right_idx_to_left_idx_sym.keys()


    def _get_connected_triangles(self, triangles:np.ndarray):
        idx_vert_to_idx_tris = {}
        for idx_triangle, triangle in enumerate(triangles):
            for idx_pt in triangle:
                self._update_vertex_to_triangles_corr(idx_pt, idx_triangle, idx_vert_to_idx_tris)
        assert len(idx_vert_to_idx_tris) == len(self._mean_shape), 'Number keys in dictionary ' \
                                                                      'should be equal number vetrices!'
        return idx_vert_to_idx_tris


    def _update_vertex_to_triangles_corr(self, idx_pt, idx_tri, idx_vert_to_idxs_tris):
        if idx_pt in idx_vert_to_idxs_tris:
            idx_vert_to_idxs_tris[idx_pt].append(idx_tri)
        else:
            idx_vert_to_idxs_tris[idx_pt] = [idx_tri]


    def _get_normal_triangle(self, pts_1, pts_2, pts_3):
        cross_product = np.cross(pts_3 - pts_2, pts_2 - pts_1)
        cross_product /= np.linalg.norm(cross_product, axis=1)[:, np.newaxis]
        return cross_product



class TrainerLandmarksProjector():


    _default_config = {"lr": 10, "momentum": 0.9, "weight_decay": 0, "epochs": 100, 'solver_type': 'PnP',
                       'beta': 1, 'gamma' : 0.0023}
    _number_landmarks = 68


    def __init__(self, path_image:Union[pathlib.Path, str],
                 camera_calibrator: CameraCalibrator,
                 model_projected: LandmarksProjectorBfmModel,
                 loss_type:str='MSE_weighted',
                 _2d_keypts_68=None,
                 ** kwargs):
        """
        Args:
            path_image:
            camera_calibrator: for good initialization
            model_projected: model, which we want train
            loss_type:
            **kwargs:
        """
        assert loss_type in ("MSE", "MSE_weighted"), f'Incorrect loss_type: {loss_type}!'
        self.config = self._default_config
        for name, value in kwargs.items():
            if name in self.config:
                if isinstance(value, type(self.config[name])):
                    self.config[name] = value
            else:
                assert False, f'Incorrect value parameter: {name} with value: {value}.'

        self._model = model_projected
        self._camera_calibrator = camera_calibrator
        if isinstance(path_image, str):
            path_image = Path(path_image)
        assert path_image.is_file(), f'The file {path_image} not found!'
        self._path_image = path_image
        self._loss_type = loss_type

        self._device = self._model.device
        self._dtype = self._model.dtype

        # need found good initialization (projected points, R and t)
        rot_matr, transl_vect, image_np = self.get_projected_init_values(_2d_keypts_68=_2d_keypts_68)
        self._image = image_np
        self._image_shape = image_np.shape[:2]
        if _2d_keypts_68 is not None:
            self._face_detection_result = [{'kps': _2d_keypts_68, 'bbox': None}]
        else:
            self._face_detection_result = self.get_2d_keypts(image_np)
        # more important place
        self._R_init = rot_matr
        self._t_init = transl_vect
        self._2d_keypts = torch.tensor(self._face_detection_result[0]['kps'].astype(np.float32), device=self._device)
        # is we want add some conditions
        self._identity_3_3 = torch.eye(3, device=self._device, dtype=self._dtype)
        if 'beta' in self.config:
            self._beta = self.config['beta']
        else:
            self._beta = 1.

        if 'gamma' in self.config:
            self._gamma = self.config['gamma']
        else:
            self._gamma = 0.0023
        # need important step (set all parameters)
        # copy from another model
        # initialize another parameters
        self.init_start_parameters()
        self._init_optimizer()
        self._init_loss()
        # scaling 2d_keypoints
        self._scaled = False
        if self._model.config['scaled']:
            self._2d_keypts = self.scale_2d_pts_forward(self._2d_keypts)
            self._scaled = True
        if hasattr(self._model, '_p'):
            self._generator_bfm = GeneratorWrapperBaselFaceModel(bfm_folder='../BFM')


    def project_all_shape_points(self):
        _2d_shape_face = self._model.forward(all_points=True)
        # need backward projection face
        _2d_shape_face_scaled = self.scale_2d_pts_backward(_2d_shape_face).detach().numpy().astype(np.int32)
        return _2d_shape_face_scaled


    def visualize_colored_mesh(self, vertices:np.ndarray, triangles:np.ndarray, vertex_colors:np.ndarray,
                               draw_pts:np.ndarray, add_circle_meshs:bool=True):
        face_mesh = MeshOperator.get_mesh_by(vertices, triangles, vertex_colors)
        frame_mesh = MeshOperator.get_mesh_coor_frame_mean_face(face_mesh)
        list_meshs = [face_mesh, frame_mesh]
        if add_circle_meshs:
            circle_meshs = []
            for pt in draw_pts:
                mesh_circle = create_sphere_mesh(center=pt, radius=0.05, color=[0, 1, 0])
                circle_meshs.append(mesh_circle)
            list_meshs += circle_meshs

        op3d.visualization.draw_geometries(list_meshs, width=800, height=600)


    def draw_face_shape_based_visibility_normals(self, test_type:str='z', eps_z:float=0.007, eps_dist:float=0.0005):
        _2d_face_shape = self.project_all_shape_points()
        _3d_face_shape_rotated_pts = self._model.get_transformed_pts(all_points=True)
        # need visualize this 3d points
        triangles = self._model._bfm.triangles
        face_shape_rotated = MeshOperator.get_mesh_by(_3d_face_shape_rotated_pts, triangles)
        frame_mesh = MeshOperator.get_mesh_coor_frame_mean_face(face_shape_rotated)
        face_shape_rotated.compute_vertex_normals()
        op3d.visualization.draw_geometries([face_shape_rotated, frame_mesh], width=800, height=600)
        #visible_pts = self._model.extract_visible_pts(strategy='all')
        visible_pts = self._model.extract_visible_3d_pts_custom()
        vertex_colors = np.zeros((_2d_face_shape.shape[0], 3)).astype(np.uint8)
        vertex_colors[visible_pts] = [0, 1, 0]
        face_mesh_rot_marked = MeshOperator.get_mesh_by(_3d_face_shape_rotated_pts, triangles, vertex_colors)
        op3d.visualization.draw_geometries([face_mesh_rot_marked, frame_mesh], width=800, height=600)
        pixel_to_idxs_3d_pts = {}
        for idx_pt in visible_pts:
            _2d_pt = _2d_face_shape[idx_pt]
            x, y = _2d_pt
            x, y = int(x), int(y)
            hash_pixel = f'{x}_{y}'
            if hash_pixel in pixel_to_idxs_3d_pts:
                pixel_to_idxs_3d_pts[hash_pixel] += [idx_pt]
            else:
                pixel_to_idxs_3d_pts[hash_pixel] = [idx_pt]
        # need select corrected points
        vertex_colors = np.zeros((_3d_face_shape_rotated_pts.shape[0], 3), dtype=np.uint8)
        number_colored_points = 0
        for idx_pixel, (hash_pixel, list_3d_pt_idxs) in enumerate(pixel_to_idxs_3d_pts.items()):
            x, y = hash_pixel.split('_')
            x, y = int(x), int(y)
            points_to_pixel = _3d_face_shape_rotated_pts[list_3d_pt_idxs]
            pt_idxs_sort = np.argsort(points_to_pixel[:, 2])
            closest_idx_pt = pt_idxs_sort[0]
            closest_z_pt_to_scr = points_to_pixel[closest_idx_pt]
            vertex_colors[list_3d_pt_idxs[closest_idx_pt]] = self._image[y, x]
            number_colored_points += 1
            for idx_pt in pt_idxs_sort[1:]:
                _3d_pt = points_to_pixel[idx_pt]
                # some test
                if test_type == 'z':
                    # check neighbour point
                    if np.abs(_3d_pt[2] - closest_z_pt_to_scr[2]) < eps_z:
                        vertex_colors[list_3d_pt_idxs[idx_pt]] = self._image[y, x]
                        number_colored_points += 1

                elif test_type == 'dist':
                    if np.linalg.norm(_3d_pt - closest_z_pt_to_scr) < eps_dist:
                        vertex_colors[list_3d_pt_idxs[idx_pt]] = self._image[y, x]
                        number_colored_points += 1

                else:
                    assert False, f'Undefined test_type: {test_type}'

            if idx_pixel % 100 == 0:
                logging.info(f'Preprocessed pixels: {idx_pixel}.')
        # before
        self._visualize_face_shape(vertices_colors=vertex_colors, visualize_mesh=True, visualize_texture=False)
        logging.info(f'Colored {number_colored_points} 3d mesh points!')
        # by symmetry we draw not visible points of face
        mask = (vertex_colors == np.zeros(3))
        mask = [all(row) for row in mask]
        zeros_color_indeces = np.where(mask)[0]
        # identify which side of colors is hidden
        left_not_colored_counts = 0
        right_not_colored_counts = 0
        left_indeces_set = set(self._model.get_left_face_glob_idxs())
        right_indeces_set = set(self._model.get_right_face_glob_idxs())
        for idx in zeros_color_indeces:
            if idx in left_indeces_set:
                left_not_colored_counts += 1
                continue
            if idx in right_indeces_set:
                right_not_colored_counts += 1
        black_color = np.zeros(3)
        # HARD Symmetry
        if right_not_colored_counts > left_not_colored_counts:
            for idx_l_glob in left_indeces_set:
                # not black color (not visible pt)
                left_face_color = vertex_colors[idx_l_glob]
                if np.all(left_face_color == black_color):
                    continue
                idx_r_glob = self._model.get_right_symmetry_idx_by_left(idx_l_glob)
                vertex_colors[idx_r_glob] = left_face_color
        else:
            for idx_r_glob in right_indeces_set:
                # not black color (not visible pt)
                right_face_color = vertex_colors[idx_r_glob]
                if np.all(right_face_color == black_color):
                    continue
                idx_l_glob = self._model.get_left_symmetry_idx_by_right(idx_r_glob)
                vertex_colors[idx_l_glob] = right_face_color
        # after
        self._visualize_face_shape(vertices_colors=vertex_colors, visualize_mesh=True, visualize_texture=False)
        return None


    def draw_face_shape_based_visibility_score(self):
        _2d_face_shape = self.project_all_shape_points()
        # calculate visibility of pixel point
        pixel_counts = Counter()
        pixel_index_3d_pts = {}
        for idx, pt in enumerate(_2d_face_shape):
            hash_pixel = f'{pt[0]}_{pt[1]}'
            pixel_counts[hash_pixel] += 1
            if hash_pixel in pixel_index_3d_pts:
                pixel_index_3d_pts[hash_pixel].append(idx)
            else:
                pixel_index_3d_pts[hash_pixel] = [idx]

        pixel_visibility = {}
        # calculate visibility score
        for pixel_hash, count in pixel_counts.items():
            pixel_visibility[pixel_hash] = 1 - np.exp(-count)
        # draw each visible point
        vertex_colors = (255. * np.zeros((_2d_face_shape.shape[0], 3))).astype(np.uint8)
        number_set_points = 0
        for pixel_hash, count in pixel_counts.items():
            if count <= 4:
                x, y = pixel_hash.split('_')
                x, y = int(x), int(y)
                rgb = self._image[y, x]
                idxs = pixel_index_3d_pts[pixel_hash]
                for idx in idxs:
                    vertex_colors[idx] = rgb
                    number_set_points += 1
        logging.info(f'Extracted rgb colors: {number_set_points}')
        # image_copy = copy.deepcopy(self._image)
        # _2d_all_pts = self._model.forward(all_points=True).detach().numpy()
        # image_all_pts = ImagesGenerator.add_keypts_and_face_dets_on_img(image_copy, [{'kps': _2d_all_pts}],
        #                                                                 add_rectangle=False)
        # cv.imshow('w', image_all_pts)
        # cv.waitKey(0)
        # visualize
        self._visualize_face_shape(vertices_colors=vertex_colors, visualize_texture=False)
        return vertex_colors


    def scale_2d_pts_forward(self, _2d_pts):
        _2d_pts[:, 0] = _2d_pts[:, 0] / self._image_shape[1]
        _2d_pts[:, 1] = _2d_pts[:, 1] / self._image_shape[0]
        return _2d_pts


    def scale_2d_pts_backward(self, _2d_pts_scaled):
        _2d_pts_scaled[:, 0] = _2d_pts_scaled[:, 0] * self._image_shape[1]
        _2d_pts_scaled[:, 1] = _2d_pts_scaled[:, 1] * self._image_shape[0]
        return _2d_pts_scaled


    def init_start_parameters(self):
        self._model.init_model_parameters(R=self._R_init, t=self._t_init, image_shape=self._image_shape)
        logging.info(f'Init LandmarksProjector model with parameters: \n')
        for name, value in self._model.named_parameters():
            logging.info(f'NAME: {name} \n{value}\n')


    def _init_loss(self, alpha:float=5.0):
        self._loss = torch.nn.MSELoss(reduction='sum')
        if self._loss_type == 'MSE_weighted':
            weights = torch.ones(self._number_landmarks, dtype=self._dtype, device=self._device)
            # higher with keypts more specific
            weights[18:] = alpha * weights[18:]
            self._weights = weights[:, None]
            logging.info(f'We use WEIGHTED strategy for fitting keypoints!')
        logging.info(f'Initialize loss function!')


    def _init_optimizer(self):
        lr = self.config['lr']
        weight_decay = self.config['weight_decay']
        momentum = self.config['momentum']
        #self._optimizer = SGD(self._model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self._optimizer = Adam(self._model.parameters(), lr=lr, weight_decay=weight_decay)
        logging.info(f'Initialize optimizer!')


    def get_2d_keypts(self, image:np.ndarray):
        return self._camera_calibrator.get_landmarks_by_image(image)


    def get_projected_init_values(self, _2d_keypts_68=None, solver_type='PnP'):
        assert solver_type in ("PnP", "calib"), f'Incorrect behavior incorrect solver_type: {solver_type}!'
        result = self._camera_calibrator.found_camera_matrix(filepath=self._path_image,
                                                             _2d_keypoints=_2d_keypts_68,
                                                             type_calibration=solver_type,
                                                             transform_to_matrix=True,
                                                             visualize=True)
        rot_matr, transl_vect = result[:2]
        image_np = result[-1]
        return rot_matr, transl_vect, image_np


    @staticmethod
    def get_necessary_objects(path_bfm_model, path_landmark_labels, path_dir_dlib_shape_preds, fit_shape:bool=True):
        bfm = BaselFaceModel2009(path_bfm_model=path_bfm_model, path_landmark_indeces=path_landmark_labels)
        lands_predictor = LandmarksPredictorDlib(path_dlib_lands_preds=path_dir_dlib_shape_preds)
        camera_calibrator = CameraCalibrator(bfm=bfm, lands_predictor=lands_predictor)
        landmark_projector = LandmarksProjectorBfmModel(path_bfm_model=path_bfm_model,
                                                        path_landmarks_indeces=path_landmark_labels,
                                                        fit_shape=fit_shape)
        return bfm, camera_calibrator, landmark_projector


    def weighted_mse(self, target, pred, reduction='sum'):
        if reduction == 'sum':
            if self._model.config['fit_shape']:
                loss_residuals = (self._weights * (target - pred) ** 2).sum()
                loss_reg_R = self._beta * torch.norm(self._model._R.T @ self._model._R - self._identity_3_3)
                loss_reg_p = self._gamma * torch.norm(self._model._p)
                print(f'Residuals: {loss_residuals.item()} Reg_R: {loss_reg_R.item()}, Reg_p: {loss_reg_p.item()}')
                return loss_residuals + loss_reg_R + loss_reg_p
            else:
                return (self._weights * (target - pred) ** 2).sum() \
                       + self._beta * torch.norm(self._model._R.T @ self._model._R - self._identity_3_3)

        elif reduction == 'mean':
            return (self._weights * (target - pred) ** 2).mean()


    def fit(self, debug:bool=False):
        epochs = self.config['epochs']
        preds_points = []

        # on the start we check generated face
        if hasattr(self._model, '_p'):
            # first
            self._visualize_face_shape()

        for epoch in range(epochs):
            pred = self._model.forward()
            if self._loss_type == 'MSE_weighted' and hasattr(self, '_weights'):
                loss = self.weighted_mse(self._2d_keypts, pred)
                #loss = self._loss(self._weights * self._2d_keypts, self._weights * pred)
            else:
                loss = self._loss(self._2d_keypts, pred)
            if debug:
                print(f'Epoch: {epoch}, loss: {loss.item() * 100}')
            preds_points.append((epoch, pred.detach().numpy()))
            self._optimizer.zero_grad()
            loss.retain_grad()
            loss.backward()
            self._optimizer.step()
            # can visualize results
            if epoch % 10 == 0:
                if hasattr(self._model, '_p'):
                    print(f'_p: {self._model._p}')
                if debug:
                    print(f'loss.grad: {loss.grad}, R.grad: {self._model._R.grad}, '
                          f't.grad: {self._model._t.grad}, p.grad: {self._model._p.grad}')
                self._optimizer.param_groups[0]['lr'] *= 0.8
                # need visualize image
                if self._scaled:
                    _2d_keypts_clone = torch.clone(self._2d_keypts).detach().numpy()
                    pred_clone = torch.clone(pred).detach().numpy()
                    _2d_keypts_unscaled = self.scale_2d_pts_backward(_2d_keypts_clone)
                    pred_unscaled = self.scale_2d_pts_backward(pred_clone)
                    result_true = {'kps': _2d_keypts_unscaled}
                    result_pred = {'kps': pred_unscaled}
                else:
                    result_true = {'kps': self._2d_keypts.detach().numpy()}
                    result_pred = {'kps': pred.detach().numpy()}

                image_true = ImagesGenerator.add_keypts_and_face_dets_on_img(image=self._image,
                                                                             result=[result_true],
                                                                             add_rectangle=False)
                image_true_preds = ImagesGenerator.add_keypts_and_face_dets_on_img(image=image_true,
                                                                                   result=[result_pred],
                                                                                   add_rectangle=False,
                                                                                   color_kps=(0, 255, 0))

                if debug:
                    cv.imshow(f'win_epoch{epoch}', image_true_preds[..., [2, 1, 0]].astype(np.uint8))
                    cv.waitKey(300)
                # check condition
                R_current = self._model._R
                logging.info(f'R.T @ R = \n{(R_current.T @ R_current).detach().numpy()}')
        # see after fitting
        if hasattr(self._model, '_p'):
            # last
            self._visualize_face_shape()


    def _get_triangles_bfm(self):
        return self._model._bfm.triangles


    def _visualize_face_shape(self, vertices_colors:np.ndarray=None, visualize_mesh:bool=False,
                              visualize_texture:bool=False):
        vertices = self._model.forward_shape(all_points=True).detach().numpy().squeeze(axis=0)
        triangles = self._model._bfm.triangles
        if vertices_colors is not None:
            if vertices_colors.dtype == np.uint8:
                vertices_colors = vertices_colors.astype(np.float32) / 255.
        face_mesh = MeshOperator.get_mesh_by(vertices=vertices,
                                             triangles=triangles,
                                             vertices_colors=vertices_colors)
        if vertices_colors is None:
            face_mesh.compute_vertex_normals()
        frame_mesh = MeshOperator.extract_mesh_coor_frame(face_mesh)
        if visualize_mesh:
            op3d.visualization.draw_geometries([face_mesh, frame_mesh], width=800, height=600)
        if visualize_texture:
            op3d.visualization.draw_geometries([face_mesh, frame_mesh], width=800, height=600)
            vertx, triangles, textures = face_mesh.vertices, face_mesh.triangles, face_mesh.vertex_colors
            mesh = Mesh(np.asarray(vertx), np.asarray(triangles), np.asarray(textures))
            render = Render()
            mesh_torch = render.get_torch_mesh_object_from_mesh(mesh)
            render_instance = render.define_render_instance()
            output = render_instance(mesh_torch)
            cv.imshow('win1', (255 * output[0, ..., :3].cpu().numpy()[..., [2, 1, 0]]).astype(np.uint8))
            cv.waitKey(0)


def test_forward_landmark_projector():
    path_test_image = Path('../datasets/examples/000037.jpg')

    bfm = BaselFaceModel2009(path_bfm_model, path_landmark_indeces=path_landmarks_bfm_indeces)
    lands_predictor = LandmarksPredictorDlib(path_dlib_lands_preds=path_dir_dlib_shape_preds)
    camera_calibr = CameraCalibrator(bfm=bfm, lands_predictor=lands_predictor)

    # returned rotation_vector and translation_vector and projected pts (Rodriguez representation)
    rotation_matrix, translation_vector, projected_pts, image_np = camera_calibr.found_camera_matrix(
                                                    path_test_image, type_calibration='PnP', visualize=False)
    model_landmarks = LandmarksProjectorBfmModel(
        path_bfm_model=path_bfm_model, path_landmarks_indeces=path_landmarks_bfm_indeces, fit_shape=True)
    model_landmarks.init_model_parameters(R=rotation_matrix, t=translation_vector, image_shape=image_np.shape[:2])

    output_pts = model_landmarks.forward()

    print(output_pts, projected_pts)


def test_fitted_frontalization(load_keypts=True):
    #path_test_image = Path('/home/users/mklochkov/projects/Deep3DFaceRecon_pytorch/datasets/train_images/celeb_4.jpeg')
    path_test_image = Path('../datasets/examples/000037.jpg')
    #path_test_image = Path('../datasets/examples/015384.jpg')
    #path_test_image = Path('../datasets/examples/015259.jpg')
    path_test_image = Path('../datasets/examples/13_Interview_Interview_2_People_Visible_13_425.jpg')
    bfm, camera_calibrator, model = TrainerLandmarksProjector.get_necessary_objects(path_bfm_model=path_bfm_model,
                                                    path_landmark_labels=path_landmarks_bfm_indeces,
                                                    path_dir_dlib_shape_preds=path_dir_dlib_shape_preds)
    _2d_keypts_68 = None
    if load_keypts:
        path_68_keypts = path_test_image.parent / (f'detections_68/{path_test_image.stem}' + '.txt')
        _2d_keypts_68 = load_68_keypts(path_68_keypts)

    trainer = TrainerLandmarksProjector(path_image=path_test_image,
                                        camera_calibrator=camera_calibrator,
                                        _2d_keypts_68=_2d_keypts_68,
                                        model_projected=model,
                                        loss_type='MSE_weighted',
                                        epochs=200,
                                        gamma=0.0023)
    trainer.fit(debug=True)
    pts_indeces_colors_another = trainer.draw_face_shape_based_visibility_normals()


if __name__ == '__main__':
    #test_forward_landmark_projector()
    test_fitted_frontalization()