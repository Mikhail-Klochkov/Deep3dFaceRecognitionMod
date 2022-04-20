import numpy as np, open3d as op3d, torch

from pathlib import Path
from typing import Callable
from scipy.optimize import minimize

from basel_face_model_2009 import BaselFaceModel2009
from frontalization.geometry_features import GeometryFeaturesGradient, GeometryFeatures, GeometryDirCalcSpaceParam
from generator_bfm_model import GeneratorWrapperBaselFaceModel
from utils_mesh import MeshOperator
from utils_mesh import load_68_keypts
from shape_extractor_model import ShapeExtractor, TestOptionsWrapper
from frontalizer import CameraCalibrator
from landmarks_predictor import LandmarksPredictorDlib


path_dir_dlib_shape_preds = Path('../checkpoints/lm_model')
path_geometry_feats = Path.home() / 'Deep3DFaceRecon_pytorch_copy/datasets/custom_geoms/geom_linear_01.txt'
path_bfm_model = Path('../BFM/01_MorphableModel.mat')
path_landmarks_bfm_indeces = Path('../BFM/landmarks68_BFM.anl')


class OptimizerShapeParameterOneDir():


    parameters_names = {'R', 't', 'K_int', 'mean_shape', 'shape_basis', 'FHyperPlane', 'alpha_init', '_2d_gt_keypts',
                        'glob_camera_matr'}
    # (2, 3)
    project_op_u_v = torch.tensor([[1, 0, 0],
                                   [0, 1, 0]], dtype=torch.float32)
    # (1, 3)
    project_op_w = torch.tensor([[0, 0, 1]], dtype=torch.float32)


    def __init__(self, bfm: BaselFaceModel2009,
                 _2d_keypts,
                 K_int,
                 R,
                 t,
                 FHyperPlane,
                 alpha_init,
                 user_direction:int=-1,
                 face_colors=None,
                 normalize_direction=False,
                 debug=True):
        assert np.abs(user_direction) == 1, f'Incorrect user direction "{user_direction}"'
        # projection params
        self.debug = debug
        self.R = R
        self.t = t.reshape((1, 3))
        self.K_int = K_int
        # for help
        self.extrinsic_matr = CameraCalibrator.get_extrinsic_matrix(self.R, self.t.reshape(3, 1))
        self.glob_camera_matr = self.K_int @ self.extrinsic_matr
        # define facemodel params (mean_face, shape_basis)
        # (#, 1)
        self.mean_shape = bfm.mean_shape.reshape((-1, 1))
        # (#, 80)
        self.shape_basis = bfm.shape_basis.reshape((-1, bfm._num_sh_basis))[:, :alpha_init.shape[0]]
        self.triangles_face_shape = bfm.triangles
        self.keypts_indeces = bfm.keypoints_indeces
        # user direction param
        self.FHyperPlane = FHyperPlane
        self.alpha_init = alpha_init
        # calculate direction by take grad
        alpha_init_copy = self.alpha_init.clone()
        alpha_init_copy.requires_grad = True
        hyperplane = self.FHyperPlane(alpha_init_copy)
        hyperplane.backward()
        # not requires grad
        self.direction = alpha_init_copy.grad.detach()
        if normalize_direction:
            self.direction /= torch.norm(self.direction)
        self.user_direction = torch.tensor(user_direction, dtype=torch.float32)
        print('The direction of change in the shape parameter space has been initialized.')
        # init extracted from image keypoints
        self._2d_gt_keypts = _2d_keypts
        # TO TENSOR TYPE all objects
        self.to()
        # SUM squared loss
        self.history_loss = []
        self.history_grads = []
        self.loss = torch.nn.MSELoss(reduction='sum')
        self.face_colors = face_colors


    def forward_homo_coors(self, vecs:torch.Tensor):
        vecs_homo = torch.ones((vecs.shape[0], 4), dtype=torch.float32)
        vecs_homo[:, :3] = vecs
        return vecs_homo


    def forward_face_shape(self, id_coefs:torch.Tensor):
        face_shape = self.shape_basis @ id_coefs[:, None] + self.mean_shape
        return face_shape.view((-1, 3))


    def to(self):
        for name in self.parameters_names:
            if not hasattr(self, name):
                assert False, f'Incorrect Initialization: {self}. Not found parameter: {name}'
            attr = getattr(self, name)
            if name == 'FHyperPlane':
                assert isinstance(attr, Callable), 'parameter FHyperPlane should be callable py object'
                continue
            if not torch.is_tensor(attr):
                attr = torch.from_numpy(attr.astype(np.float32))

            setattr(self, name, attr)


    def forward_id_coefs(self, t:torch.Tensor):
        return self.alpha_init + self.user_direction * self.direction * t


    def forward_project_keypts_two(self, _3d_keypts:torch.Tensor):
        _3d_keypts_homo = self.forward_homo_coors(_3d_keypts)
        _3d_keypts_homo_rotate = _3d_keypts_homo @ self.glob_camera_matr.T
        return _3d_keypts_homo_rotate[:, :2] / _3d_keypts_homo_rotate[:, 2:]


    # TODO: something wrong
    def forward_project_keypts(self, _3d_keypts:torch.Tensor):
        # _3d_keypts = 68 * 3, R - (3, 3), K_int - (3,3),
        homog_3d_kps = (_3d_keypts @ self.R + self.t) @ self.K_int
        proj_keypts = (self.project_op_u_v @ homog_3d_kps.T) / (self.project_op_w @ homog_3d_kps.T)
        return proj_keypts.T


    def forward(self, t:torch.Tensor):
        id_coefs = self.forward_id_coefs(t)
        # this should be (68, 3)
        _3d_keypts = self.forward_face_shape(id_coefs)[self.keypts_indeces]
        #_2d_proj_keypts = self.forward_project_keypts(_3d_keypts)
        _2d_proj_keypts = self.forward_project_keypts_two(_3d_keypts)
        return _2d_proj_keypts


    def wrapper_numpy_func(self, t:np.ndarray):
        assert t.shape == (1,), f'Incorrect input: {t}, with shape: {t.shape}'
        t = t[0]
        t = torch.tensor(t, dtype=torch.float32)
        func = self.func(t)
        self.visualize_face_shape(t)
        self.history_loss.append({'t': t.detach().numpy(), 'loss': func.detach().numpy()})
        return func.detach().numpy()


    def visualize_face_shape(self, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32)
        id_coefs = self.forward_id_coefs(t)
        # full face
        face_shape = self.forward_face_shape(id_coefs).detach().numpy()
        face_mesh = MeshOperator.get_mesh_by(vertices=face_shape,
                                             triangles=self.triangles_face_shape,
                                             vertices_colors=self.face_colors)
        face_mesh.compute_vertex_normals()
        face_shape_init = self.forward_face_shape(self.alpha_init.numpy())
        face_mesh_init = MeshOperator.get_mesh_by(vertices=face_shape_init.numpy(),
                                                  triangles=self.triangles_face_shape,
                                                  vertices_colors=self.face_colors)
        face_mesh_init.compute_vertex_normals()
        if self.debug:
            op3d.visualization.draw_geometries([face_mesh, face_mesh_init.translate((2,0,0))], width=800, height=600)


    def wrapper_numpy_func_deriv(self, t:np.ndarray):
        assert t.shape == (1,), f'Incorrect input: {t}, with shape: {t.shape}'
        t = t[0]
        t = torch.tensor(t, dtype=torch.float32)
        grad_func_by_t = self.func_deriv(t)
        grad = grad_func_by_t.detach().numpy()
        self.history_grads.append(grad)
        return grad


    def func_deriv(self, t:torch.Tensor):
        if not t.requires_grad:
            t.requires_grad = True
        func = self.func(t)
        func.backward()
        return t.grad


    def func(self, t:torch.Tensor):
        _2d_proj_keypts = self.forward(t)
        return self.loss(self._2d_gt_keypts, _2d_proj_keypts)


    # fit method
    def fit(self, method:str='SLSQP', range=(0, None)):
        # need call scipy.optimize.minimize()
        res = minimize(fun=self.wrapper_numpy_func,
                       x0=np.array([0.0], dtype=np.float32),
                       jac=self.wrapper_numpy_func_deriv,
                       bounds=[range],
                       method=method)
        t_ans = res.x
        print(f'History loss: {self.history_loss}')
        return t_ans


class OptimizerShapeParametersSeveralDir():

    parameters_names = {'R', 't', 'K_int', 'mean_shape',
                        'shape_basis', 'alpha_init', '_2d_gt_keypts',
                        'glob_camera_matr'}
    # (2, 3)
    project_op_u_v = torch.tensor([[1, 0, 0],
                                   [0, 1, 0]], dtype=torch.float32)
    # (1, 3)
    project_op_w = torch.tensor([[0, 0, 1]], dtype=torch.float32)


    def __init__(self,
                 bfm: BaselFaceModel2009,
                 _2d_keypts,
                 K_int,
                 R,
                 t,
                 FHyperSurfaces,
                 alpha_init,
                 user_directions,
                 face_colors=None,
                 normalize_directions=False,
                 debug=False):
        self._init_projection_attrs(K_int, R, t)
        # define facemodel params (mean_face, shape_basis)
        self._init_facemodel_attrs(alpha_init, bfm)
        # HyperSurfaces -> list of callable wrappers
        self._init_surface_constr_attrs(FHyperSurfaces, alpha_init, normalize_directions, user_directions)
        # init extracted from image keypoints
        self._2d_gt_keypts = _2d_keypts
        # TO torch tensor
        self.to_tensor()
        # SUM squared loss
        self.history_loss = []
        self.history_grads = []
        # for reconstruction error (optional)
        self.loss = torch.nn.MSELoss(reduction='sum')
        self.face_colors = face_colors
        self.debug = debug


    def _init_surface_constr_attrs(self, FHyperSurfaces, alpha_init, normalize_directions, user_directions):
        assert isinstance(FHyperSurfaces, list), 'Incorrect type of FHyperSurfaces'
        self.FHyperSurfaces = FHyperSurfaces
        self.alpha_init = alpha_init
        # calculate direction by take grad
        self.directions = torch.empty((len(self.FHyperSurfaces), alpha_init.shape[0]), dtype=torch.float32)
        for idx, wrapper_hypersurf in enumerate(self.FHyperSurfaces):
            alpha_init_copy = self.alpha_init.clone()
            alpha_init_copy.requires_grad = True
            hyper_surf = wrapper_hypersurf(alpha_init_copy)
            hyper_surf.backward()
            direction = alpha_init_copy.grad.detach()
            if normalize_directions:
                direction /= torch.norm(direction)
            self.directions[idx, :] = (direction)

        self.user_directions = torch.empty((len(user_directions), ), dtype=torch.float32)
        for idx, user_dir in enumerate(user_directions):
            assert np.abs(user_dir) == 1., f'Incorrect user direction: {user_dir}'
            self.user_directions[idx] = user_dir

        # we need stacked all directions to -> (n_dirs, alpha_dim) tensor
        # we need transformed user_directions -> (n_dirs, 1) tensor
        # self.user_directions = torch.tensor(self.user_directions, dtype=torch.float32).view(-1, 1)
        # assert self.user_directions.shape[0] == self.directions.shape[0], \
        #    'Incorrect behaviour in _init_surface_constr_attrs'


    def _init_facemodel_attrs(self, alpha_init, bfm):
        self.mean_shape = bfm.mean_shape.reshape((-1, 1))
        # (#, 80)
        self.shape_basis = bfm.shape_basis.reshape((-1, bfm._num_sh_basis))[:, :alpha_init.shape[0]]
        self.triangles_face_shape = bfm.triangles
        self.keypts_indeces = bfm.keypoints_indeces


    def _init_projection_attrs(self, K_int, R, t):
        self.R = R
        self.t = t.reshape((1, 3))
        self.K_int = K_int
        # for help
        self.extrinsic_matr = CameraCalibrator.get_extrinsic_matrix(self.R, self.t.reshape(3, 1))
        self.glob_camera_matr = self.K_int @ self.extrinsic_matr


    # transform to tensor
    def to_tensor(self):
        for name in self.parameters_names:
            if not hasattr(self, name):
                assert False, f'Not found parameter: {name}'
            attr = getattr(self, name)
            if not torch.is_tensor(attr):
                attr = torch.from_numpy(attr.astype(np.float32))
            setattr(self, name, attr)


    def forward_id_coeffs(self, betas:torch.Tensor):
        # self.user_directions (n, 1) and betas also
        return self.alpha_init + torch.matmul(self.directions.T, torch.multiply(self.user_directions, betas))


    def forwar_face_shape_by_betas(self, betas:torch.Tensor):
        id_coefs = self.forward_id_coeffs(betas)
        return self.shape_basis @ id_coefs[:, None] + self.mean_shape


    def forward_face_shape(self, id_coefs:torch.Tensor):
        face_shape = self.shape_basis @ id_coefs[:, None] + self.mean_shape
        return face_shape.view((-1, 3))


    def forward_homo_coors(self, vecs:torch.Tensor):
        vecs_homo = torch.ones((vecs.shape[0], 4), dtype=torch.float32)
        vecs_homo[:, :3] = vecs
        return vecs_homo


    def forward_project_keypts(self, _3d_keypts:torch.Tensor):
        _3d_keypts_homo = self.forward_homo_coors(_3d_keypts)
        _3d_keypts_homo_rotate = _3d_keypts_homo @ self.glob_camera_matr.T
        return _3d_keypts_homo_rotate[:, :2] / _3d_keypts_homo_rotate[:, 2:]


    def forward(self, betas: torch.Tensor):
        id_coefs = self.forward_id_coeffs(betas)
        _3d_keypts = self.forward_face_shape(id_coefs)[self.keypts_indeces]
        _2d_proj_keypts = self.forward_project_keypts(_3d_keypts)
        return _2d_proj_keypts


    def wrapper_numpy_func(self, betas: np.ndarray):
        betas = self.check_input(betas)
        func = self.func(betas)
        self.visualize_face_shape(betas)
        self.history_loss.append({'betas': betas.detach().numpy(), 'loss': func.detach().numpy()})
        return func.detach().numpy()


    def visualize_face_shape(self, betas):
        if not torch.is_tensor(betas):
            betas = torch.tensor(betas, dtype=torch.float32)
        id_coefs = self.forward_id_coeffs(betas)
        face_shape = self.forward_face_shape(id_coefs).detach().numpy()
        face_mesh = MeshOperator.get_mesh_by(vertices=face_shape,
                                             triangles=self.triangles_face_shape,
                                             vertices_colors=self.face_colors)
        face_mesh.compute_vertex_normals()
        face_shape_init = self.forward_face_shape(self.alpha_init.numpy())
        face_mesh_init = MeshOperator.get_mesh_by(vertices=face_shape_init.numpy(),
                                                  triangles=self.triangles_face_shape,
                                                  vertices_colors=self.face_colors)
        face_mesh_init.compute_vertex_normals()
        if self.debug:
            op3d.visualization.draw_geometries([face_mesh, face_mesh_init.translate((2, 0, 0))], width=800, height=600)


    def wrapper_numpy_func_deriv(self, betas: np.ndarray):
        betas = self.check_input(betas)
        grad_func_by_params = self.func_deriv(betas)
        grad = grad_func_by_params.detach().numpy()
        self.history_grads.append(grad)
        return grad


    def check_input(self, betas: np.ndarray):
        assert betas.shape == (self.directions.shape[0],), f'Incorrect input: {betas}, with shape: {betas.shape}'
        return torch.tensor(betas, dtype=torch.float32)


    # the same
    def func_deriv(self, betas:torch.Tensor):
        if not betas.requires_grad:
            betas.requires_grad = True
        func = self.func(betas)
        func.backward()
        return betas.grad


    def func(self, betas:torch.Tensor):
        _2d_proj_keypts = self.forward(betas)
        return self.loss(self._2d_gt_keypts, _2d_proj_keypts)


    def fit(self, method: str = 'SLSQP'):
        x0 = np.zeros((self.directions.shape[0], ), dtype=np.float32)
        ranges = [(0, None) for _ in range(self.directions.shape[0])]
        res = minimize(fun=self.wrapper_numpy_func,
                       x0=x0,
                       jac=self.wrapper_numpy_func_deriv,
                       bounds=ranges,
                       method=method)
        betas_ans = res.x
        print(f'History loss: {self.history_loss}')
        return betas_ans


class TestGeometryFeatures():


    #path_dir_images = Path('/home/users/mklochkov/projects/Deep3DFaceRecon_pytorch/datasets/examples')
    path_dir_images = Path('/home/mklochkov/Deep3DFaceRecon_pytorch_copy/datasets/examples')


    @staticmethod
    def get_face_mesh_by_classic_bfm(bfm: BaselFaceModel2009, id_coefs):
        face_shape = bfm.get_face_shape(id_coefs)
        face_mesh = MeshOperator.get_mesh_by(face_shape, triangles=bfm.triangles)
        face_mesh.compute_vertex_normals()
        return face_mesh


    @staticmethod
    def test_geometries_features(num_id_coefs:int=80):
        bfm = BaselFaceModel2009(path_bfm_model=path_bfm_model,
                                 path_landmark_indeces=path_landmarks_bfm_indeces,
                                 scale_basis=False)
        bfm_gen = GeneratorWrapperBaselFaceModel(bfm_folder='../BFM')
        id_coefs = bfm_gen.generate_id_coefs()
        face_shape = bfm.get_face_shape(id_coefs)
        geom_feats = GeometryFeatures(bfm=bfm, num_id_coefs=num_id_coefs, path_geom_descrs=path_geometry_feats)
        face_mesh = TestGeometryFeatures.get_face_mesh_by_classic_bfm(bfm, id_coefs)
        for t in np.linspace(50, 100, 5):
            id_coefs_new = geom_feats.get_face_id_params_grad(id_geom=5, id_coefs_init=id_coefs, t=t)
            face_mesh_new = TestGeometryFeatures.get_face_mesh_by_classic_bfm(bfm, id_coefs_new).translate((2, 0, 0))
            op3d.visualization.draw_geometries([face_mesh, face_mesh_new])

        f_geoms_by_id = geom_feats.get_geoms_vector_by_id_coefs(id_coefs)
        f_geoms_by_face_shape = geom_feats.get_geoms_vector_by_face_shape(face_shape)
        print(f_geoms_by_id, f_geoms_by_face_shape)


    #'13_Interview_Interview_2_People_Visible_13_425'
    @staticmethod
    def test_fine_fitting_face_shape(filename='000037',
                                     load_68_kps=False,
                                     path_bfm_model=path_bfm_model,
                                     path_geometry_feats=path_geometry_feats,
                                     path_dir_dlib_shape_preds=path_dir_dlib_shape_preds,
                                     user_direction:int=1,
                                     num_id_coefs:int=80):
        # 0 initilize BFM model
        bfm = BaselFaceModel2009(path_bfm_model,path_landmark_indeces=path_landmarks_bfm_indeces)
        lands_predictor = LandmarksPredictorDlib(path_dir_dlib_shape_preds)

        # 1 step extract shape parameters from NN
        config_nn = TestOptionsWrapper()
        shape_extractor = ShapeExtractor(config=config_nn, device_rank=1, number_id_params=num_id_coefs)
        # extract initial shape parameters
        id_coefs, exp_coefs, tex_coefs = shape_extractor.extract_shape_parameters(filename=filename)
        id_coefs = id_coefs[0]
        tex_coefs = tex_coefs[0]
        # 1.2 extract face_shape based on id_coefs
        face_shape_init = bfm.get_face_shape(id_coeff=id_coefs)
        face_tex_init = bfm.get_face_colors(tex_coefs=tex_coefs)
        face_mesh_init = TestGeometryFeatures.get_face_mesh_by_classic_bfm(bfm, id_coefs)
        # 2 step extract rotation and translation (CameraCalibration)
        path_filename = TestGeometryFeatures.path_dir_images / (filename + '.jpg')
        camera_calib = CameraCalibrator(bfm, lands_predictor, face_shape=face_shape_init)
        # need load more accurate keypoints
        _2d_kps_68 = None
        if load_68_kps:
            dir_68_kps = path_filename.parent / 'detections_68'
            path_68_kps = dir_68_kps / (filename + '.txt')
            assert path_68_kps.is_file(), f'The file {path_68_kps} not found!'
            _2d_kps_68 = load_68_keypts(path_68_kps)

        result = camera_calib.found_camera_matrix(path_filename,
                                                  _2d_keypoints=_2d_kps_68,
                                                 type_calibration='PnP',
                                                 visualize=True,
                                                 transform_to_matrix=True)
        rot_matrix, trans_vect, camera_intr = result[:3]
        image = result[-1]
        # 3 initialze Geometry features
        geom_feats = GeometryFeatures(bfm=bfm, num_id_coefs=num_id_coefs, path_geom_descrs=path_geometry_feats)
        # FHyperPlane
        geometry_grad = GeometryFeaturesGradient(geom_feats.A, geom_feats.b)
        # we can call hyperplane(x), where x from shape parameter space
        id_coefs_torch = torch.from_numpy(id_coefs.astype(np.float32))
        hyperplane = geometry_grad.wrapper_height_width(alpha_init=id_coefs_torch)
        # 4 need solve fine tuning face_shape
        if load_68_kps:
            # use it as ground thruth
            _2d_gt_keypts = _2d_kps_68
        else:
            result_kps_det = camera_calib.get_landmarks_by_image(image)
            _2d_gt_keypts = result_kps_det[0]['kps'].astype(np.float32)

        opt_finetune_faceshape = OptimizerShapeParameterOneDir(bfm,
                                                               _2d_gt_keypts,
                                                               K_int=camera_intr,
                                                               R=rot_matrix,
                                                               t=trans_vect,
                                                               FHyperPlane=hyperplane,
                                                               alpha_init=id_coefs_torch,
                                                               user_direction=user_direction,
                                                               face_colors=face_tex_init,
                                                               normalize_direction=False)
        id_coefs_finetuned = opt_finetune_faceshape.fit(method='SLSQP', range=(0, None))
        print(f'FineTuned: {id_coefs_finetuned}.')


    @staticmethod
    def test_fine_fitting_face_shape_complex(filename='51_Dresses_wearingdress_51_219',
                                     load_68_kps=True,
                                     path_bfm_model=path_bfm_model,
                                     path_geometry_feats=path_geometry_feats,
                                     path_dir_dlib_shape_preds=path_dir_dlib_shape_preds,
                                     num_id_coefs:int=80):
        # 0-initialize BFM model and landmark predictor
        bfm = BaselFaceModel2009(path_bfm_model, path_landmark_indeces=path_landmarks_bfm_indeces)
        lands_predictor = LandmarksPredictorDlib(path_dir_dlib_shape_preds)
        # 1 step extract shape parameters from NN
        config_nn = TestOptionsWrapper()
        shape_extractor = ShapeExtractor(config=config_nn, device_rank=1, number_id_params=num_id_coefs)
        # extract initial shape parameters
        id_coefs, exp_coefs, tex_coefs = shape_extractor.extract_shape_parameters(filename=filename)
        id_coefs = id_coefs[0]
        tex_coefs = tex_coefs[0]
        # 1.2 extract face_shape based on id_coefs
        face_shape_init = bfm.get_face_shape(id_coeff=id_coefs)
        face_tex_init = bfm.get_face_colors(tex_coefs=tex_coefs)
        face_mesh_init = TestGeometryFeatures.get_face_mesh_by_classic_bfm(bfm, id_coefs)
        # 2 step extract rotation and translation (CameraCalibration)
        path_filename = TestGeometryFeatures.path_dir_images / (filename + '.jpg')
        camera_calib = CameraCalibrator(bfm, lands_predictor, face_shape=face_shape_init)
        # need load more accurate keypoints
        _2d_kps_68 = None
        if load_68_kps:
            dir_68_kps = path_filename.parent / 'detections_68'
            path_68_kps = dir_68_kps / (filename + '.txt')
            assert path_68_kps.is_file(), f'The file {path_68_kps} not found!'
            _2d_kps_68 = load_68_keypts(path_68_kps)

        result = camera_calib.found_camera_matrix(path_filename,
                                                  _2d_keypoints=_2d_kps_68,
                                                  type_calibration='PnP',
                                                  visualize=True,
                                                  transform_to_matrix=True)
        rot_matrix, trans_vect, camera_intr = result[:3]
        image = result[-1]
        geom_feats = GeometryFeatures(bfm=bfm, num_id_coefs=num_id_coefs, path_geom_descrs=path_geometry_feats)
        # FHyperPlane
        geometry_grad = GeometryFeaturesGradient(geom_feats.A, geom_feats.b)
        # we can call hyperplane(x), where x from shape parameter space
        id_coefs_t_init = torch.from_numpy(id_coefs.astype(np.float32))
        if load_68_kps:
            # use it as ground thruth
            _2d_gt_keypts = _2d_kps_68
        else:
            result_kps_det = camera_calib.get_landmarks_by_image(image)
            _2d_gt_keypts = result_kps_det[0]['kps'].astype(np.float32)
        # iterative calculation a new positions for face shape
        config_opt = dict(bfm=bfm, _2d_keypts=_2d_gt_keypts, K_int=camera_intr, R=rot_matrix, t=trans_vect,
                          face_colors=face_tex_init)
        # calculate geometry features before run algorithm
        f_geom_before = geom_feats.get_geoms_vector_by_id_coefs(id_coefs)
        directions = []
        for type_fitting in ("wrapper_width_upper_width_lower", "wrapper_height_lips_width_lips"):
            hyperplane = getattr(geometry_grad, type_fitting)(id_coefs_t_init)
            config_opt.update({'alpha_init': id_coefs_t_init})
            config_opt.update({'FHyperPlane': hyperplane})
            opt_finetune_shape = OptimizerShapeParameterOneDir(bfm,
                                                               _2d_gt_keypts,
                                                               K_int=camera_intr,
                                                               R=rot_matrix,
                                                               t=trans_vect,
                                                               FHyperPlane=hyperplane,
                                                               alpha_init=id_coefs_t_init,
                                                               user_direction=1,
                                                               face_colors=face_tex_init,
                                                               normalize_direction=False,
                                                               debug=True)
            t = opt_finetune_shape.fit(method='SLSQP', range=(0, None))
            if not torch.is_tensor(t):
                t = torch.tensor(t, dtype=torch.float32)
            # new position for id_coefs
            id_coefs_t_init = opt_finetune_shape.forward_id_coefs(t)
            directions.append(opt_finetune_shape.direction)
        # get finaly geomety features after
        f_geom_after = geom_feats.get_geoms_vector_by_id_coefs(id_coefs_t_init.detach().numpy())
        print(f'Compare geometry features:\n Before: {f_geom_before}\nAfter: {f_geom_after}.')


    @staticmethod
    def test_geometries_features_by_image(filename:str='000037', num_id_coefs:int=80):
        bfm = BaselFaceModel2009(path_bfm_model=path_bfm_model, path_landmark_indeces=path_landmarks_bfm_indeces)
        config = TestOptionsWrapper()
        shape_extractor = ShapeExtractor(config=config, device_rank=1, number_id_params=num_id_coefs)
        # TODO: experiment with gradient by geometry constraints
        id_coefs, exp_coefs, tex_coefs = shape_extractor.extract_shape_parameters(filename=filename)
        id_coefs, exp_coefs, tex_coefs = id_coefs[0], exp_coefs[0], tex_coefs[0]
        id_coefs_t = torch.from_numpy(id_coefs.astype(np.float32))
        process_dirts = GeometryDirCalcSpaceParam(bfm, path_geometry_feats, num_id_coefs)
        grad_width_lips = process_dirts.get_dir_space_params('wrapper_width_height_width_lips', id_coefs_t)
        grad_width_face = process_dirts.get_dir_space_params('wrapper_height_width', id_coefs_t)
        # iterative approach / first change in one direction, after change in another
        id_coefs_new_1 = process_dirts.get_new_id_coefs(id_coefs_t, grad_width_face, t=10).detach().numpy()
        id_coefs_new_2 = process_dirts.get_new_id_coefs(id_coefs_new_1, grad_width_lips, t=50).detach().numpy()
        #id_coefs_new_1 /= np.linalg.norm(id_coefs_new_1)
        #id_coefs_new_2 /= np.linalg.norm(id_coefs_new_2)
        id_coefs_new_comb = id_coefs_new_2
        face_mesh_bfm_classic_new = TestGeometryFeatures.get_face_mesh_by_classic_bfm(bfm, id_coefs=id_coefs_new_comb)
        face_mesh_bfm_classic = TestGeometryFeatures.get_face_mesh_by_classic_bfm(bfm, id_coefs=id_coefs)
        face_mesh_bfm_classic.compute_vertex_normals()
        face_mesh_bfm_classic_new.compute_vertex_normals()
        op3d.visualization.draw_geometries([face_mesh_bfm_classic,
                                            face_mesh_bfm_classic_new.translate((2, 0, 0))], width=800, height=600)
        print('Here')
        # ####
        # face_shape_facemodel = shape_extractor.get_face_shape(id_coefs[None, :], None)
        # triangles_facemodel = shape_extractor.get_facemodel_triangles().detach().numpy()
        # face_shape_facemodel = face_shape_facemodel.detach().numpy()[0]
        # face_mesh_facemodel = MeshOperator.get_mesh_by(face_shape_facemodel, triangles_facemodel)
        # face_mesh_bfm_classic = TestGeometryFeatures.get_face_mesh_by_classic_bfm(bfm, id_coefs=id_coefs)
        # face_mesh_facemodel.compute_vertex_normals()
        # op3d.visualization.draw_geometries([face_mesh_facemodel])
        # face_mesh_bfm_classic.compute_vertex_normals()
        # op3d.visualization.draw_geometries([face_mesh_bfm_classic])
        # geom_feats = GeometryFeatures(bfm=bfm, num_id_coefs=num_id_coefs, path_geom_descrs=path_geometry_feats)
        # for t in np.linspace(0.5 * 1e5, 2 * 1e6, 10):
        #     id_coefs_new_1 = geom_feats.get_face_id_params(id_geom=5, id_coefs_init=id_coefs, t=t)
        #     face_mesh_new = TestGeometryFeatures.get_face_mesh_by_classic_bfm(bfm, id_coefs_new_1).translate(2, 0, 0)
        #     op3d.visualization.draw_geometries([face_mesh_bfm_classic, face_mesh_new])

    @staticmethod
    def test_fine_fitting_face_several_dirs(filename='51_Dresses_wearingdress_51_219',
                                            user_directions=None,
                                            name_directions=None,
                                            load_68_kps=True,
                                            path_bfm_model=path_bfm_model,
                                            path_geometry_feats=path_geometry_feats,
                                            path_dir_dlib_shape_preds=path_dir_dlib_shape_preds,
                                            num_id_coefs:int=80):
        # 0-initialize BFM model and landmark predictor
        bfm = BaselFaceModel2009(path_bfm_model, path_landmark_indeces=path_landmarks_bfm_indeces)
        lands_predictor = LandmarksPredictorDlib(path_dir_dlib_shape_preds)
        # 1 step extract shape parameters from NN
        config_nn = TestOptionsWrapper()
        shape_extractor = ShapeExtractor(config=config_nn, device_rank=1, number_id_params=num_id_coefs)
        # extract initial shape parameters
        id_coefs, exp_coefs, tex_coefs = shape_extractor.extract_shape_parameters(filename=filename)
        id_coefs = id_coefs[0]
        tex_coefs = tex_coefs[0]
        # 1.2 extract face_shape based on id_coefs
        face_shape_init = bfm.get_face_shape(id_coeff=id_coefs)
        face_tex_init = bfm.get_face_colors(tex_coefs=tex_coefs)
        face_mesh_init = TestGeometryFeatures.get_face_mesh_by_classic_bfm(bfm, id_coefs)
        # 2 step extract rotation and translation (CameraCalibration)
        path_filename = TestGeometryFeatures.path_dir_images / (filename + '.jpg')
        camera_calib = CameraCalibrator(bfm, lands_predictor, face_shape=face_shape_init)
        # need load more accurate keypoints
        _2d_kps_68 = None
        if load_68_kps:
            dir_68_kps = path_filename.parent / 'detections_68'
            path_68_kps = dir_68_kps / (filename + '.txt')
            assert path_68_kps.is_file(), f'The file {path_68_kps} not found!'
            _2d_kps_68 = load_68_keypts(path_68_kps)

        result = camera_calib.found_camera_matrix(path_filename,
                                                  _2d_keypoints=_2d_kps_68,
                                                  type_calibration='PnP',
                                                  visualize=True,
                                                  transform_to_matrix=True)
        rot_matrix, trans_vect, camera_intr = result[:3]
        image = result[-1]
        geom_feats = GeometryFeatures(bfm=bfm, num_id_coefs=num_id_coefs, path_geom_descrs=path_geometry_feats)
        geometry_grad = GeometryFeaturesGradient(geom_feats.A, geom_feats.b)
        # we can call hyperplane(x), where x from shape parameter space
        id_coefs_t_init = torch.from_numpy(id_coefs.astype(np.float32))
        if load_68_kps:
            # use it as ground thruth
            _2d_gt_keypts = _2d_kps_68
        else:
            result_kps_det = camera_calib.get_landmarks_by_image(image)
            _2d_gt_keypts = result_kps_det[0]['kps'].astype(np.float32)

        # we can initialize own
        if user_directions is None or name_directions is None:
            user_directions = [1., 1.]
            name_directions = ['wrapper_height_width', 'wrapper_width_height_width_lips']
        FHyperSurfaces = []
        for name_direction in name_directions:
            hypersurface = getattr(geometry_grad, name_direction)(id_coefs_t_init)
            FHyperSurfaces.append(hypersurface)

        assert user_directions is not None
        assert name_directions is not None
        f_geom_before = geom_feats.get_geoms_vector_by_id_coefs(id_coefs)
        opt_finetuner = OptimizerShapeParametersSeveralDir(bfm,
                                                           _2d_gt_keypts,
                                                           K_int=camera_intr,
                                                           R=rot_matrix,
                                                           t=trans_vect,
                                                           FHyperSurfaces=FHyperSurfaces,
                                                           alpha_init=id_coefs_t_init,
                                                           user_directions=user_directions,
                                                           face_colors=face_tex_init,
                                                           normalize_directions=False,
                                                           debug=True)
        #betas = torch.zeros((2, ), dtype=torch.float32)
        #opt_finetuner.visualize_face_shape(betas)
        #opt_finetuner.forward_id_coeffs(betas)
        betas_ans = opt_finetuner.fit()
        betas_ans = torch.from_numpy(betas_ans.astype(np.float32))
        id_coefs_after = opt_finetuner.forward_id_coeffs(betas_ans)
        f_geom_after = geom_feats.get_geoms_vector_by_id_coefs(id_coefs_after.detach().numpy())
        print(f'Compare geometry features:\n Before: {f_geom_before}\nAfter: {f_geom_after}.')


if __name__ == '__main__':
    TestGeometryFeatures.test_fine_fitting_face_several_dirs(user_directions=[1., 1.],
                                                             name_directions=['wrapper_width_upper_width_lower',
                                                                              'wrapper_height_lips_width_lips'])
    # TestGeometryFeatures.test_fine_fitting_face_shape_complex()