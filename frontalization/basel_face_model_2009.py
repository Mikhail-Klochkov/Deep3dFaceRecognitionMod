import open3d as op3d, numpy as np, logging

from scipy.io import loadmat
from scipy.spatial import KDTree
from typing import Union, Tuple, List
from pathlib import Path
from frontalization.mesh import Mesh
from utils_mesh import MeshOperator, create_sphere_mesh


path_bfm_model = Path('../BFM/01_MorphableModel.mat')
path_additional_bfm = Path('../BFM/facemodel_info.mat')
path_landmarks_bfm_indeces = Path('../BFM/landmarks68_BFM.anl')
path_bfm_prepared = Path('../BFM/BFM_model_front.mat')


# https://ibug.doc.ic.ac.uk/resources/300-W/
class LandmarksAnnotator():

    def get_left_eye_landmarks_indeces(self):
        return [37, 38, 39, 40, 41, 42]

    def get_right_eye_landmarks_indeces(self):
        return [43, 44, 45, 46, 47, 48]

    def get_nose_vertical_landmarks(self):
        return [28, 29, 30, 31]

    def get_nose_gorizontal_landmarks(self):
        return [32, 33, 34, 35, 36]

    def get_upper_lip_landmarks(self):
        return [49, 50, 51, 52, 53, 54, 55, 61, 62, 63, 64, 65]

    def get_bottom_lip_landmarks(self):
        return [56, 57, 58, 59, 60, 66, 67, 68]

    def get_left_border_of_the_face(self):
        return [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def get_right_border_of_the_face(self):
        return [10, 11, 12, 13, 14, 15, 16, 17]


class BaselFaceModel2009():


    _default_pathes = {"path_bfm_model" : Path('../BFM/01_MorphableModel.mat'),
                       "path_additional_bfm": Path('../BFM/facemodel_info.mat'),
                       "path_landmarks_bfm_indeces" : Path('../BFM/landmarks68_BFM.anl'),
                       "path_bfm_prepared" : Path('../BFM/BFM_model_front.mat')}


    def __init__(self, path_bfm_model, path_bfm_triangulation=None, path_bfm_prepared_model=None,
                 path_landmark_indeces=None, scale_basis:bool=True, create_mean_face_mesh:bool=False):
        self._path_bfm_model = self.check_path(path_bfm_model)
        assert self._path_bfm_model.suffix == '.mat', 'Incorrect format BFM 2009!'
        if path_bfm_triangulation:
            self._path_bfm_additional_info = self.check_path(path_bfm_triangulation)
        if path_landmark_indeces:
            self._path_landmark_label = self.check_path(path_landmark_indeces)
        self._num_sh_basis = 199
        self._num_tex_basis = 199

        # load model
        if path_bfm_prepared_model:
            path_bfm_prepared_model = self.check_path(path_bfm_prepared_model)
            self._init_bfm_model(path_bfm_prepared_model, scale_basis=scale_basis)
        else:
            # without this filepath
            self._init_bfm_model()

        if create_mean_face_mesh:
            self._path_mean_face_shape_obj = self._save_mean_face_mesh_as_obj()


    def check_path(self, path, type='file'):
        if isinstance(path, str):
            path = Path(path)
        if type=='file':
            assert path.is_file(), f'The file {path} not found!'
        else:
            assert path.is_dir(), f'The directory {path} not found!'
        return path


    @property
    def shape_basis(self):
        return self._shape_basis


    def _init_bfm_model(self, path_bfm_prepare_model=None, scale_basis:bool=True):
        self._bfm_model = loadmat(str(self._path_bfm_model))
        self._shape_basis = self._bfm_model['shapePC'].astype(np.float64).reshape((-1, 3, self._num_sh_basis))/1e5  # shape basis
        self._shape_sigmas = self._bfm_model['shapeEV'].astype(np.float64).reshape((-1, self._num_sh_basis))  # corresponding eigen values
        self._texture_basis = self._bfm_model['texPC'].astype(np.float64).reshape((-1, 3, self._num_sh_basis))  # texture basis
        self._texture_sigmas = self._bfm_model['texEV'].astype(np.float64).reshape((-1, self._num_sh_basis))  # eigen value
        if scale_basis:
            self._shape_basis = self._shape_sigmas * self._shape_basis.reshape(-1, self._num_sh_basis)
            self._shape_basis = self.shape_basis.reshape((-1, 3, self._num_sh_basis))
            self._texture_basis = self._texture_sigmas * self._texture_basis.reshape(-1, self._num_sh_basis)
            self._texture_basis = self._texture_basis.reshape((-1, 3, self._num_sh_basis))

        self._tex_mean = self._bfm_model['texMU'].astype(np.float64).reshape((-1, 3))  # mean texture
        self._shape_mean = self._bfm_model['shapeMU'].astype(np.float64).reshape((-1, 3))/1e5 # mean face
        if path_bfm_prepare_model:
            if isinstance(path_bfm_prepare_model, str):
                path_bfm_prepare_model = Path(path_bfm_prepare_model)
            assert path_bfm_prepare_model.is_file(), f'The file {path_bfm_prepare_model} not found!'
            bfm_prepared = loadmat(str(path_bfm_prepare_model)) # load triangulation
            self._triangles = bfm_prepared['tri'].astype(np.int64)-1
            logging.info(f'Extract triangles from prepared BFM model: {path_bfm_prepare_model}!')
        else:
            #self._triangles = bfm_additional_info['tri'].astype(np.int64).reshape((-1, 3))-1
            self._triangles = self._bfm_model['tl'].astype(np.int64) - 1
            self._triangles = self._triangles[:, [1, 0, 2]]

        if hasattr(self, '_path_landmark_label'):
            keypoints_indeces = []
            with self._path_landmark_label.open() as reader:
                for idx_line, line in enumerate(reader):
                    if idx_line == 0:
                        continue
                    index_keypts = int(line.rstrip(' \n'))
                    keypoints_indeces.append(index_keypts)
            self._keypts_indeces = keypoints_indeces
        logging.info(f'The basel face model is load!')


    def get_face_colors(self, tex_coefs:np.ndarray, normalize=True):
        num_basis = tex_coefs.shape[0]
        basis = self._texture_basis.reshape(-1, self._texture_basis.shape[-1])
        basis = basis[:, :num_basis]
        face_colors = basis @ tex_coefs[:, None] + self.mean_texture.reshape(-1, 1)
        if normalize:
            face_colors = face_colors / 255.
        return face_colors.reshape(-1, 3)


    def get_face_shape(self, id_coeff:np.ndarray):
        num_basis = id_coeff.shape[0]
        basis = self._shape_basis.reshape(-1, self._shape_basis.shape[-1])
        basis = basis[:, :num_basis]
        face_shape = basis @ id_coeff[:, None] + self.mean_shape.reshape(-1, 1)
        return face_shape.reshape(-1, 3)


    def get_right_glob_idx_by_left(self, glob_l_idx:int):
        if not hasattr(self, 'left_idx_to_right_idx_sym') or not hasattr(self, 'right_idx_to_left_idx_sym'):
            assert False
        if glob_l_idx not in self.left_idx_to_right_idx_sym:
            return None
        return self.left_idx_to_right_idx_sym[glob_l_idx]


    def get_left_glob_idx_by_right(self, glob_r_idx: int):
        if not hasattr(self, 'left_idx_to_right_idx_sym') or not hasattr(self, 'right_idx_to_left_idx_sym'):
            assert False
        if glob_r_idx not in self.right_idx_to_left_idx_sym:
            return None
        return self.right_idx_to_left_idx_sym[glob_r_idx]


    def get_left_symmetry_index_pt_test(self, glob_left_idx:int):
        if not hasattr(self, 'left_idx_to_right_idx_sym') or not hasattr(self, 'right_idx_to_left_idx_sym'):
            assert False
        if glob_left_idx not in self.left_idx_to_right_idx_sym:
            if glob_left_idx in self.right_idx_to_left_idx_sym:
                print(f'Glob index {glob_left_idx} correspond to right_idx_to_left_idx_sym.')
                return self.right_idx_to_left_idx_sym[glob_left_idx], 'right'
            else:
                print(f'Glob index {glob_left_idx} does not match any point.')
                return None
        return self.left_idx_to_right_idx_sym[glob_left_idx], 'left'


    def get_right_symmetry_index_pt_test(self, glob_right_idx: int):
        if not hasattr(self, 'right_idx_to_left_idx_sym') or not hasattr(self, 'left_idx_to_right_idx_sym'):
            assert False
        if glob_right_idx not in self.right_idx_to_left_idx_sym:
            if glob_right_idx in self.left_idx_to_right_idx_sym:
                print(f'Glob index {glob_right_idx} correspond to left_idx_to_right_idx_sym.')
                return self.left_idx_to_right_idx_sym[glob_right_idx], 'left'
            else:
                print(f'Glob index {glob_right_idx} does not match any point.')
                return None

        return self.right_idx_to_left_idx_sym[glob_right_idx], 'right'
    
    
    def get_symmetry_points(self, left_idx:int):
        result = self.get_left_symmetry_index_pt_test(left_idx)
        if result is None:
            print(f'Incorrect global left index: {left_idx}.')
            return None
        right_idx, flag = result
        # need swap
        if flag == 'right':
            print('Need swap left <-> right.')
            temp = left_idx
            left_idx = right_idx
            right_idx = temp

        left_pt, right_pt = self._shape_mean[left_idx], self._shape_mean[right_idx]
        symm_left_right_pts = np.vstack([left_pt[None, :], right_pt[None, :]])
        return symm_left_right_pts


    #TODO: IN COMPLEX CASE NEED CORRECT
    def build_vert_symmetry_pt_correspondence(self, complex:bool=False) -> dict:
        mean_shape = self._shape_mean
        left_part_of_face_mask = (mean_shape[:, 0] < 0).astype(np.bool)
        right_part_of_face_mask = (mean_shape[:, 0] >= 0).astype(np.bool)
        left_part_of_face = mean_shape[left_part_of_face_mask]
        right_part_of_face = mean_shape[right_part_of_face_mask]
        kdtree_r_part_yz_proj = KDTree(data=right_part_of_face[:, 1:], copy_data=True)
        left_idx_to_glob_idx = np.arange(self._shape_mean.shape[0])[left_part_of_face_mask]
        right_idx_to_glob_idx = np.arange(self._shape_mean.shape[0])[right_part_of_face_mask]
        left_idx_to_right_idx_sym = {}
        right_idx_to_left_idx_sym = {}
        for idx_pt_left, pt_left in enumerate(left_part_of_face):
            pt_left_yz = pt_left[1:]
            dist, indeces = kdtree_r_part_yz_proj.query(pt_left_yz, k = np.arange(1, 6))
            # TODO: need check also np.abs(x)
            if complex:
                pts_right_candidates = right_part_of_face[indeces]
                diffs_dists_x0_plane = np.abs(np.abs(pts_right_candidates[:, 0]) - np.abs(pt_left[0]))
                min_right_idx = np.argmin(diffs_dists_x0_plane)
                right_closest_idx = indeces[min_right_idx]
            else:
                right_closest_idx = indeces[0]
            glob_left_idx = left_idx_to_glob_idx[idx_pt_left]
            glob_right_idx = right_idx_to_glob_idx[right_closest_idx]

            left_idx_to_right_idx_sym[glob_left_idx] = glob_right_idx
            right_idx_to_left_idx_sym[glob_right_idx] = glob_left_idx

        self.left_idx_to_right_idx_sym = left_idx_to_right_idx_sym
        self.right_idx_to_left_idx_sym = right_idx_to_left_idx_sym

        return {'left_to_right': left_idx_to_right_idx_sym, 'right_to_left': right_idx_to_left_idx_sym}


    def draw_face_mesh_with_pts(self, added_pts:np.ndarray):
        face_mean_mesh = MeshOperator.get_mesh_by(vertices=self._shape_mean, triangles=self._triangles)
        face_mean_mesh.compute_vertex_normals()
        list_circle_meshs = []
        for pt in added_pts:
            mesh_circle = create_sphere_mesh(pt, radius=0.01, color=[0, 1., 0.])
            list_circle_meshs.append(mesh_circle)
        list_circle_meshs += [face_mean_mesh]
        op3d.visualization.draw_geometries(list_circle_meshs, width=800, height=600)


    @property
    def keypoints_indeces(self):
        return self._keypts_indeces


    @property
    def get_mean_face_mesh_custom(self):
        return Mesh(vertices=self._shape_mean, triangles=self._triangles, textures=self._tex_mean)


    @property
    def mean_shape(self):
        return self._shape_mean


    @property
    def mean_texture(self):
        return self._tex_mean


    @property
    def triangles(self):
        return self._triangles


    @property
    def mean_face_pcd(self):
        return MeshOperator.get_point_cloud_by(self._shape_mean, self._tex_mean)


    @property
    def mean_face_mesh_open3d(self):
        return MeshOperator.get_mesh_by(self._shape_mean, self._triangles, self._tex_mean)


    def get_mean_face_mesh_open3d_with_kpts(self, color_kpts=(0., 1., 0.),
                                                  radius:float=0.005) -> List[op3d.geometry.TriangleMesh]:
        if not hasattr(self, "_keypts_indeces"):
            assert False, "Keypoints indeces not defined!"
        keypts_vertices = self._shape_mean[self._keypts_indeces]
        meshs_keypts = []
        for idx, key_pts in enumerate(keypts_vertices):
            sphere_pt = self._create_sphere_mesh(center=key_pts, radius=radius, color=color_kpts)
            meshs_keypts += [sphere_pt]

        mean_face_mesh = self.mean_face_mesh_open3d
        return [mean_face_mesh] + meshs_keypts


    def _create_sphere_mesh(self, center: np.ndarray, radius: float, color:Union[Tuple, List]):
        mesh_sphere = op3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_sphere.paint_uniform_color(color)
        return mesh_sphere.translate(center)


    def _save_mean_face_mesh_as_obj(self, filepath='/tmp/mean_face_mesh_basel.obj', save_texture_file:bool=True):
        mean_face_mesh = self.mean_face_mesh_open3d
        if isinstance(filepath, str):
            filepath = Path(filepath)
        output = op3d.io.write_triangle_mesh(str(filepath), mesh=mean_face_mesh, write_triangle_uvs=save_texture_file,
                                    print_progress=True)
        logging.info(f'Save mean face shape as {filepath}, Success: {output}.')
        return filepath


if __name__ == '__main__':
    bfm = BaselFaceModel2009(path_bfm_model)
    bfm.build_vert_symmetry_pt_correspondence(complex=True)