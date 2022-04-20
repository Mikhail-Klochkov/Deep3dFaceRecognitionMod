import open3d as op3d, cv2 as cv, logging, numpy as np, pyrender, copy, pathlib

from pathlib import Path
from typing import Union, List, Tuple
from sklearn.metrics import pairwise_distances

from basel_face_model_2009 import BaselFaceModel2009
from landmarks_predictor import LandmarksPredictorDlib, ImagesGenerator
from utils_mesh import MeshOperator
#from renderer import Render


path_dir_dlib_shape_preds = Path('../checkpoints/lm_model')
path_bfm_model = Path('../BFM/01_MorphableModel.mat')
path_additional_bfm = Path('../BFM/facemodel_info.mat')
path_landmarks_bfm_indeces = Path('../BFM/landmarks68_BFM.anl')
path_bfm_prepared = Path('../BFM/BFM_model_front.mat')
#path_test_image = Path('../datasets/examples/000037.jpg')
path_test_image = Path('../datasets/WFLW/WFLW_images/13--Interview/13_Interview_Interview_2_People_Visible_13_425.jpg')


class CameraCalibrator():


    _distortion_shape_vector = (1, 5)

    # need pass some face_shape (from another for example system)
    def __init__(self, bfm: BaselFaceModel2009, lands_predictor: LandmarksPredictorDlib, face_shape:np.ndarray=None):
        self._bfm = bfm
        self._landmark_predictor = lands_predictor
        if face_shape is not None:
            shape_bfm_mean_face = self._bfm.mean_shape.shape
            assert face_shape.shape == shape_bfm_mean_face, f'Incorrect shape of face_shape: {face_shape.shape}.'
            self.face_shape = face_shape


    def get_landmarks_by_filepath(self, filepath: Union[str, pathlib.Path]):
        image = ImagesGenerator.try_get_numpy_img(filepath, False)
        return self.get_landmarks_by_image(image)


    def get_3d_keypoints_face_model(self):
        keypts_indeces = self._bfm.keypoints_indeces
        if hasattr(self, 'face_shape'):
            return self.face_shape[keypts_indeces]
        return self._bfm.mean_shape[keypts_indeces]


    def get_landmarks_by_image(self, image:np.ndarray):
        return self._landmark_predictor.detect_landmarks(image)


    def found_camera_matrix(self, filepath: Union[str, pathlib.Path],
                                  _2d_keypoints=None,
                                  type_calibration='calib',
                                  transform_to_matrix:bool=True,
                                  visualize:bool=False):
        image_np = ImagesGenerator.try_get_numpy_img(filepath, rgb_to_bgr=False)
        if _2d_keypoints is not None:
            _2d_keypoints_face_img = _2d_keypoints
            result_kps_det = [{'kps': _2d_keypoints_face_img, 'bbox': None}]
        else:
            result_kps_det = self.get_landmarks_by_image(image_np)
            if len(result_kps_det) > 1:
                logging.info(f'Image: {filepath} contains more than one person! By default take one!')
            _2d_keypoints_face_img = result_kps_det[0]['kps'].astype(np.float32)

        _3d_keypoints_face_model = self.get_3d_keypoints_face_model().astype(np.float32)
        _2d_keypoints_face_img_l = [_2d_keypoints_face_img]
        _3d_keypoints_mean_face_l = [_3d_keypoints_face_model]

        image_shape = image_np.shape[:2]
        if type_calibration == 'calib':
            # but not needed for calibrate_matrix_opencv()
            camera_intr = self.default_camera_matrix(img_shape=image_shape, focal_lenght=image_shape[1])
            ret, mtx, dist, rotation_vector, translation_vector = self.calibrate_matrix_opencv(
                                                                        _3d_keypoints_mean_face_l,
                                                                        _2d_keypoints_face_img_l,
                                                                        shape=image_shape)
            _2d_projected_pts, _ = self._project_points(_3dpoints=_3d_keypoints_face_model,
                                                        rvecs=rotation_vector[0],
                                                        tvecs=translation_vector[0],
                                                        camera_matrix=mtx,
                                                        _2dpoints=_2d_keypoints_face_img)

        elif type_calibration == 'PnP':
            camera_intr = self.default_camera_matrix(img_shape=image_shape, focal_lenght=image_shape[1])
            _, rotation_vector, translation_vector = cv.solvePnP(_3d_keypoints_face_model,
                                                                 _2d_keypoints_face_img,
                                                                 camera_intr,
                                                                 distCoeffs=np.zeros((4, 1)))
            _2d_projected_pts, _ = cv.projectPoints(_3d_keypoints_face_model,
                                                     rotation_vector,
                                                     translation_vector,
                                                     camera_intr,
                                                     distCoeffs=np.zeros((4, 1)))

            # test projection own
            _2d_projected_pts_my = self._project_3d_pts_to_img_plane(_3d_keypoints_face_model,
                                                                     rotation_vector,
                                                                     translation_vector,
                                                                     camera_intr,)


        else:
            camera_intr = None
            assert False, 'Incorrect type of algorithm!'

        _2d_projected_pts = np.int32(_2d_projected_pts.reshape(-1, 2))

        if visualize:
            # need visualize projected points with keypts on the image plane
            image_np_detkps = ImagesGenerator.add_keypts_and_face_dets_on_img(image_np,
                                                                              result=result_kps_det,
                                                                              add_rectangle=False)
            cv.imshow('projected_before', image_np_detkps[..., [2, 1, 0]].astype(np.uint8))
            cv.waitKey(1000)
            result_proj_det = {'kps': _2d_projected_pts, 'bbox': None}
            image_np_detkps_projdet = ImagesGenerator.add_keypts_and_face_dets_on_img(image_np_detkps,
                                                                                      result=[result_proj_det],
                                                                                      add_rectangle=False,
                                                                                      color_kps=(0, 255, 0))
            cv.imshow('projected_after', image_np_detkps_projdet[..., [2, 1, 0]].astype(np.uint8))
            cv.waitKey(1000)

        if transform_to_matrix:
            rotation_matrix, _ = cv.Rodrigues(rotation_vector.reshape((3, 1)))
            return rotation_matrix, translation_vector, camera_intr, _2d_projected_pts, image_np

        return rotation_vector, translation_vector, camera_intr, _2d_projected_pts, image_np



    def calibrate_matrix_opencv(self, _3dpoints, _2dpoints, shape:tuple):
        camera_matrix = self.default_camera_matrix(shape, focal_lenght=shape[1])
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(_3dpoints, _2dpoints, shape, camera_matrix, None,
                                                          flags=cv.CALIB_USE_INTRINSIC_GUESS)
        return ret, mtx, dist, rvecs, tvecs


    # TODO: Correct the same as OpenCV
    def _project_3d_pts_to_img_plane(self, _3dpts:np.ndarray,
                                           rotvec:np.ndarray,
                                           transvec:np.ndarray,
                                           intrisic_camera_matrix:np.ndarray):
        assert transvec.squeeze().shape == (3, ), f'Incorrect transvec shape: {transvec.shape}'
        if rotvec.shape == (3, 1) or rotvec.shape == (1, 3):
            rotvec = rotvec.reshape((3, 1))
            rotmatrix, _ = cv.Rodrigues(rotvec)
        elif rotvec.shape == (3, 3):
            rotmatrix = rotvec
        else:
            assert False, f'Incorrect shape of rotvec "{rotvec.shape}"'
        # create Extrinsic matrix
        extrinsic_matrix = self.get_extrinsic_matrix(rotmatrix, transvec)
        if intrisic_camera_matrix.shape == (3, 3):
            intrisic_camera_matrix_extend = np.zeros((3, 4), dtype=np.float32)
            intrisic_camera_matrix_extend[:3, :3] = intrisic_camera_matrix
        else:
            intrisic_camera_matrix_extend = copy.deepcopy(intrisic_camera_matrix)
        global_camera_matrix = intrisic_camera_matrix @ extrinsic_matrix
        _3dpts_homo = self._get_homogeneous_vectors(_3dpts)
        homogenious_coors = _3dpts_homo @ global_camera_matrix.T
        # devided by w coordinate
        return homogenious_coors[:, :2] / homogenious_coors[:, 2:]


    def _get_homogeneous_vectors(self, vectors:np.ndarray):
        assert len(vectors.shape) == 2, 'Incorrect vector/s format'
        assert vectors.shape[1] == 3, 'Need vector/s 3d space!'
        N, _3 = vectors.shape
        vectors_homo = np.ones((N, _3 + 1), dtype=np.float32)
        vectors_homo[:, :3] = vectors
        return vectors_homo


    @staticmethod
    def get_extrinsic_matrix(rotmatrix:np.ndarray, transvec:np.ndarray):
        assert rotmatrix.shape == (3, 3) and transvec.shape == (3, 1), 'Incorrect shapes of parameters'
        extrinsic_matrix = np.zeros((3, 4), dtype=np.float32)
        extrinsic_matrix[:3, :3] = rotmatrix
        extrinsic_matrix[:3, 3] = transvec.squeeze()
        #extrinsic_matrix[3, 3] = 1.
        return extrinsic_matrix


    def _project_points(self, _3dpoints:np.ndarray, rvecs:np.ndarray, tvecs:np.ndarray, camera_matrix:np.ndarray,
                        dist:np.ndarray=None, _2dpoints:np.ndarray=None):
        if dist is None:
            dist = np.zeros(self._distortion_shape_vector, dtype=np.float32)
        _3dpoints, rvecs, tvecs, camera_matrix, dist = self._transform_types_for_opencv_input(
                                                            _3dpoints, rvecs, tvecs, camera_matrix, dist)
        projected_3d_pts, _ = cv.projectPoints(_3dpoints, rvecs, tvecs, camera_matrix, dist)
        projected_3d_pts = projected_3d_pts.squeeze()
        if _2dpoints is not None:
            # mean rmse squared error
            rmse_error = cv.norm(_2dpoints, projected_3d_pts, cv.NORM_L2)/_3dpoints.shape[0]
            print(f'Projected points with RMSE: {rmse_error}.')
            return projected_3d_pts, rmse_error
        else:
            return projected_3d_pts


    def _transform_types_for_opencv_input(self, *args):
        args_transformed = []
        for arg in args:
            if not isinstance(arg, np.ndarray):
                assert False, f'Incorrect type "{arg}" of object.'
            if arg.dtype != np.float32:
                arg = arg.astype(np.float32)
            args_transformed.append(arg)

        return args_transformed


    @staticmethod
    def default_camera_matrix(img_shape: Union[List, Tuple], focal_lenght:float=1.):
        camera_matrix = np.zeros((3, 3), dtype=np.float32)
        camera_matrix[0, 0] = focal_lenght
        camera_matrix[0, 2] = img_shape[1] // 2
        camera_matrix[1, 1] = focal_lenght
        camera_matrix[1, 2] = img_shape[0] // 2
        camera_matrix[2, 2] = 1.

        return camera_matrix



class TestfrontalizationModule():


    @staticmethod
    def test_example_basel_face_model():
        bfm = BaselFaceModel2009(path_bfm_model,
                                 path_landmark_indeces=path_landmarks_bfm_indeces,
                                 path_bfm_triangulation=path_additional_bfm)
        meshs = bfm.get_mean_face_mesh_open3d_with_kpts()
        op3d.visualization.draw_geometries(meshs)


    @staticmethod
    def test_render_mean_face():
        bfm = BaselFaceModel2009(path_bfm_model,
                                 path_landmark_indeces=path_landmarks_bfm_indeces,
                                 create_mean_face_mesh=False)
        render = Render()
        mesh_torch = render.get_torch_mesh_object_from_mesh(mesh=bfm.get_mean_face_mesh_custom, white_texture=False)
        render_instance = render.define_render_instance()
        output = render_instance(mesh_torch)
        cv.imshow('win1', (255 * output[0, ..., :3].cpu().numpy()[..., [2,1,0]]).astype(np.uint8))
        cv.waitKey()
        logging.info(f"we get: {mesh_torch}, render: {render_instance}, output render: {output}")
        print(f"we get: {mesh_torch}, render: {render_instance}, output render: {output}")


    @staticmethod
    def test_correspondences_keypoints(number_closest_pts:int=5):
        lands_predictor = LandmarksPredictorDlib(path_dlib_lands_preds=path_dir_dlib_shape_preds)
        path_test_img = Path('../datasets/examples/000002.jpg')
        test_img = ImagesGenerator.try_get_numpy_img(path_test_img)
        bfm = BaselFaceModel2009(path_bfm_model, path_landmark_indeces=path_landmarks_bfm_indeces,
                                 create_mean_face_mesh=False)
        verteces = bfm.mean_shape
        verteces_queries = verteces[bfm.keypoints_indeces]
        vertex_colors_copy = copy.deepcopy(bfm._tex_mean)
        # dists = pairwise_distances(verteces_queries, verteces)
        # closest_indeces = np.argsort(dists, axis=1)[:, :number_closest_pts]
        result = lands_predictor.detect_landmarks(test_img)
        # test_image_labeled = ImagesGenerator.add_keypts_and_face_dets_on_img(image=test_img, result=result)
        test_image_copy = copy.deepcopy(test_img)
        for idx, pt in enumerate(result[0]['kps']):
            x, y = pt
            x, y = int(x), int(y)
            test_image_copy = cv.circle(test_image_copy, (x, y), radius=1, color=(0, 0, 255), thickness=5)
            vertex_query = verteces_queries[idx]
            dists = pairwise_distances(vertex_query[np.newaxis, :], verteces)
            closest_indeces = np.argsort(dists, axis=1)[:, :number_closest_pts]
            vertex_colors_copy[closest_indeces, :] = (0., 1., 0.)
            mesh_add_pt = MeshOperator.get_mesh_by(verteces, bfm.triangles, vertex_colors_copy)
            op3d.visualization.draw_geometries([mesh_add_pt])
            cv.imshow('iter: {idx}', test_image_copy.astype(np.uint8))
            cv.waitKey(1000)


    # with some specific face_shape
    @staticmethod
    def test_camera_calibration_face_shape(image_path,
                                           face_shape:np.ndarray,
                                           trans_to_matrix:bool=True,
                                           visualize:bool=True):
        bfm = BaselFaceModel2009(path_bfm_model,
                                 path_landmark_indeces=path_landmarks_bfm_indeces)
        lands_predictor = LandmarksPredictorDlib(path_dlib_lands_preds=path_dir_dlib_shape_preds)
        camera_calibr = CameraCalibrator(bfm=bfm, lands_predictor=lands_predictor, face_shape=face_shape)
        result = camera_calibr.found_camera_matrix(image_path,
                                                   type_calibration='PnP',
                                                   visualize=visualize,
                                                   transform_to_matrix=trans_to_matrix)
        rot_matrix, trans_vect = result[:2]
        return rot_matrix, trans_vect


    @staticmethod
    def test_camera_calibration_mean_face(image_path, trans_to_matrix:bool=True, visualize:bool=True):
        bfm = BaselFaceModel2009(path_bfm_model,
                                 path_landmark_indeces=path_landmarks_bfm_indeces)
        lands_predictor = LandmarksPredictorDlib(path_dlib_lands_preds=path_dir_dlib_shape_preds)
        camera_calibr = CameraCalibrator(bfm=bfm, lands_predictor=lands_predictor)
        result = camera_calibr.found_camera_matrix(image_path,
                                                   type_calibration='PnP',
                                                   visualize=visualize,
                                                   transform_to_matrix=trans_to_matrix)
        rotation_matrix, translation_vector = result[:2]
        return rotation_matrix, translation_vector


if __name__ == '__main__':
    #TestfrontalizationModule.test_render_mean_face()
    TestfrontalizationModule.test_camera_calibration_mean_face(path_test_image)