import logging

import dlib

from pathlib import Path

path_dir_dlib_shape_preds = Path('/home/users/mklochkov/projects/Deep3DFaceRecon_pytorch/checkpoints/lm_model')


class LandmarksPredictorDlib():


    def __init__(self, path_dlib_lands_preds, number_pts:int = 68):
        if isinstance(path_dlib_lands_preds, str):
            path_dlib_lands_preds = Path(path_dlib_lands_preds)
        assert path_dlib_lands_preds.is_dir(), f'The dir {path_dlib_lands_preds} not found!'
        self._path_dlib_dir_lands_models = path_dlib_lands_preds
        self._shape_predictor = self._init_dlib_landmark_predictor()


    def _init_dlib_landmark_predictor(self, number_pts:int=68):
        filepath = self._get_filename(number_pts)
        shape_predictor = dlib.shape_predictor(str(filepath))
        logging.info(f'Shape predictor with {number_pts} landmarks is loaded!')
        return shape_predictor


    def _get_filename(self, num_pts:int=68):
        assert num_pts in (68, 5), 'Incorrect num_pts parameter!'
        filename = f'shape_predictor_{num_pts}_face_landmarks.dat'
        filepath = self._path_dlib_dir_lands_models / filename
        assert filepath.is_file(), f'The file {filepath} not found!'
        return filepath