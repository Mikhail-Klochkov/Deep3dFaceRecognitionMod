import logging
import zipfile, joblib

from pathlib import Path
from PIL import Image
from io import BytesIO
import insightface
from insightface.app import FaceAnalysis
import PIL
from joblib import Parallel, delayed


class TrainDatasetExtractor():


    def __init__(self, path_dataset):
        if not isinstance(path_dataset, str):
            path_dataset = str(path_dataset)
        point_delemeter = path_dataset.rfind('.')
        if point_delemeter != -1:
            filename, extension = path_dataset[:point_delemeter], path_dataset[point_delemeter:]
            if extension not in ('.zip', ):
                assert False, f'Incorrect dataset file format: {path_dataset}!'
            path_dataset = Path(path_dataset)
            if not path_dataset.is_file():
                raise FileNotFoundError(f'The dataset file: {path_dataset}, not found!')
            self.is_zipped_file = True
        # can be folder
        else:
            path_dataset = Path(path_dataset)
            if not path_dataset.is_dir():
                raise NotADirectoryError(f'The dataset directory: {path_dataset}, not found!')
            self.is_zipped_file = False

        self.path_dataset = path_dataset


    def extract_images(self, parallel=False):
        extracted_data = []
        if self.is_zipped_file:
            if self.path_dataset.suffix != '.zip':
                raise ValueError(f'The file with dataset {self.path_dataset} not zip file!')
            with zipfile.ZipFile(str(self.path_dataset), 'r') as reader:
                if parallel:
                    pass
                else:
                    for filename in reader.namelist():
                        extention = self.extract_file_extention(filename)
                        if extention is None:
                            continue
                        if extention not in ('.jpg', '.png', '.JPG', '.JPEG', '.jpeg', '.PNG'):
                            continue
                        try:
                            data_binary_image = reader.read(filename)
                            data_image_encoded = BytesIO(data_binary_image)
                            image_pil = Image.open(data_image_encoded)
                        except Exception as e:
                            logging.info(f'Error {e} reading file by name: {filename}.')


    @staticmethod
    def extract_file_extention(path):
        if not isinstance(path, str):
            path = str(path)
        idx_ext_delemeter = path.rfind('.')
        if idx_ext_delemeter != -1:
            return path[idx_ext_delemeter:]
        return None


class ImageExtractor():


    def __init__(self, extract_landmarks:bool=True, transformer=None):
        self.extract_landmarks = extract_landmarks
        # some function (image preprocessing image -> transformed_image)
        self.transformer = transformer


    def _init_face_detector(self, model='insightface'):
        if model == 'insightface':
            FaceAnalysis()
        elif model == 'dlib':
            raise NotImplemented
        else:
            raise ValueError('model argument should be ("insightface", "dlib")')


    def _init_dlib_shape_predictor_landmarks(self, number_landmarks:int=5):
        pass


    def extract_parameters(self, image_pil):
        pass



if __name__ == '__main__':
    path_ffhq_dataset = Path.cwd() / 'datasets/dataset_ffhq/images_ffhq.zip'
    extractor = TrainDatasetExtractor(str(path_ffhq_dataset))
    extractor.extract_images()
