import pathlib, torch, os, numpy as np

from typing import Union
from PIL import Image
from pathlib import Path

from models import create_model
from util.load_mats import load_lm3d
from util.preprocess import align_img
from basel_face_model_2009 import BaselFaceModel2009
from frontalization.geometry_features import GeometryFeaturesGradient, GeometryFeatures

from options.test_options import TestOptions


class TestOptionsWrapper():

    #project_dir = Path.home() / 'projects/Deep3DFaceRecon_pytorch'
    project_dir = Path.home() / 'Deep3DFaceRecon_pytorch_copy'

    def __init__(self, img_folder: Union[pathlib.Path, str]='datasets/examples'):
        if isinstance(img_folder, str):
            img_folder = Path(img_folder)
        if not img_folder.is_dir():
            img_folder = self.project_dir / img_folder
        assert img_folder.is_dir(), f'The dir {img_folder} not found!'
        if not self.is_correct_organization_datafolder(img_folder):
            raise ValueError(f'Incorrect organization datafolder: {img_folder}.')
        config_test = TestOptions().parse()
        config_test.epoch = 20
        config_test.name = 'final_model'
        config_test.img_folder = img_folder
        config_test.gpu_ids = -1
        config_test.bfm_folder = self.project_dir/ 'BFM'
        config_test.checkpoints_dir = self.project_dir / 'checkpoints'
        self.config_test = config_test


    def is_correct_organization_datafolder(self, dir: pathlib.Path):
        if not (dir / 'detections').is_dir():
            return False
        return True

    @property
    def test_config(self):
        return self.config_test


class ShapeExtractor():


    def __init__(self, config: TestOptionsWrapper,
                 device_rank:int=0,
                 use_cuda:bool=False,
                 use_parallize:bool=False,
                 number_id_params:int=50,
                 bfm:BaselFaceModel2009=None,
                 path_geom_geats=None):
        self.config = config.config_test
        self.model = create_model(self.config)
        self.model.setup(config.test_config)
        if use_cuda:
            # torch.cuda.set_device(device)
            device = torch.device(device_rank)
            self.model.device = device
        if use_parallize:
            self.model.parallelize()
        self.model.eval()
        if number_id_params > 80:
            self.number_id_params = 80
        else:
            self.number_id_params = number_id_params

        # initialize data paths
        self.img_paths, self.lm_paths = self.get_all_img_lm_paths(str_representation=False)
        if bfm is not None and path_geom_geats is not None:
            self.bfm = bfm
            self.path_geom_feats = path_geom_geats


    @property
    def model_device(self):
        assert hasattr(self.model, 'device')
        return self.model.device


    def get_face_shape(self, coefs_id, coefs_exp:None):
        if coefs_exp is None:
            print('Extract face_shape without EXPRESSIONS!')
        self.facemodel_to_device()
        face_shape = self.model.facemodel.compute_face_shape(coefs_id=coefs_id, coefs_exp=coefs_exp)
        return face_shape


    def get_facemodel_triangles(self):
        return self.model.facemodel.face_buf


    def get_shape_identity_params(self, data:dict):
        # if hasattr(self.model, 'input_img'):
        #     assert False, f'set_input for model: {self.model} already called'
        self.model.set_input(data)
        if not hasattr(self.model, 'facemodel'):
            assert False, f'Incorrect model: {self.model}'
        output_coeff = self.model.net_recon(self.model.input_img)
        splitted_coefs = self.model.facemodel.split_coeff(output_coeff)
        id_coefs = splitted_coefs['id']
        tex_coefs = splitted_coefs['tex']
        exp_coefs = splitted_coefs['exp']
        id_coefs = id_coefs.squeeze()[:self.number_id_params]
        exp_coefs = exp_coefs.squeeze()
        tex_coefs = tex_coefs.squeeze()
        return id_coefs, exp_coefs, tex_coefs


    def read_data(self, im_path, lm_path, lm3d_std, to_tensor=True):
        im = Image.open(im_path).convert('RGB')
        W, H = im.size
        lm = np.loadtxt(lm_path).astype(np.float32)
        lm = lm.reshape([-1, 2])
        lm[:, -1] = H - 1 - lm[:, -1]
        _, im, lm, _ = align_img(im, lm, lm3d_std)
        if to_tensor:
            im = torch.tensor(np.array(im) / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            lm = torch.tensor(lm).unsqueeze(0)
        return im, lm


    def extract_shape_parameters(self, filename:str=None, numpy_format:bool=True):
        img_paths = self.img_paths
        lm_paths = self.lm_paths
        if filename:
            # uppercase not included
            img_path = None
            land_path = None
            for ext in (".jpg", ".png", '.jpeg'):
                img_path_current = self.image_dir / (filename + ext)
                if img_path_current.is_file():
                    img_path = img_path_current
                    print(f'Found img_path: {img_path_current}')
                    break
            if img_path:
                land_path = self.image_dir / 'detections'/ (img_path.stem + '.txt')
                if not land_path.is_file():
                    land_path = None

            if img_path and land_path:
                img_paths = [img_path]
                lm_paths = [land_path]

        id_coefs_list = []
        exp_coefs_list = []
        tex_coefs_list = []
        for path_img, path_lm in zip(img_paths, lm_paths):
            if not path_img.is_file():
                print(f'Incorrect image path {path_img} in dir: {self.image_dir}')
                continue
            if not getattr(self.config, 'bfm_model'):
                assert False, f'Incorrect config: {self.config}, not found bfm model!'
            lm3d_std = load_lm3d(self.config.bfm_folder)
            im_tensor, lm_tensor = self.read_data(str(path_img), str(path_lm), lm3d_std)
            data = {
                'imgs': im_tensor,
                'lms': lm_tensor
            }
            id_coefs, exp_coefs, tex_coefs = self.get_shape_identity_params(data)
            if path_img.stem.find('Interview') > -1:
                pass
            if numpy_format:
                id_coefs = id_coefs.detach().numpy()
                exp_coefs = exp_coefs.detach().numpy()
                tex_coefs = tex_coefs.detach().numpy()

            id_coefs_list.append(id_coefs)
            exp_coefs_list.append(exp_coefs)
            tex_coefs_list.append(tex_coefs)

        return id_coefs_list, exp_coefs_list, tex_coefs_list


    @property
    def image_dir(self):
        if 'img_folder' not in self.config:
            raise ValueError
        return getattr(self.config, 'img_folder')


    def get_all_img_lm_paths(self, str_representation:bool=True):
        image_paths = []
        for path in self.image_dir.rglob('**/*'):
            if path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                image_paths.append(path)

        landmark_paths = []
        for path in (self.image_dir / 'detections').rglob('**/*'):
            if path.suffix == '.txt':
                landmark_paths.append(path)

        landmark_names = set([path.stem for path in landmark_paths])
        data_paths = []
        for path_img in image_paths:
            name_image = path_img.stem
            if name_image in landmark_names:
                path_land = path_img.parent / 'detections' / (path_img.stem + '.txt')
                if not path_land.is_file():
                    print(f'Not found landmarks file: {path_land}!')
                    continue
                if str_representation:
                    path_land = str(path_land)
                    path_img = str(path_img)

                data_paths.append({'img_path': path_img, 'lm_path': path_land})
        paths_imgs = [d['img_path'] for d in data_paths]
        paths_lands = [d['lm_path'] for d in data_paths]

        return paths_imgs, paths_lands


    def normalize_id_coeffs(self, id_coefs):
        pass

    # def get_face_shape_from_facemodel(self, id_coefs:np.ndarray):
    #     return self.model.facemodel.compute_shape_numpy(id_coefs).detach().numpy()


    def facemodel_to_device(self):
        if not hasattr(self.model, 'facemodel'):
            assert False, f'Incorrect initialization of the model: {self.model}.'
        device_facemodel = self.model.facemodel.device
        self.model.facemodel.to_tensor(device_facemodel)


def run_shape_param_extractor_per_face(filename:str= '000037', num_params:int=80):
    config = TestOptionsWrapper()
    shape_extractor = ShapeExtractor(config=config, device_rank=1, number_id_params=num_params)
    id_coefs = shape_extractor.extract_shape_parameters(filename=filename)[0]
    return id_coefs
    #shape_extractor.facemodel_to_device()
    #return shape_extractor.get_face_shape(id_coefs=id_coefs), id_coefs


if __name__ == '__main__':
    face_shape = run_shape_param_extractor_per_face()
