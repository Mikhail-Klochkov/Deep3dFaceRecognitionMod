import copy, cv2 as cv, numpy as np

from pathlib import Path
from scipy.io import loadmat
from PIL import Image

from landmarks_predictor import ImagesGenerator

path_data_300_IBUG = Path('/data/klochkov/300W_LP/IBUG')
path_data_300_AFW = Path('/data/klochkov/300W_LP/AFW')
path_data_300_HELEN = Path('/data/klochkov/300W_LP/HELEN')


# Very Bad Annotation Database 300W for landmarks
class Dataset300WL():


    def __init__(self, dir):
        self.dir = Path(dir)
        assert self.dir.is_dir(), f'The dir {self.dir} not found!'
        self.paths = self.init_paths_pairs()


    def init_paths_pairs(self):
        self.paths = {}
        for path in self.dir.rglob('**/*'):
            name = path.stem
            if path.suffix == '.mat':
                type = 'lm'
            elif path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                type = 'im'
            else:
                continue
            self.update_paths(name, path, type)
        return self.paths


    def update_paths(self, name, path, type='im'):
        if name in self.paths:
            if type == 'im':
                assert 'im' not in self.paths[name], 'Incorrect behavior'
                self.paths[name].update({'im':  path})
            elif type == 'lm':
                assert 'lm' not in self.paths[name], 'Incorrect behavior'
                self.paths[name].update({'lm':  path})
            else:
                assert False
        else:
            if type == 'im':
                record = {'im': path}
            elif type == 'lm':
                record = {'lm': path}
            else:
                assert False
            self.paths[name] = record


    def iterate_visualizer(self, waitkey=2000):
        for filename, _ in self.paths.items():
            self.visualize(filename, waitkey)


    def visualize(self, filename, waitkey=2000):
        img, lands = self.get_data(filename)
        _2dkeypoints = lands['pt2d'].T
        if _2dkeypoints.shape == (68, 2):
            result_keypts = [{'kps': _2dkeypoints, 'bbox': None}]
            img_copy = copy.deepcopy(img)
            img_keypts = ImagesGenerator.add_keypts_and_face_dets_on_img(img_copy, result_keypts, add_rectangle=False)
            cv.imshow('projected_before', img_keypts[..., [2, 1, 0]].astype(np.uint8))
            cv.waitKey(waitkey)
        else:
            print('Several faces!')
            cv.imshow('projected_before', img[..., [2, 1, 0]].astype(np.uint8))
            cv.waitKey(waitkey)


    def get_data(self, filename):
        assert filename in self.paths, f'Not found data with name: "{filename}"'
        pair = self.paths[filename]
        if 'im' not in pair:
            assert False, f'Incorrect record for filename: {filename}'
        if 'lm' not in pair:
            assert False, f'Incorrect record for filename: {filename}'

        path_im, path_lm = pair['im'], pair['lm']

        lands = loadmat(path_lm)
        img_pil = Image.open(str(path_im))
        img = np.array(img_pil.convert('RGB'), dtype=np.uint8)
        return img, lands



if __name__ == '__main__':
    dataset300WL = Dataset300WL(dir=path_data_300_HELEN)
    print(dataset300WL.iterate_visualizer(waitkey=2000))