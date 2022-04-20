import numpy as np, cv2 as cv

from PIL import Image
from pathlib import Path

from shape_extractor_model import ShapeExtractor, TestOptionsWrapper
from landmarks_predictor import ImagesGenerator, LandmarksPredictorDlib

path_dir_imgs_WFLW = Path('../datasets/WFLW/WFLW_images')
path_dir_ann_WFLW = Path('../datasets/WFLW/WFLW_annotations')
path_dlib_lp_dir = Path('../checkpoints/lm_model/')


class DatasetWFLW():


    slice_landmarks = slice(0, 196)
    slice_rect = slice(196, 200)
    slice_attrs = slice(200, 206)
    attrs_map = {0: 'pose', 1: 'expression', 2: 'illumination', 3: 'make-up', 4: 'occlusion', 5: 'blur'}


    def __init__(self, path_dir_imgs, path_dir_ann, path_lands_preds, pose_type=0):
        self.dir_imgs = Path(path_dir_imgs)
        self.dir_anns = Path(path_dir_ann)
        self.dir_shape = Path(path_lands_preds)
        assert self.dir_anns.is_dir() and self.dir_imgs.is_dir() and self.dir_shape.is_dir()
        self.acceptable_attrs = {}
        self.acceptable_attrs.update({'expression': 0})
        self.acceptable_attrs.update({'pose': pose_type})
        self.acceptable_attrs.update({'illumination': 0})
        self.acceptable_attrs.update({'make-up': 0})
        self.acceptable_attrs.update({'blur': 0})
        self.acceptable_attrs.update({'occlusion': 0})
        self.kps_predictor = LandmarksPredictorDlib(self.dir_shape, number_pts=5)


    def transform_kps_WFLW_to_68(self, points:list):
        info_68 = []
        for j in range(17):
            x = points[j * 2 * 2 + 0]
            y = points[j * 2 * 2 + 1]
            info_68.append(x)
            info_68.append(y)
        for j in range(33, 38):
            x = points[j * 2 + 0]
            y = points[j * 2 + 1]
            info_68.append(x)
            info_68.append(y)
        for j in range(42, 47):
            x = points[j * 2 + 0]
            y = points[j * 2 + 1]
            info_68.append(x)
            info_68.append(y)
        for j in range(51, 61):
            x = points[j * 2 + 0]
            y = points[j * 2 + 1]
            info_68.append(x)
            info_68.append(y)
        point_38_x = (float(points[60 * 2 + 0]) + float(points[62 * 2 + 0])) / 2.0
        point_38_y = (float(points[60 * 2 + 1]) + float(points[62 * 2 + 1])) / 2.0
        point_39_x = (float(points[62 * 2 + 0]) + float(points[64 * 2 + 0])) / 2.0
        point_39_y = (float(points[62 * 2 + 1]) + float(points[64 * 2 + 1])) / 2.0
        point_41_x = (float(points[64 * 2 + 0]) + float(points[66 * 2 + 0])) / 2.0
        point_41_y = (float(points[64 * 2 + 1]) + float(points[66 * 2 + 1])) / 2.0
        point_42_x = (float(points[60 * 2 + 0]) + float(points[66 * 2 + 0])) / 2.0
        point_42_y = (float(points[60 * 2 + 1]) + float(points[66 * 2 + 1])) / 2.0
        point_44_x = (float(points[68 * 2 + 0]) + float(points[70 * 2 + 0])) / 2.0
        point_44_y = (float(points[68 * 2 + 1]) + float(points[70 * 2 + 1])) / 2.0
        point_45_x = (float(points[70 * 2 + 0]) + float(points[72 * 2 + 0])) / 2.0
        point_45_y = (float(points[70 * 2 + 1]) + float(points[72 * 2 + 1])) / 2.0
        point_47_x = (float(points[72 * 2 + 0]) + float(points[74 * 2 + 0])) / 2.0
        point_47_y = (float(points[72 * 2 + 1]) + float(points[74 * 2 + 1])) / 2.0
        point_48_x = (float(points[68 * 2 + 0]) + float(points[74 * 2 + 0])) / 2.0
        point_48_y = (float(points[68 * 2 + 1]) + float(points[74 * 2 + 1])) / 2.0
        info_68.append(str(point_38_x))
        info_68.append(str(point_38_y))
        info_68.append(str(point_39_x))
        info_68.append(str(point_39_y))
        info_68.append(points[64 * 2 + 0])
        info_68.append(points[64 * 2 + 1])
        info_68.append(str(point_41_x))
        info_68.append(str(point_41_y))
        info_68.append(str(point_42_x))
        info_68.append(str(point_42_y))
        info_68.append(points[68 * 2 + 0])
        info_68.append(points[68 * 2 + 1])
        info_68.append(str(point_44_x))
        info_68.append(str(point_44_y))
        info_68.append(str(point_45_x))
        info_68.append(str(point_45_y))
        info_68.append(points[72 * 2 + 0])
        info_68.append(points[72 * 2 + 1])
        info_68.append(str(point_47_x))
        info_68.append(str(point_47_y))
        info_68.append(str(point_48_x))
        info_68.append(str(point_48_y))
        for j in range(76, 96):
            x = points[j * 2 + 0]
            y = points[j * 2 + 1]
            info_68.append(x)
            info_68.append(y)

        return info_68


    def prepare_for_model(self, show=False, save_68_kps=True, number_kps=68, resize=True, cropped_image=True,
                          folder='Voter', number='56'):
        dir_imgs = Path('../datasets/examples')
        dir_lands_5 = Path('../datasets/examples/detections')
        dir_lands_68 = Path('../datasets/examples/detections_68')
        for data_face in self.data_generator(folder=folder, number=number):
            img, kps_5, kps_68, rect = data_face['img'], data_face['kps_5'], data_face['kps_68'], data_face['rect']
            filename = data_face['name']
            path_img = dir_imgs / (filename + '.jpg')
            path_5_kps_txt = dir_lands_5 / (filename + '.txt')
            path_68_kps_txt = dir_lands_68 / (filename + '.txt')
            # check this 5 landmarks
            if show:
                if number_kps == 5:
                    result_kps = [{'kps': kps_5, 'bbox': rect}]
                elif number_kps == 68:
                    result_kps = [{'kps': kps_68, 'bbox': rect}]
                else:
                    assert False,  f'Incorrect number_kps {number_kps}.'
                img_kps = ImagesGenerator.add_keypts_and_face_dets_on_img(img, result_kps)
                if resize:
                    img_kps = self.resize_image(img_kps)
                cv.imshow(f'kps_{filename}', img_kps[..., [2, 1, 0]].astype(np.uint8))
                if cv.waitKey(0) == ord('q'):
                    continue
            # need write in file cropped image
            if cropped_image:
                x_l, y_l, x_r, y_r = rect
                dx = np.abs(x_r-x_l)
                dy = np.abs(y_r-y_l)
                add_dx = dx * 1.5
                add_dy = dy * 1.5
                pad_x = add_dx // 2
                pad_y = add_dy // 2
                x_l_pad = max(x_l-pad_x, 0)
                x_r_pad = x_r+pad_x
                y_l_pad = max(y_l-pad_y, 0)
                y_r_pad = y_r+pad_y
                x_l_pad, x_r_pad, y_l_pad, y_r_pad = int(x_l_pad), int(x_r_pad), int(y_l_pad), int(y_r_pad)
                img = img[y_l_pad:y_r_pad, x_l_pad:x_r_pad]
                # need change keypoints
                kps_5[:, 0] -= x_l_pad
                kps_5[:, 1] -= y_l_pad
                kps_68[:, 0] -= x_l_pad
                kps_68[:, 1] -= y_l_pad
                result_kps = [{'kps': kps_5, 'bbox': None}]
                img_kps = ImagesGenerator.add_keypts_and_face_dets_on_img(img, result_kps, add_rectangle=False)
                cv.imshow(f'5_{filename}', img_kps[..., [2, 1, 0]].astype(np.uint8))
                if cv.waitKey(0) == ord('q'):
                    continue
            cv.imwrite(str(path_img), img[..., [2, 1, 0]])
            with path_5_kps_txt.open('w') as writer:
                for idx, pt in enumerate(kps_5):
                    x, y = pt
                    writer.write(f'{x} {y}\n')
            print(f'Save {path_5_kps_txt}.')
            if save_68_kps:
                with path_68_kps_txt.open('w') as writer:
                    for idx, pt in enumerate(kps_68):
                        x, y = pt
                        writer.write(f'{x} {y}\n')
                print(f'Save {path_68_kps_txt}.')


    def visualize_sequentially(self, resize=True):
        for data_face in self.data_generator():
            img, kps_5, kps_68, rect = data_face['img'], data_face['kps_5'], data_face['kps_68'], data_face['rect']
            filename = data_face['name']
            result_kps_68 = [{'kps': kps_68, 'bbox': rect}]
            img_kps_68 = ImagesGenerator.add_keypts_and_face_dets_on_img(img, result_kps_68)
            if resize:
                img_kps_68 = self.resize_image(img_kps_68)
            cv.imshow(f'68_{filename}', img_kps_68[..., [2, 1, 0]].astype(np.uint8))
            if cv.waitKey(0) == ord('q'):
                continue
            result_kps_5 = [{'kps': kps_5, 'bbox': rect}]
            img_kps_5 = ImagesGenerator.add_keypts_and_face_dets_on_img(img, result_kps_5)
            if resize:
                img_kps_5 = self.resize_image(img_kps_5)
            cv.imshow(f'5_{filename}', img_kps_5[..., [2, 1, 0]].astype(np.uint8))
            if cv.waitKey(0) == ord('q'):
                continue


    def resize_image(self, image):
        if max(image.shape[:2]) > 800:
            return cv.resize(image, (0, 0), fx=0.5, fy=0.5)
        return image


    def data_generator(self, folder='Interview', number='425', transform_kps_to_68_format=True, threshold_size=0.4):
        path_ann = self.dir_anns / 'list_98pt_rect_attr_train_test'/'list_98pt_rect_attr_train.txt'
        with path_ann.open() as reader:
            for idx, line in enumerate(reader):
                line_splitted = line.rstrip('\n').split(' ')
                try:
                    landmarks = line_splitted[self.slice_landmarks]
                    rect = line_splitted[self.slice_rect]
                    attrs = line_splitted[self.slice_attrs]
                    path_img = self.dir_imgs / line_splitted[-1]
                    if number and folder:
                        if not (str(path_img).find(folder) > -1 and str(path_img).find(number) > -1):
                            continue
                    if not path_img.is_file():
                        assert False, f'The file {path_img} not found!'
                except:
                    print('Error!')
                    continue
                attrs_named = {}
                for id_attr, attr_value in enumerate(attrs):
                    type_attr = self.attrs_map[id_attr]
                    attrs_named.update({type_attr: int(attr_value)})
                # we choose no-make-up, no occlusion, no-blur, normal illumination, normal expression, large pose
                is_incorrect_face = False
                for attr_name, value in attrs_named.items():
                    accept_value = self.acceptable_attrs[attr_name]
                    if accept_value != value:
                        is_incorrect_face = True
                if not is_incorrect_face or True:
                    if transform_kps_to_68_format:
                        landmarks = self.transform_kps_WFLW_to_68(landmarks)
                    landmarks = np.asarray([float(coor) for coor in landmarks]).reshape((-1, 2))
                    # x_left, y_left, x_right, y_right
                    rect = np.asarray([int(coor) for coor in rect])
                    x_l, y_l, x_r, y_r = rect
                    img = self.get_numpy_img(path_img)
                    if img is None:
                        continue
                    # some big faces on images
                    H, W = img.shape[:2]
                    dx, dy = np.abs(x_r-x_l), np.abs(y_r-y_l)
                    size_img = max(H, W)
                    size_rect = max(dx, dy)
                    # if (size_rect / size_img) < threshold_size:
                    #     continue
                    # for facerecon model
                    bbox = tuple([coor for coor in rect])
                    dets_insight = self.kps_predictor.detect_faces_insight(img)
                    if len(dets_insight) > 1:
                        continue
                    det = dets_insight[0]
                    keypts_5 = det['kps']
                    yield {'img': img,
                           'kps_5': keypts_5,
                           'kps_68': landmarks,
                           'rect': rect,
                           'pose': self.acceptable_attrs['pose'],
                           'name': path_img.stem}


    def get_numpy_img(self, path):
        try:
            pil_image = Image.open(str(path)).convert('RGB')
            image_np = np.array(pil_image, np.uint8)
        except:
            return None
        return image_np


if __name__ == '__main__':
    dataset_WFLW = DatasetWFLW(path_dir_imgs_WFLW, path_dir_ann_WFLW, path_dlib_lp_dir, pose_type=0)
    dataset_WFLW.prepare_for_model(show=True, folder='Dresses', number='219')