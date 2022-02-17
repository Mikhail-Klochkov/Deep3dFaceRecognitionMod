import requests, logging, numpy as np, sys, argparse, validators, json
from pathlib import Path


class FFHQDatasetURls():


    def __init__(self, data_dir_name='datasets/dataset_ffhq'):
        absolute_dir_path = self.get_current_directoty_file() / data_dir_name
        if isinstance(absolute_dir_path, str):
            absolute_dir_path = Path(absolute_dir_path)
        if not absolute_dir_path.is_dir():
            raise NotADirectoryError(f'The directory: {absolute_dir_path} not found!')
            #try:
            #    absolute_dir_path.mkdir()
            #except Exception as e:
            #    logging.info(f'An error {e} occurred when creating the directory: {absolute_dir_path}')
        self.ffhq_data_directory = absolute_dir_path


    def get_url_json_file(self, validator=True):
        json_ulr_google_drive = 'https://drive.google.com/file/d/16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA'
        if validator:
            if not FFHQDatasetURls.is_valid_url(json_ulr_google_drive):
                assert f'Url by path: {json_ulr_google_drive} not valid!'
        return json_ulr_google_drive, self.ffhq_data_directory / 'ffhq-dataset-v2.json'


    def get_current_directoty_file(self):
        return Path.cwd()


    @staticmethod
    def is_valid_url(url: str):
        return validators.url(url)

# TODO: want to extract files from google drive
class FFHQDatasetLoader():


    def __init__(self, path_data_dir:str='datasets/dataset_ffhq'):
        ffhqurls = FFHQDatasetURls(path_data_dir)
        json_dataset_description_url, path_json_dataset_description_file = ffhqurls.get_url_json_file()
        if isinstance(path_json_dataset_description_file, str):
            path_json_dataset_description_file = Path(path_json_dataset_description_file)
        if path_json_dataset_description_file.is_file():
            logging.info(f'File {path_json_dataset_description_file} is already exist!')
            # just open json file
            self.json_data = self.get_json_data(path_json_dataset_description_file)
            self.path_json_file = path_json_dataset_description_file
        else:
            # need download json file with desriptions about ffhq dataset
            self.json_data = self._download_json_description_file(json_dataset_description_url,
                                                                  path_json_dataset_description_file)


    def extract_link_images(self, extracted_attrs=('file_path', 'file_url')):
        data = {}
        for _, item in self.json_data.items():
            image_object = item['image']
            data_per_image = {}
            for attr in extracted_attrs:
                if attr in image_object:
                    data_per_image[attr] = image_object[attr]
            # need save by image name
            hash_key = self._extract_hash_per_image(data_per_image)
            if hash_key is None:
                assert 'We cannot extract hash without "file_path" attribute in data_per_image'
            data[hash_key] = data_per_image


    def _extract_hash_per_image(self, data_per_image: dict):
        if 'file_path' not in data_per_image:
            return None
        data_path = data_per_image['file_path']
        suffix_path = data_path.split('/')[-1]
        idx_del = suffix_path.rfind('.')[0]
        hash = suffix_path[:idx_del]
        return hash


    def _download_json_description_file(self, url: str, path_json_url_dataset_descrs, chunk_size=128):
        if isinstance(path_json_url_dataset_descrs, str):
            path_json_url_dataset_descrs = Path(path_json_url_dataset_descrs)
        datasize = 0
        with requests.Session() as session:
            try:
                with session.get(url, stream=True) as response:
                    raise_status = response.raise_for_status()
                    with path_json_url_dataset_descrs.open('wb') as writer:
                        for chunk in response.iter_content(chunk_size=chunk_size<<10):
                            writer.write(chunk)
                            datasize += len(chunk)
            except Exception as e:
                assert False, f'An error {e} occurred while downloading the json file: {path_json_url_dataset_descrs}'

        json_data = self.get_json_data(path_json_url_dataset_descrs)
        return json_data


    def get_json_data(self, path_json_url_dataset_descrs):
        with path_json_url_dataset_descrs.open(encoding='utf-8') as reader:
            json_data = json.load(reader)
        return json_data



if __name__ == '__main__':
    ffhq_dataset_loader=FFHQDatasetLoader()
    ffhq_dataset_loader.extract_link_images()
