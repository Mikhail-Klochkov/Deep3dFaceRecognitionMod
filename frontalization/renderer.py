import logging, torch, numpy as np, re, cv2 as cv, open3d as op3d

from typing import Union
from pytorch3d.structures import Meshes as Meshes_torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
        PointLights,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        SoftPhongShader,
        FoVPerspectiveCameras,
        look_at_view_transform,
        TexturesVertex,
)
from pathlib import Path


path_mean_face_mesh = Path('/tmp/mean_face_mesh_basel.obj')

class DtypeTransformator():

    _default_bit_sys: int = 32
    _available_bit_sys: tuple = (32, 64)
    _available_types: tuple = ("int", "float")
    _torch_dtypes: dict = {'float32': torch.float32, 'float64': torch.float64,
                           'int32': torch.int32, 'int64': torch.int64}

    @staticmethod
    def wrapp_tensor(x :Union[np.ndarray, torch.Tensor], device:str, dtype:str, add_batch_axis:bool=True):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if add_batch_axis:
            x = x[None]

        if str(x.device) != device:
            if device.find(':') != -1:
                try:
                    device_name, index = device.split(':')
                    x = x.to(f'{device_name}:{index}')
                    assert all(isinstance(v, str) for v in (device_name, index))
                except Exception as e:
                    logging.info(f'Incorrect device parameter: {device} with exception: {e}')
                    assert False
            else:
                x = x.to(device)
        if dtype:

            type = re.search(r'(float|int)', dtype)
            bit_system = re.search(r'\d+', dtype)
            if type:
                type = type.group()
                assert type in DtypeTransformator._available_types, f'Incorrect dtype: {dtype} parameter'
            else:
                assert False, f'Incorrect dtype: {dtype} parameter'
            if bit_system:
                bit_int = int(bit_system.group())
                assert bit_int in DtypeTransformator._available_bit_sys, f'Incorrect dtype: {dtype} parameter'
            else:
                # by default 32
                bit_int = DtypeTransformator._default_bit_sys
            key_dtype = f'{type}{bit_int}'
            assert key_dtype in DtypeTransformator._torch_dtypes, 'Incorrect class method construction!'

            dtype_torch = DtypeTransformator._torch_dtypes[key_dtype]
            x = x.to(dtype_torch)

        return x


class Render():

    def __init__(self, device:str=None):
        if device is None or device == 'cuda:0':
            if torch.cuda.is_available():
                self._device = torch.device("cuda:0")
                torch.cuda.set_device(self._device)
                logging.info(f'Set cuda:0 device!')
            else:
                self._device = torch.device("cpu")
                logging.info(f'Set cpu device!')

        assert self._device.type in ("cpu", "cuda"), f"Incorrect device parameter: {device}"


    @property
    def device_str_repr(self):
        return f"{self._device.type}:{self._device.index}"


    # TODO: something incorrect
    def get_torch_mesh_object_from_mesh(self, mesh, white_texture:bool=False):
        data = mesh.tensor_mesh_objects
        verts, faces, textures = data['vertices'], data['triangles'], data['textures']
        verts = DtypeTransformator.wrapp_tensor(verts, device=self.device_str_repr, dtype='float32',
                                                add_batch_axis=True)
        faces = DtypeTransformator.wrapp_tensor(faces, device=self.device_str_repr, dtype='float32',
                                                add_batch_axis=True)
        if white_texture:
            textures = torch.ones_like(verts, device=self.device_str_repr, dtype=torch.float32)
        else:
            textures = DtypeTransformator.wrapp_tensor(textures,
                                                       device=self.device_str_repr,
                                                       dtype='float32',
                                                       add_batch_axis=True)
        texture_vertex = TexturesVertex(verts_features=textures)
        return Meshes_torch(verts=verts, faces=faces, textures=texture_vertex)


    def get_torch_mesh_obj_from_file(self, path_mesh_obj, extract_texture:bool=True):
        if isinstance(path_mesh_obj, str):
            path_mesh_obj = Path(path_mesh_obj)
        assert path_mesh_obj.is_file(), 'The file {} not found!'
        mesh = load_objs_as_meshes([str(path_mesh_obj)], device=self._device)
        return mesh


    # for experiment purposes
    def define_render_instance(self, use_look_at_view:bool=True):
        if use_look_at_view:
            R, T = look_at_view_transform(3.0, 0, 0, degrees=True)
            cameras = FoVPerspectiveCameras(device=self._device, R=R, T=T)
        else:
            # default
            cameras = FoVPerspectiveCameras(device=self._device)

        raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1)
        lights = PointLights(device=self._device,
                             ambient_color=((0.8, 0.8, 0.8),),
                             specular_color=((0.5, 0.5, 0.5),),
                             diffuse_color=((0.1, 0.1, 0.1), ),
                             location=[[3.0, 3.0, -3.0]])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=self._device, cameras=cameras, lights=lights)
        )
        return renderer

