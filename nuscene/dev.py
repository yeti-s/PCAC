import torch
import numpy as np
import json

from PIL import Image
from os.path import join
from scipy.spatial.transform import Rotation
import struct
from torch import Tensor


# points (n, 8) -> x, y, z, intensity, point_index, red, green, blue,
# coords or xyz (n, 3) -> x, y, z
# pixel or xy (n, 2) -> x, y

MIN_DISTANCE = 2.5
MAX_DISTANCE = 25.0
D_POINTS = 8

SAMPLE_DATA_TYPE = {
    'RADAR': 0,
    'LIDAR': 1,
    'CAM_FRONT_LEFT': 2,
    'CAM_FRONT_RIGHT': 3,
    'CAM_BACK_LEFT': 4,
    'CAM_BACK_RIGHT': 5,
    'CAM_FRONT': 6,
    'CAM_BACK': 7,
}


def bin_to_tensor(path)->Tensor:
    points = []
    with open(path, 'rb') as file:
        while True:
            data = file.read(4)
            if not data:
                break
            points.append(struct.unpack('f', data)[0])
            
    points = torch.tensor(points)
    points = points.reshape(-1, 5)[:,:4] # remove ring_index
    points = torch.concatenate([points, torch.zeros(points.size(0), D_POINTS-4)], dim=1)
    
    return points

def image_to_tensor(path)->Tensor:
    image = Image.open(path)
    i_np = np.array(image)
    i_tensor = torch.from_numpy(i_np)
    
    return i_tensor.permute(2, 0, 1) 

def read_json(path:str):
    with open(path, 'r') as json_file:
        return json.load(json_file)

def get_rotation_matrix(w:float, x:float, y:float, z:float)->Tensor:
    rotation = Rotation([x, y, z, w])
    return torch.from_numpy(rotation.as_matrix()).type(torch.float32)

# img (3, h, w),  (n, 2)
def get_rgb_bilinear(img:Tensor, xy:Tensor)->Tensor:
    x = xy[:, 0]
    y = xy[:, 1]
    
    x0 = torch.floor(x).int()
    y0 = torch.floor(y).int()
    x1 = torch.ceil(x).int()
    y1 = torch.ceil(y).int()

    lt = img[:, y0, x0]
    lb = img[:, y1, x0]
    rt = img[:, y0, x1]
    rb = img[:, y1, x1]
    
    wa = ((x1.float() - x) * (y1.float() - y)).unsqueeze(0)
    wb = ((x1.float() - x) * (y - y0.float())).unsqueeze(0)
    wc = ((x - x0.float()) * (y1.float() - y)).unsqueeze(0)
    wd = ((x - x0.float()) * (y - y0.float())).unsqueeze(0)
    
    return (lt * wa + lb * wb + rt * wc + rb * wd).type(torch.uint8).transpose(0, 1)

def color_points(image:Tensor, points:Tensor, pixel:Tensor, depth:Tensor, min_dist=1.0)->Tensor:
    _, h, w = image.size()
    n_points = points.size(0)
    x = pixel[:, 0]
    y = pixel[:, 1]
    
    mask = torch.ones(n_points, dtype=bool)
    mask = torch.logical_and(mask, depth > min_dist)
    mask = torch.logical_and(mask, x > 1)
    mask = torch.logical_and(mask, x < w - 1)
    mask = torch.logical_and(mask, y > 1)
    mask = torch.logical_and(mask, y < h - 1)
    
    rgb = get_rgb_bilinear(image, pixel[mask,:2])
    points[mask,5:8] = rgb.float()
    
    return points



class EgoPose():
    def __init__(self, json_data) -> None:
        rotation = json_data['rotation']
        self.rotation:Tensor = get_rotation_matrix(rotation[0], rotation[1], rotation[2], rotation[3])
        self.timestamp:int = json_data['timestamp']
        self.tranlation:Tensor = torch.tensor(json_data['translation'])
        self.token:str = json_data['token']

class CalibratedSensor():
    def __init__(self, json_data) -> None:
        rotation = json_data['rotation']
        self.rotation:Tensor = get_rotation_matrix(rotation[0], rotation[1], rotation[2], rotation[3])
        self.tranlation:Tensor = torch.tensor(json_data['translation'])
        self.token:str = json_data['token']
        self.sensor_token:str = json_data['sensor_token']
        
        camera_intrinsic = json_data['camera_intrinsic']
        self.is_camera:bool = len(camera_intrinsic) > 0
        if self.is_camera:
            self.camera_intrinsic:Tensor = torch.tensor(camera_intrinsic)
            
            
class SampleData():
    def __init__(self, root:str, json_data) -> None:
        self.timestamp:int = json_data['timestamp']
        self.token:str = json_data['token']
        self.sample_token:str = json_data['sample_token']
        self.ego_pose_token:str = json_data['ego_pose_token']
        self.calibrated_sensor_token:str = json_data['calibrated_sensor_token']
        self.filename:str = join(root, json_data['filename'])
        self.prev:str = json_data['prev']
        self.next:str = json_data['next']
        self.height:int = json_data['height']
        self.width:int = json_data['width']
        self.is_key_frame:bool = json_data['is_key_frame']
        
        for key, value in SAMPLE_DATA_TYPE.items():
            if self.filename.find(key) > -1:
                self.type:int = value
                break
        
        
    def set_calibrated_sensor(self, calibrated_sensor:CalibratedSensor):
        self.rotation:Tensor = calibrated_sensor.rotation
        self.translation:Tensor = calibrated_sensor.tranlation
        self.is_camera:bool = calibrated_sensor.is_camera
        if self.is_camera:
            self.camera_intrinsic:Tensor = calibrated_sensor.camera_intrinsic
            
    def set_ego_pose(self, ego_pose:EgoPose):
        self.ego_rotation:Tensor = ego_pose.rotation
        self.ego_translation:Tensor = ego_pose.tranlation
        
    def get_points(self)->Tensor:
        points = bin_to_tensor(self.filename)
        
        coords = points[:,:3]
        # distance filter
        # distance = torch.sum(torch.pow(coords, 2), dim=1)
        # mask = distance > (MIN_DISTANCE**2)
        # mask = torch.logical_and(mask, distance < (MAX_DISTANCE**2))
        # points = points[mask]
        # coords = points[:,:3]
        
        # lidar to ego_vehicle
        coords = torch.matmul(coords, self.rotation.T) + self.translation
        # ego_vehicle to world
        coords = torch.matmul(coords, self.ego_rotation.T) + self.ego_translation
        points[:,:3] = coords
        
        return points
    
    def get_image(self)->Tensor:
        return image_to_tensor(self.filename)
        
    def to_world(self, points:Tensor)->Tensor:
        coords = points[:,:3]
        ego_pose = torch.matmul(coords, self.rotation.T) + self.translation
        world_pos = torch.matmul(ego_pose, self.ego_rotation.T) + self.ego_translation
        
        return torch.concatenate([world_pos, points[:,3:]], dim=1)
    
    def to_sensor(self, points:Tensor)->Tensor:
        coords = points[:,:3]
        ego_pos = torch.matmul(coords - self.ego_translation, self.ego_rotation)
        sensor_pos = torch.matmul(ego_pos - self.translation, self.rotation)
        
        return torch.concatenate([sensor_pos, points[:,3:]], dim=1)
    
    def points_to_pixel(self, points:Tensor):
        coords = self.to_sensor(points)[:,:3]
        depth = coords[:,2]
        
        viewpad = torch.eye(4)
        viewpad[:self.camera_intrinsic.shape[0], :self.camera_intrinsic.shape[1]] = self.camera_intrinsic
        
        num_points = coords.size(0)
        coords = torch.concatenate((coords.T, torch.ones(1, num_points)))
        coords = torch.matmul(viewpad, coords)
        coords = coords[:3, :]
        pixel = coords[:2, :] / coords[2:3, :].repeat(2, 1)
        
        return pixel.T, depth
    
class Sample():
    def __init__(self, json_data) -> None:
        self.timestamp:int = json_data['timestamp']
        self.token:str = json_data['token']
        self.scene_token:str = json_data['scene_token']
        self.prev = json_data['prev']
        self.next = json_data['next']
        self.cameras:list[SampleData] = []
        self.lidar = None
        
        self.data_list:list[SampleData] = []
        
    def add_sample_data(self, sample_data:SampleData)->None:
        self.data_list.append(sample_data)
        self.data_list = sorted(self.data_list, key=lambda x:x.timestamp)
        
        
class Scene():
    def __init__(self, json_data) -> None:
        self.nbr_samples:int = json_data['nbr_samples']
        self.token:str = json_data['token']
        self.log_token:str = json_data['log_token']
        self.first_sample_token:str = json_data['first_sample_token']
        self.last_sample_token:str = json_data['last_sample_token']
        self.name:str = json_data['name']
        self.description:str = json_data['description']
        self.sample_list:list[Sample] = []
    
    def get_points(self)->Tensor:
        point_index = 0
        
        last_cam_samples = [None]*8
        remain_points = torch.rand(0, D_POINTS)
        processed_points = []
        
        def color_points_with_cam(points:Tensor, cam:SampleData):
            image = cam.get_image()
            pixel, depth = cam.points_to_pixel(points)
            points = color_points(image, points, pixel, depth)
            
            mask = points[:,5]+points[:,6]
            mask = mask + points[:,7]
            mask = mask > 0
            processed_points.append(points[mask])
            return points[torch.logical_not(mask)]
        
        for sample in self.sample_list:
            print(f'get points from {sample.token}')
            data_list = sample.data_list
            
            for sample_data in data_list:
                # camera type
                if sample_data.type >= SAMPLE_DATA_TYPE['CAM_FRONT_LEFT']:
                    last_cam_samples[sample_data.type] = sample_data
                    if remain_points.size(0) > 0:
                        remain_points = color_points_with_cam(remain_points, sample_data)
                    continue
                
                # lidar
                points = sample_data.get_points()
                n_points = points.size(0)
                points[:,4] = torch.arange(point_index, point_index+n_points).float()
                point_index += n_points
                
                for cam in last_cam_samples:
                    if cam == None:
                        continue
                    points = color_points_with_cam(points, cam)
                remain_points = torch.concatenate([remain_points, points], dim=0)
        
        points = torch.concatenate(processed_points, dim=0)
        return points

class Nuscene():
    def __init__(self, root:str) -> None:
        self.root:str = root
        
        ego_pose_map = self.parse_ego_pose()
        calib_sensor_map = self.parse_calibrated_sensor()
        sample_map = self.parse_sample()
        self.parse_sample_data(ego_pose_map, calib_sensor_map, sample_map)
        self.scene_map = self.parse_scene(sample_map)
    
    def parse_ego_pose(self) -> dict[str, EgoPose]:
        data = read_json(join(self.root, 'meta', 'ego_pose.json'))
        ego_pose_map = {}
        for json_data in data:
            ego_pose = EgoPose(json_data)
            ego_pose_map[ego_pose.token] = ego_pose
            
        return ego_pose_map
            
    def parse_calibrated_sensor(self) -> dict[str, CalibratedSensor]:
        data = read_json(join(self.root, 'meta', 'calibrated_sensor.json'))
        calib_sensor_map = {}
        for json_data in data:
            calib_sensor = CalibratedSensor(json_data)
            calib_sensor_map[calib_sensor.token] = calib_sensor
            
        return calib_sensor_map
            
    def parse_sample(self) -> dict[str, Sample]:
        data = read_json(join(self.root, 'meta', 'sample.json'))
        sample_map:list[Sample] = {}
        
        for json_data in data:
            sample = Sample(json_data)
            sample_map[sample.token] = sample
        
        for _, sample in sample_map.items():
            if sample.next in sample_map:
                sample.next = sample_map[sample.next]
            else:
                sample.next = None
            if sample.prev in sample_map:
                sample.prev = sample_map[sample.prev]
            else:
                sample.prev = None
        
        return sample_map
            
            
    def parse_sample_data(self, ego_pose_map:dict[str, EgoPose], calib_sensor_map:dict[str, CalibratedSensor], 
                          sample_map:dict[str, Sample]) -> None:
        data = read_json(join(self.root, 'meta', 'sample_data.json'))
        
        for json_data in data:
            sample_data = SampleData(self.root, json_data)
            
            ego_pose = ego_pose_map[sample_data.ego_pose_token]
            sample_data.set_ego_pose(ego_pose)
            
            calib_sensor:CalibratedSensor = calib_sensor_map[sample_data.calibrated_sensor_token]
            sample_data.set_calibrated_sensor(calib_sensor)
            
            sample:Sample = sample_map[sample_data.sample_token]
            if sample_data.type == SAMPLE_DATA_TYPE['RADAR']:
                continue
            
            sample.add_sample_data(sample_data)
            
    def parse_scene(self, sample_map:dict[str, Sample]) -> dict[str, Scene]:
        data = read_json(join(self.root, 'meta', 'scene.json'))
        scene_map = {}
        for json_data in data:
            scene = Scene(json_data)
            scene_map[scene.token] = scene
            
            first_sample = sample_map[scene.first_sample_token]
            last_sample = sample_map[scene.last_sample_token]
            sample = first_sample
            while sample != None:
                scene.sample_list.append(sample)
                if sample == last_sample:
                    break
                sample = sample.next
        
        return scene_map
    
    def get_scenes(self)->dict[str,Scene]:
        return self.scene_map
    