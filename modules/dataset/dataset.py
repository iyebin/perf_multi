import os.path
import numpy as np
import torch
import trimesh
import hashlib
import cv2 as cv
import torch.nn.functional as F
import os

from os.path import join as pjoin
from utils.utils import read_dpt, read_image, write_image
from utils.camera_utils import *
from utils.geo_utils import align_scale, get_edge_mask
from modules.geo_predictors import PanoFusionInvPredictor, PanoFusionNormalPredictor, PanoGeoRefiner, PanoJointPredictor
from modules.geo_predictors import PanoVggt


class Dataset:
    def __init__(self):

        self.n_images = 0
        self.image_dir = None
        self.image_names = []
        self.ref_distance_path = None
        self.ref_normal_path = None
        self.ref_geometry_path = None
        self.images = None
        self.gt_distances = []
        self.ref_distances = []
        self.ref_normals = []
        self.height = 0
        self.width = 0
        self.data_dir = None
        self.poses = []
        self.case_name = 'wp'
        # self.reference_idx = 0

    def get_ref_distance(self):
        assert self.images is not None
        assert self.ref_distance_path is not None
        assert self.height > 0 and self.width > 0

        ref_distances = []
        distance_predictor = PanoFusionInvPredictor()

        zero_mask = torch.zeros([self.height, self.width])
        ones_mask = torch.ones([self.height, self.width])
       
        if os.path.exists(self.ref_distance_path):
            ref_distances = np.load(self.ref_distance_path)
            ref_distances = torch.from_numpy(ref_distance.astype(np.float32)).cuda()
            

        else:
            ref_distances, _ = distance_predictor(self.images[0], self.images[1],
                                                    zero_mask,
                                                    ones_mask)

        return ref_distances

    def get_ref_normal(self):
        assert self.images is not None
        assert self.ref_normal_path is not None
        assert self.height > 0 and self.width > 0

        ref_normals = []
        normal_predictor = PanoFusionNormalPredictor()

        norm_mask = torch.ones([self.height, self.width, 3]) / np.sqrt(3.)
        ones_mask = torch.ones([self.height, self.width])

        if os.path.exists(self.ref_normal_path):
            ref_normals = np.load(self.ref_normal_path)
            ref_normals = torch.from_numpy(ref_normals.astype(np.float32)).cuda()
        else:
            ref_normals = normal_predictor.inpaint_normal(self.images[0], self.images[1],
                                                          norm_mask,
                                                          ones_mask)

        return ref_normals

    def refine_geometry(self, distance_map, normal_map):
        refiner = PanoGeoRefiner()
        return refiner.refine(distance_map, normal_map)

    def get_joint_distance_normal(self, org_distance=None):
        assert self.images is not None
        assert self.ref_distance_path is not None
        assert self.ref_normal_path is not None
        assert self.height > 0 and self.width > 0

        self.poses, self.trans = load_cam_extrinsic(self.image_dir) #camera coordinate(w2c)
        # breakpoint( )

        ref_distances = []
        ref_normals = []

        ones_dim_mask = torch.ones([self.height, self.width, 1])
        ones_mask = torch.ones([self.height, self.width])
        zero_mask = torch.zeros([self.height, self.width])
        
        #image1, image2 통합
        joint_predictor = PanoJointPredictor()


        # for i in range(self.n_images):

        if os.path.exists(self.ref_distance_path) and\
        os.path.exists(self.ref_normal_path):
            ref_distances = np.load(self.ref_distance_path)
            ref_distances = torch.from_numpy(ref_distances.astype(np.float32)).cuda()
            ref_normals = np.load(self.ref_normal_path)
            ref_normals = torch.from_numpy(ref_normals.astype(np.float32)).cuda()
        

        else:
            if org_distance is None:
                # joint predictor consistency loss는 w2c 기대
                for i in range(self.n_images):
                    ref_distance, ref_normal = joint_predictor(self.images[i],
                                                            torch.ones([self.height, self.width, 1]), 
                                                            torch.ones([self.height, self.width]))
                    
                    ref_distances.append(ref_distance)
                    ref_normals.append(ref_normal)
            else:
                for i in range(self.n_images):
                    ref_distance, ref_normal = joint_predictor(self.images[i],
                                                            ref_distance=org_distance,
                                                            mask=zero_mask,
                                                            reg_loss_weight=0.)
                    ref_distances.append(ref_distance)
                    ref_normals.append(ref_normal)

        # if ref_distances.shape[0] != self.height or ref_distances.shape[1] != self.width:
        #     ref_distances = F.interpolate(
        #         ref_distances.unsqueeze(0).unsqueeze(0),
        #         size=(self.height, self.width),
        #         mode='bilinear',
        #         align_corners=False
        #     ).squeeze()

        #     ref_normals = F.interpolate(
        #         ref_normals.permute(2,0,1).unsqueeze(0),
        #         size=(self.height, self.width),
        #         mode='bilinear',
        #         align_corners=False
        #     ).squeeze().permute(1,2,0)

        # self.poses = [p.inverse() for p in self.poses]  # c2w for render/sup
            ref_distances = torch.stack(ref_distances)
            ref_normals = torch.stack(ref_normals)

        return ref_distances, ref_normals

    def get_panovggt_distance(self):
        assert self.images is not None
        assert self.ref_distance_path is not None
        assert self.ref_normal_path is not None
        assert self.height > 0 and self.width > 0

        self.poses, self.trans = load_cam_extrinsic(self.image_dir) #camera coordinate(w2c)

        #pano vggt(1 parameter)
        ref_distances = []
        # ref_normals = []

        # ones_dim_mask = torch.ones([self.height, self.width, 1])
        # ones_mask = torch.ones([self.height, self.width])
        # zero_mask = torch.zeros([self.height, self.width])

        #if distances / normals exist
        if os.path.exists(self.ref_distance_path) and\
                os.path.exists(self.ref_normal_path):
                    ref_distances = np.load(self.ref_distance_path)
                    ref_distances = torch.from_numpy(ref_distances.astype(np.float32)).cuda()
                    # ref_normals = np.load(self.ref_normal_path)
                    # ref_normals = torch.from_numpy(ref_normals.astype(np.float32)).cuda()

        else:
           
            for i in range(self.n_images):
                #pano vggt
                ref_distance = PanoVggt(self.image_dir, self.image_names)
                ref_distances.append(ref_distance)
                # ref_normals.append(ref_normal)


    '''
    python inference.py \
    --config  training/config/default.yaml \
    --checkpoint pre_checkpoints/panovggt_model.pt \
    --image_dir  img_pathes \
    --mask_dir   data/image_mask \
    --output_dir results

    '''

    def normalization(self):
        # 모든 뷰에 공통 scale을 적용해 뷰 간 절대 거리 단위를 일치시킴
        
        global_max = max(d.max().item() for d in self.ref_distances)
        scale = global_max * 1.05
      
        self.ref_distances /= scale

    def save_ref_geometry(self):
        # Save distance and normal data
        os.makedirs(pjoin(self.image_dir, 'ref_geometry'), exist_ok=True)
        os.makedirs(pjoin(self.image_dir, 'ref_distance'), exist_ok=True)
        os.makedirs(pjoin(self.image_dir, 'ref_normal'), exist_ok=True)

        if self.ref_distance_path is not None:
            np.save(self.ref_distance_path, self.ref_distances.cpu().numpy())
        if self.ref_normal_path is not None:
            np.save(self.ref_normal_path, self.ref_normals.cpu().numpy())

        # Save point cloud
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(self.height, self.width)) #just direction
        pts = pano_dirs * self.ref_distances.squeeze()[..., None] #self.ref_distances 사용
        pts = pts.cpu().numpy().reshape(-1, 3)
        #check point numbers
        points_count_path = pjoin(self.image_dir, 'ref_geometry', 'points_count.txt')
        with open(points_count_path, 'w') as f:
            f.write(f"all points number: {pts.shape[0]}\n")

        colors = torch.stack(self.images, dim=0)
        colors = colors.cpu().numpy().reshape(-1, 3)

        assert pts.shape[0] == colors.shape[0], (pts.shape, colors.shape)

        if self.images[0] is not None:
            pcd = trimesh.PointCloud(pts, vertex_colors=colors) #color는 이미지 한 장에서만 가져오고 있음 => 다중 이미지로 수정
        else:
            pcd = trimesh.PointCloud(pts)

        assert self.ref_geometry_path is not None and self.ref_geometry_path[-4:] == '.ply'
        pcd.export(self.ref_geometry_path)


    def fit_scene_to_aabb(self, aabb_extent=0.9):
        """camera(c2w) + depth가 NeRF AABB [-1,1] 안에 들어오도록 균등 스케일."""
        depths = [self.ref_distances] * self.n_images if not isinstance(self.ref_distances, list) \
            else self.ref_distances
        max_extent = max(
            p[:3, 3].norm().item() + d.max().item()
            for p, d in zip(self.poses, depths)
        )

        if max_extent > aabb_extent:
            scale = max_extent / aabb_extent
            if isinstance(self.ref_distances, list):
                self.ref_distances = [d / scale for d in self.ref_distances]
            else:
                self.ref_distances = self.ref_distances / scale
            for i in range(self.n_images):
                new_pose = self.poses[i].clone()
                new_pose[:3, 3] = self.poses[i][:3, 3] / scale
                self.poses[i] = new_pose

class WildDataset(Dataset):
    def __init__(self, conf):
        super().__init__()
         
        self.image_dir = conf.image_dir

        # png 파일 자동 탐색
        self.image_names = sorted([
            os.path.splitext(f)[0]   # 확장자 제거
            for f in os.listdir(self.image_dir)
            if f.endswith('.jpg')
        ])
        self.n_images = len(self.image_names)

        #image_dir + image_names[i] -> 이미지 저장 경로
        self.ref_distance_path = pjoin(self.image_dir, 'ref_distance', f"distance.npy") 
        self.ref_normal_path = pjoin(self.image_dir, 'ref_normal', f"normal.npy") 
        self.ref_geometry_path = pjoin(self.image_dir, 'ref_geometry', f"geometry.ply") 
        self.warped_image_dir = pjoin(self.image_dir, 'warped_images')
        # self.ref_distance_path = '.'.join(self.image_path.split('.')[:-1]) + '_ref_distance.npy'
        # self.ref_normal_path = '.'.join(self.image_path.split('.')[:-1]) + '_ref_normal.npy'
        # self.ref_geometry_path = '.'.join(self.image_path.split('.')[:-1]) + '_ref_geometry.ply'

        # self.case_name = self.image_path.split('/')[-2]
        self.images = [read_image(pjoin(self.image_dir, name + ".jpg"), to_torch=True, squeeze=True).cuda() for name in self.image_names]

        if 'image_resize' in conf:
                    self.width, self.height = conf['image_resize']
                    for i in range(self.n_images):
                        self.images[i] = cv.resize(self.images[i].cpu().numpy(), (self.width, self.height), cv.INTER_AREA)
                        self.images[i] = torch.from_numpy(self.images[i]).cuda()
        else:
            shapes = [img.shape for img in self.images]
            assert all(s == shapes[0] for s in shapes), "Shapes must be equal for all images"
            self.height, self.width, _ = self.images[0].shape

        

        self.case_name = os.path.basename(self.image_dir)

        

        self.ref_distances, self.ref_normals = self.get_joint_distance_normal()
        # self.fit_scene_to_aabb()
        self.normalization()

        self.save_ref_geometry()

