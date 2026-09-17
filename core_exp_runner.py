import os
import cv2 as cv
import numpy as np
from shutil import copyfile
from os.path import join as pjoin

import trimesh
import torch
import torch.nn.functional as F

from omegaconf import OmegaConf, DictConfig
from glob import glob
from tqdm import tqdm
import hydra

from icecream import ic

from modules.inpainters import PanoPersFusionInpainter
from modules.geo_predictors import PanoJointPredictor
# from modules.geo_predictors import RePanoJointPredictor
from modules.dataset.dataset import WildDataset
from modules.dataset.sup_info import SupInfoPool
from modules.pose_sampler import CirclePoseSampler
from modules.pose_sampler import DenseTravelPoseSampler
from modules.pose_sampler import GRFCirclePoseSampler
from modules.pose_sampler import MiddlePanoramaSampler

from modules.scene.nerf import NeRFScene

from utils.utils import write_video, write_image, colorize_single_channel_image
from utils.debug_utils import printarr
from utils.camera_utils import *

backup_file_patterns = [
    './*.py', './modules/*.py', './modules/*/*.py', './utils/*.py,'
]


class CoreRunner:
    def __init__(self, conf, device=torch.device('cuda')):
        self.conf = conf
        self.device = device

        self.dataset = WildDataset(conf.dataset)

        self.base_dir = os.getcwd()
        self.base_exp_dir = conf.device.base_exp_dir
        self.exp_dir = pjoin(self.base_exp_dir, '{}_{}'.format(conf['dataset_class_name'], self.dataset.case_name), conf.exp_name)

        os.makedirs(self.exp_dir, exist_ok=True)

        # backup codes
        file_backup_dir = os.path.join(self.exp_dir, 'record/')
        os.makedirs(file_backup_dir, exist_ok=True)

        for file_pattern in backup_file_patterns:
            file_list = glob(os.path.join(self.base_dir, file_pattern))
            for file_name in file_list:
                new_file_name = file_name.replace(self.base_dir, file_backup_dir)
                os.makedirs(os.path.dirname(new_file_name), exist_ok=True)
                copyfile(file_name, new_file_name)
        
        # self.dataset.trans 저장
        trans = self.dataset.trans

        if torch.is_tensor(trans):
            trans = trans.detach().cpu().numpy()
        else:
            trans = np.asarray(trans)

        np.savetxt(
            pjoin(file_backup_dir, 'pose_1.txt'),
            trans.reshape(1, -1),
            fmt='%.6f'
        )

        resolved_conf = OmegaConf.to_container(conf, resolve=True)
        OmegaConf.save(resolved_conf, os.path.join(file_backup_dir, 'config.yaml'))
        OmegaConf.save(resolved_conf, './config.yaml')

        self.scene = globals()[conf.scene_class_name](self.exp_dir, **conf.scene)

        # Visualization - every images
        # for i in range(self.dataset.n_images):
        #     write_image(pjoin(self.exp_dir, f'distance_vis_{self.dataset.image_names[i]}.png'),
        #                 colorize_single_channel_image(
        #                     (self.dataset.ref_distances[i].min() + 1e-6) / (self.dataset.ref_distances[i] + 1e-6)))
        #     if self.dataset.ref_normals[i] is not None:
        #         write_image(pjoin(self.exp_dir, f'normal_vis_{self.dataset.image_names[i]}.png'),
        #                     (self.dataset.ref_normals[i] * .5 + .5) * 255.)
        
        # breakpoint()
        for i in range(self.dataset.n_images):
            
            write_image(pjoin(self.exp_dir, f'distance_vis_.png'),
                        colorize_single_channel_image(
                            (self.dataset.ref_distances.min() + 1e-6) / (self.dataset.ref_distances + 1e-6)))
            if self.dataset.ref_normals is not None:
                write_image(pjoin(self.exp_dir, f'normal_vis_{self.dataset.image_names}.png'),
                            (self.dataset.ref_normals[i] * .5 + .5) * 255.)

        # self.pose_sampler_0 = MiddlePanoramaSampler(base_point=[0.0, 0.0, 0.0], another=self.dataset.trans)
        self.pose_sampler_0 = CirclePoseSampler(self.dataset.ref_distances, self.dataset.poses, **conf.pose_sampler)
        # self.pose_sampler = MiddlePanoramaSampler(self.dataset.poses)

        self.sup_pool = SupInfoPool()

        #debug
        for i in range(self.dataset.n_images):
            self.sup_pool.register_sup_info(pose=self.dataset.poses[i],
                                                mask=torch.ones([self.dataset.height, self.dataset.width]),
                                                rgb=self.dataset.images[i],
                                                distance=self.dataset.ref_distances,
                                                normal=self.dataset.ref_normals[i],
                                                log=True)
        self.sup_pool.gen_occ_grid(256)
        # grids, _ = self.sup_pool.gen_occ_grids_per_view(res=256)

        # self.sup_pool.export_two_view_occ_comparison(
        #     grids=grids,
        #     resolutions=256,
        #     save_path=pjoin(
        #         self.exp_dir,
        #         'debug',
        #         'occ_two_view_comparison.ply'
        #     )
        # )

        # self.geo_predictor = PanoJointPredictor()
        self.geo_predictor = PanoJointPredictor()
        self.inpainter = PanoPersFusionInpainter(inpainter_type=conf.pers_inpainter_type)

        self.phase = -1

        # Load checkpoint
        if conf.is_continue:
            self.load_checkpoint('ckpt.pth')

    def set_train(self):
        self.scene.set_train()

    def set_eval(self):
        self.scene.set_eval()

    def execute(self, mode):
        if mode == 'train':
            self.train()
        elif mode == 'render_dense':
            self.render_dense()

        elif mode == 'extract_panorama':
            self.extract_panorama()

        elif mode == 'render_mid':
            self.render_middle_view()

        elif mode == 'render_specific':
            self.render_specific()
        
        elif mode == 'matching_mse':
            self.matching_mse()

        elif mode == 'check_occ':
            self.check_occ(self.sup_pool, self.dataset.height, self.dataset.width)

        elif mode == 'check_grid':
            self.scene.diagnose_gray_vs_occ(self.sup_pool)
            import os
            import inspect
            import sys

            method = self.scene.diagnose_gray_vs_occ

            print("\n========== SOURCE CHECK ==========", flush=True)
            print(f"PID            : {os.getpid()}", flush=True)
            print(f"CWD            : {os.getcwd()}", flush=True)
            print(f"Python         : {sys.executable}", flush=True)
            print(f"Scene type     : {type(self.scene)}", flush=True)
            print(f"Class file     : {inspect.getfile(type(self.scene))}", flush=True)
            print(f"Method file    : {method.__func__.__code__.co_filename}", flush=True)
            print(f"First line     : {method.__func__.__code__.co_firstlineno}", flush=True)
            print("==================================\n", flush=True)

            method(self.sup_pool)

    def check_occ(self, sup_pool, h, w):
        self.scene.diagnose_culling_re(sup_pool, h, w)
        self.scene.check_grid_coverage(sup_pool)


    def train(self, raw_only=False):
        ic('Train: begin')
        if self.phase < 0:
            self.set_train()
            # breakpoint()
            self.scene.fit(self.sup_pool)

            #pose = I, 중심 좌표계의 rgb 이미지와 distance 이미지
            for i in range(self.dataset.n_images):
                render_result = self.scene.render(gen_pano_rays(self.dataset.poses[i], self.dataset.height, self.dataset.width), query_keys=['rgb', 'distance'])
                pano_rgb = render_result['rgb']
                # breakpoint()
                pano_distances = (render_result['distance'].min() / render_result['distance']).squeeze()[..., None]
                write_image(pjoin(self.exp_dir, f'{i + 1}.png'), pano_rgb * 255.)
                write_image(pjoin(self.exp_dir, f'{i + 1}_distance.png'), colorize_single_channel_image(pano_distances))

            self.phase += 1
            self.save_checkpoint()

            if raw_only:
                return

        n_anchors = self.pose_sampler_0.n_anchors

        # geo_check = True
        # geo_check = False

        # for anchor_idx in range(n_anchors):
        #     if anchor_idx < self.phase:
        #         continue
            
        #     #image0 pose
        #     pose_0 = self.pose_sampler_0.sample_pose(anchor_idx)
        #     rays_0 = gen_pano_rays(pose_0, self.dataset.height, self.dataset.width)

        #     #image1 pose
        #     # pose_1 = self.pose_sampler_1.sample_pose(anchor_idx)
        #     # rays_1 = gen_pano_rays(pose_1, self.dataset.height, self.dataset.width)

        #     visi_mask_0 = self.scene.get_pano_visibility_mask(self.sup_pool, rays_0)  # 1 visible, 0 invisible
        #     # visi_mask_1 = self.scene.get_pano_visibility_mask(self.sup_pool, rays_1)  # 1 visible, 0 invisible
        #     with torch.no_grad():
        #         render_result_0 = self.scene.render(rays_0, query_keys=['rgb', 'distance'])
        #         # render_result_1 = self.scene.render(rays_1, query_keys=['rgb', 'distance'])
        #     colors_0 = render_result_0['rgb']
        #     distances_0 = render_result_0['distance']

        #     # colors_1 = render_result_1['rgb']
        #     # distances_1 = render_result_1['distance']

        #     inpaint_mask_0 = 1. - visi_mask_0 #invisible region에 대해 inpainting 수행
        #     # inpaint_mask_1 = 1. - visi_mask_1

        #     n_repeats = 1
        #     for sub_i in range(n_repeats):
        #         if visi_mask_0.min().item() > .5:
        #             break

        #         # colors_0, distances_0, normals_0 = self.inpaint_new_panorama(sub_i, anchor_idx, colors=colors_0, distances=distances_0, mask=inpaint_mask_0)
        #         # colors_1, distances_1, normals_1 = self.inpaint_new_panorama(sub_i, anchor_idx, colors=colors_1, distances=distances_1, mask=inpaint_mask_1)
        #         if geo_check:
        #             # Perform geometric checking
        #             conflict_mask_0 = 1. - self.sup_pool.geo_check(rays_0, distances_0)    # 1 conflict, 0 not conflict
        #             inpaint_mask_0 = inpaint_mask_0 * conflict_mask_0

        #             # conflict_mask_1 = 1. - self.sup_pool.geo_check(rays_1, distances_1)    # 1 conflict, 0 not conflict
        #             # inpaint_mask_1 = inpaint_mask_1 * conflict_mask_1
        #         else:
        #             # inpaint_mask_0 *= 0
        #             # inpaint_mask_1 *= 0
        #             pass

        #         sub_i += 1

        #     # vis_dir = pjoin(self.exp_dir, 'inpaint_vis', '{:0>4d}'.format(anchor_idx))
        #     # os.makedirs(vis_dir, exist_ok=True)

        #     # Do not inpaint contents that are too close
        #     # inpaint_mask = torch.minimum(inpaint_mask, (distances > 0.05).float())
        #     # inpaint_mask_0 = torch.maximum(inpaint_mask_0, (distances_0.squeeze() < 0.1).float())
        #     # inpaint_mask_0 = torch.minimum(inpaint_mask_0, 1. - visi_mask_0)

        #     # inpaint_mask_1 = torch.maximum(inpaint_mask_1, (distances_1.squeeze() < 0.1).float())
        #     # inpaint_mask_1 = torch.minimum(inpaint_mask_1, 1. - visi_mask_1)

        #     # write_image(pjoin(vis_dir, 'final_mask_0.jpg'), inpaint_mask_0[..., None] * 255.)
        #     # write_image(pjoin(vis_dir, 'final_masked_0.jpg'), (colors_0 * (1. - inpaint_mask_0)[..., None]) * 255.)

        #     # write_image(pjoin(vis_dir, 'final_mask_1.jpg'), inpaint_mask_1[..., None] * 255.)
        #     # write_image(pjoin(vis_dir, 'final_masked_1.jpg'), (colors_1 * (1. - inpaint_mask_1)[..., None]) * 255.)

        #     # sup_mask_0 = 1. - visi_mask_0
        #     # sup_mask_0 -= torch.minimum(sup_mask_0, inpaint_mask_0)

        #     # sup_mask_1 = 1. - visi_mask_1
        #     # sup_mask_1 -= torch.minimum(sup_mask_1, inpaint_mask_1)

        #     self.sup_pool.register_sup_info(pose=pose_0, mask=inpaint_mask_0, rgb=colors_0, distance=distances_0, normal=None)
        #     # self.sup_pool.register_sup_info(pose=pose_1, mask=sup_mask_1, rgb=colors_1, distance=distances_1, normal=normals_1)
        #     self.scene.fit(self.sup_pool)

        #     self.phase += 1
        #     self.save_checkpoint()

    def inpaint_new_panorama(self, phase, anchor_idx, colors, distances, mask):
        distances = distances.squeeze()[..., None]
        mask = mask.squeeze()[..., None]

        vis_dir = pjoin(self.exp_dir, 'inpaint_vis', '{:0>4d}'.format(anchor_idx))
        os.makedirs(vis_dir, exist_ok=True)

        write_image(pjoin(vis_dir, 'uninpainted_{}.jpg'.format(phase)), colors * 255.)
        write_image(pjoin(vis_dir, 'uninpainted_disparity_{}.jpg'.format(phase)), colorize_single_channel_image(distances.min() / distances))
        write_image(pjoin(vis_dir, 'mask_{}.jpg'.format(phase)), mask * 255.)
        write_image(pjoin(vis_dir, 'masked_{}.jpg'.format(phase)), (colors * (1. - mask)) * 255.)
        
        inpainted_distances = None
        inpainted_normals = None
        if self.conf.rgbd_inpaint:
            inpainted_img, inpainted_distances = self.inpainter.inpaint_rgbd(colors, distances, mask)
            write_image(pjoin(vis_dir, 'inpainted_{}.jpg'.format(phase)), inpainted_img * 255.)
        else:
            inpainted_img = self.inpainter.inpaint(colors, mask, self.exp_dir,
                                                    anchor_idx=anchor_idx,
                                                    phase=phase)
            inpainted_img = inpainted_img.cuda()
            write_image(pjoin(vis_dir, 'inpainted_{}.jpg'.format(phase)), inpainted_img * 255.)
            inpainted_distances, inpainted_normals = self.geo_predictor(inpainted_img,
                                                                        distances,
                                                                        mask=mask,
                                                                        reg_loss_weight=0.,
                                                                        normal_loss_weight=5e-2,
                                                                        normal_tv_loss_weight=5e-2)

        inpainted_distances = inpainted_distances.squeeze()

        height, width, _ = inpainted_img.shape
        write_image(pjoin(vis_dir, 'aligned_disparity_{}.jpg'.format(phase)),
                    colorize_single_channel_image(inpainted_distances.min().item() / inpainted_distances[:, :, None]))
        if inpainted_normals is not None:
            write_image(pjoin(vis_dir, 'aligned_normals_{}.jpg'.format(phase)), (inpainted_normals * .5 + .5).clip(0., 1.) * 255.)

        return inpainted_img, inpainted_distances, inpainted_normals

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device)
        self.scene.load_state_dict(checkpoint['scene'])
        self.phase = checkpoint['phase']

    def render_dense(self, n_poses=120, cam_type='pano'):
        dense_pose_sampler = DenseTravelPoseSampler(self.pose_sampler_0, n_dense_poses=n_poses)
        out_dir = pjoin(self.exp_dir, 'dense_images_new_' + cam_type)
        os.makedirs(out_dir, exist_ok=True)

        color_frames = []
        for i in tqdm(range(dense_pose_sampler.n_poses)):
            pose = dense_pose_sampler.sample_pose(i)
            if cam_type == 'pano':
                pose[:3, :3] = torch.eye(3)
                rays = gen_pano_rays(pose, 512, 1024)
            else:
                rays = gen_pers_rays(pose, fov=np.deg2rad(75.), res=512)

            with torch.no_grad():
                render_result = self.scene.render(rays, query_keys=['rgb', 'distance'])
            colors = render_result['rgb']
            distances = render_result['distance']

            color_frames.append((colors.clip(0., 1.) * 255.).cpu().numpy().astype(np.uint8))
            write_image(pjoin(out_dir, 'image_{}.png'.format(i)), colors * 255.)
            write_image(pjoin(out_dir, 'distance_{}.png'.format(i)), colorize_single_channel_image(1. / distances))
        
        write_video(pjoin(out_dir, 'video.mp4'), color_frames, fps=30)

    def save_checkpoint(self):
        checkpoint = {
            'scene': self.scene.state_dict(),
            'sup_pool': self.sup_pool.state_dict(),
            'phase': self.phase
        }

        os.makedirs(os.path.join(self.exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.exp_dir, 'checkpoints', 'ckpt.pth'))

    @torch.no_grad()
    def extract_panorama(self, out_dir="anchor_images"):
        # breakpoint()
        device = self.device
        ratios=[0.2, 0.4, 0.6]
        # target = self.pose_sampler_0.anchor_pts
        
        # for i, t in enumerate(target):
        for ratio in ratios:
            sampler = CirclePoseSampler(self.dataset.ref_distances[0], [ratio], [8])
            target = sampler.anchor_pts
            ratio_dir = pjoin(self.exp_dir, out_dir, f"{ratio}")

            os.makedirs(pjoin(self.exp_dir, ratio_dir, 'middle'), exist_ok=True)
            os.makedirs(pjoin(self.exp_dir, ratio_dir, 'up'), exist_ok=True)
            os.makedirs(pjoin(self.exp_dir, ratio_dir, 'down'), exist_ok=True)

            for i, t in enumerate(target):
                pose = torch.eye(4, device=device)
                pose[:3, 3] = t

                render_result = self.scene.render(
                    gen_pano_rays(pose, 512, 1024),
                    query_keys=['rgb', 'distance']
                )

                pano_rgb = render_result['rgb']

                x, y, z = t.tolist()
                filename = f"anchor_{i}_{x:.3f}_{y:.3f}_{z:.3f}.png"

                if abs(z) < 1e-6:
                    path = pjoin(ratio_dir, 'middle', filename)
                elif z > 0:
                    path = pjoin(ratio_dir, 'up', filename)
                else:
                    path = pjoin(ratio_dir, 'down', filename)

                write_image(path, pano_rgb.clip(0., 1.) * 255.)
                print(f"Wrote {filename}")

    def render_specific(self):
        device = self.device
        out_dir = pjoin(self.exp_dir, 'specific_images')
        os.makedirs(out_dir, exist_ok=True)

        folder_name = pjoin(out_dir, 'x_n y_n')
        os.makedirs(folder_name, exist_ok=True)

        out_rgb = pjoin(folder_name, 'rgb')
        out_distance = pjoin(folder_name, 'distance')
        os.makedirs(out_rgb, exist_ok=True)
        os.makedirs(out_distance, exist_ok=True)
        
        x = np.arange(0, 0.1, 0.01)
        y = np.arange(0, 0.1, 0.01)
        for ys in y:
            for xs in x:
                position=[-xs, -ys, 0.0000]
                path_rgb = pjoin(out_rgb, f'rgb_{position}.png')
                path_distance = pjoin(out_distance, f'distance_{position}.png')
                pose = torch.eye(4, device=device)
                pose[:3, 3] = torch.as_tensor(position, device=device)

                render_result = self.scene.render(
                gen_pano_rays(pose, self.dataset.height, self.dataset.width),
                query_keys=['rgb', 'distance']  
                )

                pano_rgb = render_result['rgb']
                pano_distance = render_result['distance']

                write_image(path_rgb, pano_rgb.clip(0., 1.) * 255.)
                write_image(path_distance, colorize_single_channel_image(pano_distance.clip(0., 1.) * 255.))

                print(f"Wrote {position}")
        
    def matching_mse(self):
        device = self.device
        out_dir = pjoin(self.exp_dir, 'matching_mse')
        os.makedirs(out_dir, exist_ok=True)

        target_dir = pjoin(self.exp_dir, 'specific_images', 'x positive', 'rgb')
        target_files = sorted(glob(pjoin(target_dir, '*.png')))

        #x positive : row2~row6
        bmw_origin_dir = pjoin(self.exp_dir, 'bmw_origin', 'x positive')
        bmw_origin_files = glob(pjoin(bmw_origin_dir, '*.jpg'))
        bmw_origin_files.sort()

        # def mse(img1, img2):
        #     value = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
        #     value /= float(img1.shape[0] * img1.shape[1])
        #     return value

        def mse(img1, img2):
            value = torch.mean((img1.float() - img2.float()) ** 2)
            return value.item()
        
        for origin_file in bmw_origin_files:
            mse_list = []
            origin = read_image(origin_file, to_torch=True, squeeze=True).cuda()
            origin = cv.resize(origin.cpu().numpy(), (self.dataset.width, self.dataset.height), interpolation=cv.INTER_AREA)
            origin = torch.from_numpy(origin).cuda()
            for target_file in target_files:
                target = read_image(target_file, to_torch=True, squeeze=True).cuda()
                target = cv.resize(
                    target.cpu().numpy(),
                    (self.dataset.width, self.dataset.height),
                    interpolation=cv.INTER_AREA
                )
                target = torch.from_numpy(target).cuda()

                mse_result = mse(origin, target)
                mse_list.append(mse_result)

            min_mse = min(mse_list)
            min_mse_index = mse_list.index(min_mse)

            print(f"origin file: {os.path.basename(origin_file)}")
            print(f"    minimum MSE: {min_mse}")
            print(f"    minimum MSE file: {os.path.basename(target_files[min_mse_index])}")

    @torch.no_grad()
    def render_middle_view(self, n_frames=120, height=512, width=1024, fps=30):
        """이미지 0번 카메라 pose → 1번 pose로 단조 증가하는 보간 경로를 따라 연속 렌더링 후 mp4 저장.

        - α ∈ [0,1]: k=0일 때 정확히 poses[0], k=n_frames-1일 때 정확히 poses[1] (데이터셋 텐서 그대로 사용).
        - 중간 프레임: 회전 scipy SLERP(SO(3)), 평행이동 선형 보간.
        학습된 가중치가 필요하면 is_continue 등으로 ckpt 로드 권장.
        """
        from scipy.spatial.transform import Rotation as R_scipy
        from scipy.spatial.transform import Slerp

        if self.dataset.n_images < 2:
            raise ValueError('render_middle_view needs at least 2 input images.')

        pose_start = self.dataset.poses[0].detach()
        pose_end = self.dataset.poses[1].detach()
        device = pose_start.device
        dtype = pose_start.dtype

        pose0_np = pose_start.cpu().numpy().astype(np.float64)
        pose1_np = pose_end.cpu().numpy().astype(np.float64)
        R0, R1 = pose0_np[:3, :3], pose1_np[:3, :3]
        t0, t1 = pose0_np[:3, 3], pose1_np[:3, 3]

        key_rots = R_scipy.from_matrix(np.stack([R0, R1]))
        slerp = Slerp([0.0, 1.0], key_rots)

        out_dir = pjoin(self.exp_dir, 'interp_pose_video')
        os.makedirs(out_dir, exist_ok=True)
        color_frames = []

        denom = max(n_frames - 1, 1)
        for k in tqdm(range(n_frames), desc='interp_pose_render'):
            if k == 0:
                pose = pose_start.clone().to(device=device, dtype=dtype)
            elif k == n_frames - 1:
                pose = pose_end.clone().to(device=device, dtype=dtype)
            else:
                alpha = float(k) / denom
                R_i = slerp(alpha).as_matrix().astype(np.float32)
                t_i = ((1.0 - alpha) * t0 + alpha * t1).astype(np.float32)

                pose = torch.eye(4, device=device, dtype=dtype)
                pose[:3, :3] = torch.from_numpy(R_i).to(device=device, dtype=dtype)
                pose[:3, 3] = torch.from_numpy(t_i).to(device=device)

            rays = gen_pano_rays(pose, height, width)
            render_result = self.scene.render(rays, query_keys=['rgb', 'distance'])
            colors = render_result['rgb'].clip(0., 1.)
            color_frames.append((colors * 255.).cpu().numpy().astype(np.uint8))
            write_image(pjoin(self.exp_dir, out_dir, f'frame_{k:05d}.png'), colors * 255.)

        vid_path = pjoin(out_dir, 'pose_interp_0_to_1.mp4')
        write_video(vid_path, color_frames, fps=fps)
        print(f'Saved {vid_path} ({n_frames} frames @ {fps} fps)')


@hydra.main(version_base=None, config_path='./configs', config_name='nerf')
def main(conf: DictConfig) -> None:
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    mode = str(conf['mode'])
    runner = CoreRunner(conf)
    runner.set_eval()

    runner.execute(mode)


if __name__ == '__main__':
    main()
