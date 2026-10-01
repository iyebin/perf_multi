import torch
import torch.nn.functional as F
import cv2 as cv
from tqdm import tqdm
from kornia.filters import laplacian, sobel, gaussian_blur2d

import numpy as np
from trimesh.creation import icosphere as IcoSphere
from scipy.spatial.transform import Rotation

from .geo_predictor import GeoPredictor
from .omnidata_normal_predictor import OmnidataNormalPredictor

from utils.geo_utils import panorama_to_pers_directions
from utils.camera_utils import *
from utils.utils import write_image
from utils.debug_utils import printarr


def scale_unit(x):
    return (x - x.min()) / (x.max() - x.min())


class PanoFusionNormalPredictor(GeoPredictor):
    def __init__(self):
        super().__init__()
        self.normal_predictor = OmnidataNormalPredictor()

    def inpaint_normal(self, img, ref_normal, mask, gen_res=384):
        '''
        Do not support batch operation - can only inpaint one image at a time.
        :param img: [H, W, 3]
        :param ref_normal: [H, W, 3]
        :param mask: [H, W] or [H, W, 1], value 1 if need to be inpainted otherwise 0
        :return: inpainted normal [H, W, 3]
        '''

        device = img.device
        img = img.clone().squeeze().permute(2, 0, 1)                   # [3, H, W]
        mask = mask.clone().squeeze()[None]                            # [1, H, W]
        ref_normal = ref_normal.squeeze().clone().permute(2, 0, 1)     # [3, H, W]

        pers_dirs, pers_ratios, to_vecs, down_vecs, right_vecs = panorama_to_pers_directions(gen_res=gen_res, ratio=1.0)

        fx = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(right_vecs, 2, -1, True) * gen_res * .5
        fy = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(down_vecs, 2, -1, True) * gen_res * .5
        cx = torch.ones_like(fx) * gen_res * .5
        cy = torch.ones_like(fy) * gen_res * .5

        pers_dirs = pers_dirs.to(device)
        to_vecs = to_vecs.to(device)
        down_vecs = down_vecs.to(device)
        right_vecs = right_vecs.to(device)

        # pts = torch.cat([
        #     to_vecs[1:4],
        #     to_vecs[1:4] + down_vecs[1: 4] * 0.1,
        #     # to_vecs[1:4] + right_vecs[1: 4] * 0.1
        # ], dim=0)
        # import trimesh
        # pcd = trimesh.PointCloud(pts.cpu().numpy())
        # pcd.export('/home/ppwang/tmp/tmp.ply')
        # exit()

        rot_w2c = torch.stack([right_vecs / torch.linalg.norm(right_vecs, 2, -1, True),
                               down_vecs / torch.linalg.norm(down_vecs, 2, -1, True),
                               to_vecs / torch.linalg.norm(to_vecs, 2, -1, True)],
                               dim=1)
        rot_c2w = torch.linalg.inv(rot_w2c)

        n_pers = len(pers_dirs)
        img_coords = direction_to_img_coord(pers_dirs)
        sample_coords = img_coord_to_sample_coord(img_coords)

        _, pano_height, pano_width = img.shape
        pano_img_coords = torch.meshgrid(torch.linspace(.5 / pano_height, 1. - .5 / pano_height, pano_height),
                                         torch.linspace(.5 / pano_width, 1. - .5 / pano_width, pano_width),
                                         indexing='ij')
        pano_img_coords = torch.stack(list(pano_img_coords), dim=-1)
        pano_coord = img_to_pano_coord(pano_img_coords)
        distortion_weights = torch.cos(pano_coord[:, :, 0])
        pano_dirs = img_coord_to_pano_direction(pano_img_coords)
        pers_imgs = F.grid_sample(img[None].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border') # [n_pers, 3, gen_res, gen_res]
        pred_normals_raw = []

        for i in range(n_pers):
            with torch.no_grad():
                intri = {
                    'fx': fx[i].item(),
                    'fy': fy[i].item(),
                    'cx': cx[i].item(),
                    'cy': cy[i].item()
                }
                pred_normal = self.normal_predictor.predict_normal(pers_imgs[i: i+1]).cuda()
                # write_image('/home/ppwang/tmp/normal/{}_image.png'.format(i), pers_imgs[i].permute(1, 2, 0) * 255.)
                # write_image('/home/ppwang/tmp/normal/{}_normal.png'.format(i), pred_normal[0].permute(1, 2, 0) * 255.)
                pred_normal = pred_normal * 2. - 1.
                pred_normal = pred_normal / torch.linalg.norm(pred_normal, ord=2, dim=1, keepdim=True)

                # OpenGL -> OpenCV
                # pred_normal = torch.stack([pred_normal[:, 0], -pred_normal[:, 1], -pred_normal[:, 2]], dim=1)  # [1, 3, res, res]
                pred_normal = pred_normal.permute(0, 2, 3, 1)  # [1, res, res, 3]
                pred_normal = apply_rot(pred_normal, rot_c2w[i])
                pred_normal = pred_normal.permute(0, 3, 1, 2)  # [1, 3, res, res]
                pred_normals_raw.append(pred_normal)

        pred_normals_raw = torch.cat(pred_normals_raw, dim=0)

        proj_coords = []
        proj_masks  = []

        for i in range(n_pers):
            proj_coord, proj_mask = direction_to_pers_img_coord(pano_dirs, to_vecs[i], down_vecs[i], right_vecs[i])
            proj_coord = img_coord_to_sample_coord(proj_coord)
            proj_coords.append(proj_coord[None])
            proj_masks.append(proj_mask[None])

        proj_coords = torch.cat(proj_coords, dim=0)   # [n_pers, pano_height, pano_width, 2]
        proj_masks = torch.cat(proj_masks, dim=0)     # [n_pers, pano_height, pano_width, 1]
        proj_coords = proj_coords.clip(-1., 1.)       # not necessary
        proj_masks = proj_masks.permute(0, 3, 1, 2)

        bias_params = torch.zeros([n_pers, 3, gen_res, gen_res], requires_grad=True)

        # pano_normal_params = torch.zeros([3, pano_height, pano_width], requires_grad=True)
        pano_normal_params = -img_coord_to_pano_direction(img_coord_from_hw(pano_height, pano_width))
        pano_normal_params = pano_normal_params.permute(2, 0, 1).contiguous()
        pano_normal_params.requires_grad_(True)

        all_iter_steps = 1000
        lr_alpha = 5e-3
        init_lr = 2e-1

        optimizer_a = torch.optim.Adam([pano_normal_params], lr=init_lr)
        optimizer_b = torch.optim.Adam([bias_params, pano_normal_params], lr=init_lr)

        for iter_step in tqdm(range(all_iter_steps)):
            if iter_step * 2 < all_iter_steps:
                optimizer = optimizer_a
                lr_progress = iter_step * 2 / all_iter_steps
            else:
                optimizer = optimizer_b
                lr_progress = (iter_step - all_iter_steps / 2) / (all_iter_steps / 2)

            lr_ratio = (np.cos(lr_progress * np.pi) + 1.) * (1. - lr_alpha) + lr_alpha
            for g in optimizer.param_groups:
                g['lr'] = init_lr * lr_ratio

            ii, jj = torch.meshgrid(torch.linspace(-1., 1., gen_res),
                                    torch.linspace(-1., 1., gen_res))

            pano_normal = pano_normal_params * mask + ref_normal * (1. - mask)
            pano_normal_norm = torch.linalg.norm(pano_normal, ord=2, dim=0, keepdim=True)
            pano_normal = pano_normal / pano_normal_norm

            pred_normals = (pred_normals_raw + bias_params)
            pred_normals_norm = torch.linalg.norm(pred_normal, ord=2, dim=1, keepdim=True)
            pred_normals = pred_normals / pred_normals_norm
            pred_normals = F.grid_sample(pred_normals, proj_coords, padding_mode='border')

            #debug
            # pred_normals = pred_normals * proj_masks
            # for i in range(20):
            #     write_image('/home/ppwang/tmp/normal/pano_{}.png'.format(i), (pred_normals[i].permute(1, 2, 0) * .5 + .5)[..., 0:1] * 255.)
            # print(pred_normals.shape)
            # exit()

            align_error = (pred_normals - pano_normal[None]) * proj_masks
            align_error = F.smooth_l1_loss(align_error, torch.zeros_like(align_error), beta=0.5, reduction='none')
            weight_masks = proj_masks * distortion_weights[None, None]
            align_loss = (align_error * weight_masks).sum() / weight_masks.sum()

            bias_tv_loss = F.smooth_l1_loss(bias_params[:, :, 1:, :], bias_params[:, :, :-1, :], beta=0.5) + \
                           F.smooth_l1_loss(bias_params[:, :, :, 1:], bias_params[:, :, :, :-1], beta=0.5)

            reg_loss = ((pano_normal_norm - 1.)**2).mean() + ((pred_normals_norm - 1.)**2).mean()
            loss = align_loss + bias_tv_loss * 1. + reg_loss * 1e-2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pano_normal = pano_normal_params * mask + ref_normal * (1. - mask)

        # 최종 normal normalize
        pano_normal = pano_normal / (
            torch.linalg.norm(pano_normal, ord=2, dim=0, keepdim=True) + 1e-8
        )

        # [3, H, W] -> [H, W, 3]
        pano_normal = pano_normal.detach().permute(1, 2, 0).contiguous()

        # normal + block 평균값 시각화
        normal_vis = self.visualize_normal_block_mean(
            pano_normal,
            block_size=28,
            decimals=2,
            font_scale=0.4,
            thickness=1,
            draw_grid=True,
            scale=4,
        )

        cv.imwrite(
            "/data/intern01/normal_block_mean.png",
            normal_vis
        )

        return pano_normal
        # return pano_normal.detach().permute(1, 2, 0).contiguous()
    @staticmethod
    def visualize_normal_block_mean(
        normal,
        block_size=28, decimals=2,
        font_scale=0.8, thickness=1,
        draw_grid=True, scale=4,
    ):
        if isinstance(normal, torch.Tensor):
            normal = normal.detach().cpu().numpy()

        normal = normal.astype(np.float32)

        H, W, C = normal.shape
        assert C == 3, f"Expected (H, W, 3), got {normal.shape}"

        # normal [-1, 1] → RGB [0, 255] → BGR
        vis = np.clip((normal + 1.0) * 127.5, 0, 255).astype(np.uint8)
        vis = cv.cvtColor(vis, cv.COLOR_RGB2BGR)
        vis = cv.resize(
            vis, (W * scale, H * scale),
            interpolation=cv.INTER_NEAREST
        )

        font = cv.FONT_HERSHEY_SIMPLEX

        for y in range(0, H, block_size):
            for x in range(0, W, block_size):
                y2 = min(y + block_size, H)
                x2 = min(x + block_size, W)

                vecs = normal[y:y2, x:x2].reshape(-1, 3)
                vecs = vecs[np.isfinite(vecs).all(axis=1)]

                if len(vecs) == 0:
                    continue

                mean_n = vecs.mean(axis=0)
                norm = np.linalg.norm(mean_n)

                if norm < 1e-6:
                    continue

                nx, ny, nz = mean_n / norm
                text = f"{nx:+.{decimals}f},{ny:+.{decimals}f},{nz:+.{decimals}f}"

                x1s, y1s = x * scale, y * scale
                x2s, y2s = x2 * scale, y2 * scale
                cx, cy = (x1s + x2s) // 2, (y1s + y2s) // 2

                (tw, th), base = cv.getTextSize(
                    text, font, font_scale, thickness
                )

                tx = int(np.clip(
                    cx - tw / 2,
                    0,
                    W * scale - tw - 1
                ))
                ty = int(np.clip(
                    cy + th / 2,
                    th,
                    H * scale - base - 1
                ))

                cv.rectangle(
                    vis,
                    (tx - 2, ty - th - 2),
                    (tx + tw + 2, ty + base + 2),
                    (0, 0, 0),
                    -1
                )

                cv.putText(
                    vis, text, (tx, ty),
                    font, font_scale,
                    (255, 255, 255),
                    thickness,
                    cv.LINE_AA
                )

                if draw_grid:
                    cv.rectangle(
                        vis,
                        (x1s, y1s),
                        (x2s - 1, y2s - 1),
                        (40, 40, 40),
                        1
                    )

        return vis


