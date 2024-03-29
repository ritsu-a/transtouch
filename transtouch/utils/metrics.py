import os
import os.path as osp

_ROOT_DIR = os.path.abspath(osp.join(osp.dirname(__file__), "../.."))

import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as conf_mapF
from path import Path
from tabulate import tabulate

from transtouch.models.build_model import MODEL_LIST
from transtouch.utils.geometry import cal_normal_map, depth2pts_np
from transtouch.utils.reprojection import apply_disparity, apply_disparity_v2

with open(osp.join(_ROOT_DIR, "data_rendering/materials/objects_v2.csv"), "r") as f:
    OBJECT_INFO = csv.reader(f)
    OBJECT_INFO = list(OBJECT_INFO)[1:]
    OBJECT_NAMES = [_[0] for _ in OBJECT_INFO]

TRANSPARENT_OBJECTS = [
    "beaker_big",
    "beaker_small",
    "flask1",
    "flask2",
    "flask3",
    "flask4",
    "funnel",
    "jack_daniels",
    "round_bottom_flask",
    "spellegrino",
    "voss",
]

SPECULAR_OBJECTS = [
    "coca_cola",
    "coffee_cup",
    "gold_ball",
    "steel_ball",
]

DIFFUSE_OBJECTS = [
    "beer_can",
    "camera",
    "cellphone",
    "champagne",
    "coke_bottle",
    "hammer",
    "pepsi_bottle",
    "rubik",
    "sharpener",
    "tennis_ball",
]


class ErrorMetric(object):
    def __init__(
        self,
        model_type: str,
        use_mask: bool = True,
        max_disp: int = 192,
        depth_range=(0.2, 2.0),
        num_classes: int = 25,
        is_depth: bool = False,
        realsense_mask: bool = False,
    ):
        assert (
            model_type in ("RealSense", "Inpainting", "Depth2Depth", "RealSense", "Cleargrasp", "Transcg") + MODEL_LIST
        ), f"Unknown model type: [{model_type}]"
        self.model_type = model_type
        self.use_mask = use_mask
        self.max_disp = max_disp
        self.is_depth = is_depth
        self.depth_range = depth_range
        self.num_classes = num_classes
        assert len(OBJECT_NAMES) == num_classes
        # load real robot masks
        mask_dir = Path(_ROOT_DIR) / "transtouch/assets/real_robot_masks"
        self.real_robot_masks = {}
        for mask_file in sorted(mask_dir.listdir("m*.png")):
            rgb_mask = cv2.imread(mask_file)
            rgb_mask = cv2.resize(rgb_mask[:, :, 0], (960, 540))
            mask = (rgb_mask == rgb_mask.max())
            self.real_robot_masks[mask_file.name[1:-4]] = mask
        # load realsense mask
        self.realsense_mask = realsense_mask
        if realsense_mask:
            self.realsense_mask_dir = Path(_ROOT_DIR) / "trantouch/assets/realsense_masks"
            self.realsense_masks = {}
            assert self.realsense_mask_dir.exists()
            for mask_file in sorted(self.realsense_mask_dir.listdir("*.png")):
                mask = cv2.imread(mask_file, flags=cv2.IMREAD_GRAYSCALE)
                mask = mask > 0
                self.realsense_masks[mask_file.name[:-4]] = mask

        self.cmap = plt.get_cmap("jet")

        # cutoff threshold
        self.disp_diff_threshold = 8
        self.depth_diff_threshold = 32e-3

    def reset(self):
        self.mean_epe = []
        self.object_epe = []
        self.diffuse_epe = []
        self.specular_epe = []
        self.transparent_epe = []
        self.table_epe = []

        self.mean_weight = []
        self.object_weight = []
        self.diffuse_weight = []
        self.specular_weight = []
        self.transparent_weight = []
        self.table_weight = []

        self.mean_bad1 = []
        self.object_bad1 = []
        self.diffuse_bad1 = []
        self.specular_bad1 = []
        self.transparent_bad1 = []
        self.table_bad1 = []

        self.mean_bad2 = []
        self.object_bad2 = []
        self.diffuse_bad2 = []
        self.specular_bad2 = []
        self.transparent_bad2 = []
        self.table_bad2 = []

        self.bad1 = []
        self.bad2 = []
        self.depth_abs_err = []
        self.delta_05 = []
        self.delta_10 = []
        self.delta_15 = []
        self.depth_err2 = []
        self.depth_err4 = []
        self.depth_err8 = []
        self.depth_err16 = []
        self.depth_err32 = []
        self.normal_err = []
        self.normal_err10 = []
        self.normal_err20 = []
        self.obj_disp_err = np.zeros(self.num_classes)
        self.obj_bad1 = np.zeros(self.num_classes)
        self.obj_bad2 = np.zeros(self.num_classes)
        self.obj_depth_err = np.zeros(self.num_classes)
        self.obj_delta_05 = np.zeros(self.num_classes)
        self.obj_delta_10 = np.zeros(self.num_classes)
        self.obj_delta_15 = np.zeros(self.num_classes)
        self.obj_depth_err4 = np.zeros(self.num_classes)
        self.obj_depth_err8 = np.zeros(self.num_classes)
        self.obj_depth_err16 = np.zeros(self.num_classes)
        self.obj_depth_err32 = np.zeros(self.num_classes)
        self.obj_normal_err = np.zeros(self.num_classes)
        self.obj_normal_err10 = np.zeros(self.num_classes)
        self.obj_count = np.zeros(self.num_classes)
        self.transparent_disp_err = 0.0
        self.transparent_bad1 = 0.0
        self.transparent_bad2 = 0.0
        self.transparent_depth_err = 0.0
        self.transparent_delta_05 = 0.0
        self.transparent_delta_10 = 0.0
        self.transparent_delta_15 = 0.0
        self.transparent_depth_err4 = 0.0
        self.transparent_depth_err8 = 0.0
        self.transparent_depth_err16 = 0.0
        self.transparent_depth_err32 = 0.0
        self.transparent_normal_err = 0.0
        self.transparent_normal_err10 = 0.0
        self.transparent_count = 0
        self.specular_disp_err = 0.0
        self.specular_bad1 = 0.0
        self.specular_bad2 = 0.0
        self.specular_depth_err = 0.0
        self.specular_delta_05 = 0.0
        self.specular_delta_10 = 0.0
        self.specular_delta_15 = 0.0
        self.specular_depth_err4 = 0.0
        self.specular_depth_err8 = 0.0
        self.specular_depth_err16 = 0.0
        self.specular_depth_err32 = 0.0
        self.specular_normal_err = 0.0
        self.specular_normal_err10 = 0.0
        self.specular_count = 0
        self.diffuse_disp_err = 0.0
        self.diffuse_bad1 = 0.0
        self.diffuse_bad2 = 0.0
        self.diffuse_depth_err = 0.0
        self.diffuse_delta_05 = 0.0
        self.diffuse_delta_10 = 0.0
        self.diffuse_delta_15 = 0.0
        self.diffuse_depth_err4 = 0.0
        self.diffuse_depth_err8 = 0.0
        self.diffuse_depth_err16 = 0.0
        self.diffuse_depth_err32 = 0.0
        self.diffuse_normal_err = 0.0
        self.diffuse_normal_err10 = 0.0
        self.diffuse_count = 0

    def compute(self, data_batch, pred_dict, save_folder="", real_data=False):
        """
        Compute the error metrics for predicted disparity map or depth map
        """
        if "-300" in data_batch["dir"] and real_data:
            assert self.use_mask, "Use_mask should be True when evaluating real data"

        left_right = False
        if self.model_type in (
            "PSMNetConfidence",
            "PSMNetDilation",
        ):
            prediction = pred_dict["pred3"]
            prediction = prediction.detach().cpu().numpy()[0, 0, 2:-2]

 
            # prediction_r = pred_dict["pred3_r"]
            # prediction_r = prediction_r.detach().cpu().numpy()[0, 0, 2:-2]
            # left_right = True
        elif self.model_type in (
            "PSMNetEdgeNormal",
        ):
            prediction = pred_dict["pred3"]
            prediction = prediction.detach().cpu().numpy()[0, 0, 2:-2]
        elif self.model_type in (
            "RealSense",
            "Cleargrasp",
            "Transcg"
        ):
            prediction = pred_dict["depth"]

        focal_length = data_batch["focal_length"][0].cpu().numpy()
        baseline = data_batch["baseline"][0].cpu().numpy()
        disp_gt = data_batch["img_disp_l"].cpu().numpy()[0, 0, 2:-2]
        depth_gt = data_batch["img_depth_l"].cpu().numpy()[0, 0, 2:-2]

        
        if self.is_depth:
            disp_pred = focal_length * baseline / (prediction + 1e-7)

            depth_pred = prediction
            if left_right:
                disp_pred_r = focal_length * baseline / (prediction_r + 1e-7)
        else:
            depth_pred = focal_length * baseline / (prediction + 1e-7)
            disp_pred = prediction
            if left_right:
                disp_pred_r = prediction_r


        if self.use_mask:
            mask = np.logical_and(disp_gt > 1e-1, disp_gt < self.max_disp)
            x_base = np.arange(0, disp_gt.shape[1]).reshape(1, -1)
            mask = np.logical_and(mask, x_base > disp_gt)
            mask = np.logical_and(mask, depth_gt > self.depth_range[0])
            mask = np.logical_and(mask, depth_gt < self.depth_range[1])
            if real_data:
                view_id = data_batch["dir"].split("-")[-1]
                mask = np.logical_and(mask, np.logical_not(self.real_robot_masks[view_id]))
            if self.realsense_mask:
                assert "-300" in data_batch["dir"] and real_data
                scene = str(data_batch["dir"])
                if self.model_type == "RealSense" and scene not in self.realsense_masks:
                    # save realsense mask
                    realsense_mask = depth_pred > self.depth_range[0]
                    realsense_mask = np.logical_and(realsense_mask, depth_pred < self.depth_range[1])
                    self.realsense_masks[scene] = realsense_mask
                    cv2.imwrite(self.realsense_mask_dir / f"{scene}.png", (realsense_mask.astype(np.uint8) * 255))
                mask = np.logical_and(mask, self.realsense_masks[scene])
        else:
            mask = np.ones_like(disp_gt).astype(np.bool)

        disp_diff = disp_gt - disp_pred
        depth_diff = depth_gt - depth_pred
        disp_diff = np.clip(disp_diff, -self.disp_diff_threshold, self.disp_diff_threshold)
        depth_diff = np.clip(depth_diff, -self.depth_diff_threshold, self.depth_diff_threshold)

        
        mean_epe = 0
        object_epe = 0
        diffuse_epe = 0
        specular_epe = 0
        transparent_epe = 0

        all_object_mask = np.zeros_like(disp_diff).astype(bool)
        diffuse_mask = np.zeros_like(disp_diff).astype(bool)
        specular_mask = np.zeros_like(disp_diff).astype(bool)
        transparent_mask = np.zeros_like(disp_diff).astype(bool)

        bad1 = (np.abs(disp_diff[mask]) > 1).sum() / mask.sum()
        bad2 = (np.abs(disp_diff[mask]) > 2).sum() / mask.sum()

        depth_abs_err = np.abs(depth_diff[mask]).mean()
        depth_err2 = (np.abs(depth_diff[mask]) > 2e-3).sum() / mask.sum()
        depth_err4 = (np.abs(depth_diff[mask]) > 4e-3).sum() / mask.sum()
        depth_err8 = (np.abs(depth_diff[mask]) > 8e-3).sum() / mask.sum()
        depth_err16 = (np.abs(depth_diff[mask]) > 16e-3).sum() / mask.sum()
        depth_err32 = (np.abs(depth_diff[mask]) > 32e-3).sum() / mask.sum()

        delta_05 = ((np.abs(depth_diff[mask]) / depth_gt[mask]) < 0.05).sum() / mask.sum()
        delta_10 = ((np.abs(depth_diff[mask]) / depth_gt[mask]) < 0.10).sum() / mask.sum()
        delta_15 = ((np.abs(depth_diff[mask]) / depth_gt[mask]) < 0.15).sum() / mask.sum()


        if "img_normal_l" in data_batch and "intrinsic_l" in data_batch:
            normal_gt = data_batch["img_normal_l"][0].permute(1, 2, 0).cpu().numpy()[2:-2]
            valid_mask = np.abs(normal_gt).sum(-1) > 0
            if self.use_mask:
                valid_mask = np.logical_and(valid_mask, mask)
            invalid_mask = np.logical_not(valid_mask)
            intrinsic_l = data_batch["intrinsic_l"][0].cpu().numpy().copy()
            intrinsic_l[1, 2] -= 2
            normal_pred = cal_normal_map(depth_pred, intrinsic_l, radius=0.005, max_nn=100)
            normal_diff = np.arccos(np.clip(np.sum(normal_gt * normal_pred, axis=-1), -1, 1))
            normal_diff[invalid_mask] = 0
            normal_err = normal_diff.sum() / valid_mask.sum()
            normal_err10 = (normal_diff > 10 / 180 * np.pi).sum() / valid_mask.sum()
            normal_err20 = (normal_diff > 20 / 180 * np.pi).sum() / valid_mask.sum()
            self.normal_err.append(normal_err)
            self.normal_err10.append(normal_err10)
            self.normal_err20.append(normal_err20)

        if "img_label_l" in data_batch:
            label_l = data_batch["img_label_l"].cpu().numpy()[0][2:-2]
            
            for i in range(self.num_classes):
                obj_mask = label_l == i
                all_object_mask += obj_mask
                if self.use_mask:
                    obj_mask = np.logical_and(obj_mask, mask)
                if obj_mask.sum() > 0:
                    self.obj_count[i] += 1
                    self.obj_disp_err[i] += np.abs(disp_diff[obj_mask]).mean()
                    self.obj_bad1[i] += (np.abs(disp_diff[obj_mask]) > 1).sum() / obj_mask.sum()
                    self.obj_bad2[i] += (np.abs(disp_diff[obj_mask]) > 2).sum() / obj_mask.sum()
                    self.obj_delta_05[i] += (np.abs(depth_diff[obj_mask]) / depth_gt[obj_mask] < 0.05).sum() / obj_mask.sum()
                    self.obj_delta_10[i] += (np.abs(depth_diff[obj_mask]) / depth_gt[obj_mask] < 0.10).sum() / obj_mask.sum()
                    self.obj_delta_15[i] += (np.abs(depth_diff[obj_mask]) / depth_gt[obj_mask] < 0.15).sum() / obj_mask.sum()

                    self.obj_depth_err[i] += np.abs(depth_diff[obj_mask]).mean()
                    self.obj_depth_err4[i] += (np.abs(depth_diff[obj_mask]) > 4e-3).sum() / obj_mask.sum()
                    self.obj_depth_err8[i] += (np.abs(depth_diff[obj_mask]) > 8e-3).sum() / obj_mask.sum()
                    self.obj_depth_err16[i] += (np.abs(depth_diff[obj_mask]) > 16e-3).sum() / obj_mask.sum()
                    self.obj_depth_err32[i] += (np.abs(depth_diff[obj_mask]) > 32e-3).sum() / obj_mask.sum()

                    if OBJECT_NAMES[i] in TRANSPARENT_OBJECTS:
                        transparent_mask += obj_mask
                    elif OBJECT_NAMES[i] in DIFFUSE_OBJECTS:
                        diffuse_mask += obj_mask
                    elif OBJECT_NAMES[i] in SPECULAR_OBJECTS:
                        specular_mask += obj_mask



                    if OBJECT_NAMES[i] in DIFFUSE_OBJECTS:
                        self.diffuse_count += 1
                        self.diffuse_disp_err += np.abs(disp_diff[obj_mask]).mean()
                        self.diffuse_bad1 += (np.abs(disp_diff[obj_mask]) > 1).sum() / obj_mask.sum()
                        self.diffuse_bad2 += (np.abs(disp_diff[obj_mask]) > 2).sum() / obj_mask.sum()
                        self.diffuse_delta_05 += (np.abs(depth_diff[obj_mask]) / depth_gt[obj_mask] < 0.05).sum() / obj_mask.sum()
                        self.diffuse_delta_10 += (np.abs(depth_diff[obj_mask]) / depth_gt[obj_mask] < 0.10).sum() / obj_mask.sum()
                        self.diffuse_delta_15 += (np.abs(depth_diff[obj_mask]) / depth_gt[obj_mask] < 0.15).sum() / obj_mask.sum()
                        self.diffuse_depth_err += np.abs(depth_diff[obj_mask]).mean()
                        self.diffuse_depth_err4 += (np.abs(depth_diff[obj_mask]) > 4e-3).sum() / obj_mask.sum()
                        self.diffuse_depth_err8 += (np.abs(depth_diff[obj_mask]) > 8e-3).sum() / obj_mask.sum()
                        self.diffuse_depth_err16 += (np.abs(depth_diff[obj_mask]) > 16e-3).sum() / obj_mask.sum()
                        self.diffuse_depth_err32 += (np.abs(depth_diff[obj_mask]) > 32e-3).sum() / obj_mask.sum()
                    elif OBJECT_NAMES[i] in SPECULAR_OBJECTS:
                        self.specular_count += 1
                        self.specular_disp_err += np.abs(disp_diff[obj_mask]).mean()
                        self.specular_bad1 += (np.abs(disp_diff[obj_mask]) > 1).sum() / obj_mask.sum()
                        self.specular_bad2 += (np.abs(disp_diff[obj_mask]) > 2).sum() / obj_mask.sum()
                        self.specular_delta_05 += (np.abs(depth_diff[obj_mask]) / depth_gt[obj_mask] < 0.05).sum() / obj_mask.sum()
                        self.specular_delta_10 += (np.abs(depth_diff[obj_mask]) / depth_gt[obj_mask] < 0.10).sum() / obj_mask.sum()
                        self.specular_delta_15 += (np.abs(depth_diff[obj_mask]) / depth_gt[obj_mask] < 0.15).sum() / obj_mask.sum()
                        self.specular_depth_err += np.abs(depth_diff[obj_mask]).mean()
                        self.specular_depth_err4 += (np.abs(depth_diff[obj_mask]) > 4e-3).sum() / obj_mask.sum()
                        self.specular_depth_err8 += (np.abs(depth_diff[obj_mask]) > 8e-3).sum() / obj_mask.sum()
                        self.specular_depth_err16 += (np.abs(depth_diff[obj_mask]) > 16e-3).sum() / obj_mask.sum()
                        self.specular_depth_err32 += (np.abs(depth_diff[obj_mask]) > 32e-3).sum() / obj_mask.sum()
                    elif OBJECT_NAMES[i] in TRANSPARENT_OBJECTS:
                        self.transparent_count += 1
                        self.transparent_disp_err += np.abs(disp_diff[obj_mask]).mean()
                        self.transparent_bad1 += (np.abs(disp_diff[obj_mask]) > 1).sum() / obj_mask.sum()
                        self.transparent_bad2 += (np.abs(disp_diff[obj_mask]) > 2).sum() / obj_mask.sum()
                        self.transparent_delta_05 += (np.abs(depth_diff[obj_mask]) / depth_gt[obj_mask] <  0.05).sum() / obj_mask.sum()
                        self.transparent_delta_10 += (np.abs(depth_diff[obj_mask]) / depth_gt[obj_mask] <  0.10).sum() / obj_mask.sum()
                        self.transparent_delta_15 += (np.abs(depth_diff[obj_mask]) / depth_gt[obj_mask] <  0.15).sum() / obj_mask.sum()
                        self.transparent_depth_err += np.abs(depth_diff[obj_mask]).mean()
                        self.transparent_depth_err4 += (np.abs(depth_diff[obj_mask]) > 4e-3).sum() / obj_mask.sum()
                        self.transparent_depth_err8 += (np.abs(depth_diff[obj_mask]) > 8e-3).sum() / obj_mask.sum()
                        self.transparent_depth_err16 += (np.abs(depth_diff[obj_mask]) > 16e-3).sum() / obj_mask.sum()
                        self.transparent_depth_err32 += (np.abs(depth_diff[obj_mask]) > 32e-3).sum() / obj_mask.sum()


        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            plt.imsave(
                os.path.join(save_folder, "disp_pred.png"),
                np.clip(disp_pred, 0, self.max_disp),
                vmin=0.0,
                vmax=self.max_disp,
                cmap="jet",
            )
            # np.save(os.path.join(save_folder, "disp_pred.npy"), disp_pred)
            # np.save(os.path.join(save_folder, "depth_pred.npy"), depth_pred)
            # np.save(os.path.join(save_folder, "disp_gt.npy"), disp_gt)
            plt.imsave(os.path.join(save_folder, "disp_gt.png"), disp_gt, vmin=0.0, vmax=self.max_disp, cmap="jet")
            disp_err = - disp_diff + 1e5 * (1 - mask)
            plt.imsave(
                os.path.join(save_folder, "disp_err3.png"), disp_err, vmin=-8, vmax=8, cmap="jet"
            )

            
            disp_err_blur = cv2.GaussianBlur(disp_diff, (0, 0), 3, 3)

            disp_err_mask = (disp_err_blur > -4) * (disp_err_blur < 4) + (1 - mask)

            disp_err_mask = np.clip(disp_err_mask, 0, 1)
            
            plt.imsave(
                os.path.join(save_folder, "disp_err_mask2.png"), disp_err_mask, vmin=0, vmax=1, cmap="jet"
            )


            #plt.imsave(
            #    os.path.join(save_folder, "depth_err.png"),
            #    depth_diff - 1e5 * (1 - mask),
            #    vmin=-self.depth_diff_threshold,
            #    vmax=self.depth_diff_threshold,
            #    cmap="jet",
            #)

            # left right consistency
            if left_right:
                np.save(os.path.join(save_folder, "disp_pred_r.npy"), disp_pred_r)
                disp_pred_l_reprojed = apply_disparity_v2(
                    torch.tensor(disp_pred_r).unsqueeze(0).unsqueeze(0),
                    -torch.tensor(disp_pred).unsqueeze(0).unsqueeze(0),
                ).numpy()[0, 0]
                disp_pred_consistency = disp_pred - disp_pred_l_reprojed
                plt.imsave(
                    os.path.join(save_folder, "disp_pred_consistency.png"),
                    disp_pred_consistency,
                    vmin=-4,
                    vmax=4,
                    cmap="jet",
                )
                plt.imsave(
                    os.path.join(save_folder, "disp_pred_consistency_mask.png"),
                    np.abs(disp_pred_consistency) > 2,
                    cmap="jet",
                )
                depth_pred_l_reprojed = focal_length * baseline / (disp_pred_l_reprojed + 1e-7)
                depth_diff_l_reprojed = depth_gt - depth_pred_l_reprojed

            if self.use_mask:
                plt.imsave(os.path.join(save_folder, "mask.png"), mask.astype(float), vmin=0.0, vmax=1.0, cmap="jet")

            if "conf_map" in pred_dict:
                conf_map = pred_dict["conf_map"].cpu().numpy()[0, :, 2:-2]
                np.save(os.path.join(save_folder, f"confidence.npy"), conf_map)
                for i in range(conf_map.shape[0]):
                   plt.imsave(
                       os.path.join(save_folder, f"confidence_{i+1}.png"), conf_map[i], vmin=0.0, vmax=1.0, cmap="jet"
                   )
            if "conf_map_r" in pred_dict:
                conf_map = pred_dict["conf_map_r"].cpu().numpy()[0, :, 2:-2]
                np.save(os.path.join(save_folder, f"confidence_r.npy"), conf_map)
                for i in range(conf_map.shape[0]):
                    plt.imsave(
                        os.path.join(save_folder, f"confidence_{i+1}_r.png"),
                        conf_map[i],
                        vmin=0.0,
                        vmax=1.0,
                        cmap="jet",
                    )


            if "intrinsic_l" in data_batch:
                intrinsic_l = data_batch["intrinsic_l"][0].cpu().numpy().copy()
                intrinsic_l[1, 2] -= 2
                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(depth2pts_np(depth_gt, intrinsic_l))
                pcd_gt = pcd_gt.crop(
                    o3d.geometry.AxisAlignedBoundingBox(
                        min_bound=np.array([-10, -10, 0.1]), max_bound=np.array([10, 10, 1.8])
                    )
                )
                # o3d.io.write_point_cloud(os.path.join(save_folder, "gt.pcd"), pcd_gt)
    
                
                pcd_pred = o3d.geometry.PointCloud()
                pcd_pred.points = o3d.utility.Vector3dVector(depth2pts_np(depth_pred, intrinsic_l))
                
                print(depth_pred.shape)
                print(depth_diff.shape)

                
                pcd_pred.colors = o3d.utility.Vector3dVector(
                    self.cmap(
                        np.clip(
                            (depth_diff + self.depth_diff_threshold - 1e5 * (1 - mask)) / self.depth_diff_threshold / 2,
                            0,
                            1,
                        )
                    )[..., :3].reshape(-1, 3)
                )             
                in_scene = mask.reshape(-1)
                points = []
                colors = []
                for idx in range(len(pcd_pred.points)):
                    if in_scene[idx]:
                        points.append(pcd_pred.points[idx])
                        colors.append(pcd_pred.colors[idx])

                pcd_pred.points = o3d.utility.Vector3dVector(points)
                pcd_pred.colors = o3d.utility.Vector3dVector(colors)

                pcd_pred = pcd_pred.crop(
                    o3d.geometry.AxisAlignedBoundingBox(
                        min_bound=np.array([-10, -10, 0.1]), max_bound=np.array([10, 10, 1.8])
                    )
                )
                pth = data_batch["dir"]
                o3d.io.write_point_cloud(os.path.join(save_folder, f"{pth}.pcd"), pcd_pred)

                if left_right:
                    pcd_pred = o3d.geometry.PointCloud()
                    pcd_pred.points = o3d.utility.Vector3dVector(depth2pts_np(depth_pred_l_reprojed, intrinsic_l))
                    pcd_pred.colors = o3d.utility.Vector3dVector(
                        self.cmap(np.clip((depth_diff_l_reprojed + 16e-3 - 1e5 * (1 - mask)) / 32e-3, 0, 1))[
                            ..., :3
                        ].reshape(-1, 3)
                    )
                    pcd_pred = pcd_pred.crop(
                        o3d.geometry.AxisAlignedBoundingBox(
                            min_bound=np.array([-10, -10, 0.1]), max_bound=np.array([10, 10, 1.8])
                        )
                    )
                    o3d.io.write_point_cloud(os.path.join(save_folder, "pred_reprojed.pcd"), pcd_pred)

            if "img_normal_l" in data_batch and "intrinsic_l" in data_batch:
                cv2.imwrite(os.path.join(save_folder, "normal_gt.png"), ((normal_gt + 1) * 127.5).astype(np.uint8))
                cv2.imwrite(os.path.join(save_folder, "normal_pred.png"), ((normal_pred + 1) * 127.5).astype(np.uint8))
                cv2.imwrite(
                   os.path.join(save_folder, "normal_pred_u16.png"), ((normal_pred + 1) * 1000.0).astype(np.uint16)
                )
                plt.imsave(
                   os.path.join(save_folder, "normal_diff.png"),
                   normal_diff * mask,
                   vmin=0.0,
                   vmax=np.pi,
                   cmap="jet",
                )

                if "normal3" in pred_dict:
                    normal3 = pred_dict["normal3"][0].permute(1, 2, 0).cpu().numpy()[2:-2]
                    cv2.imwrite(os.path.join(save_folder, "normal_pred3.png"), ((normal3 + 1) * 127.5).astype(np.uint8))
                    cv2.imwrite(
                        os.path.join(save_folder, "normal_pred3_u16.png"), ((normal3 + 1) * 1000.0).astype(np.uint16)
                    )

                    normal_diff3 = np.arccos(np.clip(np.sum(normal_gt * normal3, axis=-1), -1, 1))
                    plt.imsave(
                        os.path.join(save_folder, "normal_diff3.png"),
                        normal_diff3 * mask,
                        vmin=0.0,
                        vmax=np.pi,
                        cmap="jet",
                    )

            # if "edge" in pred_dict: TODO
            if False:  
                edge_pred = pred_dict["edge"]
                edge_pred = torch.argmax(edge_pred, dim=1)[0].detach().cpu().numpy()
                edge_gt = data_batch["img_disp_edge_l"][0].detach().cpu().numpy()

                cv2.imwrite(os.path.join(save_folder, "edge_pred.png"), (edge_pred.astype(np.uint8)) * 255)
                cv2.imwrite(os.path.join(save_folder, "edge_gt.png"), (edge_gt.astype(np.uint8)) * 255)

        if "img_label_l" in data_batch.keys():
            label_l = data_batch["img_label_l"].cpu().numpy()[0][2:-2]
        else:
            label_l = None
        

        table_mask = mask * (1 - all_object_mask)

        mask = mask.astype(bool)
        all_object_mask = all_object_mask.astype(bool)
        diffuse_mask = diffuse_mask.astype(bool)
        specular_mask = specular_mask.astype(bool)
        transparent_mask = transparent_mask.astype(bool)
        table_mask = table_mask.astype(bool)
        

        mean_epe = np.abs(disp_diff[mask]).sum()
        object_epe = np.abs(disp_diff[all_object_mask]).sum()
        diffuse_epe = np.abs(disp_diff[diffuse_mask]).sum()
        specular_epe = np.abs(disp_diff[specular_mask]).sum()
        transparent_epe = np.abs(disp_diff[transparent_mask]).sum()
        table_epe = np.abs(disp_diff[table_mask]).sum()

        self.mean_weight.append(mask.sum())
        self.object_weight.append(all_object_mask.sum())
        self.diffuse_weight.append(diffuse_mask.sum())
        self.specular_weight.append(specular_mask.sum())
        self.transparent_weight.append(transparent_mask.sum())
        self.table_weight.append(table_mask.sum())


        self.mean_epe.append(mean_epe)
        self.object_epe.append(object_epe)
        self.diffuse_epe.append(diffuse_epe)
        self.specular_epe.append(specular_epe)
        self.transparent_epe.append(transparent_epe)
        self.table_epe.append(table_epe)


        bad1 = (np.abs(disp_diff[mask]) > 1).sum() / mask.sum()
        bad2 = (np.abs(disp_diff[mask]) > 2).sum() / mask.sum()

        # self.mean_bad1.append((np.abs(disp_diff[mask]) > 1).sum())
        # self.object_bad1.append((np.abs(disp_diff[all_object_mask]) > 1).sum())
        # self.diffuse_bad1.append((np.abs(disp_diff[diffuse_mask]) > 1).sum())
        # self.specular_bad1.append((np.abs(disp_diff[specular_mask]) > 1).sum())
        # self.transparent_bad1.append((np.abs(disp_diff[transparent_mask]) > 1).sum())
        # self.table_bad1.append((np.abs(disp_diff[table_mask]) > 1).sum())

        # self.mean_bad2.append((np.abs(disp_diff[mask]) > 2).sum())
        # self.object_bad2.append((np.abs(disp_diff[all_object_mask]) > 2).sum())
        # self.diffuse_bad2.append((np.abs(disp_diff[diffuse_mask]) > 2).sum())
        # self.specular_bad2.append((np.abs(disp_diff[specular_mask]) > 2).sum())
        # self.transparent_bad2.append((np.abs(disp_diff[transparent_mask]) > 2).sum())
        # self.table_bad2.append((np.abs(disp_diff[table_mask]) > 2).sum())
        
        self.bad1.append(bad1)
        self.bad2.append(bad2)
        self.delta_05.append(delta_05)
        self.delta_10.append(delta_10)
        self.delta_15.append(delta_15)
        self.depth_abs_err.append(depth_abs_err)
        self.depth_err2.append(depth_err2)
        self.depth_err4.append(depth_err4)
        self.depth_err8.append(depth_err8)
        self.depth_err16.append(depth_err16)
        self.depth_err32.append(depth_err32)

        return {
            "epe": mean_epe / mask.sum(),
            "object_epe": object_epe / (all_object_mask.sum() + 1e-7),
            "diffuse_epe": diffuse_epe / (diffuse_mask.sum() + 1e-7),
            "specular_epe": specular_epe / (specular_mask.sum() + 1e-7),
            "transparent_epe": transparent_epe / (transparent_mask.sum() + 1e-7),
            "table_epe": table_epe / (table_mask.sum() + 1e-7),
            "bad1": bad1,
            "bad2": bad2,
            "depth_abs_err": depth_abs_err,
            "depth_err2": depth_err2,
            "depth_err4": depth_err4,
            "depth_err8": depth_err8,
            "depth_err16": depth_err16,
            "depth_err32": depth_err32,
            "delta_05": delta_05,
            "delta_10": delta_10,
            "delta_15": delta_15,
        }

    def summary(self):
        s = ""
        #headers = ["epe", "object_epe", "ordinary_epe", "printed_epe", "transparent_epe", "bad1", "bad2", "depth_abs_err", "depth_err2", "depth_err4", "depth_err8", "depth_err16", "depth_err32", "delta_05", "delta_10", "delta_15"]
        # headers = ["", "Mean", "Object", "Ordinary", "Printed", "Transparent", "Surface", "Table", "Edge"]
        headers = ["", "Mean", "Object", "Diffuse", "Specular", "Transparent", "Table"]

        mean_w = np.sum(self.mean_weight) / np.sum(self.mean_weight)
        object_w = np.sum(self.object_weight) / np.sum(self.mean_weight)
        diffuse_w = np.sum(self.diffuse_weight) / np.sum(self.mean_weight)
        specular_w = np.sum(self.specular_weight) / np.sum(self.mean_weight)
        transparent_w = np.sum(self.transparent_weight) / np.sum(self.mean_weight)
        table_w = np.sum(self.table_weight) / np.sum(self.mean_weight)

        table = [
            [
                "weight",
                mean_w,
                object_w,
                diffuse_w,
                specular_w,
                transparent_w,
                table_w,
            ],
            [   "epe",
                np.sum(self.mean_epe) / np.sum(self.mean_weight),
                np.sum(self.object_epe) / np.sum(self.object_weight),
                np.sum(self.diffuse_epe) / np.sum(self.diffuse_weight),
                np.sum(self.specular_epe) / np.sum(self.specular_weight),
                np.sum(self.transparent_epe) / np.sum(self.transparent_weight),
                np.sum(self.table_epe) / np.sum(self.table_weight),
            ],
            [
                "bad1",
                np.sum(self.mean_bad1) / np.sum(self.mean_weight),
                np.sum(self.object_bad1) / np.sum(self.object_weight),
                np.sum(self.diffuse_bad1) / np.sum(self.diffuse_weight),
                np.sum(self.specular_bad1) / np.sum(self.specular_weight),
                np.sum(self.transparent_bad1) / np.sum(self.transparent_weight),
                np.sum(self.table_bad1) / np.sum(self.table_weight),
            ],
            [
                "bad2",
                np.sum(self.mean_bad2) / np.sum(self.mean_weight),
                np.sum(self.object_bad2) / np.sum(self.object_weight),
                np.sum(self.diffuse_bad2) / np.sum(self.diffuse_weight),
                np.sum(self.specular_bad2) / np.sum(self.specular_weight),
                np.sum(self.transparent_bad2) / np.sum(self.transparent_weight),
                np.sum(self.table_bad2) / np.sum(self.table_weight),
            ],
        ]
        
        s += tabulate(table, headers=headers, tablefmt="fancy_grid", floatfmt=".6f")

        if self.obj_count.sum() > 0:
            headers = ["class_id", "name", "count", "disp_err", "bad1", "bad2", "depth_err", "depth_err4", "depth_err8", "depth_err16", "depth_err32", "delta_05", "delta_10", "delta_15"]
            if self.normal_err:
                headers += ["obj_norm_err", "obj_norm_err10"]
            table = []
            for i in range(self.num_classes):
                t = [
                    i,
                    OBJECT_NAMES[i],
                    self.obj_count[i],
                    self.obj_disp_err[i] / (self.obj_count[i] + 1e-7),
                    self.obj_bad1[i] / (self.obj_count[i] + 1e-7),
                    self.obj_bad2[i] / (self.obj_count[i] + 1e-7),
                    self.obj_depth_err[i] / (self.obj_count[i] + 1e-7),
                    self.obj_depth_err4[i] / (self.obj_count[i] + 1e-7),
                    self.obj_depth_err8[i] / (self.obj_count[i] + 1e-7),
                    self.obj_depth_err16[i] / (self.obj_count[i] + 1e-7),
                    self.obj_depth_err32[i] / (self.obj_count[i] + 1e-7),
                    self.obj_delta_05[i] / (self.obj_count[i] + 1e-7),
                    self.obj_delta_10[i] / (self.obj_count[i] + 1e-7),
                    self.obj_delta_15[i] / (self.obj_count[i] + 1e-7),
                ]
                if self.normal_err:
                    t += [
                        self.obj_normal_err[i] / (self.obj_count[i] + 1e-7),
                        self.obj_normal_err10[i] / (self.obj_count[i] + 1e-7),
                    ]
                table.append(t)
            t = [
                "-",
                "DIFFUSE",
                self.diffuse_count,
                self.diffuse_disp_err / (self.diffuse_count + 1e-7),
                self.diffuse_bad1 / (self.diffuse_count + 1e-7),
                self.diffuse_bad2 / (self.diffuse_count + 1e-7),
                self.diffuse_depth_err / (self.diffuse_count + 1e-7),
                self.diffuse_depth_err4 / (self.diffuse_count + 1e-7),
                self.diffuse_depth_err8 / (self.diffuse_count + 1e-7),
                self.diffuse_depth_err16 / (self.diffuse_count + 1e-7),
                self.diffuse_depth_err32 / (self.diffuse_count + 1e-7),
                self.diffuse_delta_05 / (self.diffuse_count + 1e-7),
                self.diffuse_delta_10 / (self.diffuse_count + 1e-7),
                self.diffuse_delta_15 / (self.diffuse_count + 1e-7),
            ]
            if self.normal_err:
                t += [
                    self.diffuse_normal_err / (self.diffuse_count + 1e-7),
                    self.diffuse_normal_err10 / (self.diffuse_count + 1e-7),
                ]
            table.append(t)


            t = [
                "-",
                "SPECULAR",
                self.specular_count,
                self.specular_disp_err / (self.specular_count + 1e-7),
                self.specular_bad1 / (self.specular_count + 1e-7),
                self.specular_bad2 / (self.specular_count + 1e-7),
                self.specular_depth_err / (self.specular_count + 1e-7),
                self.specular_depth_err4 / (self.specular_count + 1e-7),
                self.specular_depth_err8 / (self.specular_count + 1e-7),
                self.specular_depth_err16 / (self.specular_count + 1e-7),
                self.specular_depth_err32 / (self.specular_count + 1e-7),
                self.specular_delta_05 / (self.specular_count + 1e-7),
                self.specular_delta_10 / (self.specular_count + 1e-7),
                self.specular_delta_15 / (self.specular_count + 1e-7),
            ]
            if self.normal_err:
                t += [
                    self.specular_normal_err / (self.specular_count + 1e-7),
                    self.specular_normal_err10 / (self.specular_count + 1e-7),
                ]
            table.append(t)

            t = [
                "-",
                "TRANSPARENT",
                self.transparent_count,
                self.transparent_disp_err / (self.transparent_count + 1e-7),
                self.transparent_bad1 / (self.transparent_count + 1e-7),
                self.transparent_bad2 / (self.transparent_count + 1e-7),
                self.transparent_depth_err / (self.transparent_count + 1e-7),
                self.transparent_depth_err4 / (self.transparent_count + 1e-7),
                self.transparent_depth_err8 / (self.transparent_count + 1e-7),
                self.transparent_depth_err16 / (self.transparent_count + 1e-7),
                self.transparent_depth_err32 / (self.transparent_count + 1e-7),
                self.transparent_delta_05 / (self.transparent_count + 1e-7),
                self.transparent_delta_10 / (self.transparent_count + 1e-7),
                self.transparent_delta_15 / (self.transparent_count + 1e-7),
            ]
            if self.normal_err:
                t += [
                    self.transparent_normal_err / (self.transparent_count + 1e-7),
                    self.transparent_normal_err10 / (self.transparent_count + 1e-7),
                ]
            table.append(t)


            t = [
                "-",
                "ALL",
                self.diffuse_count + self.specular_count + self.transparent_count,
                (self.diffuse_disp_err + self.specular_disp_err + self.transparent_disp_err) / 
                    (self.diffuse_count + self.specular_count + self.transparent_count + 1e-7),
                (self.diffuse_bad1 + self.specular_bad1 + self.transparent_bad1) / 
                    (self.diffuse_count + self.specular_count + self.transparent_count + 1e-7),
                (self.diffuse_bad2 + self.specular_bad2 + self.transparent_bad2) / 
                    (self.diffuse_count + self.specular_count + self.transparent_count + 1e-7),
                (self.diffuse_depth_err + self.specular_depth_err + self.transparent_depth_err) / 
                    (self.diffuse_count + self.specular_count + self.transparent_count + 1e-7),
                (self.diffuse_depth_err4 + self.specular_depth_err4 + self.transparent_depth_err4) / 
                    (self.diffuse_count + self.specular_count + self.transparent_count + 1e-7),
                (self.diffuse_depth_err8 + self.specular_depth_err8 + self.transparent_depth_err8) / 
                    (self.diffuse_count + self.specular_count + self.transparent_count + 1e-7),
                (self.diffuse_depth_err16 + self.specular_depth_err16 + self.transparent_depth_err16) / 
                    (self.diffuse_count + self.specular_count + self.transparent_count + 1e-7),
                (self.diffuse_depth_err32 + self.specular_depth_err32 + self.transparent_depth_err32) / 
                    (self.diffuse_count + self.specular_count + self.transparent_count + 1e-7),
                (self.diffuse_delta_05 + self.specular_delta_05 + self.transparent_delta_05) / 
                    (self.diffuse_count + self.specular_count + self.transparent_count + 1e-7),
                (self.diffuse_delta_10 + self.specular_delta_10 + self.transparent_delta_10) / 
                    (self.diffuse_count + self.specular_count + self.transparent_count + 1e-7),
                (self.diffuse_delta_15 + self.specular_delta_15 + self.transparent_delta_15) / 
                    (self.diffuse_count + self.specular_count + self.transparent_count + 1e-7),
                ]
            if self.normal_err:
                t += [
                    (self.diffuse_normal_err + self.specular_normal_err + self.transparent_normal_err) / 
                        (self.diffuse_count + self.specular_count + self.transparent_count + 1e-7),
                    (self.diffuse_normal_err10 + self.specular_normal_err10 + self.transparent_normal_err10) / 
                        (self.diffuse_count + self.specular_count + self.transparent_count + 1e-7),
                ]
            table.append(t)
            s += "\n"
            s += tabulate(table, headers=headers, tablefmt="fancy_grid", floatfmt=".6f")

        return s

    def save_preds_only(self, data_batch, pred_dict, save_folder="", real_data=False):
        if self.model_type in (
            "PSMNetConfidence",
            "PSMNetDilation",
        ):
            prediction = pred_dict["pred3"]
            prediction = prediction.detach().cpu().numpy()[0, 0, 2:-2]
        
        if "focal_length" in data_batch and "baseline" in data_batch and "intrinsic_l" in data_batch:
            focal_length = data_batch["focal_length"][0].cpu().numpy()
            baseline = data_batch["baseline"][0].cpu().numpy()
            intrinsic_l = data_batch["intrinsic_l"][0].cpu().numpy()
            intrinsic_l[1, 2] -= 2
        else:
            focal_length = 672.0
            baseline = 0.055
            intrinsic_l = np.array([[672, 0.0, 476], [0, 672, 273], [0, 0, 1]])
            intrinsic_l[1, 2] -= 2

        if self.is_depth:
            disp_pred = focal_length * baseline / (prediction + 1e-7)
            depth_pred = prediction
        else:
            depth_pred = focal_length * baseline / (prediction + 1e-7)
            disp_pred = prediction

        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            plt.imsave(
                os.path.join(save_folder, "disp_pred.png"),
                np.clip(disp_pred, 0, self.max_disp),
                vmin=0.0,
                vmax=self.max_disp,
                cmap="jet",
            )
            if "conf_map" in pred_dict:
                conf_map = pred_dict["conf_map"].cpu().numpy()[0, :, 2:-2]
                np.save(os.path.join(save_folder, f"confidence.npy"), conf_map)
                for i in range(conf_map.shape[0]):
                    plt.imsave(
                        os.path.join(save_folder, f"confidence_{i}.png"), conf_map[i], vmin=0.0, vmax=1.0, cmap="jet"
                    )
            np.save(os.path.join(save_folder, "disp_pred.npy"), disp_pred)
            pcd_pred = o3d.geometry.PointCloud()
            pcd_pred.points = o3d.utility.Vector3dVector(depth2pts_np(depth_pred, intrinsic_l))
            if "conf_map" in pred_dict:
                pcd_pred.colors = o3d.utility.Vector3dVector(self.cmap(conf_map)[..., :3].reshape(-1, 3))
            pcd_pred = pcd_pred.crop(
                o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=np.array([-10, -10, 0.1]), max_bound=np.array([10, 10, 1.8])
                )
            )
            o3d.io.write_point_cloud(os.path.join(save_folder, "pred.pcd"), pcd_pred)

        # prob cost volume
        if self.model_type in [
            "PSMNetConfidence",
            "PSMNetDilation",
        ]:
            cost_volume = pred_dict["prob_volume"][0].detach().cpu().numpy()
            save_prob_volume(cost_volume, os.path.join(save_folder, "prob_volume.pcd"))

        # 2D Discriminator
        if "d_gt" in pred_dict and "d_pred" in pred_dict:
            d_pred = pred_dict["d_pred"][0, 0].detach().cpu().numpy()
            plt.imsave(os.path.join(save_folder, "d_pred.png"), d_pred, vmin=-0.02, vmax=0.02, cmap="jet")

    def save_for_train(self, data_batch, pred_dict, save_folder="", real_data=False):
        if self.model_type in (
            "PSMNetConfidence",
            "PSMNetDilation",
        ):
            prediction = pred_dict["pred3"]
            prediction = prediction.detach().cpu().numpy()[0, 0, 2:-2]
        

        focal_length = data_batch["focal_length"][0].cpu().numpy()
        baseline = data_batch["baseline"][0].cpu().numpy()

        if self.is_depth:
            disp_pred = focal_length * baseline / (prediction + 1e-7)
            depth_pred = prediction
        else:
            depth_pred = focal_length * baseline / (prediction + 1e-7)
            disp_pred = prediction

        if save_folder:
            os.makedirs(save_folder, exist_ok=True)

            conf_map = pred_dict["conf_map"].cpu().numpy()[0, :, 2:-2]
            # np.save(os.path.join(save_folder, f"confidence.npy"), conf_map)
            # for i in range(conf_map.shape[0]):
            #     plt.imsave(
            #         os.path.join(save_folder, f"confidence_{i}.png"), conf_map[i], vmin=0.0, vmax=1.0, cmap="jet"
            #     )
            conf_map = (conf_map[1] * 60000).astype(np.uint16)
            cv2.imwrite(os.path.join(save_folder, "confidence_1_u16.png"), conf_map)
            

            # np.save(os.path.join(save_folder, "disp_pred.npy"), disp_pred)
            edge_pred = pred_dict["edge"]
            edge_pred = torch.argmax(edge_pred, dim=1)[0].detach().cpu().numpy()
            cv2.imwrite(os.path.join(save_folder, "edge_pred.png"), (edge_pred.astype(np.uint8)) * 255)

            intrinsic_l = data_batch["intrinsic_l"][0].cpu().numpy()
            intrinsic_l[1, 2] -= 2
            intrinsic = data_batch["intrinsic"][0].cpu().numpy()
            extrinsic_l = data_batch["extrinsic_l"][0].cpu().numpy()
            extrinsic = data_batch["extrinsic"][0].cpu().numpy()

            rl_mainl = extrinsic @ np.linalg.inv(extrinsic_l)
            depth = cv2.resize(depth_pred, (1920, 1080), interpolation=cv2.INTER_NEAREST)
            intrinsic_l[:2] *= 2
            depth_rgb = cv2.rgbd.registerDepth(
                intrinsic_l, intrinsic, None, rl_mainl, depth, (1920, 1080), depthDilation=True
            )
            # depth_rgb = cv2.medianBlur(depth_rgb, 5)
            depth_rgb = (depth_rgb * 1000.0).astype(np.uint16)
            cv2.imwrite(os.path.join(save_folder, "depth_pred_rgb.png"), depth_rgb)
            vis_depth_rgb = visualize_depth(depth_rgb)
            cv2.imwrite(os.path.join(save_folder, "depth_pred_rgb_colored.png"), vis_depth_rgb)

            depth = (depth_pred * 1000.0).astype(np.uint16)
            depth = cv2.resize(depth, (1920, 1080), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(save_folder, "depth_pred.png"), depth)
            vis_depth = visualize_depth(depth)
            cv2.imwrite(os.path.join(save_folder, "depth_pred_colored.png"), vis_depth)

    def save_normal(self, data_batch, pred_dict, save_folder, suffix=""):
        os.makedirs(save_folder, exist_ok=True)
        if self.model_type == "PSMNetClearGrasp":
            disp_gt = data_batch["img_disp_l"].cpu().numpy()[0, 0, 2:-2]
            mask = np.logical_and(disp_gt > 1e-1, disp_gt < self.max_disp)
            x_base = np.arange(0, disp_gt.shape[1]).reshape(1, -1)
            mask = np.logical_and(mask, x_base > disp_gt)
            if "-300" in data_batch["dir"]:
                view_id = data_batch["dir"].split("-")[-1]
                mask = np.logical_and(mask, np.logical_not(self.real_robot_masks[view_id]))
            normal_gt = data_batch["img_normal_l"][0].permute(1, 2, 0).cpu().numpy()[2:-2]
            normal3 = pred_dict["normal3"][0].permute(1, 2, 0).cpu().numpy()[2:-2]
            cv2.imwrite(os.path.join(save_folder, "normal_pred3.png"), ((normal3 + 1) * 127.5).astype(np.uint8))
            cv2.imwrite(os.path.join(save_folder, "normal_pred3_u16.png"), ((normal3 + 1) * 1000.0).astype(np.uint16))

            normal_diff3 = np.arccos(np.clip(np.sum(normal_gt * normal3, axis=-1), -1, 1))
            plt.imsave(
                os.path.join(save_folder, "normal_diff3.png"),
                normal_diff3 * mask,
                vmin=0.0,
                vmax=np.pi,
                cmap="jet",
            )

            edge_pred = pred_dict["edge"]
            edge_pred = torch.argmax(edge_pred, dim=1)[0].detach().cpu().numpy()
            edge_gt = data_batch["img_disp_edge_l"][0].detach().cpu().numpy()

            cv2.imwrite(os.path.join(save_folder, "edge_pred.png"), (edge_pred.astype(np.uint8)) * 255)
            cv2.imwrite(os.path.join(save_folder, "edge_gt.png"), (edge_gt.astype(np.uint8)) * 255)
            return
        if self.model_type in (
            "PSMNetConfidence",
            "PSMNetDilation",
            "PSMNetEdgeNormal",
        ):
            prediction = pred_dict["pred3"]
            prediction = prediction.detach().cpu().numpy()[0, 0, 2:-2]
        
        focal_length = data_batch["focal_length"][0].cpu().numpy()
        baseline = data_batch["baseline"][0].cpu().numpy()
        if self.is_depth:
            disp_pred = focal_length * baseline / (prediction + 1e-7)
            depth_pred = prediction
        else:
            depth_pred = focal_length * baseline / (prediction + 1e-7)
            disp_pred = prediction

        conf_map = pred_dict["conf_map"].cpu().numpy()[0, :, 2:-2]
        for i in range(5):
            plt.imsave(
                        os.path.join(save_folder, f"confidence_{i+1}.png"), conf_map[i], vmin=0, vmax=conf_map[i].max(), cmap="jet"
                    )
        conf_map = (conf_map[1] * 60000).astype(np.uint16)
        cv2.imwrite(os.path.join(save_folder, f"confidence_1_u16{suffix}.png"), conf_map)
        
        intrinsic_l = data_batch["intrinsic_l"][0].cpu().numpy().copy()
        intrinsic_l[1, 2] -= 2
        normal_pred = cal_normal_map(depth_pred, intrinsic_l, radius=0.005, max_nn=100)
        cv2.imwrite(os.path.join(save_folder, f"normal_pred{suffix}.png"), ((normal_pred + 1) * 127.5).astype(np.uint8))
        cv2.imwrite(
            os.path.join(save_folder, f"normal_pred_u16{suffix}.png"), ((normal_pred + 1) * 1000.0).astype(np.uint16)
        )

        print("a" * 50)
        print(save_folder)


# Error metric for messy-table-dataset object error
def compute_obj_err(disp_gt, depth_gt, disp_pred, focal_length, baseline, label, mask, obj_total_num=17):
    """
    Compute error for each object instance in the scene
    :param disp_gt: GT disparity map, [bs, 1, H, W]
    :param depth_gt: GT depth map, [bs, 1, H, W]
    :param disp_pred: Predicted disparity map, [bs, 1, H, W]
    :param focal_length: Focal length, [bs, 1]
    :param baseline: Baseline of the camera, [bs, 1]
    :param label: Label of the image [bs, 1, H, W]
    :param obj_total_num: Total number of objects in the dataset
    :return: obj_disp_err, obj_depth_err - List of error of each object
             obj_count - List of each object appear count
    """
    depth_pred = focal_length * baseline / disp_pred  # in meters

    obj_list = label.unique()  # TODO this will cause bug if bs > 1, currently only for testing
    obj_num = obj_list.shape[0]

    # Array to store error and count for each object
    total_obj_disp_err = np.zeros(obj_total_num)
    total_obj_depth_err = np.zeros(obj_total_num)
    total_obj_depth_4_err = np.zeros(obj_total_num)
    total_obj_count = np.zeros(obj_total_num)

    for i in range(obj_num):
        obj_id = int(obj_list[i].item())
        obj_mask = label == obj_id
        obj_disp_err = F.l1_loss(disp_gt[obj_mask * mask], disp_pred[obj_mask * mask], reduction="mean").item()
        obj_depth_err = torch.clip(
            torch.abs(depth_gt[obj_mask * mask] * 1000 - depth_pred[obj_mask * mask] * 1000),
            min=0,
            max=100,
        )
        obj_depth_err = torch.mean(obj_depth_err).item()
        obj_depth_diff = torch.abs(depth_gt[obj_mask * mask] - depth_pred[obj_mask * mask])
        obj_depth_err4 = obj_depth_diff[obj_depth_diff > 4e-3].numel() / obj_depth_diff.numel()

        total_obj_disp_err[obj_id] += obj_disp_err
        total_obj_depth_err[obj_id] += obj_depth_err
        total_obj_depth_4_err[obj_id] += obj_depth_err4
        total_obj_count[obj_id] += 1
    return (
        total_obj_disp_err,
        total_obj_depth_err,
        total_obj_depth_4_err,
        total_obj_count,
    )


def save_prob_volume(
    prob_volume,
    file_path,
    mask=None,
    threshold=0.01,
):
    d, h, w = prob_volume.shape
    custom_cmap = plt.get_cmap("jet")
    if mask is None:
        mask = (prob_volume > threshold).reshape(-1)
    else:
        valid_mask = (prob_volume > threshold).reshape(-1)
        mask = mask[None, :]
        mask = np.repeat(mask, d, axis=0).reshape(-1)
        mask = np.logical_and(mask, valid_mask)

    color = custom_cmap(prob_volume)[..., :3].reshape(-1, 3)

    coor = np.zeros((d, h, w, 3))
    for i in range(h):
        coor[:, i, :, 0] = i
    for i in range(w):
        coor[:, :, i, 1] = i
    for i in range(d):
        coor[i, :, :, 2] = i

    coor = coor.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coor[mask])
    pcd.colors = o3d.utility.Vector3dVector(color[mask])

    o3d.io.write_point_cloud(file_path, pcd)


def visualize_depth(depth):
    MAX_DEPTH = 2.0
    cmap = plt.get_cmap("rainbow")
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 1000.0
    if len(depth.shape) == 3:
        depth = depth[..., 0]
    depth = np.clip(depth / MAX_DEPTH, 0.0, 1.0)
    vis_depth = cmap(depth)
    vis_depth = (vis_depth[:, :, :3] * 255.0).astype(np.uint8)
    vis_depth = cv2.cvtColor(vis_depth, cv2.COLOR_RGB2BGR)
    return vis_depth
