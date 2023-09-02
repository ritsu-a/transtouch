import argparse
import copy
import os
import time

import cv2
import numpy as np
import torch
from path import Path
from tqdm import tqdm

from transtouch.config import cfg
from transtouch.datasets.messytable import MessyTableDataset
from transtouch.utils.io import load_pickle
from transtouch.utils.loguru_logger import setup_logger
from transtouch.utils.metrics import ErrorMetric
from data_rendering.utils.render_utils import visualize_depth

from transtouch.utils.cfg_utils import purge_cfg

import os.path as osp
import warnings



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate cleargrasp performance")
    parser.add_argument("-d", "--data-folder", type=str, required=True)
    parser.add_argument(
        "-s",
        "--split-file",
        type=str,
        metavar="FILE",
        required=True,
    )
    parser.add_argument(
        "--cfg",
        dest="config_file",
        default="",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("-r", "--realsense-mask", action="store_true", help="whether use realsense mask")

    args = parser.parse_args()
    return args


def register_depth(view_folder, data_dir):
    view_folder = Path(view_folder)
    #if (view_folder / "1024_Cleargrasp_real.png").exists():
    if False:
        depth_u16 = cv2.imread(view_folder / "1024_depthL_real.png", cv2.IMREAD_UNCHANGED)
        assert depth_u16.shape == (540, 960)
        depth = (depth_u16.astype(float)) / 1000.0
    else:
        img_meta = load_pickle(view_folder / "meta.pkl")
        extrinsic_l = img_meta["extrinsic_l"]
        extrinsic_r = img_meta["extrinsic_r"]
        intrinsic_l = img_meta["intrinsic_l"][:3, :3]
        # intrinsic_l[:2] /= 2
        intrinsic_l[2] = np.array([0.0, 0.0, 1.0])
        if os.path.exists(f"/share/pengyang/cleargrasp/eval_depth_completion/results/exp-000/output-depth/{data_dir}-output-depth.npy"):
            rgb_cam_depth = np.load(f"/share/pengyang/cleargrasp/eval_depth_completion/results/exp-000/output-depth/{data_dir}-output-depth.npy")
        else:
            rgb_cam_depth = np.load(f"/share/pengyang/cleargrasp/eval_depth_completion/results/exp-002/output-depth/{data_dir}-output-depth.npy")
        rgb_cam_depth = cv2.resize(rgb_cam_depth, (960, 540), interpolation=cv2.INTER_NEAREST)
        w, h = 960, 540
        rt_mainl = img_meta["extrinsic"] @ np.linalg.inv(img_meta["extrinsic_l"])
        rt_lmain = np.linalg.inv(rt_mainl)
        depth = cv2.rgbd.registerDepth(
            img_meta["intrinsic"][:3, :3], intrinsic_l, None, rt_lmain, rgb_cam_depth, (w, h), depthDilation=True
        )
        depth[np.isnan(depth)] = 0
        depth[np.isinf(depth)] = 0
        depth[depth < 0] = 0

        depth = cv2.resize(depth, (960, 540), interpolation=cv2.INTER_NEAREST)
        depth_u16 = copy.deepcopy(depth)
        depth_u16 = (depth_u16 * 1000.0).astype(np.uint16)

        cv2.imwrite(view_folder / "1024_Cleargrasp_real.png", depth_u16)
        vis_depth = visualize_depth(depth_u16)
        cv2.imwrite(view_folder / "1024_Cleargrasp_real_colored.png", vis_depth)

    return depth


def main():
    args = parse_args()
    # Load the configuration
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    # cfg.freeze()
    config_name = args.config_file.split("/")[-1].split(".")[0]
    output_dir = cfg.OUTPUT_DIR
    # Replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace("@", config_path.replace("configs", "outputs"))
        if osp.isdir(output_dir):
            warnings.warn("Output directory exists.")
        os.makedirs(output_dir, exist_ok=True)
        cfg.OUTPUT_DIR = output_dir
    # run name
    timestamp = time.strftime("%y-%m-%d_%H-%M-%S")
    run_name = "{:s}".format(timestamp)
    logger = setup_logger(f"transtouch.testCleargrasp", output_dir, rank=0, filename=f"log.cleargrasp.{run_name}.txt")
    logger.info(args)

    # use default test config
    cfg.TEST.IS_DEPTH = True
    # Build metrics
    metric = ErrorMetric(
        model_type="Cleargrasp",
        use_mask=cfg.TEST.USE_MASK,
        max_disp=cfg.TEST.MAX_DISP,
        depth_range=cfg.TEST.DEPTH_RANGE,
        num_classes=cfg.DATA.NUM_CLASSES,
        is_depth=cfg.TEST.IS_DEPTH,
        realsense_mask=args.realsense_mask,
    )
    metric.reset()

    dataset = MessyTableDataset(
        cfg=cfg,
        mode="test",
        domain="real",
        root_dir=args.data_folder,
        split_file=args.split_file,
        height=544,
        width=960,
        meta_name="meta.pkl",
        depth_name="depthL.png",
        normal_name="",
        left_names=["1024_irL_real_360.png"],
        right_names=["1024_irR_real_360.png"],
        left_pattern_name="",
        right_pattern_name="",
        label_name="labelL.png",
        normal_conf_name="",
    )

    file_dir = osp.join(output_dir, "cleargrasp_test")
    eval_tic = time.time()
    for data in tqdm(dataset):
        view_folder = data["dir"]
        data_dir = data["dir"]
        data = {k: v.unsqueeze(0) for k, v in data.items() if isinstance(v, torch.Tensor)}
        data["dir"] = view_folder
        view_folder = os.path.join(args.data_folder, view_folder)
        depth = register_depth(view_folder, data_dir)
        print(depth.mean())
        pred_dict = {"depth": depth}
        print(osp.join(file_dir, data_dir + "_real"))
        curr_metric = metric.compute(
            data, pred_dict,
            save_folder=osp.join(file_dir, data_dir + "_real"),
            real_data=True)
        logger.info(view_folder + "\t" + "\t".join([f"{k}: {v:.3f}" for k, v in curr_metric.items()]))

    epoch_time_eval = time.time() - eval_tic
    logger.info("Cleargrasp Test total_time: {:.2f}s".format(epoch_time_eval))
    logger.info("Cleargrasp Eval Metric: \n" + metric.summary())


if __name__ == "__main__":
    main()
