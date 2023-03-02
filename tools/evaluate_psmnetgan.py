import argparse
import copy
import os
import time

import cv2
import numpy as np
import torch
from path import Path
from tqdm import tqdm

from active_zero2.config import cfg
from active_zero2.datasets.messytable import MessyTableDataset
from active_zero2.utils.io import load_pickle
from active_zero2.utils.loguru_logger import setup_logger
from active_zero2.utils.metrics import ErrorMetric
from data_rendering.utils.render_utils import visualize_depth


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate real realsense performance")
    parser.add_argument("-d", "--data-folder", type=str, required=True)
    parser.add_argument(
        "-s",
        "--split-file",
        type=str,
        metavar="FILE",
        required=True,
    )
    parser.add_argument("-r", "--realsense-mask", action="store_true", help="whether use realsense mask")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # run name
    timestamp = time.strftime("%y-%m-%d_%H-%M-%S")
    run_name = "{:s}".format(timestamp)
    logger = setup_logger(f"ActiveZero2.test PSMNetGAN", args.data_folder, rank=0, filename=f"log.test.{run_name}.txt")
    logger.info(args)

    # use default test config
    cfg.TEST.IS_DEPTH = True
    # Build metrics
    metric = ErrorMetric(
        model_type="RealSense",
        use_mask=cfg.TEST.USE_MASK,
        max_disp=cfg.TEST.MAX_DISP,
        depth_range=cfg.TEST.DEPTH_RANGE,
        num_classes=cfg.DATA.NUM_CLASSES,
        is_depth=cfg.TEST.IS_DEPTH,
        realsense_mask=args.realsense_mask,
    )
    metric.reset()

    dataset = MessyTableDataset(
        mode="test",
        domain="real",
        root_dir="/media/DATA/LINUX_DATA/ICCV2021_Diagnosis/real_data_v10/",
        split_file=args.split_file,
        height=544,
        width=960,
        meta_name="meta.pkl",
        depth_name="depthL.png",
        normal_name="",
        left_name="1024_irL_real.png",
        right_name="1024_irR_real.png",
        left_pattern_name="",
        right_pattern_name="",
        label_name="irL_label_image.png",
        normal_conf_name="",
    )

    eval_tic = time.time()
    for data in tqdm(dataset):
        view_folder = data["dir"]
        data = {k: v.unsqueeze(0) for k, v in data.items() if isinstance(v, torch.Tensor)}
        data["dir"] = view_folder
        depth = cv2.imread(os.path.join(args.data_folder, f"{view_folder}.png"), cv2.IMREAD_UNCHANGED)
        if args.realsense_mask:
            view_folder = os.path.join(args.data_folder, f"{view_folder}_realsense_mask")
        else:
            view_folder = os.path.join(args.data_folder, view_folder)
        view_folder = Path(view_folder)
        view_folder.makedirs_p()

        depth = (depth.astype(float)) / 1000.0
        pred_dict = {"depth": depth}

        curr_metric = metric.compute(data, pred_dict, save_folder=view_folder, real_data=True)
        logger.info(view_folder + "\t" + "\t".join([f"{k}: {v:.3f}" for k, v in curr_metric.items()]))

    epoch_time_eval = time.time() - eval_tic
    logger.info("PSMNetGAN Test total_time: {:.2f}s".format(epoch_time_eval))
    logger.info("PSMNetGAN Eval Metric: \n" + metric.summary())


if __name__ == "__main__":
    main()
