import os
import os.path as osp
import sys

_ROOT_DIR = os.path.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, _ROOT_DIR)

import argparse
import hashlib
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from path import Path
from tqdm import tqdm

from active_zero2.utils.io import load_pickle
from data_rendering.utils.sim_depth import calc_main_depth_from_left_right_ir

from .utils.render_utils import visualize_depth


def get_md5(filename):
    hash_obj = hashlib.md5()
    with open(filename, "rb") as f:
        hash_obj.update(f.read())
    return hash_obj.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", "--split-file", type=str, required=True)
    parser.add_argument("-d", "--dir", type=str, required=True)
    parser.add_argument("-l", "--left-name", type=str, default="0128_irL_kuafu_half.png")
    parser.add_argument("-r", "--right-name", type=str, default="0128_irR_kuafu_half.png")
    parser.add_argument("-n", "--ndisp", type=int, default=96)
    parser.add_argument("-w", "--wsize", type=int, default=7)
    parser.add_argument("--depth-name", type=str, default="0128_depth_kuafu.png")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_dir = Path(args.dir)
    img_dirs = []
    with open(args.split_file, "r") as f_split_file:
        for l in f_split_file.readlines():
            img_dirs.append(data_dir / l.strip())

    timestamp = time.strftime("%y-%m-%d_%H-%M-%S")
    run_name = "{:s}".format(timestamp)
    log_file = open(data_dir / f"stereo_depth_{run_name}.txt", "w")
    log_file.write(args)
    log_file.write("\n")

    for img_dir in tqdm(img_dirs):
        meta = load_pickle(img_dir / "meta.pkl")
        irl = cv2.imread(img_dir / args.left_name)
        irr = cv2.imread(img_dir / args.right_name)
        assert irl.shape[:2] == (540, 960) and irr.shape[:2] == (540, 960)

        cam_ir_intrinsic = meta["intrinsic_l"]
        assert (
            900 < cam_ir_intrinsic[0, 2] < 1000 and 500 < cam_ir_intrinsic[1, 2] < 600
        ), f"illegal cam_ir_intrinsic: {cam_ir_intrinsic} in {img_dir}"

        cam_ir_intrinsic[:2] /= 2

        depth = calc_main_depth_from_left_right_ir(
            irl,
            irr,
            meta["extrinsic_l"],
            meta["extrinsic_r"],
            meta["extrinsic"],
            cam_ir_intrinsic,
            cam_ir_intrinsic,
            meta["intrinsic"],
            lr_consistency=False,
            main_cam_size=(1920, 1080),
            ndisp=args.ndisp,
            use_census=True,
            register_depth=True,
            census_wsize=args.wsize,
            use_noise=False,
        )

        depth = (depth * 1000).astype(np.uint16)
        cv2.imwrite(os.path.join(img_dir, args.depth_name), depth)
        vis_depth = visualize_depth(depth)
        cv2.imwrite(os.path.join(img_dir, f"{args.depth_name[:-4]}_colored.png"), vis_depth)
        log_file.write(os.path.join(img_dir, args.depth_name))
        log_file.write(": ")
        log_file.write(get_md5(os.path.join(img_dir, args.depth_name)))
        log_file.write("\n")

    log_file.close()


if __name__ == "__main__":
    main()
