import os
import os.path as osp
import sys
import time

import numpy as np
import sapien.core as sapien
from loguru import logger
from path import Path
import json
import pickle

CUR_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from data_rendering.utils.folder_paths import *
from data_rendering.utils.render_utils import NUM_OBJECTS, OBJECT_NAMES, OBJECT_DB

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-s",
        "--split-file",
        type=str,
        metavar="FILE",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--data-folder",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    timestamp = time.strftime("%y-%m-%d_%H-%M-%S")
    filename = f"log.meta.{timestamp}.txt"
    logger.remove()
    fmt = (
        f"<green>{{time:YYYY-MM-DD HH:mm:ss.SSS}}</green> | "
        f"<cyan>meta</cyan> | "
        f"<lvl>{{level}}</lvl> | "
        f"<lvl>{{message}}</lvl>"
    )
    log_file = Path(args.data_folder) / filename
    logger.add(log_file, format=fmt)

    # logger to std stream
    logger.add(sys.stdout, format=fmt)
    logger.info(f"Args: {args}")
    with open(args.split_file, "r") as f:
        prefix = [line.strip() for line in f]

    n = len(prefix)
    start = time.time()

    data_folder = Path(args.data_folder)

    for idx in range(n):
        p = prefix[idx]
        d = data_folder / p
        if not (d / "meta.pkl").exists():
            logger.warning(f"{d}/meta.pkl not exists.")
            continue
        try:
            meta = pickle.load(open(d / "meta.pkl", "rb"))
        except:
            logger.error(f"fail to open {d}/meta.pkl")
            continue
        if "poses_world" in meta:
            logger.info(f"Skip {d}/meta.pkl")
            continue

        scene_id = p.split("-")[0] + "-" + p.split("-")[1]
        world_js = json.load(open(os.path.join(SCENE_DIR, f"{scene_id}/input.json"), "r"))
        assets = world_js.keys()
        poses_world = [None for _ in range(NUM_OBJECTS)]
        extents = [None for _ in range(NUM_OBJECTS)]
        scales = [None for _ in range(NUM_OBJECTS)]
        obj_ids = []
        object_names = []

        for obj_name in assets:
            obj_id = OBJECT_NAMES.index(obj_name)
            obj_ids.append(obj_id)
            object_names.append(obj_name)
            poses_world[obj_id] = world_js[obj_name]
            extents[obj_id] = np.array(
                [
                    float(OBJECT_DB[obj_name]["x_dim"]),
                    float(OBJECT_DB[obj_name]["y_dim"]),
                    float(OBJECT_DB[obj_name]["z_dim"]),
                ],
                dtype=np.float32,
            )
            scales[obj_id] = np.ones(3)
        obj_info = {
            "poses_world": poses_world,
            "extents": extents,
            "scales": scales,
            "object_ids": obj_ids,
            "object_names": object_names,
        }
        meta.update(obj_info)
        with open(d / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        logger.info(f"Update {d}/meta.pkl")
