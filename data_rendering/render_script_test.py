import os
import os.path as osp
import sys
import time

import numpy as np
import sapien.core as sapien
from loguru import logger
from path import Path

CUR_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from data_rendering.render_scene import render_gt_depth_label, render_scene

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--target-root", type=str, required=True)
    args = parser.parse_args()

    spp = 128
    num_view = 21

    repo_root = REPO_ROOT
    target_root = args.target_root
    data_root = osp.join(args.target_root, "data")
    Path(data_root).makedirs_p()

    timestamp = time.strftime("%y-%m-%d_%H-%M-%S")
    name = "render_test_" + target_root.split("/")[-1]
    filename = f"log.render.test.{timestamp}.txt"
    # set up logger
    logger.remove()
    fmt = (
        f"<green>{{time:YYYY-MM-DD HH:mm:ss.SSS}}</green> | "
        f"<cyan>{name}</cyan> | "
        f"<lvl>{{level}}</lvl> | "
        f"<lvl>{{message}}</lvl>"
    )

    # logger to file
    log_file = Path(target_root) / filename
    logger.add(log_file, format=fmt)

    # logger to std stream
    logger.add(sys.stdout, format=fmt)
    logger.info(f"Args: {args}")

    sub_scene_list = [
        "0-300002",
        "0-300100",
        "0-300103",
        "0-300104",
        "0-300106",
        "0-300110",
        "0-300113",
        "0-300116",
        "0-300122",
        "0-300124",
        "0-300126",
        "0-300128",
        "0-300135",
        "0-300138",
        "0-300155",
        "0-300158",
        "0-300163",
        "0-300169",
        "0-300172",
        "0-300183",
        "0-300197",
        "1-300101",
        "1-300117",
        "1-300135",
    ]

    logger.info(f"Generating {len(sub_scene_list)} scenes from {sub_scene_list[0]} to {sub_scene_list[-1]}")

    # build scene
    sim = sapien.Engine()
    sim.set_log_level("warning")
    sapien.KuafuRenderer.set_log_level("warning")

    render_config = sapien.KuafuConfig()
    render_config.use_viewer = False
    render_config.use_denoiser = True
    render_config.spp = spp
    render_config.max_bounces = 8

    renderer = sapien.KuafuRenderer(render_config)
    sim.set_renderer(renderer)

    for sc in sub_scene_list:
        if osp.exists(osp.join(data_root, f"{sc}-{num_view-1}/meta.pkl")):
            logger.info(f"Skip scene {sc} rendering")
            continue
        logger.info(f"Rendering scene {sc}")
        render_scene(
            sim=sim,
            renderer=renderer,
            scene_id=sc,
            repo_root=repo_root,
            target_root=data_root,
            spp=spp,
            num_views=num_view,
            rand_pattern=False,
            fixed_angle=True,
            primitives=False,
            primitives_v2=False,
            rand_lighting=False,
            rand_table=False,
            rand_env=False
        )

    renderer = None
    sim = None

    sim_vk = sapien.Engine()
    sim_vk.set_log_level("warning")

    renderer = sapien.VulkanRenderer(offscreen_only=True)
    renderer.set_log_level("warning")
    sim_vk.set_renderer(renderer)

    for sc in sub_scene_list:
        if osp.exists(osp.join(data_root, f"{sc}-{num_view-1}/depthR_colored.png")):
            logger.info(f"Skip scene {sc} gt depth and seg")
            continue
        logger.info(f"Generating scene {sc} gt depth and seg")
        render_gt_depth_label(
            sim=sim_vk,
            renderer=renderer,
            scene_id=sc,
            repo_root=repo_root,
            target_root=data_root,
            spp=spp,
            num_views=num_view,
            rand_pattern=False,
            fixed_angle=True,
            primitives=False,
            primitives_v2=False,
        )
    sim_vk = None
    renderer = None
