#!/usr/bin/env python
import os
import os.path as osp
import sys
import pickle

import tensorboardX

_ROOT_DIR = os.path.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, _ROOT_DIR)
import argparse
import gc
import time
import warnings
# import wandb
import cv2
import matplotlib.pyplot as plt
import numpy as np


import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn import SyncBatchNorm
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from active_zero2.config import cfg
from active_zero2.datasets.build_dataset import build_dataset
from active_zero2.models.build_model import build_model
from active_zero2.utils.cfg_utils import purge_cfg
from active_zero2.utils.checkpoint import CheckpointerV2
from active_zero2.utils.loguru_logger import setup_logger
from active_zero2.utils.metric_logger import MetricLogger
from active_zero2.utils.reduce import set_random_seed, synchronize
from active_zero2.utils.sampler import IterationBasedBatchSampler
from active_zero2.utils.solver import build_lr_scheduler, build_optimizer
from active_zero2.utils.metrics import ErrorMetric
from active_zero2.utils.torch_utils import worker_init_fn


def parse_args():
    parser = argparse.ArgumentParser(description="ActiveZero2")
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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # ---------------------------------------------------------------------------- #
    # Setup the experiment
    # ---------------------------------------------------------------------------- #
    args = parse_args()
    world_size = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    torch.cuda.set_device(local_rank)

    # Load the configuration
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    config_name = args.config_file.split("/")[-1].split(".")[0]


    # run name
    timestamp = time.strftime("%y-%m-%d_%H-%M-%S")
    run_name = "{:s}".format(timestamp)

    # Parse the output directory
    output_dir = cfg.OUTPUT_DIR
    # Replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace("@", config_path.replace("configs", "outputs"))
        os.makedirs(output_dir, exist_ok=True)
        cfg.OUTPUT_DIR = output_dir

    logger = setup_logger(
        f"ActiveZero2.gradient_reverse [{config_name}]", output_dir, rank=local_rank, filename=f"log.gradient_reverse.{run_name}.txt"
    )
    logger.info(args)
    from active_zero2.utils.collect_env import collect_env_info

    # Build tensorboard logger
    summary_writer = SummaryWriter(f"{output_dir}/{run_name}")

    # ---------------------------------------------------------------------------- #
    # Build models, optimizer, scheduler, checkpointer, etc.
    # ---------------------------------------------------------------------------- #
    # Build model
    set_random_seed(cfg.RNG_SEED)
    model = build_model(cfg)

    # logger.info(f"Model: \n{model}")
    model = model.cuda()
    # wandb.watch(model)

    # Build checkpointer
    # Note that checkpointer will load state_dict of model, optimizer and scheduler.
    checkpointer = CheckpointerV2(
        model,
        save_dir=output_dir,
        logger=logger,
        max_to_keep=cfg.TUNE.MAX_TO_KEEP,
        local_rank=local_rank,
    )

    # Build dataloader
    # Reset the random seed again in case the initialization of models changes the random state.
    set_random_seed(cfg.RNG_SEED)

    tune_real_dataset = build_dataset(cfg, mode="tune", domain="real")


    sampler = RandomSampler(tune_real_dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=cfg.TUNE.BATCH_SIZE, drop_last=True)
    tune_real_loader = iter(
        DataLoader(
            tune_real_dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.TUNE.NUM_WORKERS,
            worker_init_fn=lambda worker_id: worker_init_fn(
                worker_id, base_seed=cfg.RNG_SEED if cfg.RNG_SEED >= 0 else None
            ),
        )
    )


    ### loading masks

    masks = {}
    for view_id in range(1, 18):
        rgb_mask = cv2.imread(f"/share/liuyu/activezero2/active_zero2/assets/real_robot_masks/m{view_id}.png")
        rgb_mask = cv2.resize(rgb_mask[:, :, 0], (960, 540))
        mask = (rgb_mask == rgb_mask.min())
        masks[view_id] = mask


    # ---------------------------------------------------------------------------- #
    # Gradient reverse test begins.
    # ---------------------------------------------------------------------------- #
    train_meters = MetricLogger()

    metric = ErrorMetric(
        model_type=cfg.MODEL_TYPE,
        use_mask=cfg.VAL.USE_MASK,
        max_disp=cfg.VAL.MAX_DISP,
        depth_range=cfg.VAL.DEPTH_RANGE,
        num_classes=cfg.DATA.NUM_CLASSES,
        is_depth=cfg.VAL.IS_DEPTH,
    )

    model.eval()
    max_iter = cfg.TUNE.MAX_ITER
    tic = time.time()


    for idx in range(len(tune_real_dataset)):
        checkpointer.load(cfg.TUNE.WEIGHT, resume= False, strict = cfg.RESUME_STRICT, log=False)
        optimizer = build_optimizer(cfg, model)
        data_batch = tune_real_dataset[idx]
        data_dir = data_batch["dir"]
        full_dir = data_batch["full_dir"]
        view_id = int(data_dir.split("-")[-1])
        mask = masks[view_id]
        data_folder = output_dir + "/" + data_dir
        os.makedirs(data_folder, exist_ok=True)

        tactile_pth = "/share/liuyu/activezero2/tactile_files"+ "/" + data_dir + "_real"
        os.makedirs(tactile_pth, exist_ok=True)

        data = {k: v.unsqueeze(0).cuda(non_blocking=True) for k, v in data_batch.items() if k == "img_l" or k == "img_r" or k == "img_pattern_r" or k == "img_pattern_l" or k == "focal_length" or k == "baseline" or k == "intrinsic_l" or k =="extrinsic_l"}
        
        pred_dict = model(data)

        con_before = pred_dict["conf_map"].detach().cpu().numpy()[0, 5, 2:-2, :]
        pred3_before = pred_dict["pred3"].detach().cpu().numpy()[0, 0, 2:-2, :]


        con = pred_dict["conf_map"].detach().cpu().numpy()[0, 5, 2:-2, :]
        con_clip = con < cfg.TUNE.CONF_THRESHOLD
        con_clip *= mask
        con_clip = np.array(con_clip, dtype = np.float32)
        con_touch_map = con_clip.copy()
        clip_blur = cv2.GaussianBlur(con_clip, (cfg.TUNE.TOUCH_SIZE, cfg.TUNE.TOUCH_SIZE), 0) * con_clip

        plt.imsave(
                        os.path.join(data_folder, f"confidence_clip_0.png"), con_clip, vmin=0, vmax=1, cmap="jet"
                    )
        plt.imsave(
                        os.path.join(data_folder, f"confidence_0.png"), con, vmin=0, vmax=1, cmap="jet"
                    )
        plt.imsave(
                        os.path.join(data_folder, f"confidence_blur_0.png"), clip_blur, vmin=0, vmax=1, cmap="jet"
                    )

        touch_info = {}
        focal_length = data_batch["focal_length"]
        baseline = data_batch["baseline"]
        touch_info["extrinsic"] = data_batch["extrinsic_l"].numpy()
        touch_info["intrinsic"] = data_batch["intrinsic_l"].numpy() * 1280 / 960
        touch_info["intrinsic"][2, 2] = 1

        # touch_interval = 10
        touches = []
        for i in range(cfg.TUNE.NUM_TOUCH):
            if (i > 0):
                checkpointer.load(cfg.TUNE.WEIGHT, resume = False, strict = cfg.RESUME_STRICT, log = False)
                optimizer = build_optimizer(cfg, model)
            x, y = np.where(clip_blur == clip_blur.max())
            x = x[0]
            y = y[0]
            print(i, clip_blur.max(), x, y)
            touches.append((x, y))
            num_tune = 0
            while True:
                if (num_tune > 0 or i > 0):
                    del pred_dict
                    torch.cuda.empty_cache()
                    pred_dict = model(data)
                if (pred_dict["conf_map"][0, 5, x + 2, y].cpu() > (con_before[x, y] + 2) / 3.0):
                    break
                num_tune += 1
                loss = 0
                disp_gt = pred3_before[x, y] - cfg.PSMNetEdgeNormal.MIN_DISP
                disp_gt = disp_gt / (cfg.PSMNetEdgeNormal.MAX_DISP - cfg.PSMNetEdgeNormal.MIN_DISP) * cfg.PSMNetEdgeNormal.NUM_DISP
                prob_cost3 = pred_dict["prob_cost3"][0, :, x + 2, y]
                for j in range(cfg.PSMNetEdgeNormal.NUM_DISP):
                    if (prob_cost3[j] != 0):
                        loss -= prob_cost3[j] * torch.log(prob_cost3[j])
                # for j in range(cfg.PSMNetEdgeNormal.NUM_DISP):
                #     loss += (np.abs(disp_gt - j) ) * prob_cost3[j]
                print(loss, i, num_tune)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            con_diff = pred_dict["conf_map"].detach().cpu().numpy()[0, 5, 2:-2, :] - con_before
            print("confidence change from ", con_before[x, y], " to ", pred_dict["conf_map"].detach().cpu().numpy()[0, 5, x + 2, y])
            plt.imsave(
                            os.path.join(data_folder, f"con_diff_{i + 1}.png"), con_diff, vmin=-1, vmax=1, cmap="jet"
                        )
            con_diff_percent = con_diff / (1 - con_before + 1e-6) * con_clip
            plt.imsave(
                            os.path.join(data_folder, f"con_diff_percent_{i + 1}.png"), con_diff_percent, vmin=-1, vmax=1, cmap="jet"
                        )

            con_clip -= con_diff_percent > 0.5
            clip_blur = cv2.GaussianBlur(con_clip, (cfg.TUNE.TOUCH_SIZE, cfg.TUNE.TOUCH_SIZE), 0) * con_clip

            plt.imsave(
                            os.path.join(data_folder, f"confidence_clip_{i + 1}.png"), con_clip, vmin=0, vmax=1, cmap="jet"
                        )
            plt.imsave(
                            os.path.join(data_folder, f"confidence_blur_{i + 1}.png"), clip_blur, vmin=0, vmax=1, cmap="jet"
                        )


            touch_info["depth"] = focal_length.numpy() * baseline.numpy() / pred3_before[x, y] * 1000
            x_scaled = int(x / 960 * 1280)
            y_scaled = int(y / 960 * 1280)
            # touch_info["pos"] = (x_scaled, y_scaled)
            touch_info["pos"] = (x, y)

            with open(f"{tactile_pth}/entropy_{i}.pkl", "wb") as f:
                pickle.dump(touch_info, f)

        print(touches)

        for touch in touches:
            (x, y) = touch

        touch = 0
        for x, y in touches:
            touch += 1
            cv2.circle(con_touch_map,(int(y),int(x)),20,(0.5,),2)
            cv2.circle(con_touch_map,(int(y),int(x)),1,(0.5,),2)
            cv2.putText(con_touch_map, str(touch), (y,x), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        plt.imsave(
                        os.path.join(data_folder, f"entropy_touch.png"), con_touch_map, vmin=0, vmax=1, cmap="jet"
                    )

        plt.imsave(
                        os.path.join(tactile_pth, f"entropy_touch.png"), con_touch_map, vmin=0, vmax=1, cmap="jet"
                    )

        rgb = cv2.imread(os.path.join(full_dir, "1024_rgb_real.png"))

        cv2.imwrite(os.path.join(data_folder, f"rgb.png"), rgb)

        logger.info(f"{data_folder}, confidence position:{touches}")

        del pred_dict
        torch.cuda.empty_cache()

