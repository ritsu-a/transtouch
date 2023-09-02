#!/usr/bin/env python
import os
import os.path as osp
import sys

import tensorboardX

_ROOT_DIR = os.path.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, _ROOT_DIR)
import argparse
import gc
import time
import warnings
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

from transtouch.config import cfg
from transtouch.datasets.build_dataset import build_dataset
from transtouch.models.build_model import build_model
from transtouch.utils.cfg_utils import purge_cfg
from transtouch.utils.checkpoint import CheckpointerV2
from transtouch.utils.loguru_logger import setup_logger
from transtouch.utils.metric_logger import MetricLogger
from transtouch.utils.reduce import set_random_seed, synchronize
from transtouch.utils.sampler import IterationBasedBatchSampler
from transtouch.utils.solver import build_lr_scheduler, build_optimizer
from transtouch.utils.metrics import ErrorMetric
from transtouch.utils.torch_utils import worker_init_fn


def parse_args():
    parser = argparse.ArgumentParser(description="transtouch")
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
        f"transtouch.generate mask [{config_name}]", output_dir, rank = 0, filename=f"log.train.{run_name}.txt"
    )

    set_random_seed(cfg.RNG_SEED)
    
    tune_real_dataset = build_dataset(cfg, mode="tune", domain="real")

    test_data_loader=DataLoader(
        tune_real_dataset,
        batch_size=1,
        num_workers=cfg.TUNE.NUM_WORKERS,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )

    model = build_model(cfg)
    model = model.cuda()
    checkpointer = CheckpointerV2(
        model,
        save_dir=output_dir,
        logger=logger,
        max_to_keep=cfg.TUNE.MAX_TO_KEEP,
        local_rank=0,
    )
    model.eval()
    checkpoint_data = checkpointer.load(cfg.TUNE.WEIGHT, resume = False, strict = cfg.RESUME_STRICT)
    f = open(osp.join(output_dir, "position.txt"),'w')

    for iteration_val, data_batch in enumerate(test_data_loader):
        data_dir = data_batch["dir"][0]
        data_batch_i = {k: v.cuda(non_blocking=True) for k, v in data_batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            pred = model(data_batch_i)
        disp_gt = data_batch_i["img_disp_l"].cpu().numpy()[0, 0, 2:-2]
        mask = np.load(os.path.join(cfg.OUTPUT_DIR, "_CACHE", data_dir, "mask_new.npy"))
        mask_obj = mask.copy()
        con = pred["conf_map"].cpu().numpy()[0, cfg.TUNE.CONF_RANGE-1, 2 : -2, :]
        con_blur = cv2.GaussianBlur(con, (cfg.TUNE.TOUCH_SIZE, cfg.TUNE.TOUCH_SIZE), 0)
        disp_err = np.abs(pred["pred3"].cpu().numpy()[0, 0, 2 : -2, :] - disp_gt)
        disp_err_blur = cv2.GaussianBlur(disp_err, (cfg.TUNE.TOUCH_SIZE, cfg.TUNE.TOUCH_SIZE), 0)
        (n, m) = con.shape

        save_folder=osp.join(output_dir, data_dir + "_real")
        os.makedirs(save_folder, exist_ok=True)


        for i in range(n):
            for j in range(m):
                if i < 25 or j < 25 or i > n - 25 or j > m - 25:
                    mask[i, j] = 0
                    mask_obj[i, j] = 0
                else:
                    (l1,r1) = (i - (cfg.TUNE.TOUCH_SIZE // 2) - 3, i + ((cfg.TUNE.TOUCH_SIZE + 1) // 2) + 3)
                    (l2,r2) = (j - (cfg.TUNE.TOUCH_SIZE // 2) - 3, j + ((cfg.TUNE.TOUCH_SIZE + 1) // 2) + 3)
                    label=data_batch["img_label_l"][0, l1 + 2 : r1 + 2, l2 : r2]
                    if (label.max() > label.min() or label.max() == 17):
                        mask_obj[i, j] = 0

        touch=[]
        if (cfg.TUNE.MODE == "confidence"):
            for i in range(cfg.TUNE.NUM_TOUCH):
                x, y = np.where(con_blur == con_blur[mask].min())
                p = np.random.randint(len(x))
                x, y = x[p], y[p]
                touch.append((x+2, y))
                con_blur[x-cfg.TUNE.TOUCH_SIZE: x+cfg.TUNE.TOUCH_SIZE, y-cfg.TUNE.TOUCH_SIZE: y+cfg.TUNE.TOUCH_SIZE] = 1
        elif (cfg.TUNE.MODE=="random"):
            for i in range(cfg.TUNE.NUM_TOUCH):
                pos = np.random.randint(len(con_blur[mask]))
                x, y = con_blur[mask][pos]
                touch.append((x+2,y))
        elif (cfg.TUNE.MODE=="oracle_loss"):
            for i in range(cfg.TUNE.NUM_TOUCH):
                x, y = np.where(disp_err_blur == disp_err_blur[mask].max())
                p = np.random.randint(len(x))
                x, y = x[p], y[p]
                touch.append((x+2, y))
                disp_err_blur[x-cfg.TUNE.TOUCH_SIZE: x+cfg.TUNE.TOUCH_SIZE, y-cfg.TUNE.TOUCH_SIZE: y+cfg.TUNE.TOUCH_SIZE] = 0
        elif (cfg.TUNE.MODE=="sameobject_loss"):
            for i in range(cfg.TUNE.NUM_TOUCH):
                x, y = np.where(disp_err_blur == disp_err_blur[mask_obj].max())
                p = np.random.randint(len(x))
                x, y = x[p], y[p]
                touch.append((x+2, y))
                disp_err_blur[x-cfg.TUNE.TOUCH_SIZE: x+cfg.TUNE.TOUCH_SIZE, y-cfg.TUNE.TOUCH_SIZE: y+cfg.TUNE.TOUCH_SIZE] = 0
        st = data_dir
        for (x,y) in touch:
            st += ' '+'('+str(x)+','+str(y)+')'
        print(st, file = f)

        disp_diff_before = disp_gt - pred["pred3"].detach().cpu().numpy()[0, 0, 2:-2]
        
        disp_diff_before = np.clip(disp_diff_before, -8, 8)
        disp_err_before = np.clip(disp_diff_before + 1e5 * (1-mask), -8, 8)
        disp_err_before=((disp_err_before+8)/16*255).astype('uint8')
        disp_err_before=cv2.applyColorMap(disp_err_before, cv2.COLORMAP_JET)

        for (x,y) in touch:
            x+=2
            cv2.circle(disp_err_before,(y,x),20,(255,255,255),2)
            cv2.circle(label,(y,x),20,(255,),2)
            cv2.circle(disp_err_before,(y,x),1,(255,255,255),2)
            cv2.circle(label,(y,x),1,(255,),2)

        plt.imsave(os.path.join(save_folder, "disp_before.png"), disp_err_before)