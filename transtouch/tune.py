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
import wandb
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


def gaussian_kernel(size, sigma, x, y):
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-((i - size // 2) ** 2 + (j - size // 2) ** 2) / (2 * sigma ** 2))
    kernel /= kernel[size // 2, size // 2]
    l1, l2, r1, r2 = 0, 0, size, size
    l1 = max(l1, size // 2 - x)
    l2 = max(l2, size // 2 - y)
    r1 = min(r1, size // 2 + 540 - x)
    r2 = min(r2, size // 2 + 960 - y)
    return kernel[l1: r1, l2: r2]

if __name__ == "__main__":
    # ---------------------------------------------------------------------------- #
    # Setup the experiment
    # ---------------------------------------------------------------------------- #
    args = parse_args()
    wandb.init(project="sim2real_active_tactile", entity="sim2real_tactile")
    world_size = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    torch.cuda.set_device(local_rank)

    # Load the configuration
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    # cfg.freeze()
    config_name = args.config_file.split("/")[-1].split(".")[0]

    wandb.config = cfg

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
        f"ActiveZero2.tuning [{config_name}]", output_dir, rank=local_rank, filename=f"log.tune.{run_name}.txt"
    )
    logger.info(args)
    from active_zero2.utils.collect_env import collect_env_info

    logger.info("Collecting env info (might take some time)\n" + collect_env_info())
    logger.info(f"Loaded config file: '{args.config_file}'")
    logger.info(f"Running with configs:\n{cfg}")

    # Build tensorboard logger
    summary_writer = SummaryWriter(f"{output_dir}/{run_name}")

    # ---------------------------------------------------------------------------- #
    # Build models, optimizer, scheduler, checkpointer, etc.
    # ---------------------------------------------------------------------------- #
    # Build model
    set_random_seed(cfg.RNG_SEED)
    model = build_model(cfg)

    # Enable CUDNN benchmark
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # logger.info(f"Model: \n{model}")
    model = model.cuda()
    wandb.watch(model)

    # Build optimizer
    optimizer = build_optimizer(cfg, model)
    # Build lr_scheduler
    scheduler = build_lr_scheduler(cfg, optimizer)

    # Build checkpointer
    # Note that checkpointer will load state_dict of model, optimizer and scheduler.
    checkpointer = CheckpointerV2(
        model,
        save_dir=output_dir,
        logger=logger,
        max_to_keep=cfg.TUNE.MAX_TO_KEEP,
        local_rank=local_rank,
    )


    checkpoint_data = checkpointer.load(cfg.TUNE.WEIGHT, resume= False, strict = cfg.RESUME_STRICT)
    ckpt_period = cfg.TUNE.CHECKPOINT_PERIOD
    # Build dataloader
    # Reset the random seed again in case the initialization of models changes the random state.
    set_random_seed(cfg.RNG_SEED)
    
    tune_real_dataset = build_dataset(cfg, mode="tune", domain="real")

    if cfg.TUNE.USE_SIM:
        tune_sim_dataset = build_dataset(cfg, mode="tune", domain="sim")
    else:
        tune_sim_dataset = None

    val_real_dataset = build_dataset(cfg, mode="val", domain="real")

    if tune_real_dataset:
        sampler = RandomSampler(tune_real_dataset, replacement=False)
        batch_sampler = BatchSampler(sampler, batch_size=cfg.TUNE.BATCH_SIZE, drop_last=True)
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iterations=cfg.TUNE.MAX_ITER
        )
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
    else:
        tune_real_loader = None

    if tune_sim_dataset:
        sampler = RandomSampler(tune_sim_dataset, replacement=False)
        batch_sampler = BatchSampler(sampler, batch_size=cfg.TUNE.BATCH_SIZE, drop_last=True)
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iterations=cfg.TRAIN.MAX_ITER
        )
        tune_sim_loader = iter(
            DataLoader(
                tune_sim_dataset,
                batch_sampler=batch_sampler,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                worker_init_fn=lambda worker_id: worker_init_fn(
                    worker_id, base_seed=cfg.RNG_SEED if cfg.RNG_SEED >= 0 else None
                ),
            )
        )
    else:
        tune_sim_loader = None

    if val_real_dataset:
        val_real_loader = DataLoader(
            val_real_dataset,
            batch_size=cfg.VAL.BATCH_SIZE,
            num_workers=cfg.VAL.NUM_WORKERS,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
        )
    else:
        val_real_loader = None


    ### loading masks

    masks = {}
    for view_id in range(1, 18):
        rgb_mask = cv2.imread(f"/share/liuyu/activezero2/active_zero2/assets/real_robot_masks/m{view_id}.png")
        rgb_mask = cv2.resize(rgb_mask[:, :, 0], (960, 540))
        mask = (rgb_mask == rgb_mask.min())
        masks[view_id] = mask


    
    
    # ---------------------------------------------------------------------------- #
    # Training begins.
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


    # Validation at the beginning

    # metric.reset()
    # logger.info("Validation begins at the beginning.")


    # if val_real_loader:

    #     for iteration_val, data_batch in enumerate(val_real_loader):
    #         # copy data from cpu to gpu
    #         data_dir = data_batch["dir"][0]
    #         data_batch = {
    #             k: v.cuda(non_blocking=True) for k, v in data_batch.items() if isinstance(v, torch.Tensor)
    #         }
    #         data_batch["dir"] = data_dir
    #         data_batch["full_dir"] = os.path.join(cfg.OUTPUT_DIR, "_CACHE", data_dir)
    #         # Forward
    #         with torch.no_grad():
    #             pred_dict = model(data_batch)
    #         metric.compute(data_batch, pred_dict, save_folder="", real_data=True,)
    #         del pred_dict
    #         torch.cuda.empty_cache()
            
    # logger.info("Real Eval Metric: \n" + metric.summary())
    # train_meters.reset()


    for iteration in range(max_iter):
        # model.train()
        cur_iter = iteration + 1
        loss_dict = {}
        time_dict = {}
        optimizer.zero_grad()

        real_tic = time.time()

        loss = 0

        if tune_real_loader:
            tactile_loss = 0
            pseudo_loss = 0
            data_batch = next(tune_real_loader)
            real_data_time = time.time() - real_tic
            time_dict["time_data_real"] = real_data_time
            # Copy data from cpu to gpu
            data = {k: v.cuda(non_blocking=True) for k, v in data_batch.items() if isinstance(v, torch.Tensor)}
            pred_dict = model(data) 

            for i in range(cfg.TUNE.BATCH_SIZE):
                full_dir = data_batch["full_dir"][i]
                data_dir = data_batch["dir"][i]

                view_id = data_dir.split("-")[-1]
                mask = torch.tensor(masks[int(view_id)]).cuda()

                focal_length = data_batch["focal_length"][i]
                baseline = data_batch["baseline"][i]

                if cfg.TUNE.MODE != "None":
                    for idx in range(cfg.TUNE.NUM_TOUCH):
                        if cfg.TUNE.MODE == "Confidence":
                            with open(f"/share/datasets/sim2real_tactile/tactile/{data_dir}_real/confidence_{idx}.pkl", "rb") as file:
                                touch_dict = pickle.load(file)
                        elif cfg.TUNE.MODE == "Entropy":
                            with open(f"/share/datasets/sim2real_tactile/tactile/{data_dir}_real/entropy_{idx}.pkl", "rb") as file:
                                touch_dict = pickle.load(file)
                        elif cfg.TUNE.MODE == "Human":
                            with open(f"/share/datasets/sim2real_tactile/tactile/{data_dir}_real/human_{idx}.pkl", "rb") as file:
                                touch_dict = pickle.load(file)
                        elif cfg.TUNE.MODE == "Random":
                            with open(f"/share/datasets/sim2real_tactile/tactile/{data_dir}_real/random_{idx}.pkl", "rb") as file:
                                touch_dict = pickle.load(file)
                        else:
                            exit(0)
                        

                        x, y = touch_dict["pos"]
                        x, y = int(x), int(y)
                        disp = focal_length * baseline / touch_dict["depth"]

                        size = cfg.TUNE.GAUSSIAN_KERNEL
                        sigma = cfg.TUNE.GAUSSIAN_SIGMA


                        pred1 = pred_dict["pred1"][i, 0, 2:-2, :][x - size // 2 :x + size // 2 + 1, y - size // 2: y + size // 2 + 1]
                        pred2 = pred_dict["pred2"][i, 0, 2:-2, :][x - size // 2 :x + size // 2 + 1, y - size // 2: y + size // 2 + 1]
                        pred3 = pred_dict["pred3"][i, 0, 2:-2, :][x - size // 2 :x + size // 2 + 1, y - size // 2: y + size // 2 + 1]
                        
                        kernel = torch.tensor(gaussian_kernel(size, sigma, x, y)).cuda()
                        (mask1, gt) = data_batch["pseudo_pos"]
                        gt = gt[0, 2:-2, :][x - size // 2 :x + size // 2 + 1, y - size // 2: y + size // 2 + 1].cuda(non_blocking = True)
                    
                        target = disp * kernel + gt * (1 - kernel)

                        res = cfg.TUNE.LOSS.PRED1 * F.smooth_l1_loss(pred1, target)
                        res += cfg.TUNE.LOSS.PRED2 * F.smooth_l1_loss(pred2, target)
                        res += cfg.TUNE.LOSS.PRED3 * F.smooth_l1_loss(pred3, target)
                        # print(pred1.shape, kernel.shape, x, y)

                        tactile_loss += (res * mask[x - size // 2 :x + size // 2 + 1, y - size // 2:y + size // 2 + 1]).mean()
                
                tactile_loss /= cfg.TUNE.NUM_TOUCH

                if ((cfg.TUNE.PSEUDO_THRESHOLD < 1.0) and cfg.LOSS.TUNE_PSEUDO.WEIGHT > 0) or cfg.LOSS.TUNE_PSEUDO.WEIGHT==-1.0:
                    (mask1, gt) = data_batch["pseudo_pos"]
                    gt = gt.cuda(non_blocking = True)
                    pseudo_loss += cfg.TUNE.LOSS.PRED1*F.smooth_l1_loss(pred_dict["pred1"][i, :, :, :][mask1], gt[mask1])
                    pseudo_loss += cfg.TUNE.LOSS.PRED2*F.smooth_l1_loss(pred_dict["pred2"][i, :, :, :][mask1], gt[mask1])
                    pseudo_loss += cfg.TUNE.LOSS.PRED3*F.smooth_l1_loss(pred_dict["pred3"][i, :, :, :][mask1], gt[mask1])
            if (cfg.TUNE.PSEUDO_THRESHOLD < 1.0 and pseudo_loss != 0.0):            
                if (cfg.LOSS.TUNE_PSEUDO.WEIGHT==-1.0): 
                    ratio = (tactile_loss / pseudo_loss).clone().detach()
                    # ratio = np.exp(np.round(np.log(ratio)))
                    # print(f"pseudo loss : {pseudo_loss}, ratio : {ratio}")
                    pseudo_loss *= ratio / 4
                elif cfg.LOSS.TUNE_PSEUDO.WEIGHT > 0:
                    pseudo_loss *= cfg.LOSS.TUNE_PSEUDO.WEIGHT
                loss += pseudo_loss
                loss_dict["pseudo_loss"] = pseudo_loss

            if cfg.LOSS.REAL_REPROJ.WEIGHT != 0:
                real_reproj = model.compute_reproj_loss(
                    data,
                    pred_dict,
                    use_mask=cfg.LOSS.REAL_REPROJ.USE_MASK,
                    patch_size=cfg.LOSS.REAL_REPROJ.PATCH_SIZE,
                    only_last_pred=cfg.LOSS.REAL_REPROJ.ONLY_LAST_PRED,
                )
                if (cfg.LOSS.REAL_REPROJ.WEIGHT==-1.0):
                    ratio = (tactile_loss / real_reproj).clone().detach()
                    # print(f"real_reproj loss : {real_reproj}, ratio : {ratio}")
                    real_reproj *= ratio / 8

                else:
                    real_reproj *= cfg.LOSS.REAL_REPROJ.WEIGHT
                loss += real_reproj
                loss_dict["loss_real_reproj"] = real_reproj
            
            tactile_loss *= cfg.LOSS.TUNE_TACTILE.WEIGHT

            loss_dict["tactile_loss"] = tactile_loss
            loss += tactile_loss

            loss.backward()
        
            del pred_dict
            torch.cuda.empty_cache()
        
        # TODO recover
        if tune_sim_loader and cfg.LOSS.SIM_DISP.WEIGHT > 0:
            sim_tic = time.time()
        
            loss_sim = 0
            data_batch = next(tune_sim_loader)

            # Copy data from cpu to gpu
            data = {k: v.cuda(non_blocking=True) for k, v in data_batch.items() if isinstance(v, torch.Tensor)}
            
            pred_dict = model(data) 

            if cfg.LOSS.SIM_DISP.WEIGHT > 0:
                sim_disp = model.compute_disp_loss(data, pred_dict)
                sim_disp *= cfg.LOSS.SIM_DISP.WEIGHT
                loss_sim += sim_disp
                loss_dict["loss_sim_disp"] = sim_disp


            loss_dict["loss_sim_total"] = loss_sim 

            sim_time = time.time() - sim_tic
            time_dict["time_sim"] = sim_time

            loss_sim.backward()
        
            del pred_dict
            torch.cuda.empty_cache()

        # TODO : scale up sim loss 

        optimizer.step()
        train_meters.update(**loss_dict)

        time_dict["time_batch"] = time.time() - tic
        train_meters.update(**time_dict)

        # Logging
        log_period = cfg.TUNE.LOG_PERIOD
        if log_period > 0 and (cur_iter % log_period == 0 or cur_iter == 1):
            logger.info(
                train_meters.delimiter.join(
                    [
                        "iter: {iter:6d}",
                        "{meters}",
                        "lr: {lr:.2e}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    iter = cur_iter,
                    meters = str(train_meters),
                    lr = optimizer.param_groups[0]["lr"],
                    memory = torch.cuda.max_memory_allocated() / (1024.0**2),
                )
            )
            keywords = (
                "loss",
                "acc",
                "heading",
            )
            for name, metrics in train_meters.metrics.items():
                if all(k not in name for k in keywords):
                    continue
                summary_writer.add_scalar("train/" + name, metrics.result, global_step=cur_iter)
            summary_writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step=cur_iter)
        
        wandb.log(loss_dict)

        keywords = (
                "loss",
                "acc",
                "heading",
            )
        for name, metrics in train_meters.metrics.items():
            if all(k not in name for k in keywords):
                continue
            wandb.log({name + "_mean" : metrics.result})

        model.eval()
        
        if cfg.VAL.PERIOD != 0 and (cur_iter % cfg.VAL.PERIOD == 0 or cur_iter == max_iter):
            
            metric.reset()
            logger.info("Validation begins at iteration {}.".format(cur_iter))

            start_time_val = time.time()
            with torch.no_grad():
            # sim data
                if val_real_loader:
                    for iteration_val, data_batch in enumerate(val_real_loader):
                        # copy data from cpu to gpu
                        data_dir = data_batch["dir"][0]
                        data_batch = {
                            k: v.cuda(non_blocking=True) for k, v in data_batch.items() if isinstance(v, torch.Tensor)
                        }
                        data_batch["dir"] = data_dir
                        data_batch["full_dir"] = os.path.join(cfg.OUTPUT_DIR, "_CACHE", data_dir)
                        # Forward
                        pred_dict = model(data_batch)
                        metric.compute(
                            data_batch,
                            pred_dict,
                            save_folder="",
                            real_data=True,
                        )
                        del pred_dict
                        torch.cuda.empty_cache()

            # END: validation loop
            epoch_time_val = time.time() - start_time_val
            logger.info(
                "Iteration[{}]-Val total_time: {:.2f}s".format(cur_iter, epoch_time_val)
            )

            logger.info("Real Eval Metric: \n" + metric.summary())
            # restore training
            train_meters.reset()

        # ---------------------------------------------------------------------------- #
        # After validation
        # ---------------------------------------------------------------------------- #
        # Checkpoint
        if (ckpt_period > 0 and cur_iter % ckpt_period == 0) or cur_iter == max_iter:
            checkpoint_data["iteration"] = cur_iter
            checkpointer.save("model_{:06d}".format(cur_iter), **checkpoint_data)

        # ---------------------------------------------------------------------------- #
        # Finalize one step
        # ---------------------------------------------------------------------------- #
        # Since pytorch v1.1.0, lr_scheduler is called after optimization.
        if scheduler is not None:
            scheduler.step()
        tic = time.time()
