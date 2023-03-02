import random

import numpy as np
import torch
import torchvision.transforms as Transforms
import torchvision.transforms.functional as TF


class DataAug:
    def __init__(self, data_aug_cfg=None, domain="sim"):
        self.data_aug_cfg = data_aug_cfg
        self.domain = domain
        self.aug = (data_aug_cfg is not None) and (domain in data_aug_cfg.DOMAINS)

    def __call__(self, x):
        if x.ndim == 2:
            x = torch.from_numpy(x).unsqueeze(0)
        else:
            x = torch.from_numpy(x).permute(2, 0, 1)

        if self.aug:
            if self.data_aug_cfg.GAUSSIAN_BLUR:
                gaussian_sig = random.uniform(self.data_aug_cfg.GAUSSIAN_MIN, self.data_aug_cfg.GAUSSIAN_MAX)
                x = TF.gaussian_blur(x, kernel_size=self.data_aug_cfg.GAUSSIAN_KERNEL, sigma=gaussian_sig)
            if self.data_aug_cfg.COLOR_JITTER:
                n = x.shape[0]
                xs = []
                brightness = random.uniform(self.data_aug_cfg.BRIGHT_MIN, self.data_aug_cfg.BRIGHT_MAX)
                contrast = random.uniform(self.data_aug_cfg.CONTRAST_MIN, self.data_aug_cfg.CONTRAST_MAX)
                saturation = random.uniform(self.data_aug_cfg.SATURATION_MIN, self.data_aug_cfg.SATURATION_MAX)
                gamma = random.uniform(self.data_aug_cfg.GAMMA_MIN, self.data_aug_cfg.GAMMA_MAX)
                for i in range(n):
                    xi = x[i:i+1]
                    xi = TF.adjust_brightness(xi, brightness)
                    xi = TF.adjust_contrast(xi, contrast)
                    xi = TF.adjust_saturation(xi, saturation)
                    xi = TF.adjust_gamma(xi, gamma)
                    xs.append(xi)
                x = torch.concat(xs, dim=0)
        x = TF.normalize(x, mean=0.45, std=0.224)
        return x


def data_augmentation(data_aug_cfg=None, domain="sim"):
    """ """
    transform_list = [Transforms.ToTensor()]
    if data_aug_cfg and domain in data_aug_cfg.DOMAINS:
        if data_aug_cfg.GAUSSIAN_BLUR:
            gaussian_sig = random.uniform(data_aug_cfg.GAUSSIAN_MIN, data_aug_cfg.GAUSSIAN_MAX)
            transform_list += [Transforms.GaussianBlur(kernel_size=data_aug_cfg.GAUSSIAN_KERNEL, sigma=gaussian_sig)]
        if data_aug_cfg.COLOR_JITTER:
            transform_list += [
                Transforms.ColorJitter(
                    brightness=[data_aug_cfg.BRIGHT_MIN, data_aug_cfg.BRIGHT_MAX],
                    contrast=[data_aug_cfg.CONTRAST_MIN, data_aug_cfg.CONTRAST_MAX],
                    saturation=[data_aug_cfg.SATURATION_MIN, data_aug_cfg.SATURATION_MAX],
                    hue=[data_aug_cfg.HUE_MIN, data_aug_cfg.HUE_MAX],
                )
            ]
    # Normalization
    transform_list += [
        Transforms.Normalize(
            mean=[0.45],
            std=[0.224],
        )
    ]
    custom_augmentation = Transforms.Compose(transform_list)
    return custom_augmentation


class SimIRNoise(object):
    def __init__(self, data_aug_cfg=None, domain="sim"):
        if data_aug_cfg is not None and domain == "sim":
            self.sim_ir = data_aug_cfg.SIM_IR
            self.speckle_shape_min = data_aug_cfg.SPECKLE_SHAPE_MIN
            self.speckle_shape_max = data_aug_cfg.SPECKLE_SHAPE_MAX
            self.gaussian_mu = data_aug_cfg.GAUSSIAN_MU
            self.gaussian_sigma = data_aug_cfg.GAUSSIAN_SIGMA
        else:
            self.sim_ir = False

    def apply(self, img):
        if self.sim_ir:
            speckle_shape = np.random.uniform(self.speckle_shape_min, self.speckle_shape_max)
            img = img * np.random.gamma(shape=speckle_shape, scale=1.0 / speckle_shape, size=img.shape)
            img = img + self.gaussian_mu + self.gaussian_sigma * np.random.standard_normal(img.shape)
        return img
