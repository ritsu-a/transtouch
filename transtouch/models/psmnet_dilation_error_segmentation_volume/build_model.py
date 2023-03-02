from active_zero2.models.psmnet_dilation_error_segmentation_volume.psmnet_3 import PSMNetDilationErrorSegmentationVolume


def build_model(cfg):
    model = PSMNetDilationErrorSegmentationVolume(
        num_ir=cfg.DATA.NUM_IR,
        min_disp=cfg.PSMNetDilation.MIN_DISP,
        max_disp=cfg.PSMNetDilation.MAX_DISP,
        num_disp=cfg.PSMNetDilation.NUM_DISP,
        set_zero=cfg.PSMNetDilation.SET_ZERO,
        dilation=cfg.PSMNetDilation.DILATION,
        use_off=cfg.PSMNetDilation.USE_OFF,
        alpha=cfg.PSMNetDilation.ALPHA
    )
    return model
