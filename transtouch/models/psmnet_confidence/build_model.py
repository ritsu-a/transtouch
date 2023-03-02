from active_zero2.models.psmnet_confidence.psmnet_3 import PSMNetConfidence


def build_model(cfg):
    model = PSMNetConfidence(
        min_disp=cfg.PSMNetConfidence.MIN_DISP,
        max_disp=cfg.PSMNetConfidence.MAX_DISP,
        num_disp=cfg.PSMNetConfidence.NUM_DISP,
        set_zero=cfg.PSMNetConfidence.SET_ZERO,
        alpha=cfg.PSMNetConfidence.ALPHA
    )
    return model
