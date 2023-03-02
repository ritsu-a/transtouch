from active_zero2.models.psmnet_confidence.build_model import build_model as build_psmnet_confidence
from active_zero2.models.psmnet_dilation.build_model import build_model as build_psmnet_dilation
from active_zero2.models.deeplabv3.build_model import build_model as build_deeplabv3
from active_zero2.models.psmnet_dilation_error_segmentation.build_model import build_model as build_psmnet_dilation_error_segmentation
from active_zero2.models.psmnet_dilation_error_segmentation_volume.build_model import build_model as build_psmnet_dilation_error_segmentation_volume
from active_zero2.models.psmnet_edge_normal.build_model import build_model as build_psmnet_edge_normal

MODEL_LIST = (
    "PSMNetConfidence",
    "PSMNetDilation",
    "DeepLabV3",
    "PSMNetDilationErrorSegmentation",
    "PSMNetDilationErrorSegmentationVolume",
    "PSMNetEdgeNormal"
)


def build_model(cfg):
    if cfg.MODEL_TYPE == "PSMNetConfidence":
        model = build_psmnet_confidence(cfg)
    elif cfg.MODEL_TYPE == "PSMNetDilation":
        model = build_psmnet_dilation(cfg)
    elif cfg.MODEL_TYPE == 'DeepLabV3':
        model = build_deeplabv3(cfg)
    elif cfg.MODEL_TYPE == 'PSMNetDilationErrorSegmentation' :
        model = build_psmnet_dilation_error_segmentation(cfg)
    elif cfg.MODEL_TYPE == 'PSMNetDilationErrorSegmentationVolume' :
        model = build_psmnet_dilation_error_segmentation_volume(cfg)
    elif cfg.MODEL_TYPE == 'PSMNetEdgeNormal':
        model = build_psmnet_edge_normal(cfg)
    else:
        raise ValueError(f"Unexpected model type: {cfg.MODEL_TYPE}")

    return model
