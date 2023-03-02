from active_zero2.models.deeplabv3.deeplabv3 import DeepLabV3


def build_model(cfg):
    model = DeepLabV3()
    return model
