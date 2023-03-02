from yacs.config import CfgNode as CN
from yacs.config import load_cfg

_C = CN()
cfg = _C

# ---------------------------------------------------------------------------- #
# Resume
# ---------------------------------------------------------------------------- #
# Automatically resume weights from last checkpoints
_C.AUTO_RESUME = True
# Whether to resume the optimizer and the scheduler
_C.RESUME_STATES = True
# Path of weights to resume
_C.RESUME_PATH = ""
_C.RESUME_STRICT = True

# if set to @, the filename of config will be used by default
_C.OUTPUT_DIR = "@"

# For reproducibility...but not really because modern fast GPU libraries use
# non-deterministic op implementations
# -1 means not to set explicitly.
_C.RNG_SEED = 1

# ---------------------------------------------------------------------------- #
# DATA
# ---------------------------------------------------------------------------- #
_C.DATA = CN()
_C.DATA.NUM_CLASSES = 25
_C.DATA.NUM_IR = 1

_C.DATA.TRAIN = CN()

_C.DATA.TRAIN.SIM = CN()
_C.DATA.TRAIN.SIM.ROOT_DIR = ""
_C.DATA.TRAIN.SIM.SPLIT_FILE = ""
_C.DATA.TRAIN.SIM.HEIGHT = 256
_C.DATA.TRAIN.SIM.WIDTH = 512
_C.DATA.TRAIN.SIM.META_NAME = ""
_C.DATA.TRAIN.SIM.DEPTH_NAME = ""
_C.DATA.TRAIN.SIM.NORMAL_NAME = ""
_C.DATA.TRAIN.SIM.NORMAL_CONF_NAME = ""
_C.DATA.TRAIN.SIM.LEFT_NAMES = ()
_C.DATA.TRAIN.SIM.LEFT_NAME = ""
_C.DATA.TRAIN.SIM.RIGHT_NAMES = ()
_C.DATA.TRAIN.SIM.RIGHT_NAME = ""
_C.DATA.TRAIN.SIM.LEFT_PATTERN_NAME = ""
_C.DATA.TRAIN.SIM.RIGHT_PATTERN_NAME = ""
_C.DATA.TRAIN.SIM.LABEL_NAME = ""
_C.DATA.TRAIN.SIM.DEPTH_R_NAME = ""
_C.DATA.TRAIN.SIM.LEFT_OFF_NAME = ""
_C.DATA.TRAIN.SIM.RIGHT_OFF_NAME = ""
_C.DATA.TRAIN.SIM.EDGE_SIGMA_RANGE = (0.5, 2.5)
_C.DATA.TRAIN.REAL = CN()
_C.DATA.TRAIN.REAL.ROOT_DIR = ""
_C.DATA.TRAIN.REAL.SPLIT_FILE = ""
_C.DATA.TRAIN.REAL.HEIGHT = 256
_C.DATA.TRAIN.REAL.WIDTH = 512
_C.DATA.TRAIN.REAL.META_NAME = ""
_C.DATA.TRAIN.REAL.DEPTH_NAME = ""
_C.DATA.TRAIN.REAL.NORMAL_NAME = ""
_C.DATA.TRAIN.REAL.NORMAL_CONF_NAME = ""
_C.DATA.TRAIN.REAL.LEFT_NAMES = ()
_C.DATA.TRAIN.REAL.LEFT_NAME = ""
_C.DATA.TRAIN.REAL.RIGHT_NAMES = ()
_C.DATA.TRAIN.REAL.RIGHT_NAME = ""
_C.DATA.TRAIN.REAL.LEFT_PATTERN_NAME = ""
_C.DATA.TRAIN.REAL.RIGHT_PATTERN_NAME = ""
_C.DATA.TRAIN.REAL.LABEL_NAME = ""
_C.DATA.TRAIN.REAL.DEPTH_R_NAME = ""
_C.DATA.TRAIN.REAL.LEFT_OFF_NAME = ""
_C.DATA.TRAIN.REAL.RIGHT_OFF_NAME = ""
_C.DATA.TRAIN.REAL.EDGE_SIGMA_RANGE = (0.5, 2.5)

_C.DATA.VAL = CN()

_C.DATA.VAL.SIM = CN()
_C.DATA.VAL.SIM.ROOT_DIR = ""
_C.DATA.VAL.SIM.SPLIT_FILE = ""
_C.DATA.VAL.SIM.HEIGHT = 256
_C.DATA.VAL.SIM.WIDTH = 512
_C.DATA.VAL.SIM.META_NAME = ""
_C.DATA.VAL.SIM.DEPTH_NAME = ""
_C.DATA.VAL.SIM.NORMAL_NAME = ""
_C.DATA.VAL.SIM.NORMAL_CONF_NAME = ""
_C.DATA.VAL.SIM.LEFT_NAMES = ()
_C.DATA.VAL.SIM.LEFT_NAME = ""
_C.DATA.VAL.SIM.RIGHT_NAMES = ()
_C.DATA.VAL.SIM.RIGHT_NAME = ""
_C.DATA.VAL.SIM.LEFT_PATTERN_NAME = ""
_C.DATA.VAL.SIM.RIGHT_PATTERN_NAME = ""
_C.DATA.VAL.SIM.LABEL_NAME = ""
_C.DATA.VAL.SIM.DEPTH_R_NAME = ""
_C.DATA.VAL.SIM.LEFT_OFF_NAME = ""
_C.DATA.VAL.SIM.RIGHT_OFF_NAME = ""
_C.DATA.VAL.SIM.EDGE_SIGMA_RANGE = (0.5, 2.5)
_C.DATA.VAL.REAL = CN()
_C.DATA.VAL.REAL.ROOT_DIR = ""
_C.DATA.VAL.REAL.SPLIT_FILE = ""
_C.DATA.VAL.REAL.HEIGHT = 256
_C.DATA.VAL.REAL.WIDTH = 512
_C.DATA.VAL.REAL.META_NAME = ""
_C.DATA.VAL.REAL.DEPTH_NAME = ""
_C.DATA.VAL.REAL.NORMAL_NAME = ""
_C.DATA.VAL.REAL.NORMAL_CONF_NAME = ""
_C.DATA.VAL.REAL.LEFT_NAMES = ()
_C.DATA.VAL.REAL.RIGHT_NAMES = ()
_C.DATA.VAL.REAL.LEFT_PATTERN_NAME = ""
_C.DATA.VAL.REAL.RIGHT_PATTERN_NAME = ""
_C.DATA.VAL.REAL.LABEL_NAME = ""
_C.DATA.VAL.REAL.DEPTH_R_NAME = ""
_C.DATA.VAL.REAL.LEFT_OFF_NAME = ""
_C.DATA.VAL.REAL.RIGHT_OFF_NAME = ""
_C.DATA.VAL.REAL.EDGE_SIGMA_RANGE = (0.5, 2.5)

_C.DATA.TEST = CN()

_C.DATA.TEST.SIM = CN()
_C.DATA.TEST.SIM.ROOT_DIR = ""
_C.DATA.TEST.SIM.SPLIT_FILE = ""
_C.DATA.TEST.SIM.HEIGHT = 544
_C.DATA.TEST.SIM.WIDTH = 960
_C.DATA.TEST.SIM.META_NAME = ""
_C.DATA.TEST.SIM.DEPTH_NAME = ""
_C.DATA.TEST.SIM.NORMAL_NAME = ""
_C.DATA.TEST.SIM.NORMAL_CONF_NAME = ""
_C.DATA.TEST.SIM.LEFT_NAMES = ()
_C.DATA.TEST.SIM.LEFT_NAME = ""
_C.DATA.TEST.SIM.RIGHT_NAMES = ()
_C.DATA.TEST.SIM.RIGHT_NAME = ""
_C.DATA.TEST.SIM.LEFT_PATTERN_NAME = ""
_C.DATA.TEST.SIM.RIGHT_PATTERN_NAME = ""
_C.DATA.TEST.SIM.LABEL_NAME = ""
_C.DATA.TEST.SIM.DEPTH_R_NAME = ""
_C.DATA.TEST.SIM.LEFT_OFF_NAME = ""
_C.DATA.TEST.SIM.RIGHT_OFF_NAME = ""
_C.DATA.TEST.SIM.EDGE_SIGMA_RANGE = (0.5, 2.5)
_C.DATA.TEST.REAL = CN()
_C.DATA.TEST.REAL.ROOT_DIR = ""
_C.DATA.TEST.REAL.SPLIT_FILE = ""
_C.DATA.TEST.REAL.HEIGHT = 544
_C.DATA.TEST.REAL.WIDTH = 960
_C.DATA.TEST.REAL.META_NAME = ""
_C.DATA.TEST.REAL.DEPTH_NAME = ""
_C.DATA.TEST.REAL.NORMAL_NAME = ""
_C.DATA.TEST.REAL.NORMAL_CONF_NAME = ""
_C.DATA.TEST.REAL.LEFT_NAMES = ()
_C.DATA.TEST.REAL.LEFT_NAME = ""
_C.DATA.TEST.REAL.RIGHT_NAMES = ()
_C.DATA.TEST.REAL.RIGHT_NAME = ""
_C.DATA.TEST.REAL.LEFT_PATTERN_NAME = ""
_C.DATA.TEST.REAL.RIGHT_PATTERN_NAME = ""
_C.DATA.TEST.REAL.LABEL_NAME = ""
_C.DATA.TEST.REAL.DEPTH_R_NAME = ""
_C.DATA.TEST.REAL.LEFT_OFF_NAME = ""
_C.DATA.TEST.REAL.RIGHT_OFF_NAME = ""
_C.DATA.TEST.REAL.EDGE_SIGMA_RANGE = (0.5, 2.5)

_C.DATA.TUNE = CN()

_C.DATA.TUNE.SIM = CN()
_C.DATA.TUNE.SIM.ROOT_DIR = ""
_C.DATA.TUNE.SIM.SPLIT_FILE = ""
_C.DATA.TUNE.SIM.HEIGHT = 544
_C.DATA.TUNE.SIM.WIDTH = 960
_C.DATA.TUNE.SIM.META_NAME = ""
_C.DATA.TUNE.SIM.DEPTH_NAME = ""
_C.DATA.TUNE.SIM.NORMAL_NAME = ""
_C.DATA.TUNE.SIM.NORMAL_CONF_NAME = ""
_C.DATA.TUNE.SIM.LEFT_NAMES = ()
_C.DATA.TUNE.SIM.LEFT_NAME = ""
_C.DATA.TUNE.SIM.RIGHT_NAMES = ()
_C.DATA.TUNE.SIM.RIGHT_NAME = ""
_C.DATA.TUNE.SIM.LEFT_PATTERN_NAME = ""
_C.DATA.TUNE.SIM.RIGHT_PATTERN_NAME = ""
_C.DATA.TUNE.SIM.LABEL_NAME = ""
_C.DATA.TUNE.SIM.DEPTH_R_NAME = ""
_C.DATA.TUNE.SIM.LEFT_OFF_NAME = ""
_C.DATA.TUNE.SIM.RIGHT_OFF_NAME = ""
_C.DATA.TUNE.SIM.EDGE_SIGMA_RANGE = (0.5, 2.5)
_C.DATA.TUNE.REAL = CN()
_C.DATA.TUNE.REAL.ROOT_DIR = ""
_C.DATA.TUNE.REAL.SPLIT_FILE = ""
_C.DATA.TUNE.REAL.HEIGHT = 544
_C.DATA.TUNE.REAL.WIDTH = 960
_C.DATA.TUNE.REAL.META_NAME = ""
_C.DATA.TUNE.REAL.DEPTH_NAME = ""
_C.DATA.TUNE.REAL.NORMAL_NAME = ""
_C.DATA.TUNE.REAL.NORMAL_CONF_NAME = ""
_C.DATA.TUNE.REAL.LEFT_NAMES = ()
_C.DATA.TUNE.REAL.LEFT_NAME = ""
_C.DATA.TUNE.REAL.RIGHT_NAMES = ()
_C.DATA.TUNE.REAL.RIGHT_NAME = ""
_C.DATA.TUNE.REAL.LEFT_PATTERN_NAME = ""
_C.DATA.TUNE.REAL.RIGHT_PATTERN_NAME = ""
_C.DATA.TUNE.REAL.LABEL_NAME = ""
_C.DATA.TUNE.REAL.DEPTH_R_NAME = ""
_C.DATA.TUNE.REAL.LEFT_OFF_NAME = ""
_C.DATA.TUNE.REAL.RIGHT_OFF_NAME = ""
_C.DATA.TUNE.REAL.EDGE_SIGMA_RANGE = (0.5, 2.5)

# data augmentation
_C.DATA_AUG = CN()
_C.DATA_AUG.DOMAINS = (
    "sim",
    "real",
)
_C.DATA_AUG.COLOR_JITTER = True
_C.DATA_AUG.GAUSSIAN_BLUR = True
_C.DATA_AUG.GAUSSIAN_MIN = 0.1
_C.DATA_AUG.GAUSSIAN_MAX = 2.0
_C.DATA_AUG.GAUSSIAN_KERNEL = 9
_C.DATA_AUG.BRIGHT_MIN = 0.4
_C.DATA_AUG.BRIGHT_MAX = 1.4
_C.DATA_AUG.CONTRAST_MIN = 0.8
_C.DATA_AUG.CONTRAST_MAX = 1.2
_C.DATA_AUG.SATURATION_MIN = 0.4
_C.DATA_AUG.SATURATION_MAX = 1.6
_C.DATA_AUG.HUE_MIN = -0.4
_C.DATA_AUG.HUE_MAX = 0.4
_C.DATA_AUG.GAMMA_MIN = 0.5
_C.DATA_AUG.GAMMA_MAX = 1.5

_C.DATA_AUG.SIM_IR = False
_C.DATA_AUG.SPECKLE_SHAPE_MIN = 350
_C.DATA_AUG.SPECKLE_SHAPE_MAX = 450
_C.DATA_AUG.GAUSSIAN_MU = -1e-3
_C.DATA_AUG.GAUSSIAN_SIGMA = 4e-3

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #

# PSMNet, PSMNetRay
_C.MODEL_TYPE = ""


_C.PSMNetConfidence = CN()
_C.PSMNetConfidence.MIN_DISP = 0
_C.PSMNetConfidence.MAX_DISP = 0
_C.PSMNetConfidence.NUM_DISP = 0
_C.PSMNetConfidence.SET_ZERO = False
_C.PSMNetConfidence.ALPHA = 0

_C.PSMNetDilation = CN()
_C.PSMNetDilation.MIN_DISP = 0
_C.PSMNetDilation.MAX_DISP = 0
_C.PSMNetDilation.NUM_DISP = 0
_C.PSMNetDilation.SET_ZERO = False
_C.PSMNetDilation.DILATION = 3
_C.PSMNetDilation.USE_OFF = False
_C.PSMNetDilation.ALPHA = 0.0


_C.PSMNetEdgeNormal = CN()
_C.PSMNetEdgeNormal.MIN_DISP = 0
_C.PSMNetEdgeNormal.MAX_DISP = 0
_C.PSMNetEdgeNormal.NUM_DISP = 0
_C.PSMNetEdgeNormal.SET_ZERO = False
_C.PSMNetEdgeNormal.DILATION = 3
_C.PSMNetEdgeNormal.EPSILON = 1.0
_C.PSMNetEdgeNormal.GRAD_THRESHOLD = 100.0
_C.PSMNetEdgeNormal.USE_OFF = False
_C.PSMNetEdgeNormal.USE_VOLUME = False
_C.PSMNetEdgeNormal.EDGE_WEIGHT = 10.0


_C.SMDNet = CN()
# data
_C.SMDNet.NUM_SAMPLES = 0
_C.SMDNet.DILATION = 10
# network
_C.SMDNet.OUTPUT_REPRESENTATION = "bimodal"
_C.SMDNet.MAX_DISP = 192
_C.SMDNet.NO_SINE = False
_C.SMDNet.NO_RESIDUAL = False

# ---------------------------------------------------------------------------- #
# Losses
# ---------------------------------------------------------------------------- #

_C.LOSS = CN()
_C.LOSS.SIM_REPROJ = CN()
_C.LOSS.SIM_REPROJ.WEIGHT = 0.0
_C.LOSS.SIM_REPROJ.USE_MASK = True
_C.LOSS.SIM_REPROJ.PATCH_SIZE = 11
_C.LOSS.SIM_REPROJ.ONLY_LAST_PRED = True
_C.LOSS.SIM_DISP = CN()
_C.LOSS.SIM_DISP.WEIGHT = 1.0

_C.LOSS.REAL_REPROJ = CN()
_C.LOSS.REAL_REPROJ.WEIGHT = 1.0
_C.LOSS.REAL_REPROJ.USE_MASK = False
_C.LOSS.REAL_REPROJ.PATCH_SIZE = 11
_C.LOSS.REAL_REPROJ.ONLY_LAST_PRED = True
_C.LOSS.REAL_DISP = CN()
_C.LOSS.REAL_DISP.WEIGHT = 0.0

_C.LOSS.CONFIDENCE = CN()
_C.LOSS.CONFIDENCE.WEIGHT = 50.0
_C.LOSS.CONFIDENCE.USE_MASK = True

_C.LOSS.TUNE_TACTILE = CN()
_C.LOSS.TUNE_TACTILE.WEIGHT = 1.0

_C.LOSS.TUNE_PSEUDO = CN()
_C.LOSS.TUNE_PSEUDO.WEIGHT = 1.0


# ---------------------------------------------------------------------------- #
# Optimizer
# ---------------------------------------------------------------------------- #
_C.OPTIMIZER = CN()
_C.OPTIMIZER.TYPE = ""

# Basic parameters of the optimizer
# Note that the learning rate should be changed according to batch size
_C.OPTIMIZER.LR = 1e-3
_C.OPTIMIZER.WEIGHT_DECAY = 0.0
# Maximum norm of gradients. Non-positive for disable
_C.OPTIMIZER.MAX_GRAD_NORM = 0.0

# Specific parameters of optimizers
_C.OPTIMIZER.SGD = CN()
_C.OPTIMIZER.SGD.momentum = 0

_C.OPTIMIZER.Adam = CN()
_C.OPTIMIZER.Adam.betas = (0.9, 0.999)

# ---------------------------------------------------------------------------- #
# Scheduler (learning rate schedule)
# ---------------------------------------------------------------------------- #
_C.LR_SCHEDULER = CN()
_C.LR_SCHEDULER.TYPE = ""

# Specific parameters of schedulers
_C.LR_SCHEDULER.StepLR = CN()
_C.LR_SCHEDULER.StepLR.step_size = 0
_C.LR_SCHEDULER.StepLR.gamma = 0.1

_C.LR_SCHEDULER.MultiStepLR = CN()
_C.LR_SCHEDULER.MultiStepLR.milestones = ()
_C.LR_SCHEDULER.MultiStepLR.gamma = 0.1

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

# Batch size
_C.TRAIN.BATCH_SIZE = 1
# Number of workers (dataloader)
_C.TRAIN.NUM_WORKERS = 1
# Period to save checkpoints. 0 for disable
_C.TRAIN.CHECKPOINT_PERIOD = 1
# Period to log training status. 0 for disable
_C.TRAIN.LOG_PERIOD = 100
# Max number of checkpoints to keep
_C.TRAIN.MAX_TO_KEEP = 5
# Max number of iteration
_C.TRAIN.MAX_ITER = 1


# ---------------------------------------------------------------------------- #
# Specific validation options
# ---------------------------------------------------------------------------- #
_C.VAL = CN()

# Batch size
_C.VAL.BATCH_SIZE = 1
# Number of workers (dataloader)
_C.VAL.NUM_WORKERS = 1
# Period to validate. 0 for disable
_C.VAL.PERIOD = 0
# Period to log validation status. 0 for disable
_C.VAL.LOG_PERIOD = 100

# The metric for best validation performance
_C.VAL.METRIC = ""
_C.VAL.METRIC_ASCEND = True

_C.VAL.USE_MASK = True
_C.VAL.MAX_DISP = 192
_C.VAL.DEPTH_RANGE = (0.2, 1.25)
_C.VAL.IS_DEPTH = False


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

_C.TEST.BATCH_SIZE = 1
_C.TEST.NUM_WORKERS = 1
# The path of weights to be tested. "@" has similar syntax as OUTPUT_DIR.
# If not set, the last checkpoint will be used by default.
_C.TEST.WEIGHT = ""

_C.TEST.LOG_PERIOD = 10
_C.TEST.METRIC = ""

_C.TEST.USE_MASK = True
_C.TEST.MAX_DISP = 192
_C.TEST.DEPTH_RANGE = (0.2, 1.25)
_C.TEST.IS_DEPTH = False


_C.TUNE = CN()

_C.TUNE.BATCH_SIZE = 1
_C.TUNE.NUM_WORKERS = 1
# The path of weights to be tested. "@" has similar syntax as OUTPUT_DIR.
# If not set, the last checkpoint will be used by default.
_C.TUNE.WEIGHT = ""
_C.TUNE.MODE = "confidence"
_C.TUNE.AREA = "none"

_C.TUNE.NUM_TOUCH = 1
_C.TUNE.TOUCH_SIZE = 5
_C.TUNE.GAUSSIAN_SIGMA = 1.0
_C.TUNE.GAUSSIAN_KERNEL = 5
_C.TUNE.PSEUDO_THRESHOLD = 1.0
_C.TUNE.CONF_THRESHOLD = 1.0

_C.TUNE.LR = 1E-4

_C.TUNE.MAX_ITER = 2000
_C.TUNE.LOG_PERIOD = 50
_C.TUNE.MAX_TO_KEEP = 5
_C.TUNE.CHECKPOINT_PERIOD = 200
_C.TUNE.METRIC = ""


_C.TUNE.USE_MASK = True
_C.TUNE.MAX_DISP = 192
_C.TUNE.CONF_RANGE = 2
_C.TUNE.DEPTH_RANGE = (0.2, 1.25)
_C.TUNE.IS_DEPTH = False

_C.TUNE.LOSS = CN()
_C.TUNE.LOSS.PRED1 = 0.5
_C.TUNE.LOSS.PRED2 = 0.7
_C.TUNE.LOSS.PRED3 = 1.0

_C.TUNE.USE_SIM = False

_C.TUNE.TACTILE_PATH = "tune"

# ---------------------------------------------------------------------------- #
# Adversarial Learning
# ---------------------------------------------------------------------------- #

PI = 3.1415926
_C.PSMNetADV = CN()
_C.PSMNetADV.MIN_DISP = 0
_C.PSMNetADV.MAX_DISP = 0
_C.PSMNetADV.NUM_DISP = 0
_C.PSMNetADV.SET_ZERO = False
_C.PSMNetADV.DILATION = 3
_C.PSMNetADV.EPSILON = 1.0
_C.PSMNetADV.D_CHANNELS = 16
_C.PSMNetADV.DISP_ENCODING = (PI / 32, PI / 8, PI / 2)
_C.PSMNetADV.WGANGP_NORM = 1.0
_C.PSMNetADV.WGANGP_LAMBDA = 10.0
_C.PSMNetADV.USE_SIM_PRED = False  # ADV between sim disp pred and real disp pred or gt disp and real disp pred

_C.PSMNetADV4 = CN()
_C.PSMNetADV4.MIN_DISP = 0
_C.PSMNetADV4.MAX_DISP = 0
_C.PSMNetADV4.NUM_DISP = 0
_C.PSMNetADV4.SET_ZERO = False
_C.PSMNetADV4.DILATION = 3
_C.PSMNetADV4.EPSILON = 1.0
_C.PSMNetADV4.GRAD_THRESHOLD = 100.0
_C.PSMNetADV4.D_CHANNELS = 16
_C.PSMNetADV4.DISP_ENCODING = (PI / 32, PI / 8, PI / 2)
_C.PSMNetADV4.WGANGP_NORM = 1.0
_C.PSMNetADV4.WGANGP_LAMBDA = 10.0
_C.PSMNetADV4.USE_SIM_PRED = False  # ADV between sim disp pred and real disp pred or gt disp and real disp pred
_C.PSMNetADV4.D_TYPE = "D3"
_C.PSMNetADV4.DISP_GRAD_NORM = "L1"

_C.PSMNetGrad2DADV = CN()
_C.PSMNetGrad2DADV.MIN_DISP = 0
_C.PSMNetGrad2DADV.MAX_DISP = 0
_C.PSMNetGrad2DADV.NUM_DISP = 0
_C.PSMNetGrad2DADV.SET_ZERO = False
_C.PSMNetGrad2DADV.DILATION = 3
_C.PSMNetGrad2DADV.EPSILON = 1.0
_C.PSMNetGrad2DADV.GRAD_THRESHOLD = 100.0
_C.PSMNetGrad2DADV.D_CHANNELS = 16
_C.PSMNetGrad2DADV.WGANGP_NORM = 1.0
_C.PSMNetGrad2DADV.WGANGP_LAMBDA = 10.0
_C.PSMNetGrad2DADV.USE_SIM_PRED = False  # ADV between sim disp pred and real disp pred or gt disp and real disp pred
_C.PSMNetGrad2DADV.SUB_AVG_SIZE = 0
_C.PSMNetGrad2DADV.DISP_GRAD_NORM = "L1"

_C.PSMNetInpainting = CN()
_C.PSMNetInpainting.MIN_DISP = 0
_C.PSMNetInpainting.MAX_DISP = 0
_C.PSMNetInpainting.NUM_DISP = 0
_C.PSMNetInpainting.SET_ZERO = False
_C.PSMNetInpainting.DILATION = 3
_C.PSMNetInpainting.EPSILON = 2.0
_C.PSMNetInpainting.GRAD_THRESHOLD = 100.0
_C.PSMNetInpainting.SUB_AVG_SIZE = 0
_C.PSMNetInpainting.DISP_GRAD_NORM = "L1"
_C.PSMNetInpainting.USE_OFF = True
_C.PSMNetInpainting.USE_EDGE = True
_C.PSMNetInpainting.CONF_RANGE = (0.65, 0.85)


_C.G_OPTIMIZER = CN()
_C.G_OPTIMIZER.TYPE = ""

# Basic parameters of the optimizer
# Note that the learning rate should be changed according to batch size
_C.G_OPTIMIZER.LR = 1e-3
_C.G_OPTIMIZER.WEIGHT_DECAY = 0.0

_C.G_OPTIMIZER.Adam = CN()
_C.G_OPTIMIZER.Adam.betas = (0.5, 0.9)

_C.D_OPTIMIZER = CN()
_C.D_OPTIMIZER.TYPE = ""

# Basic parameters of the optimizer
# Note that the learning rate should be changed according to batch size
_C.D_OPTIMIZER.LR = 1e-4

_C.D_OPTIMIZER.Adam = CN()
_C.D_OPTIMIZER.Adam.betas = (0.0, 0.9)

_C.STEREO_OPTIMIZER = CN()
_C.STEREO_OPTIMIZER.TYPE = ""

# Basic parameters of the optimizer
# Note that the learning rate should be changed according to batch size
_C.STEREO_OPTIMIZER.LR = 1e-4

_C.STEREO_OPTIMIZER.Adam = CN()
_C.STEREO_OPTIMIZER.Adam.betas = (0.9, 0.999)


_C.INP_OPTIMIZER = CN()
_C.INP_OPTIMIZER.TYPE = ""

# Basic parameters of the optimizer
# Note that the learning rate should be changed according to batch size
_C.INP_OPTIMIZER.LR = 1e-4

_C.INP_OPTIMIZER.Adam = CN()
_C.INP_OPTIMIZER.Adam.betas = (0.0, 0.9)

# use the adversarial loss after ADV_ITER
_C.ADV_ITER = 0
_C.LOSS.ADV = 1.0

_C.LOSS.SIM_GRAD = 0.01
_C.LOSS.REAL_GRAD = 0.03

_C.LOSS.EDGE = 1.0
_C.LOSS.SIM_NORMAL = 1.0
_C.LOSS.REAL_NORMAL = 1.0


### pseudo loss
_C.USE_PSEUDO_LOSS = False