from path import Path
import random
import sys
import os
import os.path as osp
REPO_ROOT = os.path.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from sapien.core import VulkanRenderer
from tqdm import tqdm
import cv2
import numpy as np

CUR_DIR = os.path.dirname(__file__)

from data_rendering.utils.folder_paths import TEXTURE_SQ_FOLDER, TEXTURE_SQ_LIST, ENV_MAP_FOLDER, ENV_MAP_LIST

# ENV_MAP_FOLDER = "/media/DATA/LINUX_DATA/activezero2/datasets/rand_env/"
Path(ENV_MAP_FOLDER).makedirs_p()
# ENV_MAP_LIST = "/media/DATA/LINUX_DATA/activezero2/datasets/rand_env/list.txt"

with open(TEXTURE_SQ_LIST, "r") as f:
    texture_sq_list = [line.strip() for line in f]

fenv = open(ENV_MAP_LIST, "w")


def get_random_sq_texture():
    random_file = random.choice(texture_sq_list)
    path = os.path.join(TEXTURE_SQ_FOLDER, random_file)
    return path


num_env = 10000

renderer = VulkanRenderer(offscreen_only=True)

for i in range(6):
    img = cv2.imread(get_random_sq_texture())
    img = img.astype(np.float32)
    scale = 255 / img.max()
    img *= scale
    img *= np.random.rand() * 0.2
    img = img.astype(np.uint8)
    cv2.imwrite(os.path.join(CUR_DIR, f"{i}.png"), img)


for e in tqdm(range(num_env)):
    renderer.create_ktx_environment_map(
        os.path.join(CUR_DIR, "0.png"),
        os.path.join(CUR_DIR, "1.png"),
        os.path.join(CUR_DIR, "2.png"),
        os.path.join(CUR_DIR, "3.png"),
        os.path.join(CUR_DIR, "4.png"),
        os.path.join(CUR_DIR, "5.png"),
        os.path.join(ENV_MAP_FOLDER, f"{e:04d}.ktx"),
    )
    fenv.write(f"{e:04d}.ktx\n")
