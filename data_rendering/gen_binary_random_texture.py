import os
import os.path as osp
import sys

_ROOT_DIR = os.path.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, _ROOT_DIR)
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from data_rendering.utils.folder_paths import TEXTURE_FOLDER, TEXTURE_LIST


with open(TEXTURE_LIST, "r") as f:
    texture_list = [line.strip() for line in f]

for l in tqdm(texture_list):
    img = cv2.imread(osp.join(TEXTURE_FOLDER, l), cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (21, 21), sigmaX=9)
    img = ((img > 100) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(osp.join(TEXTURE_FOLDER, l[:-4] + "_bin.png"), img)
