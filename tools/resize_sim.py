import argparse
import multiprocessing
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

parser = argparse.ArgumentParser(description="Extract LCN IR pattern from IR images")
parser.add_argument(
    "-s",
    "--split-file",
    type=str,
    metavar="FILE",
    required=True,
)
parser.add_argument(
    "-d",
    "--data-folder",
    type=str,
    required=True,
)
parser.add_argument("--mp", type=int, default=1, help="multi-process")
args = parser.parse_args()



def sub_main(prefix_list):
    n = len(prefix_list)
    start = time.time()
    for idx in range(n):
        for direction in ["irL", "irR"]:
            p = prefix_list[idx]
            f6 = os.path.join(args.data_folder, p, f"0128_{direction}_kuafu_half.png")
            img_6 = np.array(Image.open(f6).convert(mode="L"))
            h = img_6.shape[0]
            assert h in (540, 720, 1080), f"Illegal img shape: {img_6.shape}"
            if h in (720, 1080):
                img_6 = cv2.resize(img_6, (960, 540), interpolation=cv2.INTER_CUBIC)

            print(f"Generating {p} resized {direction} pattern {idx}/{n} time: {time.time() - start:.2f}s")
            cv2.imwrite(os.path.join(args.data_folder, p, f"0128_{direction}_kuafu_half_540.png"), img_6)


def main():
    print("Multi-processing: ", args.mp)
    with open(args.split_file, "r") as f:
        prefix = [line.strip() for line in f]
    num = len(prefix)
    assert num % args.mp == 0
    l = num // args.mp

    p_list = []
    for i in range(args.mp):
        p = multiprocessing.Process(target=sub_main, args=(prefix[i * l : (i + 1) * l],))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()


if __name__ == "__main__":
    main()
