CUDA_VISIBLE_DEVICES=5 python transtouch/test.py --cfg configs/baseline2.yml -s TEST.WEIGHT /share/liuyu/useful_checkpoints/baseline.pth

CUDA_VISIBLE_DEVICES=3 python transtouch/test.py --cfg configs/activezero2.yml -s TEST.WEIGHT /share/pengyang/pretrained/model_070000.pth