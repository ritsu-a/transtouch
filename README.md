CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port='29505' active_zero2/train.py --cfg configs/pengyang_normal.yml

CUDA_VISIBLE_DEVICES= python active_zero2/train.py --cfg configs/liuyu.yml

test:
CUDA_VISIBLE_DEVICES=4 python active_zero2/test.py --cfg configs/tune_confidence10.yml -s TEST.WEIGHT /share/pengyang/sim2real_active_tactile/outputs/tune_confidence10/model_002000.pth

CUDA_VISIBLE_DEVICES=0 python active_zero2/test.py --cfg configs/tune_confidence40_5_3e-5_sim0.5.yml -s TEST.WEIGHT /share/pengyang/sim2real_active_tactile/outputs_before0217/tune_confidence40_5_3e-5_sim0.5/model_005600.pth

CUDA_VISIBLE_DEVICES=1 python active_zero2/test.py --cfg configs/tune_baseline.yml -s TEST.WEIGHT /share/pengyang/pretrained/model_070000.pth 



test normal:
CUDA_VISIBLE_DEVICES=1 python active_zero2/test.py -n --cfg configs/pretrained_edge.yml -s TEST.WEIGHT /share/pengyang/pretrained/model_070000.pth 


fine-tuning:

generating temporal IR images
python tools/temporal_lcn_ir.py -s /share/liuyu/activezero2/split_files/ts_tune_16.txt -d /share/datasets/sim2real_tactile/touch -p 11 -t 0.005


render data
python data_rendering/render_script.py --sub 1 --total 1 --target-root /share/pengyang/data/test --primitives-v2

train on psmnet edge normal

CUDA_VISIBLE_DEVICES=1 python active_zero2/tuning.py --cfg configs/baseline.yml TUNE.WEIGHT /share/pengyang/pretrained/model_070000.pth

CUDA_VISIBLE_DEVICES=2 python active_zero2/tuning.py --cfg configs/baseline_confidence.yml TUNE.WEIGHT /share/pengyang/pretrained/model_070000.pth

CUDA_VISIBLE_DEVICES=3 python active_zero2/tuning.py --cfg configs/baseline_human.yml TUNE.WEIGHT /share/pengyang/pretrained/model_070000.pth

CUDA_VISIBLE_DEVICES=4 python active_zero2/tuning.py --cfg configs/baseline_random.yml TUNE.WEIGHT /share/pengyang/pretrained/model_070000.pth

CUDA_VISIBLE_DEVICES=5 python active_zero2/tuning.py --cfg configs/baseline_nopseudo.yml TUNE.WEIGHT /share/pengyang/pretrained/model_070000.pth

CUDA_VISIBLE_DEVICES=7 python active_zero2/tuning.py --cfg configs/baseline_nospread.yml TUNE.WEIGHT /share/pengyang/pretrained/model_070000.pth

CUDA_VISIBLE_DEVICES=0 python active_zero2/tuning.py --cfg configs/baseline1.yml TUNE.WEIGHT /share/pengyang/pretrained/model_070000.pth

CUDA_VISIBLE_DEVICES=1 python active_zero2/tuning.py --cfg configs/baseline2.yml TUNE.WEIGHT /share/pengyang/pretrained/model_070000.pth


How to run tuning:
1.Pick patches
2.Generate tuning mask with tactile_patch.py in render
3.Run tuning, change path of the mask above



gradient_test:
CUDA_VISIBLE_DEVICES=5 python active_zero2/gradient_test.py --cfg configs/gradient_0214_3.yml TUNE.WEIGHT /share/pengyang/pretrained/model_070000.pth OPTIMIZER.LR 1e-4


generate normal
python tools/gen_normal_map.py -s /share/pengyang/sim2real_active_tactile/split_files/ts_all.txt -d /share/datasets/sim2real_tactile/real/dataset/ --sub 1 --total 1 >gen_normal.log


gradient reverse test
CUDA_VISIBLE_DEVICES=7 python active_zero2/gradient_reverse.py --cfg configs/gradient_reverse.yml TUNE.WEIGHT /share/pengyang/pretrained/model_070000.pth OPTIMIZER.LR 1e-7


temporal consistency test
CUDA_VISIBLE_DEVICES=0 python active_zero2/temporal_consistency.py --cfg configs/pengyang_temporal_consistency.yml -s TEST.WEIGHT /share/pengyang/sim2real_active_tactile/outputs_before/outputs_before0105/pengyang_dilation/model_060000.pth

test with normal
CUDA_VISIBLE_DEVICES=7 python active_zero2/gradient_temp.py --cfg configs/gradient_temp.yml TUNE.WEIGHT /share/pengyang/pretrained/model_070000.pth OPTIMIZER.LR 1e-5

CUDA_VISIBLE_DEVICES=4 python active_zero2/confidence_and_random.py --cfg configs/gradient_temp.yml TUNE.WEIGHT /share/pengyang/pretrained/model_070000.pth OPTIMIZER.LR 1e-5