## Train Pre Activated ResNets on tiny-imagenet-200
pip install tensorboardX
# preactresnet18
python train.py --name tiny-imagenet-200 --arch preactresnet18 --seed 1
python train.py --name tiny-imagenet-200 --arch preactresnet18 --seed 1 --mixmethod mixup --alpha 1
python train.py --name tiny-imagenet-200 --arch preactresnet18 --seed 1 --mixmethod ManifoldMixup --alpha 1
python train.py --name tiny-imagenet-200 --arch preactresnet18 --seed 1 --mixmethod ShuffleMix --alpha 1 --mix_type soft --ratio 0.5 --index_type random --k_layer1 -1 --k_layer2 4

# preactresnet34
python train.py --name tiny-imagenet-200 --arch preactresnet34 --seed 1
python train.py --name tiny-imagenet-200 --arch preactresnet34 --seed 1 --mixmethod mixup --alpha 1
python train.py --name tiny-imagenet-200 --arch preactresnet34 --seed 1 --mixmethod ManifoldMixup --alpha 1
python train.py --name tiny-imagenet-200 --arch preactresnet34 --seed 1 --mixmethod ShuffleMix --alpha 1 --mix_type soft --ratio 0.5 --index_type random --k_layer1 -1 --k_layer2 4

# preactwideresnet18
python train.py --name tiny-imagenet-200 --arch preactwideresnet18 --seed 1
python train.py --name tiny-imagenet-200 --arch preactwideresnet18 --seed 1 --mixmethod mixup --alpha 1
python train.py --name tiny-imagenet-200 --arch preactwideresnet18 --seed 1 --mixmethod ManifoldMixup --alpha 1
python train.py --name tiny-imagenet-200 --arch preactwideresnet18 --seed 1 --mixmethod ShuffleMix --alpha 1 --mix_type soft --ratio 0.5 --index_type random --k_layer1 -1 --k_layer2 4