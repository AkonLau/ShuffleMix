## Train Pre Activated ResNets on cifar100
pip install tensorboardX
# preactresnet18
python train.py --name cifar100 --arch preactresnet18 --seed 1 --add_noise_level 0.4 --mult_noise_level 0.2
python train.py --name cifar100 --arch preactresnet18 --seed 1 --mixmethod mixup --alpha 1 --add_noise_level 0.4 --mult_noise_level 0.2
python train.py --name cifar100 --arch preactresnet18 --seed 1 --mixmethod ManifoldMixup --alpha 1 --add_noise_level 0.4 --mult_noise_level 0.2
python train.py --name cifar100 --arch preactresnet18 --seed 1 --mixmethod ShuffleMix --alpha 1 --mix_type soft --ratio 0.5 --index_type random --k_layer1 -1 --k_layer2 4 --add_noise_level 0.4 --mult_noise_level 0.2

# preactwideresnet18
python train.py --name cifar100 --arch preactwideresnet18 --seed 1 --add_noise_level 0.4 --mult_noise_level 0.2
python train.py --name cifar100 --arch preactwideresnet18 --seed 1 --mixmethod mixup --alpha 1 --add_noise_level 0.4 --mult_noise_level 0.2
python train.py --name cifar100 --arch preactwideresnet18 --seed 1 --mixmethod ManifoldMixup --alpha 1 --add_noise_level 0.4 --mult_noise_level 0.2
python train.py --name cifar100 --arch preactwideresnet18 --seed 1 --mixmethod ShuffleMix --alpha 1 --mix_type soft --ratio 0.5 --index_type random --k_layer1 -1 --k_layer2 4 --add_noise_level 0.4 --mult_noise_level 0.2
