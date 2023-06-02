import torch
from torchvision import datasets, transforms
from src.data.tiny_imagenet_200 import TinyImageNet

def getData(name='cifar10', train_bs=128, test_bs=512, num_workers=8, data_root=None):
    # data root
    if data_root is None:
        data_root = "./data"

    if name == 'cifar10':
        num_classes=10
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_data = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        test_data = datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)

    elif name == 'cifar100':
        num_classes=100
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_data = datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
        test_data = datasets.CIFAR100(root=data_root, train=False, download=False, transform=transform_test)

    elif name == 'svhn':
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
        train_transform = transforms.Compose(
            [ transforms.RandomCrop(32, padding=2), transforms.ToTensor(),
              transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_data = datasets.SVHN(root=data_root, split='train', transform=train_transform, download=False)
        test_data = datasets.SVHN(root=data_root, split='test', transform=test_transform, download=False)

    elif name == 'tiny-imagenet-200':
        num_classes = 200
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(64, padding=4),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])

        train_data = TinyImageNet(data_root, split='train', download=False, transform=train_transform)
        test_data = TinyImageNet(data_root, split='val', download=False, transform=test_transform)


    elif name == 'imagenet':
        assert False, 'Do not finish imagenet code'
    else:
        assert False, 'Do not support dataset : {}'.format(name)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_bs, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_bs, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, num_classes
