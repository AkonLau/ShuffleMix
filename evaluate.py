import os
import torch
import torch.optim
import torch.utils.data
import argparse
from src.data.get_data import getData
import numpy as np

def white_noisy_validate(val_loader, model, time_begin=None):
    perturbed_test_accs = []
    noise_levels = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35]
    # noise_levels = [0, 0.1, 0.2, 0.3]
    for eps in noise_levels:
        model.eval()
        acc1_val = 0
        n = 0
        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                images += eps * torch.cuda.FloatTensor(images.shape).normal_()
                output = model(images)

                model_logits = output[0] if (type(output) is tuple) else output
                pred = model_logits.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                acc1_val += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                n += len(images)

        avg_acc1 = (acc1_val / n)
        perturbed_test_accs.append(avg_acc1)
    return perturbed_test_accs

def sp(image, amount):
    row,col = image.shape
    s_vs_p = 0.5
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    idx = np.random.choice(range(32*32), np.int(num_salt), False)
    out = out.reshape(image.size, -1)
    out[idx] = np.min(out)
    out = out.reshape(32,32)

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    idx = np.random.choice(range(32*32), np.int(num_pepper), False)
    out = out.reshape(image.size, -1)
    out[idx] = np.max(out)
    out = out.reshape(32,32)
    return out

def sp_wrapper(data, amount):
    np.random.seed(12345)
    for i in range(data.shape[0]):
        data_numpy = data[i,0,:,:].data.cpu().numpy()
        noisy_input = sp(data_numpy, amount)
        data[i,0,:,:] = torch.tensor(noisy_input).float().to('cuda')

    return data

def sp_noise_validate(val_loader, model, time_begin=None):
    perturbed_test_accs = []
    noise_levels = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35]
    # noise_levels = [0, 0.02, 0.04, 0.1]

    for eps in noise_levels:
        model.eval()
        acc1_val = 0
        n = 0
        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)


                images = sp_wrapper(images, eps)
                output = model(images)

                model_logits = output[0] if (type(output) is tuple) else output
                pred = model_logits.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                acc1_val += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                n += len(images)

        avg_acc1 = (acc1_val / n)
        perturbed_test_accs.append(avg_acc1)
    return perturbed_test_accs

def cls_validate(val_loader, model, time_begin=None):
    model.eval()
    acc1_val = 0
    n = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(images)

            model_logits = output[0] if (type(output) is tuple) else output
            pred = model_logits.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            acc1_val += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            n += len(images)

    avg_acc1 = (acc1_val / n)
    return avg_acc1

def main(data='cifar10', folder=None, batch_size=512, noise_type='none'):

    models = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    models = sorted(models)
    print('************************')
    print("Beginning evaluation")
    print('************************')
    print("Noise type:", noise_type)
    print('************************')
    print("Data type:", data)

    _, test_loader, num_classes = getData(name=data, train_bs=batch_size, test_bs=batch_size, data_root='.\data')

    for index, m in enumerate(models):
        print(m)
        model = torch.load(folder + m)
        model.eval()
        if noise_type == 'white':
            test_acc = white_noisy_validate(test_loader, model)
        elif noise_type == 'sp':
            test_acc = sp_noise_validate(test_loader, model)
        else:
            test_acc = cls_validate(test_loader, model)
        print(test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("ShuffleMix for evaluation")
    parser.add_argument("--data", type=str, default='cifar10', required=False, help='dataset')
    parser.add_argument("--dir", type=str, default='saved_model', required=False, help='model dir')
    parser.add_argument("--noise", default='none', type=str, help='noise type: none, white, sp')

    args = parser.parse_args()
    main(data=args.data, folder=args.dir, noise_type=args.noise)
