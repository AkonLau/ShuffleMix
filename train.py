import os
import argparse
import timeit
import logging

import src.models
from src.tools import *
from src.shufflemix import *
from src.data.get_data import getData
from tensorboardX import SummaryWriter

#==============================================================================
# Training settings
#==============================================================================
parser = argparse.ArgumentParser(description='ShuffleMix')
parser.add_argument('--name', type=str, default='cifar10', metavar='N', help='dataset, cifra10, cifar100 or imagenet')
parser.add_argument('--data_root',type=str, default=None, help='the data root for training and testing')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--test_batch_size', type=int, default=512, metavar='N', help='input batch size for testing (default: 512)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--lr_decay', type=float, default=0.1, help='learning rate decay value (default: 0.1)')
parser.add_argument('--lr_decay_epoch', type=int, nargs='+', default=[100, 150, 180], help='decrease learning rate at these epochs.')
parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--arch', type=str, default='preactresnet18', metavar='N', help='model name')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')

# mix method
parser.add_argument('--log_path',type=str, default='log')
parser.add_argument('--mixmethod', default=None, type=str, help='mix type: mixup, ManiflodMixup or ShuffleMix...')
parser.add_argument('--alpha', type=float, default=0.0, metavar='S', help='hyper parameter for mixup')

# Generalized-Manifold-Mix
parser.add_argument('--ratio', type=float, default=0.0, metavar='S', help='ratio for ShuffleMix')
parser.add_argument('--mix_type', type=str, default='soft', metavar='N', help='mix type: hard or soft')
parser.add_argument('--index_type', type=str, default='random', metavar='N', help='location type: fixed or random')
parser.add_argument('--k_layer1', type=int, default=0, metavar='N', help='manifold layer:-1~3')
parser.add_argument('--k_layer2', type=int, default=0, metavar='N', help='manifold layer:0~4')
# Noise-Feature-Mix
parser.add_argument('--add_noise_level', type=float, default=0.0, metavar='S', help='level of additive noise')
parser.add_argument('--mult_noise_level', type=float, default=0.0, metavar='S', help='level of multiplicative noise')
# dropout
parser.add_argument('--dropout', dest='dropout', action='store_true', help='dropout')
parser.add_argument('--ada_dropout', dest='ada_dropout', action='store_true', help='ada_dropout')

args = parser.parse_args()
#==============================================================================
# set random seed to reproduce the work
#==============================================================================
seed_everything(args.seed)
#==============================================================================
# get device
#==============================================================================

device = get_device()
#==============================================================================
# get dataset
#==============================================================================
train_loader, test_loader, num_classes = getData(name=args.name, train_bs=args.batch_size,
                                                 test_bs=args.test_batch_size, data_root=args.data_root)
#==============================================================================
# get model
#==============================================================================
model = src.models.__dict__[args.arch](num_classes=num_classes, is_dropout=args.dropout).cuda()
# mix method select
if args.mixmethod is not None:
    if args.mixmethod == 'ShuffleMix' or args.mixmethod == 'ManifoldMixup':
        model = eval(args.mixmethod)(model, args).cuda()
        MixMethod = None
    else:
        MixMethod = eval(args.mixmethod)
else:
    MixMethod = None
#==============================================================================
# Model summary
#==============================================================================
# print(model)
print('**** Setup ****')
print('Total params: %.2fK' % (sum(p.numel() for p in model.parameters())*10**-3))
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())*10**-6))
print('************')
#==============================================================================
# setup optimizer
#==============================================================================
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
#==============================================================================
# criterion
#==============================================================================
criterion = nn.CrossEntropyLoss().to(device)
#==============================================================================
# logger setting
store_name = '_'.join([args.name, args.arch, str(args.mixmethod), str(args.alpha), args.mix_type,
                       str(args.ratio), args.index_type, 'k', str(args.k_layer1), str(args.k_layer2),
                       'noise', str(args.add_noise_level), str(args.mult_noise_level), 'seed', str(args.seed)])

if args.dropout is True:
    store_name += '_dropout'
if args.ada_dropout is True:
    store_name += '_ada_dropout'
log_dir = os.path.join(args.log_path, args.name, store_name)
# set tensorboard
tf_writer = SummaryWriter(log_dir=log_dir)
# logger writer
logger = set_logger(log_path=log_dir)
logging.info(print_conf(args))
#==============================================================================
# start training
best_acc1 = 0
t0 = timeit.default_timer()
for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    correct = 0.0
    total_num = 0
    
    for step, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if args.alpha == 0.0:
            inputs = add_noise(inputs, args.add_noise_level, args.mult_noise_level) # add noise when level > 0
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        else:
            if args.mixmethod == 'ShuffleMix' or args.mixmethod == 'ManifoldMixup':
                outputs, targets_a, targets_b, lam = model(inputs, targets) # add noise among model when level > 0
            else:
                inputs, targets_a, targets_b, lam = MixMethod(inputs, targets, args.alpha)
                inputs = add_noise(inputs, args.add_noise_level, args.mult_noise_level) # add noise when level > 0
                outputs = model(inputs)

            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        optimizer.zero_grad()
        loss.backward()          
        optimizer.step() # update weights

        # compute statistics
        train_loss += loss.item() * targets.size()[0]
        total_num += targets.size()[0]
        _, predicted = outputs.max(1)

    train_loss = train_loss / total_num
    test_acc1, test_acc5, test_nll_loss = validate(test_loader, model, criterion)
    logging.info({'Epoch: ', epoch, '| lr: %.4f' % optimizer.param_groups[0]['lr'],
          '| train loss: %.3f' % train_loss, '| test acc.: %.3f' % test_acc1, '| test nll loss.: %.3f' % test_nll_loss})
    # save model
    best_acc1 = max(test_acc1, best_acc1)
    # plot training precessing
    tf_writer.add_scalar('loss/train', train_loss, epoch)
    tf_writer.add_scalar('loss/test_nll', test_nll_loss, epoch)

    tf_writer.add_scalar('acc/test_top1', test_acc1, epoch)
    tf_writer.add_scalar('acc/test_top5', test_acc5, epoch)

    tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

    # schedule learning rate decay
    optimizer = lr_scheduler(epoch, optimizer, decay_eff=args.lr_decay, decayEpoch=args.lr_decay_epoch)
logging.info({'total time: ', timeit.default_timer() - t0})
#==============================================================================
# store final results
#==============================================================================
DESTINATION_PATH = 'saved_model/' + args.name + '_models/'
if not os.path.isdir('saved_model/'):
    os.mkdir('saved_model/')
if not os.path.isdir(DESTINATION_PATH):
    os.mkdir(DESTINATION_PATH)
save_checkpoint(model, DESTINATION_PATH, store_name)