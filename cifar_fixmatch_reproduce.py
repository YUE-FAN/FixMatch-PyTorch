"""
cifar100 weight decay 0.0005 is actually better
bnwd is actually better
cv2 RA is the same as PIL RA

cifar100 mean and std is different
cutout uses gray?
translate x is int or float?
sharp 1.8
Posterize 0-4

python cifar_fixmatch_reproduce.py -a wideresnet_leaky -d cifar100 -j 8 --epochs 1024 --train_batch 64 --lr 0.03 --init_data 100 --val_data 1 --mu 7 --lambda_u 1 --threshold 0.95 --n_imgs_per_epoch 65536 --checkpoint /fan/fixmatch/cifar100/reproduce/wresnetleaky_100_1_RA_ema/ --manualSeed 1 --datasetSeed 1 --use_ema --ema_decay 0.999 --wd 0.0005 --gpu-id 4
"""
from __future__ import print_function

import argparse
import os
import math
import time
import random
import copy
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar_mnist as models
from dataloaders import CIFAR_Semi
import torch.nn.functional as F

from customDA.fixmatch_transforms import TransformFix_CIFAR
from torch.optim.lr_scheduler import LambdaLR
from dataloaders.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader

from customDA import np_transforms
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p
from utils.ema import EMA

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
# Datasets
parser.add_argument('-a', '--arch', metavar='ARCH', choices=model_names, help='model architecture: ' + ' | '.join(model_names))
parser.add_argument('-d', '--dataset', default='mnist', type=str)
parser.add_argument('--datapath', default='/BS/databases00/', type=str)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N')
# Optimizer
parser.add_argument('--epochs', default=1024, type=int, help='number of total epochs to run')
parser.add_argument('--warmup', default=0, type=int, help='number of warmup epochs to run')
parser.add_argument('--train_batch', default=64, type=int, metavar='N', help='train batchsize')
parser.add_argument('--test_batch', default=200, type=int, metavar='N', help='test batchsize')
parser.add_argument('--lr', default=0.03, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W',
                    help='weight decay (default: 5e-4 on cifar)')
parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
# Others
parser.add_argument('--k', type=int, help='2 for cifar10, 8 for cifar100')
parser.add_argument('--n', type=int, help='on cifar datasets, 28')
parser.add_argument('--init_data', type=int)
parser.add_argument('--val_data', type=int)
parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
parser.add_argument('--lambda_u', default=1, type=float, help='coefficient of unlabeled loss')
parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
parser.add_argument('--n_imgs_per_epoch', type=int, default=64 * 1024,
                    help='number of training images for each epoch')
parser.add_argument('--use_ema', action='store_true', help='use EMA model')
parser.add_argument('--ema_decay', default=0.999, type=float, help='EMA decay rate')
# Checkpoint
parser.add_argument('--checkpoint',
                    help='directory to output the result')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')

# Seeds
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--datasetSeed', type=int, help='datasetSeed for the initial split')
parser.add_argument('--gpu-id', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
cudnn.benchmark = False
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    torch.backends.cudnn.deterministic = True


def main():
    print("***** Running training *****")
    print(f"  Task = {args.dataset}@{args.init_data}")
    print(f"  Num Epochs = {args.epochs}")
    print(f"  Batch size per GPU = {args.train_batch}")
    print(f"  Total train batch size = {args.train_batch}")

    num_classes, transform_labeled, transform_unlabeled, transform_test = setup_constants()

    labeled_data, labeled_targets, unlabeled_data, unlabeled_targets, val_data, val_targets, test_data, test_targets = init_split()

    model = init_model(num_classes)

    train_model_from_scratch(model, 'log.txt', 'checkpoint.pth.tar', use_cuda,
                             labeled_data, labeled_targets, transform_labeled,
                             unlabeled_data, unlabeled_targets, transform_unlabeled,
                             val_data, val_targets, transform_test,
                             test_data, test_targets, transform_test)


def setup_constants():
    """
    setup all kinds of constants here, just to make it cleaner :)
    """
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    if args.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
        num_classes = 10
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        num_classes = 100


    transform_labeled = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),  # with p = 0.5
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),  # with p = 1
        np_transforms.PILToNumpy(),  # PIL Image -> np.uint8
        np_transforms.NumpyToTensor(),  # np.uint8 -> torch.float32
        np_transforms.Normalize_11()
    ])

    transform_unlabeled = TransformFix_CIFAR(mean, std)

    transform_test = transforms.Compose([
        np_transforms.NumpyToTensor(),  # np.uint8 -> torch.float32
        np_transforms.Normalize_11()
    ])

    return num_classes, transform_labeled, transform_unlabeled, transform_test


def init_split():
    """
    Split the CIFAR dataset according to args.init_data
    Note that the returned tensors are all on CPU in the form of np.ndarray, uint8 and int
    """
    if args.dataset == 'cifar10':
        tmp = datasets.CIFAR10(root=os.path.join(args.datapath, 'cifar-10'), train=True)
        testset = datasets.CIFAR10(root=os.path.join(args.datapath, 'cifar-10'), train=False)
        num_classes = 10
    else:
        tmp = datasets.CIFAR100(root=os.path.join(args.datapath, 'cifar-100'), train=True)
        testset = datasets.CIFAR100(root=os.path.join(args.datapath, 'cifar-100'), train=False)
        num_classes = 100
    total_data = tmp.data  # np.ndarray, uint8
    total_targets = np.array(tmp.targets, dtype=np.int)

    test_data = testset.data
    test_targets = testset.targets

    shuffle_index = list(range(len(total_data)))
    random.Random(args.datasetSeed).shuffle(shuffle_index)
    total_data = total_data[shuffle_index]
    total_targets = total_targets[shuffle_index]

    def balanced_selection(total_data, total_targets, num_classes, per_class_data):
        select_index_set = np.zeros(num_classes * per_class_data, dtype=np.int) - 1
        label_counter = [0] * num_classes
        j = 0
        for i, label in enumerate(total_targets):
            if label_counter[label] != per_class_data:
                label_counter[label] += 1
                select_index_set[j] = i
                j += 1
            if label_counter == [per_class_data] * num_classes:
                break
        unselected_index_set = np.ones(total_targets.shape).astype(np.bool)
        unselected_index_set[select_index_set] = 0
        unselected_index_set, = np.where(unselected_index_set)

        selected_data = total_data[select_index_set]
        selected_targets = total_targets[select_index_set]
        unselected_data = total_data[unselected_index_set]
        unselected_targets = total_targets[unselected_index_set]
        return selected_data, selected_targets, unselected_data, unselected_targets

    val_data, val_targets, train_data, train_targets = balanced_selection(total_data, total_targets,
                                                                          num_classes, args.val_data)
    labeled_data, labeled_targets, _, _ = balanced_selection(train_data, train_targets,
                                                             num_classes, args.init_data)


    labeled_data = copy.deepcopy(labeled_data)
    labeled_targets = copy.deepcopy(labeled_targets)
    unlabeled_data = copy.deepcopy(train_data)
    unlabeled_targets = copy.deepcopy(train_targets)

    return labeled_data, labeled_targets, unlabeled_data, unlabeled_targets, val_data, val_targets, test_data, test_targets


def init_model(num_classes):
    print("==> creating model '{}'".format(args.arch))
    if args.arch.endswith('resnet18'):
        model = models.__dict__[args.arch](num_classes=num_classes)
    elif args.arch.endswith('vgg16'):
        model = models.__dict__[args.arch](num_classes=num_classes, grayscale=False)
    elif args.arch.endswith('wideresnet'):
        model = models.__dict__[args.arch](depth=args.n,
                                           widen_factor=args.k,
                                           drop_rate=0,
                                           num_classes=num_classes)
    elif args.arch.endswith('wideresnetleaky'):
        model = models.__dict__[args.arch](depth=args.n,
                                           widen_factor=args.k,
                                           drop_rate=0,
                                           num_classes=num_classes)
    else:
        raise Exception('choose wrong model!!!!')
    return model


def train_model_from_scratch(model, logger_name, checkpoint_name, use_cuda,
                             labeled_data, labeled_targets, tfms_labeled,
                             unlabeled_data, unlabeled_targets, tfms_unlabeled,
                             val_data, val_targets, tfms_val,
                             test_data, test_targets, tfms_test):
    """
    This function trains a pre-defined model from scratch and test and log the info.
    The training scheme is defined in args.
    """
    # define the trainloader, valloader, testloader
    labeledset = CIFAR_Semi(labeled_data, labeled_targets, transform=tfms_labeled)
    unlabeledset = CIFAR_Semi(unlabeled_data, unlabeled_targets, transform=tfms_unlabeled)
    valset = CIFAR_Semi(val_data, val_targets, transform=tfms_val)
    testset = CIFAR_Semi(test_data, test_targets, transform=tfms_test)

    per_epoch_steps = args.n_imgs_per_epoch // args.train_batch

    sampler_x = RandomSampler(labeledset, replacement=True, num_samples=per_epoch_steps * args.train_batch)
    batch_sampler_x = BatchSampler(sampler_x, batch_size=args.train_batch, drop_last=True)
    labeledloader = DataLoader(labeledset, batch_sampler=batch_sampler_x, num_workers=args.workers)

    sampler_u = RandomSampler(unlabeledset, replacement=True, num_samples=per_epoch_steps * args.train_batch * args.mu)
    batch_sampler_u = BatchSampler(sampler_u, batch_size=args.train_batch * args.mu, drop_last=True)
    unlabeledloader = DataLoader(unlabeledset, batch_sampler=batch_sampler_u, num_workers=args.workers)

    valloader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch, shuffle=False,
                                            num_workers=args.workers, drop_last=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False,
                                             num_workers=args.workers, drop_last=False)

    # define optimizer and learning rate scheduler
    model = torch.nn.DataParallel(model).cuda()
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        # if len(param.size()) == 1:
        if 'bn' in name or 'bias' in name:
            non_wd_params.append(param)  # bn.weight, bn.bias and classifier.bias, conv2d.bias
            # print(name)
        else:
            wd_params.append(param)
    param_list = [
        {'params': wd_params, 'weight_decay': args.weight_decay}, {'params': non_wd_params, 'weight_decay': 0}]
    optimizer = optim.SGD(param_list, lr=args.lr, momentum=args.momentum, nesterov=args.nesterov)
    total_steps = args.epochs * per_epoch_steps
    warmup_steps = args.warmup * per_epoch_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # train the model from scratch
    best_val_acc = 0
    best_test_acc = 0
    start_epoch = 0
    if args.use_ema:
        ema_model = EMA(model, args.ema_decay)
    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_test_acc = checkpoint['best_test_acc']
        best_val_acc = checkpoint['best_val_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if args.use_ema:
            ema_model.load_state_dict(checkpoint['ema_state_dict'])
        logger = Logger(os.path.join(args.checkpoint, logger_name), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, logger_name), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Train Loss X', 'Train Loss U', 'Mask',
                          'Total Acc', 'Used Acc', 'Valid Loss', 'Valid Acc.', 'Test Loss', 'Test Acc.'])

    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))
        loss, loss_x, loss_u, mask_prob, total_c, used_c = train(labeledloader, unlabeledloader, model,
                                                                 ema_model if args.use_ema else None,
                                                                 optimizer, scheduler, epoch, use_cuda)

        if args.use_ema:
            ema_model.apply_shadow()
        val_loss, val_acc = test(valloader, model, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, epoch, use_cuda)
        if args.use_ema:
            ema_model.restore()

        logger.append([lr, loss, loss_x, loss_u, mask_prob, total_c, used_c, val_loss, val_acc, test_loss, test_acc])

        is_best = val_acc > best_val_acc
        if is_best:
            best_test_acc = test_acc
            best_model = copy.deepcopy(model)
        if checkpoint_name is not None:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.shadow if args.use_ema else None,
                'best_val_acc': best_val_acc,
                'best_test_acc': best_test_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, checkpoint=args.checkpoint, filename=checkpoint_name)
        best_val_acc = max(val_acc, best_val_acc)
    logger.close()
    print('Best test acc:', best_test_acc)
    return best_model, best_test_acc


def train(labeled_trainloader, unlabeled_trainloader, model, ema, optimizer, scheduler, epoch, use_cuda):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    mask_prob = AverageMeter()
    total_c = AverageMeter()
    used_c = AverageMeter()
    end = time.time()

    train_loader = zip(labeled_trainloader, unlabeled_trainloader)
    bar = Bar('Processing', max=args.n_imgs_per_epoch // args.train_batch)

    for batch_idx, (data_x, data_u) in enumerate(train_loader):
        inputs_x, targets_x = data_x
        (inputs_u_w, inputs_u_s), targets_u = data_u
        data_time.update(time.time() - end)
        bs = inputs_x.size(0)

        inputs_train = torch.cat((inputs_x, inputs_u_s, inputs_u_w), dim=0).contiguous()

        if use_cuda:
            inputs_train = inputs_train.cuda()
            targets_x = targets_x.cuda()

        logits_train = model(inputs_train)[-1]
        logits_x = logits_train[:bs]
        logits_u_s, logits_u_w = logits_train[bs:].chunk(2)

        with torch.no_grad():
            pseudo_label = torch.softmax(logits_u_w, dim=-1)
            max_probs, targets_u_w = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()


        Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
        Lu = (F.cross_entropy(logits_u_s, targets_u_w, reduction='none') * mask).mean()

        total_acc = targets_u_w.cpu().detach().eq(targets_u).float().view(-1)
        if mask.sum() != 0:
            used_acc = total_acc[mask != 0]
            used_acc = used_acc.mean(0)

        else:
            used_acc = torch.tensor(0)
        total_acc = total_acc.mean(0)

        loss = Lx + args.lambda_u * Lu

        optimizer.zero_grad()
        loss.backward()

        losses.update(loss.item())
        losses_x.update(Lx.item())
        losses_u.update(Lu.item())
        mask_prob.update(mask.mean().item())
        total_c.update(total_acc.item())
        used_c.update(used_acc.item())

        scheduler.step()
        optimizer.step()

        if args.use_ema:
            ema.update_params()

        batch_time.update(time.time() - end)
        end = time.time()
        # plot progress
        lr = optimizer.param_groups[0]['lr']
        bar.suffix = ("Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.4f}. ".format(
                batch=batch_idx + 1,
                iter=args.n_imgs_per_epoch // args.train_batch,
                lr=lr,
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                loss_x=losses_x.avg,
                loss_u=losses_u.avg,
                mask=mask_prob.avg))
        bar.next()
    bar.finish()
    if args.use_ema:
        ema.update_buffer()
    return losses.avg, losses_x.avg, losses_u.avg, mask_prob.avg, total_c.avg, used_c.avg


def test(testloader, model, epoch, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():

        bar = Bar('Processing', max=len(testloader))
        for batch_idx, (inputs, targets) in enumerate(testloader):
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs = model(inputs)[-1]
            loss = F.cross_entropy(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg)


def save_checkpoint(state, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

# def get_cosine_schedule_with_warmup(optimizer,
#                                     num_warmup_steps,
#                                     num_training_steps,
#                                     num_cycles=1.,
#                                     last_epoch=-1):
#     def _lr_lambda(current_step):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
#         no_progress = float(current_step - num_warmup_steps) / \
#             float(max(1, num_training_steps - num_warmup_steps))
#         return max(0., math.cos(math.pi * num_cycles * no_progress) + 1) / 2
#
#     return LambdaLR(optimizer, _lr_lambda, last_epoch)

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))  # this is correct
    return LambdaLR(optimizer, _lr_lambda, last_epoch)


if __name__ == '__main__':
    main()