"""
NOTE: MUST USE conda activate fixmatch
CUDA_VISIBLE_DEVICES=0,1 python cifar_fixmatch_reproduce_multigpu.py -a wideresnet -d cifar10 -j 8 --epochs 1024 --train_batch 64 --lr 0.03 --init_data 400 --val_data 500 --mu 7 --lambda_u 1 --threshold 0.95 --n_imgs_per_epoch 65536 --checkpoint /BS/yfan/work/trained-models/fixmatch/cifar10/reproduce/wresnet_400_RA_ema_multigpu/ --manualSeed 6 --datasetSeed 6 --use_ema --ema_decay 0.999 --ngpus_per_node 2 --multiprocessing-distributed
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python cifar_fixmatch_reproduce_multigpu.py -a wideresnet -d cifar100 -j 16 --epochs 1024 --train_batch 64 --lr 0.03 --init_data 100 --val_data 1 --mu 7 --lambda_u 1 --threshold 0.95 --n_imgs_per_epoch 65536 --checkpoint /fan/fixmatch/cifar100/reproduce/wresnet_400_1_RA_ema_multigpu/ --manualSeed 6 --datasetSeed 6 --use_ema --ema_decay 0.999 --ngpus_per_node 8 --multiprocessing-distributed --wd 0.001

"""
from __future__ import print_function

import warnings
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
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from customDA import np_transforms
import models.cifar_mnist as models
from dataloaders import CIFAR_Semi, DistributedSampler
from customDA.fixmatch_transforms import TransformFix_CIFAR
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
parser.add_argument('--checkpoint', help='directory to output the result')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')

# Seeds
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--datasetSeed', type=int, help='datasetSeed for the initial split')
# Device options
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:22334', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--ngpus_per_node', default=8, type=int,
                    help='number of GPUs to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'


def main():
    global args

    print("***** Running training *****")
    print(f"  Task = {args.dataset}@{args.init_data}")
    print(f"  Num Epochs = {args.epochs}")
    print(f"  Batch size per GPU = {args.train_batch // args.ngpus_per_node}")
    print(f"  Total train batch size = {args.train_batch}")

    num_classes, transform_labeled, transform_unlabeled, transform_test = setup_constants()

    labeled_data, labeled_targets, unlabeled_data, unlabeled_targets, val_data, val_targets, test_data, test_targets = init_split()

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
        random.seed(args.manualSeed)
        np.random.seed(args.manualSeed)
        torch.manual_seed(args.manualSeed)

    if args.gpu is not None:
        print('You have chosen a specific GPU. This will completely '
              'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = args.ngpus_per_node  # torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, num_classes,
                                                           labeled_data, labeled_targets, transform_labeled,
                                                           unlabeled_data, unlabeled_targets, transform_unlabeled,
                                                           val_data, val_targets, transform_test,
                                                           test_data, test_targets, transform_test
                                                           ))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args, num_classes,
                labeled_data, labeled_targets, transform_labeled,
                unlabeled_data, unlabeled_targets, transform_unlabeled,
                val_data, val_targets, transform_val,
                test_data, test_targets, transform_test
                ):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                             world_size=args.world_size, rank=args.rank)

    model = init_model(args, num_classes)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.train_batch = int(args.train_batch / ngpus_per_node)
            args.n_imgs_per_epoch = int(args.n_imgs_per_epoch / ngpus_per_node)
            args.test_batch = int(args.test_batch / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # sync BN layers
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            warnings.warn('NOT DISTRIBUTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        warnings.warn('NOT DISTRIBUTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        warnings.warn('NOT DISTRIBUTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    per_epoch_steps = args.n_imgs_per_epoch // args.train_batch
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
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, total_steps)
    cudnn.benchmark = True

    labeledset = CIFAR_Semi(labeled_data, labeled_targets, transform=transform_labeled)
    unlabeledset = CIFAR_Semi(unlabeled_data, unlabeled_targets, transform=transform_unlabeled)
    valset = CIFAR_Semi(val_data, val_targets, transform=transform_val)
    testset = CIFAR_Semi(test_data, test_targets, transform=transform_test)

    if args.distributed:
        labeled_sampler = DistributedSampler(labeledset,
                                             num_samples=per_epoch_steps * args.train_batch * ngpus_per_node)
        unlabeled_sampler = DistributedSampler(unlabeledset,
                                               num_samples=per_epoch_steps *
                                                           args.train_batch * ngpus_per_node * args.mu)
    else:
        labeled_sampler = None
        unlabeled_sampler = None
    labeledloader = DataLoader(labeledset, batch_size=args.train_batch, shuffle=(labeled_sampler is None),
                               num_workers=args.workers, pin_memory=True, sampler=labeled_sampler)
    unlabeledloader = DataLoader(unlabeledset, batch_size=args.train_batch * args.mu,
                                 shuffle=(unlabeled_sampler is None), num_workers=args.workers,
                                 pin_memory=True, sampler=unlabeled_sampler)

    valloader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch, shuffle=False,
                                            num_workers=args.workers, drop_last=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False,
                                             num_workers=args.workers, drop_last=False)

    best_val_acc = 0
    best_test_acc = 0
    start_epoch = 0
    if args.use_ema:  # everybody ema, but only the rank 0 save model
        ema_model = EMA(model, args.ema_decay)

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
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
            logger.set_names(['Learning Rate', 'Train Loss', 'Train Loss X', 'Train Loss U', 'Mask',
                              'Total Acc', 'Used Acc', 'Valid Loss', 'Valid Acc.', 'Test Loss', 'Test Acc.'])

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            labeled_sampler.set_epoch(epoch)
            unlabeled_sampler.set_epoch(epoch)
        lr = optimizer.param_groups[0]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))
        loss, loss_x, loss_u, mask_prob, total_c, used_c = train(labeledloader, unlabeledloader, model,
                                                                 ema_model if args.use_ema else None,
                                                                 optimizer, scheduler, epoch, args)

        if args.use_ema:
            ema_model.apply_shadow()
        val_loss, val_acc = test(valloader, model, epoch, args)
        test_loss, test_acc = test(testloader, model, epoch, args)
        if args.use_ema:
            ema_model.restore()


        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            logger.append([lr, loss, loss_x, loss_u, mask_prob, total_c, used_c, val_loss, val_acc, test_loss, test_acc])
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.shadow if args.use_ema else None,
                'best_val_acc': best_val_acc,
                'best_test_acc': best_test_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, checkpoint=args.checkpoint, filename='checkpoint.pth.tar')
        best_val_acc = max(val_acc, best_val_acc)
        best_test_acc = max(test_acc, best_test_acc)
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        logger.close()
    print('Best test acc:', best_test_acc)


def setup_constants():
    """
    setup all kinds of constants here, just to make it cleaner :)
    """
    mem = os.popen('"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().split('\n')
    total = mem[0].split(',')[0]
    total = int(total)
    max_mem = int(total * 0.3)
    # x = torch.rand((256, 1024, max_mem)).cuda()
    # del x

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


def init_model(args, num_classes):
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


def train(labeled_trainloader, unlabeled_trainloader, model, ema, optimizer, scheduler, epoch, args):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    total_c = AverageMeter()
    used_c = AverageMeter()
    end = time.time()

    train_loader = zip(labeled_trainloader, unlabeled_trainloader)
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % args.ngpus_per_node == 0):
        bar = Bar('Processing', max=args.n_imgs_per_epoch // args.train_batch)

    for batch_idx, (data_x, data_u) in enumerate(train_loader):
        inputs_x, targets_x = data_x
        (inputs_u_w, inputs_u_s), targets_u = data_u
        data_time.update(time.time() - end)
        bs = inputs_x.size(0)

        if args.gpu is not None:
            inputs_x = inputs_x.cuda(args.gpu, non_blocking=True)
            inputs_u_s = inputs_u_s.cuda(args.gpu, non_blocking=True)
            inputs_u_w = inputs_u_w.cuda(args.gpu, non_blocking=True)
            targets_x = targets_x.cuda(args.gpu, non_blocking=True)

        inputs_train = torch.cat((inputs_x, inputs_u_s, inputs_u_w), dim=0).contiguous()

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
        total_c.update(total_acc.item())
        used_c.update(used_acc.item())

        optimizer.step()  # after torch1.1.0, scheduler should after opt (its the case when using conda fixmatch)
        scheduler.step()

        if args.use_ema:
            ema.update_params()

        batch_time.update(time.time() - end)
        end = time.time()
        mask_prob = mask.mean().item()

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % args.ngpus_per_node == 0):
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
                    mask=mask_prob))
            bar.next()
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % args.ngpus_per_node == 0):
        bar.finish()
    if args.use_ema:
        ema.update_buffer()
    return losses.avg, losses_x.avg, losses_u.avg, mask_prob, total_c.avg, used_c.avg


def test(testloader, model, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % args.ngpus_per_node == 0):
            bar = Bar('Processing', max=len(testloader))
        for batch_idx, (inputs, targets) in enumerate(testloader):
            data_time.update(time.time() - end)

            if args.gpu is not None:
                inputs = inputs.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)

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

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % args.ngpus_per_node == 0):
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
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % args.ngpus_per_node == 0):
            bar.finish()
    return (losses.avg, top1.avg)


def save_checkpoint(state, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


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