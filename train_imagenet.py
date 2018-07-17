import argparse
import os
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import models
from imagenet import config as config, utils as utils
from modules import ABN

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('config', metavar='CONFIG_FILE',
                    help='path to configuration file')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--log-dir', type=str, default='.', metavar='PATH',
                    help='output directory for Tensorboard log')
parser.add_argument('--log-hist', action='store_true',
                    help='log histograms of the weights')

best_prec1 = 0
args = None
conf = None
logger = None


def main():
    global args, best_prec1, logger, conf
    args = parser.parse_args()

    args.distributed = args.world_size > 1
    logger = SummaryWriter(args.log_dir)

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # Load configuration
    conf = config.load_config(args.config)

    # Create model
    model_params = utils.get_model_params(conf["network"])
    model = models.__dict__["net_" + conf["network"]["arch"]](**model_params)

    if not args.distributed:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer, scheduler = utils.create_optimizer(conf["optimizer"], model)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        init_weights(model)
        args.start_epoch = 0

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    train_transforms, val_transforms = utils.create_transforms(conf["input"])
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose(train_transforms))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=conf["optimizer"]["batch_size"], shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose(val_transforms)),
        batch_size=conf["optimizer"]["batch_size"], shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, conf["optimizer"]["schedule"]["epochs"]):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, it=epoch * len(train_loader))

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': conf["network"]["arch"],
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    global logger, conf
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if conf["optimizer"]["schedule"]["mode"] == "epoch":
        scheduler.step(epoch)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if conf["optimizer"]["schedule"]["mode"] == "step":
            scheduler.step(i + epoch * len(train_loader))

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if conf["optimizer"]["clip"] != 0.:
            nn.utils.clip_grad_norm(model.parameters(), conf["optimizer"]["clip"])
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

        logger.add_scalar("train/loss", losses.val, i + epoch * len(train_loader))
        logger.add_scalar("train/lr", scheduler.get_lr()[0], i + epoch * len(train_loader))
        logger.add_scalar("train/top1", top1.val, i + epoch * len(train_loader))
        logger.add_scalar("train/top5", top5.val, i + epoch * len(train_loader))
        if args.log_hist and i % 10 == 0:
            for name, param in model.named_parameters():
                if name.find("fc") != -1 or name.find("bn_out") != -1:
                    logger.add_histogram(name, param.clone().cpu().data.numpy(), i + epoch * len(train_loader))


def validate(val_loader, model, criterion, it=None):
    global logger
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    if it is not None:
        logger.add_scalar("val/loss", losses.avg, it)
        logger.add_scalar("val/top1", top1.avg, it)
        logger.add_scalar("val/top5", top5.avg, it)

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def init_weights(model):
    global conf
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            init_fn = getattr(nn.init, conf["network"]["weight_init"])
            if conf["network"]["weight_init"].startswith("xavier") or conf["network"]["weight_init"] == "orthogonal":
                gain = conf["network"]["weight_gain_multiplier"]
                if conf["network"]["activation"] == "relu" or conf["network"]["activation"] == "elu":
                    gain *= nn.init.calculate_gain("relu")
                elif conf["network"]["activation"] == "leaky_relu":
                    gain *= nn.init.calculate_gain("leaky_relu", conf["network"]["leaky_relu_slope"])
                init_fn(m.weight, gain)
            elif conf["network"]["weight_init"].startswith("kaiming"):
                if conf["network"]["activation"] == "relu" or conf["network"]["activation"] == "elu":
                    init_fn(m.weight, 0)
                else:
                    init_fn(m.weight, conf["network"]["leaky_relu_slope"])

            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant(m.bias, 0.)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, ABN):
            nn.init.constant(m.weight, 1.)
            nn.init.constant(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight, .1)
            nn.init.constant(m.bias, 0.)


if __name__ == '__main__':
    main()
