import argparse
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import models
from imagenet import config as config, utils as utils

parser = argparse.ArgumentParser(description='PyTorch ImageNet Testing.')
parser.add_argument('config', metavar='CONFIG_FILE',
                    help='path to configuration file. NOTE: validation-related settings are ignored')
parser.add_argument('checkpoint', metavar='PATH', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--crop', '-c', metavar='N', type=int, default=224,
                    help='crop size')
parser.add_argument('--scale', '-s', metavar='N', type=int, default=256,
                    help='scale size, if -1 do not scale input')
parser.add_argument('--ten_crops', action='store_true',
                    help='run ten-crops testing instead of center-crop testing')

args = None
conf = None
cudnn.benchmark = True


def get_transforms(config):
    global args

    normalize = transforms.Normalize(mean=config["mean"], std=config["std"])

    val_transforms = []
    if args.scale != -1:
        val_transforms.append(transforms.Resize(args.scale))
    if args.ten_crops:
        val_transforms += [
            transforms.TenCrop(args.crop),
            transforms.Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops]),
            transforms.Lambda(lambda crops: [normalize(crop) for crop in crops]),
            transforms.Lambda(lambda crops: torch.stack(crops))
        ]
    else:
        val_transforms += [
            transforms.CenterCrop(args.crop),
            transforms.ToTensor(),
            normalize
        ]

    return val_transforms


def main():
    global args, conf
    args = parser.parse_args()

    # Load configuration
    conf = config.load_config(args.config)

    # Create model
    model_params = utils.get_model_params(conf["network"])
    model = models.__dict__["net_" + conf["network"]["arch"]](**model_params)
    model = torch.nn.DataParallel(model).cuda()

    # Resume from checkpoint
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    val_transforms = get_transforms(conf["input"])

    batch_size = conf["optimizer"]["batch_size"] if not args.ten_crops else conf["optimizer"]["batch_size"] // 10
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose(val_transforms)),
        batch_size=batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()
    validate(val_loader, model, criterion)


def validate(val_loader, model, criterion):
    global args
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            if args.ten_crops:
                bs, ncrops, c, h, w = input.size()
                input = input.view(-1, c, h, w)

            target = target.cuda(non_blocking=True)

            # compute output
            if args.ten_crops:
                output = model(input).view(bs, ncrops, -1).mean(1)
            else:
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

    return top1.avg


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


if __name__ == '__main__':
    main()
