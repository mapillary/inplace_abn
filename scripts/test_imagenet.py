import argparse
import os

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

import models
from dataset.sampler import TestDistributedSampler
from imagenet import config, utils
from modules import SingleGPU

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
parser.add_argument('--local_rank', default=0, type=int,
                    help='process rank on node')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')

args = None
conf = None
cudnn.benchmark = True


def main():
    global args, conf
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)

    try:
        world_size = int(os.environ['WORLD_SIZE'])
        distributed = world_size > 1
    except:
        distributed = False
        world_size = 1

    if distributed:
        dist.init_process_group(backend=args.dist_backend, init_method='env://')

    # Load configuration
    conf = config.load_config(args.config)

    # Create model
    model_params = utils.get_model_params(conf["network"])
    model = models.__dict__["net_" + conf["network"]["arch"]](**model_params)
    model.cuda()
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    else:
        model = SingleGPU(model)

    # Resume from checkpoint
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    val_transforms = utils.create_test_transforms(conf["input"], args.crop, args.scale, args.ten_crops)

    batch_size = conf["optimizer"]["batch_size"] if not args.ten_crops else conf["optimizer"]["batch_size"] // 10
    dataset = datasets.ImageFolder(valdir, transforms.Compose(val_transforms))
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size // world_size, shuffle=False, sampler=TestDistributedSampler(dataset),
        num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()
    utils.validate(val_loader, model, criterion, args.ten_crops, args.print_freq)


if __name__ == '__main__':
    main()
