from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from dataset import MPSGaze360, detection_collate, preproc
from config import cfg_re50
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.gazeonce360 import GazeOnce360

from tensorboardX import SummaryWriter
writer = SummaryWriter(comment='newproj')

parser = argparse.ArgumentParser(description='GazeOnce360 Training')
parser.add_argument('--training_dataset', default=['data/MPSGaze360_train_0/labels.txt',], help='Training dataset directory')
parser.add_argument('--num_workers', default=6, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./checkpoints/', help='Location to save checkpoint models')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = cfg_re50

rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder

net = GazeOnce360(cfg=cfg)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict,strict=False)

if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()

def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = MPSGaze360(training_dataset, preproc(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0
    
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                  torch.save(net.state_dict(), save_folder + cfg['name']+ '_epoch_' + str(epoch) + '.pth', _use_new_zipfile_serialization=False)
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, images_hr, masks, targets, offsets, images_hr_full = next(batch_iterator)
        images = images.cuda()
        images_hr = images_hr.cuda()
        masks = masks.cuda()
        targets = [anno.cuda() for anno in targets]
        offsets = offsets.cuda()
        images_full_hr = images_hr_full.cuda()

        # forward
        out = net(images, images_hr, offsets, images_full_hr)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm, loss_gaze, loss_hdps, loss_dist, loss_fc_landmark, loss_fc_gaze, loss_fc_hdps, loss_l_pre, loss_c_pre = criterion(out, priors, targets, masks)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm + cfg['gaze_weight'] * loss_gaze + loss_hdps + loss_dist + loss_fc_landmark + loss_fc_gaze + loss_fc_hdps + loss_l_pre + loss_c_pre

        loss.backward()
        optimizer.step()

        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        writer.add_scalar('loss_loc', loss_l, iteration)
        writer.add_scalar('loss_class', loss_c, iteration)
        writer.add_scalar('loss_ldmk', loss_landm, iteration)
        writer.add_scalar('loss_gaze', loss_gaze, iteration)
        writer.add_scalar('loss_hdps', loss_hdps, iteration)
        writer.add_scalar('loss_dist', loss_dist, iteration)
        writer.add_scalar('loss_fc_landmark', loss_fc_landmark, iteration)
        writer.add_scalar('loss_fc_gaze', loss_fc_gaze, iteration)
        writer.add_scalar('loss_fc_hdps', loss_fc_hdps, iteration)
        writer.add_scalar('loss_loc_pre', loss_l_pre, iteration)
        writer.add_scalar('loss_class_pre', loss_c_pre, iteration)
        writer.add_scalar('loss_total', loss, iteration)
        writer.add_scalar('lr', lr, iteration)

        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} LocPre: {:.4f} ClaPre: {:.4f} Landm: {:.4f} Gaze: {:.4f} Hdps: {:.4f} Dist: {:.4f} FCLandm: {:.4f} FCGaze: {:.4f} FCHdps: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'.format(epoch, \
            max_epoch,(iteration % epoch_size) + 1,epoch_size, iteration + 1, max_iter,loss_l.item(), loss_c.item(), loss_l_pre.item(), loss_c_pre.item(), loss_landm.item(),loss_gaze.item(), loss_hdps.item(), loss_dist.item(), loss_fc_landmark.item(), loss_fc_gaze.item(), loss_fc_hdps.item(), \
            lr, batch_time, str(datetime.timedelta(seconds=eta))))
    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth', _use_new_zipfile_serialization=False)


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    train()
