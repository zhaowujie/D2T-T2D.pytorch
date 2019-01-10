# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Thomas Balestri; based off Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import datetime
import pickle

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.resnet import resnet

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a detect-to-track network')
    parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='imagenet_vid', type=str)
    parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='res101', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=10, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="output/models",
                      type=str)
    parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=4, type=int)
    parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
    parser.add_argument('--use_det', dest='use_det',
                      help='whether to use detection dataset in training.',
                      action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                      help='whether use tensorflow tensorboard',
                      default=False, type=bool)

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0,batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
          self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
          self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
          self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data

if __name__ == '__main__':


    args = parse_args()

    print('Called with args:')
    print(args)

    if args.use_tfboard:
        from model.utils.logger import Logger
        # Set the logger
        logger = Logger('./logs')

    if args.dataset == "imagenet_vid":
        args.imdb_name = "imagenet_vid_train"
        args.imdbval_name = "imagenet_vid_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
    elif args.dataset == "imagenet_vid+imagenet_det":
        args.imdb_name = "imagenet_vid_train+imagenet_det_train"
        args.imdbval_name = "imagenet_vid_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        args.use_det = True
    else:
        raise NotImplementedError

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

        print('Using config:')
        pprint.pprint(cfg)
        np.random.seed(cfg.RNG_SEED)

        #torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available() and not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = False
    cfg.USE_GPU_NMS = args.cuda
    if args.use_det:
        print("Using VID and DET datasets.")
        imdb_names = args.imdb_name.split('+')
        assert len(imdb_names) < 3, 'Cannot handle > 2 datasets when using VID and DET.'
        # vid and det
        vid_imdb, vid_roidb, vid_ratio_list, vid_ratio_index = combined_roidb(imdb_names[0])
        det_imdb, det_roidb, det_ratio_list, det_ratio_index = combined_roidb(imdb_names[1], duplicate_frames=True)
        vid_train_size, det_train_size = len(vid_roidb), len(det_roidb)
        train_size = min(vid_train_size, det_train_size)
        assert vid_imdb.num_classes==det_imdb.num_classes, 'VID and DET must have same number of classes.'
        imdb_classes = vid_imdb.classes
    else:
        # if os.path.exists('./data/imdb.pkl'):
        #     imdb = pickle.load('./data/imdb.pkl')
        #     roidb = pickle.load('./data/roidb_pairs.pkl')
        #     ratio_list = pickle.load('./data/ratio_list.pkl')
        #     ratio_index = pickle.load('./ratio_index.pkl')
        # else:
        #     imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name) #get imdb roidb_pairs ratio and index
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)  # get imdb roidb_pairs ratio and index
        train_size = len(roidb)
        imdb_classes = imdb.classes

    print('{:d} roidb frame pairs'.format(train_size))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.use_det:
        vid_sampler_batch = sampler(vid_train_size, args.batch_size)
        det_sampler_batch = sampler(det_train_size, args.batch_size)

        # Build VID and DET datasets
        vid_dataset = roibatchLoader(vid_roidb,vid_ratio_list, vid_ratio_index,
                args.batch_size, vid_imdb.num_classes, training=True)
        det_dataset = roibatchLoader(det_roidb,det_ratio_list, det_ratio_index,
                args.batch_size, det_imdb.num_classes, training=True)

        vid_dataloader = torch.utils.data.DataLoader(vid_dataset,
                batch_size=args.batch_size,
                sampler=vid_sampler_batch,
                num_workers=args.num_workers)
        det_dataloader = torch.utils.data.DataLoader(det_dataset,
                batch_size=args.batch_size,
                sampler=det_sampler_batch,
                num_workers=args.num_workers)
    else:
        sampler_batch = sampler(train_size, args.batch_size)

        dataset = roibatchLoader(roidb,ratio_list, ratio_index, args.batch_size, imdb.num_classes, training=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                              sampler=sampler_batch, num_workers=args.num_workers)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)


    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    if args.net == 'res101':
        RFCN = resnet(imdb_classes, 101, pretrained_rfcn=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        RFCN = resnet(imdb_classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        RFCN = resnet(imdb_classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()
    RFCN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    #tr_momentum = cfg.TRAIN.MOMENTUM
    #tr_momentum = args.momentum

    params = []
    for key, value in dict(RFCN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
        else:
            params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, RFCN.parameters()), momentum=cfg.TRAIN.MOMENTUM)
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.resume:
        load_name = os.path.join(output_dir,
          'rfcn_detect_track_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        RFCN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
          cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        RFCN = nn.DataParallel(RFCN)

    if args.cuda:
        RFCN.cuda()

    if args.use_det:
        iters_per_epoch = 2*int(train_size / args.batch_size)
    else:
        iters_per_epoch = int(train_size / args.batch_size)

    for epoch in range(args.start_epoch, args.max_epochs):
        # setting to train mode
        RFCN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        if args.use_det:
          vid_data_iter = iter(vid_dataloader)
          det_data_iter = iter(det_dataloader)
        else:
          data_iter = iter(dataloader)

        for step in range(iters_per_epoch):
            # data loader emits tensor of dim (batch_size, images_per_pass (e.g. 2), channels, height, width)
            if args.use_det:
                # Alternate training with samples from VID and DET
                if step%2==0:
                    data = next(vid_data_iter)
                else:
                    data = next(det_data_iter)
            else:
                data = next(data_iter)

            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])
            # Permute batch size and number of items per leg in the siamese network.
            # New dims for im_data will be (n_legs, batch_size, C, H, W)
            #else:
            #    im_data = im_data.permute(1,0,2,3,4).contiguous()
            #    im_info = im_info.permute(1,0,2).contiguous()
            #    gt_boxes = gt_boxes.permute(1,0,2,3).contiguous()
            #    num_boxes = num_boxes.permute(1,0,2).contiguous()


            RFCN.zero_grad()
            rois, cls_prob, bbox_pred, tracking_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, tracking_loss_bbox = RFCN(im_data, im_info, gt_boxes, num_boxes)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() + tracking_loss_bbox.mean()
            loss_temp += loss.data[0]
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().data[0]
                    loss_rpn_box = rpn_loss_box.mean().data[0]
                    loss_rcnn_cls = RCNN_loss_cls.mean().data[0]
                    loss_rcnn_box = RCNN_loss_bbox.mean().data[0]
                    loss_tracking_box = tracking_loss_bbox.mean().data[0]
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.data[0]
                    loss_rpn_box = rpn_loss_box.data[0]
                    loss_rcnn_cls = RCNN_loss_cls.data[0]
                    loss_rcnn_box = RCNN_loss_bbox.data[0]
                    loss_tracking_box = tracking_loss_bbox.data[0]
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("[%s][epoch %2d][iter %4d] loss: %.4f, lr: %.2e" \
                                        % (nowTime, epoch, step, loss_temp, lr), end=' ')
                print("\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start), end=' ')
                print("\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f trk_box %.4f" \
                              % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, loss_tracking_box))
                if args.use_tfboard:
                    info = {
                    'loss': loss_temp,
                    'loss_rpn_cls': loss_rpn_cls,
                    'loss_rpn_box': loss_rpn_box,
                    'loss_rcnn_cls': loss_rcnn_cls,
                    'loss_rcnn_box': loss_rcnn_box,
                    'loss_tracking_box': loss_tracking_box
                    }
                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, step)

                loss_temp = 0
                start = time.time()

        if args.mGPUs:
            save_name = os.path.join(output_dir, 'rfcn_detect_track_{}_{}_{}.pth'.format(args.session, epoch, step))
            save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': RFCN.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
            }, save_name)
        else:
            save_name = os.path.join(output_dir, 'rfcn_detect_track_{}_{}_{}.pth'.format(args.session, epoch, step))
            save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': RFCN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
            }, save_name)
        print('save model: {}'.format(save_name))

        end = time.time()
        print(end - start)
