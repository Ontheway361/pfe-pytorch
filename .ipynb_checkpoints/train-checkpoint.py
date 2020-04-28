#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import torch
import random
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader

import model as mlib
import dataset as dlib
from config import training_args

torch.backends.cudnn.bencmark = True
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7" # TODO

from IPython import embed


def my_collate_fn(batch):
    
    imgs, gtys = [], []
    for pid_imgs, gty in batch:
        imgs.extend(pid_imgs)
        gtys.extend([gty] * len(pid_imgs))
    return (torch.stack(imgs, dim=0), torch.Tensor(gtys).long())
        


class MetricFace(dlib.VerifyFace):

    def __init__(self, args):

        dlib.VerifyFace.__init__(self, args)
        self.args   = args
        self.model  = dict()
        self.data   = dict()
        self.softmax= torch.nn.Softmax(dim=1)
        self.device = args.use_gpu and torch.cuda.is_available()


    def _report_settings(self):
        ''' Report the settings '''

        str = '-' * 16
        print('%sEnvironment Versions%s' % (str, str))
        print("- Python    : {}".format(sys.version.strip().split('|')[0]))
        print("- PyTorch   : {}".format(torch.__version__))
        print("- TorchVison: {}".format(torchvision.__version__))
        print("- USE_GPU   : {}".format(self.device))
        print('-' * 52)


    def _model_loader(self):

        self.model['backbone']  = mlib.MobileFace(self.args.in_feats, self.args.drop_ratio)
        # self.model['backbone'] = mlib.iresnet_zoo(self.args.backbone, drop_ratio=self.args.drop_ratio, use_se = self.args.use_se) # SEBlock
        # self.model['backbone']  = mlib.resnet_zoo(self.args.backbone, drop_ratio=self.args.drop_ratio)  # ResBlock
        # self.model['metric']    = mlib.FullyConnectedLayer(self.args)
        self.model['uncertain'] = mlib.UncertaintyHead(self.args.in_feats)
        # self.model['criterion'] = mlib.FaceLoss(self.args)
        self.model['criterion'] = mlib.MLSLoss(mean=False)
        
        if self.args.freeze_backbone:
            for p in self.model['backbone'].parameters():
                p.requires_grad = False
        
        self.model['optimizer'] = torch.optim.SGD(
                                      [# {'params': self.model['backbone'].parameters()},
                                       #{'params': self.model['metric'].parameters()},
                                      {'params': self.model['uncertain'].parameters()}],
                                      lr=self.args.base_lr,
                                      weight_decay=self.args.weight_decay,
                                      momentum=0.9,
                                      nesterov=True)
        self.model['scheduler'] = torch.optim.lr_scheduler.MultiStepLR(
                                      self.model['optimizer'], milestones=self.args.lr_adjust, gamma=self.args.gamma)
        if self.device:
            self.model['backbone']  = self.model['backbone'].cuda()
            self.model['uncertain'] = self.model['uncertain'].cuda()
            # self.model['metric']    = self.model['metric'].cuda()
            self.model['criterion'] = self.model['criterion'].cuda()

        if self.device and len(self.args.gpu_ids) > 1:
            self.model['backbone']  = torch.nn.DataParallel(self.model['backbone'], device_ids=self.args.gpu_ids)
            self.model['uncertain'] = torch.nn.DataParallel(self.model['uncertain'], device_ids=self.args.gpu_ids)
            # self.model['metric']    = torch.nn.DataParallel(self.model['metric'],   device_ids=self.args.gpu_ids)
            print('Parallel mode was going ...')
        elif self.device:
            print('Single-gpu mode was going ...')
        else:
            print('CPU mode was going ...')

        if len(self.args.resume) > 2:
            checkpoint = torch.load(self.args.resume, map_location=lambda storage, loc: storage)
            # self.args.start_epoch = checkpoint['epoch']
            self.model['backbone'].load_state_dict(checkpoint['backbone'])
            # self.model['uncertain'].load_state_dict(checkpoint['uncertain'])
            # self.model['metric'].load_state_dict(checkpoint['metric'])
            print('Resuming the train process at %3d epoches ...' % self.args.start_epoch)
        print('Model loading was finished ...')


    def _data_loader(self):

        self.data['train_loader'] = DataLoader(
                                        dlib.CASIAWebFacePFE(self.args, mode='train'),
                                        batch_size=self.args.batch_size, \
                                        shuffle=True,
                                        collate_fn=my_collate_fn,
                                    )
        # self.data['lfw']   = dlib.LFW(self.args)  # TODO
        print('Data loading was finished ...')


    def _model_train(self, epoch = 0):

        self.model['backbone'].eval()
        # self.model['metric'].train()
        self.model['uncertain'].train()

        loss_recorder, batch_acc = [], []
        for idx, (img, gty) in enumerate(self.data['train_loader']):
            
            img.requires_grad = False
            gty.requires_grad = False

            if self.device:
                img = img.cuda()
                gty = gty.cuda()

            feature, sig_feat = self.model['backbone'](img)  # TODO
            # output  = self.model['metric'](feature, gty)
            # loss    = self.model['criterion'](output, gty)
            log_sig_sq = self.model['uncertain'](sig_feat)
            loss = self.model['criterion'](feature, log_sig_sq, gty)
            self.model['optimizer'].zero_grad()
            loss.backward()
            self.model['optimizer'].step()
            # predy   = np.argmax(output.data.cpu().numpy(), axis=1)  # TODO
            # it_acc  = np.mean((predy == gty.data.cpu().numpy()).astype(int))
            # batch_acc.append(it_acc)
            loss_recorder.append(loss.item())
            if (idx + 1) % self.args.print_freq == 0:
                print('epoch : %2d|%2d, iter : %4d|%4d,  loss : %.4f' % \
                      (epoch, self.args.end_epoch, idx+1, len(self.data['train_loader']), np.mean(loss_recorder)))
                '''
                print('epoch : %2d|%2d, iter : %4d|%4d,  loss : %.4f, batch_ave_acc : %.4f' % \
                      (epoch, self.args.end_epoch, idx+1, len(self.data['train_loader']), \
                       np.mean(loss_recorder), np.mean(batch_acc)))
                '''
        train_loss = np.mean(loss_recorder)
        print('train_loss : %.4f' % train_loss)
        return train_loss


    def _verify_lfw(self):

        self._eval_lfw()

        self._k_folds()

        best_thresh, lfw_acc = self._eval_runner()

        return best_thresh, lfw_acc


    def _main_loop(self):

        if not os.path.exists(self.args.save_to):
                os.mkdir(self.args.save_to)

        max_lfw_acc, min_train_loss = 0.0, 100
        for epoch in range(self.args.start_epoch, self.args.end_epoch + 1):

            start_time = time.time()

            train_loss = self._model_train(epoch)
            self.model['scheduler'].step()
            # lfw_thresh, lfw_acc = self._verify_lfw()

            end_time = time.time()
            print('Single epoch cost time : %.2f mins' % ((end_time - start_time)/60))

            if min_train_loss > train_loss:

                print('%snew SOTA was found%s' % ('*'*16, '*'*16))
                # max_lfw_acc = max(max_lfw_acc, lfw_acc)
                min_train_loss = train_loss
                filename = os.path.join(self.args.save_to, 'sota.pth.tar')
                torch.save({
                    'epoch'     : epoch,
                    'backbone'  : self.model['backbone'].state_dict(),
                    'uncertain' : self.model['uncertain'].state_dict(),
                    'train_loss': min_train_loss,
                }, filename)

            if epoch % self.args.save_freq == 0:
                filename = 'epoch_%d_train_loss_%.4f.pth.tar' % (epoch, train_loss)
                savename = os.path.join(self.args.save_to, filename)
                torch.save({
                    'epoch'     : epoch,
                    'backbone'  : self.model['backbone'].state_dict(),
                    'uncertain' : self.model['uncertain'].state_dict(),
                    'train_loss': train_loss,
                }, savename)

            if self.args.is_debug:
                break


    def train_runner(self):

        self._report_settings()

        self._model_loader()

        self._data_loader()

        self._main_loop()


if __name__ == "__main__":

    faceu = MetricFace(training_args())
    faceu.train_runner()
