import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler, prepare_dataset, prepare_dataset_crop
from utils import losses, metrics, ramps
from val_2D import test_single_volume_ds, test_single_volume_ds_cr
from networks.net_factory import net_factory

def dice_score(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return loss

num_classes = 2

testing_data_path = "/home/azka/sharjeel/Thyroid/code/testing_data.npy"

model_name = 'unet_urpc'
model = net_factory(net_type=model_name, in_chns=1, class_num=num_classes)

device = torch.device('cuda:1')
model = model.to(device)

model_path = "/home/azka/sharjeel/Thyroid/code/saved_models_with_combined/unet_urpc_best_stage2_model.pth"
model.load_state_dict(torch.load(model_path))

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

db_test = prepare_dataset(testing_data_path, state='test')

testloader = DataLoader(db_test, batch_size=1, shuffle=False,
                       num_workers=1)


#writer = SummaryWriter(snapshot_path + '/log')
logging.info("{} iterations per epoch".format(len(testloader)))
print('start')
metric_list = 0.0
for i_batch, sampled_batch in enumerate(testloader):
    model.eval()

    volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
    print(i_batch)
    print('-------', volume_batch.shape)
    #volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)

    metric_i = test_single_volume_ds(sampled_batch["image"], sampled_batch["label"], model, device, i_batch, classes=num_classes)
    print('DSC ' + str(i_batch) + ': ', metric_i)
    metric_list += np.array(metric_i)
    print('metric_list: ',metric_list)
    
metric_list = metric_list / len(db_test)
print('metric_list_com: ',metric_list)
performance = np.mean(metric_list, axis=0)[0]
mean_hd95 = np.mean(metric_list, axis=0)[1]

#logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
print('Mean Dice: ', performance)
