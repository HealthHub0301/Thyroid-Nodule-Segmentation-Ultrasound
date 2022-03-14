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
from dataloaders.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler, prepare_dataset
from utils import losses, metrics, ramps
from val_2D import test_single_volume_ds
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/centos/Farhan/pickle/ultrasound_2d/data/DDTI_HH_mix_val', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='/pyramid_net_label_DDTI_+HH', help='experiment_name') #change everytime for different folds
parser.add_argument('--model', type=str,
                    default='unet_urpc', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=6,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)



def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    train_data_path = "/home/azka/sharjeel/Thyroid/code/training_data.npy"
    validation_data_path = "/home/azka/sharjeel/Thyroid/code/validation_data.npy"

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    device = torch.device('cuda:2')
    model = model.to(device)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    
    db_train = prepare_dataset(train_data_path)
    db_val = prepare_dataset(validation_data_path)
    
    #db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)

    #labeled_idxs = list(range(559))
    #unlabeled_idxs = list(range(1,559))
    labeled_idxs = list(range(400)) # 538
    unlabeled_idxs = list(range(400,500)) #654
    
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=False, worker_init_fn=worker_init_fn)


    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
                          
    #optimizer = optim.Adam(model.parameters(), lr=base_lr)
    
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    kl_distance = nn.KLDivLoss(reduction='none')
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            model.train()

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)

            outputs, outputs_aux1, outputs_aux2, outputs_aux3 = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
            outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
            outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)
            #print('\n', 'ce loss')
            #print('shapes: ', outputs[:args.labeled_bs].shape, ' and ', label_batch[:args.labeled_bs][:].shape)
            #print('unique vals: ',  ' and ', torch.unique(label_batch[:args.labeled_bs][:]))

            loss_ce = ce_loss(outputs[:args.labeled_bs],
                              label_batch[:args.labeled_bs][:].long())
            loss_ce_aux1 = ce_loss(outputs_aux1[:args.labeled_bs],
                                   label_batch[:args.labeled_bs][:].long())
            loss_ce_aux2 = ce_loss(outputs_aux2[:args.labeled_bs],
                                   label_batch[:args.labeled_bs][:].long())
            loss_ce_aux3 = ce_loss(outputs_aux3[:args.labeled_bs],
                                   label_batch[:args.labeled_bs][:].long())
            #print('dice loss')
            #print('shapes: ', outputs_soft[:args.labeled_bs].shape, ' and ', label_batch[:args.labeled_bs].shape)
            #np_pred = outputs_soft[:args.labeled_bs].detach().cpu().numpy()
            #np_tar = label_batch[:args.labeled_bs].detach().cpu().numpy()
            #print('target unique: ', np.unique(np_pred))
            #print('input unique: ', np.unique(np_tar))

            loss_dice = dice_loss(
                outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_dice_aux1 = dice_loss(
                outputs_aux1_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_dice_aux2 = dice_loss(
                outputs_aux2_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_dice_aux3 = dice_loss(
                outputs_aux3_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            #print('dice loss done')
            supervised_loss = (loss_ce+loss_ce_aux1+loss_ce_aux2+loss_ce_aux3 +
                               loss_dice+loss_dice_aux1+loss_dice_aux2+loss_dice_aux3)/8

            preds = (outputs_soft+outputs_aux1_soft +
                     outputs_aux2_soft+outputs_aux3_soft)/4
            #print('kl distance')

            variance_main = torch.sum(kl_distance(
                torch.log(outputs_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_main = torch.exp(-variance_main)

            variance_aux1 = torch.sum(kl_distance(
                torch.log(outputs_aux1_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux1 = torch.exp(-variance_aux1)

            variance_aux2 = torch.sum(kl_distance(
                torch.log(outputs_aux2_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux2 = torch.exp(-variance_aux2)

            variance_aux3 = torch.sum(kl_distance(
                torch.log(outputs_aux3_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux3 = torch.exp(-variance_aux3)

            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_dist_main = (
                preds[args.labeled_bs:] - outputs_soft[args.labeled_bs:]) ** 2

            consistency_loss_main = torch.mean(
                consistency_dist_main * exp_variance_main) / (torch.mean(exp_variance_main) + 1e-8) + torch.mean(variance_main)

            consistency_dist_aux1 = (
                preds[args.labeled_bs:] - outputs_aux1_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux1 = torch.mean(
                consistency_dist_aux1 * exp_variance_aux1) / (torch.mean(exp_variance_aux1) + 1e-8) + torch.mean(variance_aux1)

            consistency_dist_aux2 = (
                preds[args.labeled_bs:] - outputs_aux2_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux2 = torch.mean(
                consistency_dist_aux2 * exp_variance_aux2) / (torch.mean(exp_variance_aux2) + 1e-8) + torch.mean(variance_aux2)

            consistency_dist_aux3 = (
                preds[args.labeled_bs:] - outputs_aux3_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux3 = torch.mean(
                consistency_dist_aux3 * exp_variance_aux3) / (torch.mean(exp_variance_aux3) + 1e-8) + torch.mean(variance_aux3)

            consistency_loss = (consistency_loss_main + consistency_loss_aux1 +
                                consistency_loss_aux2 + consistency_loss_aux3) / 4
            loss = (supervised_loss) + consistency_weight * consistency_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            if iter_num%10000 == 0:
                base_lr = base_lr/5
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)
            '''logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))'''

#             if iter_num % 20 == 0:
#                 image = volume_batch[1, 0:1, :, :]
#                 writer.add_image('train/Image', image, iter_num)
#                 outputs = torch.argmax(torch.softmax(
#                     outputs, dim=1), dim=1, keepdim=True)
#                 writer.add_image('train/Prediction',
#                                  outputs[1, ...] * 50, iter_num)
#                 labs = label_batch[1, ...].unsqueeze(0) * 50
#                 writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_ds(sampled_batch["image"], sampled_batch["label"], model, device, classes=num_classes)
                    metric_list += np.array(metric_i)
                    
                metric_list = metric_list / len(db_val)
                
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                #print(performance)

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"
    
if __name__ == "__main__":
    import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "/home/azka/sharjeel/Thyroid/code/saved_models_with_combined/"

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)