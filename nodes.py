

import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from utils import init_model, PerturbedGradientDescent
from utils import init_optimizer, model_parameter_vector


class Node(object):

    def __init__(self,args, client_id, client_name, train_loader, val_loader, test_loader):
        self.client_id = client_id
        self.client_name = client_name
        self.args = args
        self.node_num = self.args.node_num
        self.num_classes = args.num_classes

        self.train_loader = train_loader
        self.val_loader  = val_loader
        self.test_loader = test_loader

        self.model = init_model(self.args.local_model, self.args).cuda()
        
        if args.client_method == 'fedrod':
            if args.local_model == 'UNet2D':
                self.p_head = copy.deepcopy(self.model.seg1)
            if args.local_model == 'UNet3D':
                self.p_head = copy.deepcopy(self.model.out)
            if args.local_model == 'SANet':
                self.p_head = copy.deepcopy(self.model.last)
        # node init for feddyn
        if args.client_method == 'feddyn':
            self.old_grad = None
            self.old_grad = copy.deepcopy(self.model)
            self.old_grad = model_parameter_vector(args, self.old_grad)
            self.old_grad = torch.zeros_like(self.old_grad)
        if 'feddyn' in args.server_method:
            self.server_state = copy.deepcopy(self.model)
            for param in self.server_state.parameters():
                param.data = torch.zeros_like(param.data)
        #########################Scaffold##############################
        if args.method == 'Scaffold':
            self.control = {}
            self.delta_control = {}
            self.delta_y = {}
            for k, v in self.model.named_parameters():
                self.control[k] = torch.zeros_like(v.data)
                self.delta_control[k] = torch.zeros_like(v.data)
                self.delta_y[k] = torch.zeros_like(v.data)
        #########################MOON##############################
        if args.method == 'MOON':
            self.pre_model = copy.deepcopy(self.model)
                
        #########################FedRDN##############################
        if args.method == 'FedRDN':
            if self.train_loader:
                self.local_stats = compute_dataset_statistics(self.train_loader, args=args) #mean, std
                self.FedRDNTransform_train = None
                self.FedRDNTransform_test = None
        
        # p_model and p_optimizer for ditto
        if args.method == 'Ditto':
            self.p_model = init_model(self.args.local_model, self.args).cuda()
            if args.optimizer == 'sgd':
                self.p_optimizer = torch.optim.SGD(self.p_model.parameters(), lr=args.lr_per, momentum=args.momentum, weight_decay=args.local_wd_rate)
            elif args.optimizer == 'adam':
                self.p_optimizer = torch.optim.Adam(self.p_model.parameters(), lr=args.lr_per, weight_decay=args.local_wd_rate)

        self.optimizer = init_optimizer(self.client_id, self.model, args)
        
        self.averager = Averager(client_id, client_name)
        self.maxer = Maxer(client_id, client_name)
        
        if args.dataset == 'FeTS2022' or args.dataset == 'FL_Skin' or args.dataset == 'Fundus' or args.dataset == 'Prostate' or args.dataset == 'Meibomian_Gland' or args.dataset == 'Pancreas' or args.dataset == 'Polyp' or args.dataset == 'Pathology_COSAS2024' or args.dataset == 'FL_Ultrasound' or args.dataset == 'FL_Breast_Ultrasound' or args.dataset == 'KiTS19' or args.dataset == 'MMS':
            self.averager = Averager_seg(client_id, client_name)
            self.maxer = Maxer_seg(client_id, client_name)
            self.recorder = Recorder_seg(client_id, client_name)
                

def compute_dataset_statistics(dataloader, args=None):
    """
    计算数据集的统计信息（均值和标准差）
    
    Args:
        dataloader: 数据加载器
        
    Returns:
        (mean, std): 通道级别的均值和标准差
    """
    # 初始化统计变量
    n_samples = 0
    if args.dataset == 'MMS' or args.dataset == 'Pancreas' or args.dataset == 'FeTS2022':
        
        if args.dataset == 'MMS' or args.dataset == 'Pancreas':
            channel_sum = torch.zeros(1).cuda()  # 假设单通道图像
            channel_sum_sq = torch.zeros(1).cuda()
        if args.dataset == 'FeTS2022':
            channel_sum = torch.zeros(4).cuda()  # 假设单通道图像
            channel_sum_sq = torch.zeros(4).cuda()
            
        for batch in dataloader:
            
            images = batch['img']
                
            # 确保图像是浮点数
            images = images.float().cuda()
            
            # 更新统计
            n_samples += images.size(0)
            channel_sum += images.sum(dim=[0, 2, 3, 4])  # 按批次和空间维度求和
            channel_sum_sq += (images ** 2).sum(dim=[0, 2, 3, 4])
        
        # 计算均值和标准差
        mean = channel_sum / (n_samples * images.size(2) * images.size(3) * images.size(4))
        std = torch.sqrt(channel_sum_sq / (n_samples * images.size(2) * images.size(3) * images.size(4)) - mean ** 2)
    
    else:
        channel_sum = torch.zeros(3).cuda()  # 假设单通道图像
        channel_sum_sq = torch.zeros(3).cuda()
        
        for batch in dataloader:
            
            images = batch['img']
                
            # 确保图像是浮点数
            images = images.float().cuda()
            
            # 更新统计
            n_samples += images.size(0)
            channel_sum += images.sum(dim=[0, 2, 3])  # 按批次和空间维度求和
            channel_sum_sq += (images ** 2).sum(dim=[0, 2, 3])
        
        # 计算均值和标准差
        mean = channel_sum / (n_samples * images.size(2) * images.size(3))
        std = torch.sqrt(channel_sum_sq / (n_samples * images.size(2) * images.size(3)) - mean ** 2)
    
    return mean, std

class Averager():
    
    def __init__(self, client_id, client_name):
        #val_loss, val_acc, val_recall, val_prec, val_f1, val_auc
        self.client_id = client_id
        self.client_name = client_name
        self.loss = 0
        self.acc = 0
        self.recall = 0
        self.prec = 0
        self.f1 = 0
        self.auc = 0
        self.num = 0
    
    def update(self, loss, acc, recall, prec, f1, auc):
        self.loss = self.loss + loss
        self.acc = self.acc + acc
        self.recall = self.recall + recall
        self.prec = self.prec + prec
        self.f1 = self.f1 + f1
        self.auc = self.auc + auc
        self.num = self.num + 1
        
    def log(self, is_log = False):
        if self.num == 0:
            loss = self.loss
            acc = self.acc
            recall = self.recall
            prec = self.prec
            f1 = self.f1
            auc = self.auc
        else:
            loss = self.loss/self.num
            acc = self.acc/self.num
            recall = self.recall/self.num
            prec = self.prec/self.num
            f1 = self.f1/self.num
            auc = self.auc/self.num
        if is_log:
            print('Test  {:<12.12}, loss:{:.5f}, ACC:{:.3f}, Recall:{:.3f}, Prec:{:.3f}, F1:{:.3f}, AUC:{:.3f}'.format(self.client_name, loss, acc, recall, prec, f1, auc), flush=True)
        
        return loss, acc, recall, prec, f1, auc
    
    
    
class Maxer():
    
    def __init__(self, client_id, client_name):
        #val_loss, val_acc, val_recall, val_prec, val_f1, val_auc
        self.client_id = client_id
        self.client_name = client_name
        self.loss = 0
        self.acc = 0
        self.recall = 0
        self.prec = 0
        self.f1 = 0
        self.auc = 0
        self.epoch = 0
    
    
    def update(self, epoch, loss, acc, recall, prec, f1, auc):
        
        if acc>self.acc:
            self.loss = loss
            self.acc = acc
            self.recall = recall
            self.prec = prec
            self.f1 = f1
            self.auc = auc
            self.epoch = epoch
        
    def log(self, is_log = False):

        loss = self.loss
        acc = self.acc
        recall = self.recall
        prec = self.prec
        f1 = self.f1
        auc = self.auc
        epoch = self.epoch

        if is_log:
            print('Test  {:<12.12}, loss:{:.5f}, ACC:{:.3f}, Recall:{:.3f}, Prec:{:.3f}, F1:{:.3f}, AUC:{:.3f}, Epoch:{}'.format(self.client_name, loss, acc, recall, prec, f1, auc, epoch), flush=True)
        
        return loss, acc, recall, prec, f1, auc, epoch
    


class Seed_Averager():
    
    def __init__(self, client_id, client_name):
        #val_loss, val_acc, val_recall, val_prec, val_f1, val_auc
        self.client_id = client_id
        self.client_name = client_name
        self.acc = []
        self.recall = []
        self.prec = []
        self.f1 = []
        self.auc = []
    
    def update(self, acc = 0, recall = 0, prec = 0, f1 = 0, auc = 0):
        self.acc.append(acc)
        self.recall.append(recall)
        self.prec.append(prec)
        self.f1.append(f1)
        self.auc.append(auc)

    def log(self, is_log = False, details = False):
        mean_acc = np.mean(self.acc)
        std_acc = np.std(self.acc)

        mean_recall = np.mean(self.recall)
        std_recall = np.std(self.recall)

        mean_prec = np.mean(self.prec)
        std_prec = np.std(self.prec)
        
        mean_f1 = np.mean(self.f1)
        std_f1 = np.std(self.f1)
        
        mean_auc = np.mean(self.auc)
        std_auc = np.std(self.auc)
        
        if details:
            print('Client:', self.client_name)
            print('ACC   :', self.acc)
            print('Recall:', self.recall)
            print('Prec  :', self.prec)
            print('F1    :', self.f1)
            print('AUC   :', self.auc)
            print('-----------------------------------------------------')
        
        if is_log:
            print('Test  {:<12.12}, ACC:{:.3f}[{:.3f}], Recall:{:.3f}[{:.3f}], Prec:{:.3f}[{:.3f}], F1:{:.3f}[{:.3f}], AUC:{:.3f}[{:.3f}]'.format(self.client_name, mean_acc, std_acc, mean_recall, std_recall, mean_prec, std_prec, mean_f1, std_f1, mean_auc, std_auc), flush=True)
        
        return mean_acc, std_acc, mean_recall, std_recall, mean_prec, std_prec, mean_f1, std_f1, mean_auc, std_auc



class Averager_seg():
    
    def __init__(self, client_id, client_name):
        #val_loss, val_acc, val_recall, val_prec, val_f1, val_auc
        self.client_id = client_id
        self.client_name = client_name
        self.loss = 0
        self.dice = 0
        self.iou = 0
        self.sensitivity = 0  
        self.specificity = 0
        
        self.num = 0
    
    def update(self, loss, dice, iou, sensitivity, specificity):
        self.loss = self.loss + loss
        self.dice = self.dice + dice
        self.iou = self.iou + iou
        self.sensitivity = self.sensitivity + sensitivity
        self.specificity = self.specificity + specificity
        self.num = self.num + 1
        
    def log(self, is_log = False):
        if self.num == 0:
            loss = self.loss
            dice = self.dice
            iou = self.iou
            sensitivity = self.sensitivity
            specificity = self.specificity
        else:
            loss = self.loss/self.num
            dice = self.dice/self.num
            iou = self.iou/self.num
            sensitivity = self.sensitivity/self.num
            specificity = self.specificity/self.num
            
        if is_log:
            print('Test  {:<12.12}, loss:{:.5f}, Dice:{:.5f}, IOU:{:.5f}, Sen:{:.5f}, Spe:{:.5f}'.format(self.client_name, loss, dice, iou, sensitivity, specificity), flush=True)
        
        return loss, dice, iou, sensitivity, specificity
    
    
    
class Maxer_seg():
    
    def __init__(self, client_id, client_name):
        #val_loss, val_acc, val_recall, val_prec, val_f1, val_auc
        self.client_id = client_id
        self.client_name = client_name
        self.loss = 0
        self.dice = 0
        self.iou = 0
        self.sensitivity = 0
        self.specificity = 0
        self.epoch = 0
    
    
    def update(self, epoch, loss, dice, iou, sensitivity, specificity):
        
        if dice>self.dice:
            self.loss = loss
            self.dice = dice
            self.iou = iou
            self.sensitivity = sensitivity
            self.specificity = specificity
            self.epoch = epoch
        
    def log(self, is_log = False):

        loss = self.loss
        dice = self.dice
        iou = self.iou
        sensitivity = self.sensitivity
        specificity = self.specificity
        epoch = self.epoch

        if is_log:
            print('Test  {:<12.12}, loss:{:.5f}, Dice:{:.5f}, IOU:{:.5f}, Sen:{:.5f}, Spe:{:.5f}, Epoch:{}'.format(self.client_name, loss, dice, iou,sensitivity,specificity, epoch), flush=True)
        
        return loss, dice, iou, sensitivity, specificity, epoch


class Seed_Averager_seg():
    
    def __init__(self, client_id, client_name):
        #val_loss, val_acc, val_recall, val_prec, val_f1, val_auc
        self.client_id = client_id
        self.client_name = client_name
        self.dice = []
        self.iou = []
        self.sensitivity = []
        self.specificity = []
    
    def update(self, dice = 0, iou = 0, sensitivity=0, specificity=0):
        self.dice.append(dice)
        self.iou.append(iou)
        self.sensitivity.append(sensitivity)
        self.specificity.append(specificity)

    def log(self, is_log = False, details = False):

        
        mean_dice = np.mean(self.dice)
        std_dice = np.std(self.dice)
        
        mean_iou = np.mean(self.iou)
        std_iou = np.std(self.iou)
        
        mean_sensitivity = np.mean(self.sensitivity)
        std_sensitivity = np.std(self.sensitivity)
        
        mean_specificity = np.mean(self.specificity)
        std_specificity = np.std(self.specificity)
        
        if details:
            print('Dice  :', self.dice)
            print('IOU   :', self.iou)
            print('Sen  :', self.sensitivity)
            print('Spe   :', self.specificity)
            print('-----------------------------------------------------')
        
        if is_log:
            print('Test  {:<12.12}, Dice:{:.3f}[{:.3f}], IOU:{:.3f}[{:.3f}], Sen:{:.3f}[{:.3f}], Spe:{:.3f}[{:.3f}]'.format(self.client_name, mean_dice, std_dice, mean_iou, std_iou, mean_sensitivity, std_sensitivity, mean_specificity, std_specificity), flush=True)
        
        return mean_dice, std_dice, mean_iou, std_iou, mean_sensitivity, std_sensitivity, mean_specificity, std_specificity




class Recorder_seg():
    
    def __init__(self, client_id, client_name):
        #val_loss, val_acc, val_recall, val_prec, val_f1, val_auc
        self.client_id = client_id
        self.client_name = client_name
        self.loss = 0
        self.dice = 0
        self.iou = 0
        self.sensitivity = 0
        self.specificity = 0
        self.epoch = 0
    
    
    def update(self, epoch, loss, dice, iou, sensitivity, specificity):
        
        self.loss = loss
        self.dice = dice
        self.iou = iou
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.epoch = epoch
        
    def log(self, is_log = False):

        loss = self.loss
        dice = self.dice
        iou = self.iou
        sensitivity = self.sensitivity
        specificity = self.specificity
        epoch = self.epoch

        if is_log:
            print('Test  {:<12.12}, loss:{:.5f}, Dice:{:.5f}, IOU:{:.5f}, Sen:{:.5f}, Spe:{:.5f}, Epoch:{}'.format(self.client_name, loss, dice, iou,sensitivity,specificity, epoch), flush=True)
        
        return loss, dice, iou, sensitivity, specificity, epoch


