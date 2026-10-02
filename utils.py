
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import random
from torch.backends import cudnn
import math
from torch.optim import Optimizer
import copy

from loss import DiceBCELoss, BinaryDistillationLoss, DiceCELoss

##############################################################################
# Tools
##############################################################################

def set_server_method(args):
    """Map the public method name to its client and server updates."""
    methods = {
        'FedAvg': ('local_train', 'fedavg'),
        'FedBN': ('local_train', 'fedbn'),
        'FedProx': ('fedprox', 'fedavg'),
        'SioBN': ('local_train', 'siobn'),
        'SingleSet': ('local_train', 'singleset'),
        'FedRoD': ('fedrod', 'fedavg'),
        'FedDYN': ('feddyn', 'feddyn'),
        'Scaffold': ('scaffold', 'scaffold'),
        'Ditto': ('ditto', 'fedavg'),
        'MOON': ('moon', 'fedavg'),
        'FedNova': ('local_train', 'fednova'),
        'FedPer': ('local_train', 'fedper'),
        'PN': ('local_train', 'fedavg'),
        'FedRDN': ('fedrdn', 'fedavg'),
        'FedLWS': ('local_train', 'fedlws'),
        'FedAWA': ('local_train', 'fedawa'),
        'FedGM': ('local_train', 'fedgm'),
    }
    args.client_method, args.server_method = methods[args.method]
    return args


class Model(nn.Module):
    """For classification problem"""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def get_params(self):
        return self.state_dict()

    def get_gradients(self, dataloader):
        raise NotImplementedError


def set_params(model, model_state_dict, exclude_keys=set()):
    """
        Reference: Be careful with the state_dict[key].
        https://discuss.pytorch.org/t/how-to-copy-a-modified-state-dict-into-a-models-state-dict/64828/4.
    """
    with torch.no_grad():
        for key in model_state_dict.keys():
            if key not in exclude_keys:
                model.state_dict()[key].copy_(model_state_dict[key])
    return model

def freeze_layers(model, layers_to_freeze):
    for name, p in model.named_parameters():
        try:
            if name in layers_to_freeze:
                p.requires_grad = False
            else:
                p.requires_grad = True
        except:
            pass
    return model

class ModelWrapper(Model):
    def __init__(self, base, head, config):
        """
            head and base should be nn.module
        """
        super(ModelWrapper, self).__init__(config)

        self.base = base
        self.head = head

    def forward(self, x, return_embedding):
        feature_embedding = self.base(x)
        out = self.head(feature_embedding)
        if return_embedding:
            return feature_embedding, out
        else:
            return out


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def value(self):
        return self.total / float(self.steps)

def softmax_fuct(lrs):
    '''
    lrs is dict as {0:3, 1:3, 2:4}
    '''
    exp_cache = []
    softmax_lrs = {}
    for i in range(len(lrs)):
        exp_cache.append(math.exp(lrs[i]))
    
    for i in range(len(lrs)):
        softmax_lrs[i] = exp_cache[i]/sum(exp_cache)
    
    return softmax_lrs

def cos(x, y):
    fuct = nn.CosineSimilarity(dim=0)
    result = fuct(x, y)
    result = result.detach().cpu().numpy().tolist()
    return result

def get_cosGrad_matrix(gradients):
    client_num = len(gradients)
    matrix = [[0.0 for _ in range(client_num)] for _ in range(client_num)]

    for i in range(client_num):
        for j in range(client_num):
            if matrix[j][i] != 0.0:
                matrix[i][j] = matrix[j][i]
            else:
                matrix[i][j] = cos(gradients[i], gradients[j])
    
    return matrix

def model_parameter_vector(args, model):
    param = [p.view(-1) for p in model.parameters()]
    # vector = torch.concat(param, dim=0)
    vector = torch.cat(param, dim=0)
    return vector

##############################################################################
# Initialization function
##############################################################################

def init_model(model_type, args):
    if model_type != 'UNet2D':
        raise ValueError(f"This release supports UNet2D, got {model_type!r}")
    from models_dict.unet2d import Unet2D
    norm = 'pn' if args.method == 'PN' else 'gn'
    return Unet2D(norm=norm, num_classes=args.num_classes)


def init_optimizer(num_id, model, args):

    if args.client_method == 'scaffold':
        optimizer = ScaffoldOptimizer(model.parameters(), lr=args.lr, weight_decay=args.local_wd_rate)
    else:
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.local_wd_rate)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.local_wd_rate)

    return optimizer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

##############################################################################
# Training function
##############################################################################

def generate_matchlist(client_node, ratio = 0.5):
    candidate_list = [i for i in range(len(client_node))]
    select_num = int(ratio * len(client_node))
    match_list = np.random.choice(candidate_list, select_num, replace = False).tolist()
    return match_list

def lr_scheduler(rounds, node_list, args):
    # learning rate scheduler for decaying
    if (rounds+1)%args.stepsize == 0:
        args.lr /= 2.0 #0.99
        for i in range(len(node_list)):
            node_list[i].args.lr = args.lr
            node_list[i].optimizer.param_groups[0]['lr'] = args.lr
    print('Learning rate={:.4f}'.format(args.lr))
    return args
    

class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')

        default = dict(lr=lr, mu=mu)

        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                # g = g.cuda()
                if p.grad != None:
                    d_p = p.grad.data + group['mu'] * (p.data - g.data)
                    p.data.add_(d_p, alpha=-group['lr'])

##############################################################################
# Validation function
##############################################################################
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

def compute_metrics(pre, gt): #D, H, W
    pred = pre.cpu().numpy()
    gt = gt.cpu().numpy()
    acc=accuracy_score(gt, pred)
    recall=recall_score(gt, pred, average='micro')
    prec = precision_score(gt, pred, average='macro')
    f1 = f1_score(gt, pred, average='macro')

    return acc, recall, prec, f1

def compute_auc(pre_scores, gt, num_classes = 8):

    pre_scores = pre_scores.cpu().numpy()
    gt = gt.cpu().numpy()
    
    gt_one_hot = np.eye(num_classes)[gt]
    auc_score = roc_auc_score(gt_one_hot, pre_scores)
    
    return auc_score
    

def validate(args, node, test_loader):
    '''
    Generally, 'validate' refers to the local datasets of clients and 'local' refers to the server's testset.
    '''
    node.model.eval() 
    with torch.no_grad():
        preds = []
        targets = []
        pred_scores = []
        loss = 0
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output, feature = node.model(data)
            
            loss_local =  F.cross_entropy(output, target)
            loss = loss + loss_local.item()
            pred = output.argmax(dim=1)
            pred_scores.append(output.softmax(dim=1)) # B, C
            preds.append(pred)
            targets.append(target.view_as(pred))
            
        
        
        pred_scores = torch.cat(pred_scores)
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        acc, recall, prec, f1 = compute_metrics(preds, targets)
        auc = compute_auc(pred_scores, targets, num_classes=args.num_classes)
        loss = loss/preds.shape[0]
        
    return loss, acc*100, recall*100, prec*100, f1*100, auc*100


def validate_seg(args, node, test_loader):
    '''
    Generally, 'validate' refers to the local datasets of clients and 'local' refers to the server's testset.
    '''
    node.model.eval() 

    if args.dataset == 'Digit':
        criterion = torch.nn.CrossEntropyLoss()
    if args.dataset == 'FeTS2022' or args.dataset == 'Fundus' or args.dataset == 'Prostate' or args.dataset == 'Meibomian_Gland' or args.dataset == 'Pancreas' or args.dataset == 'Polyp' or args.dataset == 'Pathology_COSAS2024' or args.dataset == 'FL_Ultrasound' or args.dataset == 'FL_Breast_Ultrasound' or args.dataset == 'FL_Skin' or args.dataset == 'KiTS19' or args.dataset == 'MMS':
        if args.loss == 'bce':
            criterion = torch.nn.BCEWithLogitsLoss()
        elif args.loss == 'dice_bce':
            criterion = DiceBCELoss()
        else:
            assert False
            
    with torch.no_grad():

        dices = []
        ious = []
        sensitivitys = []
        specificitys = []
        loss = 0
        num = 0
        if args.dataset == 'Pancreas' or args.dataset == 'MMS': #
            for idx, batch in enumerate(test_loader):
                
                data = batch['img']
                target = batch['mask']
                
                data, target = data.cuda(), target.cuda()
                
                if args.dataset == 'Pancreas':
                    patch_size=(32, 256, 256)
                if args.dataset == 'MMS':
                    patch_size=(16, 256, 256)
                
                output = sliding_window_inference_3d(args, node, data, patch_size=patch_size, overlap=0.5,)
                
                #print('output-target',output.shape,target.shape)
                #output-target torch.Size([1, 4, 12, 196, 240]) torch.Size([1, 1, 12, 196, 240])
                loss_local =  criterion(output, target)
                
                loss = loss + loss_local.item()
                
                dice = dice_fn(output, target)
                iou = iou_fn(output, target)
                sensitivity, specificity = sensitivity_specificity_fn(output, target)
                dices.append(dice.item())
                ious.append(iou.item())
                sensitivitys.append(sensitivity.item())
                specificitys.append(specificity.item())
                num += data.shape[0]
        else:
            for idx, batch in enumerate(test_loader):
                
                data = batch['img']
                target = batch['mask']
                
                data, target = data.cuda(), target.cuda()
                
                if args.method == 'FedRoD':
                    logit, feature = node.model(data)
                    logit_p = node.p_head(feature)
                    output = logit + logit_p
                elif args.method == 'Ditto':
                    output, feature = node.p_model(data)
                elif args.method == 'FedRDN':
                    data = node.FedRDNTransform_test(data)
                    output, _ = node.model(data)
                else:
                    output, _ = node.model(data)
                #print('output-target',output.shape,target.shape, np.unique(target.cpu().numpy()))
                loss_local =  criterion(output, target)
                
                loss = loss + loss_local.item()
                
                dice = dice_fn(output, target)
                iou = iou_fn(output, target)
                sensitivity, specificity = sensitivity_specificity_fn(output, target)
                dices.append(dice.item())
                ious.append(iou.item())
                sensitivitys.append(sensitivity.item())
                specificitys.append(specificity.item())
                num += data.shape[0]
            
        dice = sum(dices) / num
        iou = sum(ious) / num
        sensitivity = sum(sensitivitys) / num
        specificity = sum(specificitys) / num
        loss = loss / num
        
    return loss, dice*100, iou*100, sensitivity*100, specificity*100




def sliding_window_inference_3d(
    args,
    node,
    volume,           # [1, C, D, H, W] 或 [C, D, H, W]
    patch_size=(32, 256, 256),
    overlap=0.5,      # 重叠比例
    ):
    """
    3D滑动窗口推理
    """
    # 参数设置
    stride = [int(p * (1 - overlap)) for p in patch_size]
    volume_shape = volume.shape[-3:]  # D, H, W
    
    # 初始化输出和权重矩阵
    output_shape = (args.num_classes, *volume_shape)
    output = torch.zeros(output_shape).cuda()
    count = torch.zeros(output_shape).cuda()
    
    # 高斯权重（平滑重叠区域）
    gaussian_weight = create_gaussian_weight(patch_size)
    gaussian_weight = gaussian_weight.cuda()
    
    # 滑动窗口
    for d in range(0, volume_shape[0], stride[0]):
        for h in range(0, volume_shape[1], stride[1]):
            for w in range(0, volume_shape[2], stride[2]):
                # 计算patch范围
                d_end = min(d + patch_size[0], volume_shape[0])
                h_end = min(h + patch_size[1], volume_shape[1])
                w_end = min(w + patch_size[2], volume_shape[2])
                
                # 处理边界
                d_start = d_end - patch_size[0]
                h_start = h_end - patch_size[1]
                w_start = w_end - patch_size[2]
                
                if d_start < 0: d_start = 0
                if h_start < 0: h_start = 0
                if w_start < 0: w_start = 0
                
                d_end = min(d_start + patch_size[0], volume_shape[0])
                h_end = min(h_start + patch_size[1], volume_shape[1])
                w_end = min(w_start + patch_size[2], volume_shape[2])
                
                # 提取patch
                patch = volume[...,
                              d_start:d_end,
                              h_start:h_end,
                              w_start:w_end]

                # 填充到固定尺寸（如果边界不足）
                original_shape = patch.shape[-3:]
                if original_shape != patch_size:
                    pad_d = patch_size[0] - patch.shape[-3]
                    pad_h = patch_size[1] - patch.shape[-2]
                    pad_w = patch_size[2] - patch.shape[-1]
                    
                    patch = torch.nn.functional.pad(
                        patch,
                        (0, pad_w, 0, pad_h, 0, pad_d),
                        mode='constant'
                    )
                # 推理
                with torch.no_grad():
                    if args.method == 'FedRoD':
                        logit, feature = node.model(patch)
                        logit_p = node.p_head(feature)
                        pred_patch = logit + logit_p
                    elif args.method == 'Ditto':
                        pred_patch, feature = node.p_model(patch)
                    elif args.method == 'FedRDN':
                        data = node.FedRDNTransform_test(patch)
                        pred_patch, _ = node.model(data)
                    else:
                        pred_patch, _ = node.model(patch)
                    pred_patch = pred_patch.squeeze(0)

                # 裁剪回原始大小（如果是填充过的）
                if original_shape != patch_size:
                    pred_patch = pred_patch[..., :d_end-d_start,
                                               :h_end-h_start,
                                               :w_end-w_start]
                    
                # 加权累加
                weight = gaussian_weight[..., :d_end-d_start,
                                            :h_end-h_start,
                                            :w_end-w_start]

                output[..., d_start:d_end, h_start:h_end, w_start:w_end] += (pred_patch * weight)
                count[..., d_start:d_end, h_start:h_end, w_start:w_end] += weight

    # 归一化
    output = output / (count + 1e-8)
    return output.unsqueeze(0)

def create_gaussian_weight(patch_size, sigma=0.125):
    """创建3D高斯权重矩阵"""
    center = [p // 2 for p in patch_size]
    meshgrid = torch.meshgrid(
        [torch.arange(p, dtype=torch.float32) for p in patch_size],
        indexing='ij'
    )
    
    weight = torch.ones(patch_size, dtype=torch.float32)
    for i, (grid, c) in enumerate(zip(meshgrid, center)):
        weight *= torch.exp(-((grid - c) ** 2) / (2 * (sigma * patch_size[i]) ** 2))
    
    return weight / weight.max()





def dice_fn(y_pred, y_true, smooth=1e-6):
    """
    PyTorch 实现的 Dice 系数（支持批量计算）
    Args:
        y_pred: 预测概率图 (B, C, H, W), (B, C, D, H, W)
        y_true: Ground Truth 标签 (B, H, W), (B, D, H, W)
        smooth: 平滑系数避免除以零
    Returns:
        dice: Dice 系数
    """
    num_classes = y_pred.shape[1]
    if num_classes==1:
        y_pred = torch.sigmoid(y_pred) #(B, 1, H, W), (B, 1, D, H, W)
        y_pred = (y_pred > 0.5).float()  # (B, 1, H, W), (B, 1, D, H, W)
        y_pred = y_pred.squeeze(1) # (B, H, W), (B, D, H, W)
        if y_pred.dim() == 3:
            intersection = (y_pred * y_true).sum(dim=(1, 2))
            sum_true_pred = y_pred.sum(dim=(1, 2)) + y_true.sum(dim=(1, 2))
        if y_pred.dim() == 4:
            intersection = (y_pred * y_true).sum(dim=(1, 2, 3))
            sum_true_pred = y_pred.sum(dim=(1, 2, 3)) + y_true.sum(dim=(1, 2, 3))
    else:
        y_pred = torch.softmax(y_pred, dim = 1)#B,C,H,W    B,C,D,H,W   
        y_pred = torch.argmax(y_pred, dim = 1)#B,H,W     B,D,H,W
        if y_pred.dim() == 3:
            y_pred = F.one_hot(y_pred, num_classes).permute(0, 3, 1, 2).float() #B,C,H,W
            y_true = F.one_hot(y_true.long(), num_classes).permute(0, 3, 1, 2).float()#B,C,H,W
            intersection = (y_pred * y_true).sum(dim=(2, 3))
            sum_true_pred = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
        
        if y_pred.dim() == 4:
            y_pred = F.one_hot(y_pred, num_classes).permute(0, 4, 1, 2, 3).float()#B,C,D,H,W
            y_true = F.one_hot(y_true.long(), num_classes).permute(0, 4, 1, 2, 3).float()#B,C,D,H,W
            intersection = (y_pred * y_true).sum(dim=(2, 3, 4))
            sum_true_pred = y_pred.sum(dim=(2, 3, 4)) + y_true.sum(dim=(2, 3, 4))
        
    dice = (2.0 * intersection + smooth) / (sum_true_pred + smooth)
    dice = dice/num_classes
        
    return dice.sum()  # 返回批量的均值



def dice_fn_v1(y_pred, y_true, smooth=1e-6, num_class=2):
    """
    PyTorch 实现的 Dice 系数（支持批量计算）
    Args:
        y_pred: 预测概率图 (B, 1, H, W), 值范围 [0, 1]
        y_true: Ground Truth 标签 (B, 1, H, W), 值范围 {0, 1}
        smooth: 平滑系数避免除以零
    Returns:
        dice: Dice 系数
    """
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float().squeeze(dim=1)  # 二值化
    all_dice = 0
    batch_size = y_true.shape[0]
    y_true = y_true.squeeze(dim=1)
    for i in range(num_class):
        each_pred = torch.zeros_like(y_pred)
        each_pred[y_pred==i] = 1

        each_gt = torch.zeros_like(y_true)
        each_gt[y_true==i] = 1            
        
        intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1)
        
        union = each_pred.view(batch_size,-1).sum(1) + each_gt.view(batch_size,-1).sum(1)
        dice = (2. *  intersection )/ (union + 1e-5)
        
        all_dice += torch.sum(dice) # B, 1
    
    return all_dice * 1.0 / num_class



def iou_fn(y_pred, y_true):
    """
    计算二分类IoU
    Args:
        y_pred: 预测概率图 (B, C, H, W), (B, C, D, H, W)
        y_true: Ground Truth 标签 (B, H, W), (B, D, H, W)
        threshold: 二值化阈值
    Returns:
        iou: 交并比
    """
    num_classes = y_pred.shape[1]
    
    if num_classes == 1:
        # 二值化预测结果
        y_pred = torch.sigmoid(y_pred)#(B, 1, H, W), (B, 1, D, H, W)
        y_pred = (y_pred > 0.5).float()#(B, 1, H, W), (B, 1, D, H, W)
        y_pred = y_pred.squeeze(1) # (B, H, W), (B, D, H, W)
        y_true = y_true.float() # (B, H, W), (B, D, H, W)
        
        # 计算交集和并集
        if y_pred.dim() == 3:
            intersection = (y_pred * y_true).sum(dim=(1, 2))
            union = (y_pred + y_true).sum(dim=(1, 2)) - intersection
        if y_pred.dim() == 4:
            intersection = (y_pred * y_true).sum(dim=(1, 2, 3))
            union = (y_pred + y_true).sum(dim=(1, 2, 3)) - intersection
    else:
        y_pred = torch.softmax(y_pred, dim = 1)#(B, C, H, W), (B, C, D, H, W)
        y_pred = torch.argmax(y_pred, dim = 1)#(B, H, W), (B, D, H, W)
        if y_pred.dim() == 3:
            y_pred = F.one_hot(y_pred, num_classes).permute(0, 3, 1, 2).float() #B,C,H,W
            y_true = F.one_hot(y_true.long(), num_classes).permute(0, 3, 1, 2).float() #B,C,H,W
            intersection = (y_pred * y_true).sum(dim=(2, 3))
            union = (y_pred + y_true).sum(dim=(2, 3)) - intersection
            
        if y_pred.dim() == 4:
            y_pred = F.one_hot(y_pred, num_classes).permute(0, 4, 1, 2, 3).float()
            y_true = F.one_hot(y_true.long(), num_classes).permute(0, 4, 1, 2, 3).float()
            intersection = (y_pred * y_true).sum(dim=(2, 3, 4))
            union = (y_pred + y_true).sum(dim=(2, 3, 4)) - intersection
            
    # 避免除以零
    iou = (intersection + 1e-6) / (union + 1e-6)
    iou = iou/num_classes
        
    return iou.sum()





def iou_fn_v1(y_pred, y_true, num_class=2):
    """
    计算二分类IoU
    Args:
        y_pred: 预测概率图 (B, 1, H, W)，值范围[0,1]
        y_true: 真实标签 (B, 1, H, W)，值范围{0,1}
        threshold: 二值化阈值
    Returns:
        iou: 交并比
    """
    # 二值化预测结果
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float().squeeze(dim=1)  # 二值化
    y_true = y_true.float().squeeze(dim=1)
    all_iou = 0
    batch_size = y_true.shape[0]
    for i in range(num_class):
        each_pred = torch.zeros_like(y_pred)
        each_pred[y_pred==i] = 1

        each_gt = torch.zeros_like(y_true)
        each_gt[y_true==i] = 1            
        
        # 计算交集和并集
        intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1)
        union = each_pred.view(batch_size,-1).sum(1) + each_gt.view(batch_size,-1).sum(1)
        
        # 避免除以零
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        all_iou += torch.sum(iou)
        
    
    return all_iou * 1.0 / num_class



def sensitivity_specificity_fn(y_pred, y_true):
    """
    计算二分类任务的 Sensitivity 和 Specificity
    Args:
        y_pred: 预测概率图 (B, C, H, W), (B, C, D, H, W)
        y_true: Ground Truth 标签 (B, H, W), (B, D, H, W)
        threshold: 二值化阈值
    Returns:
        sensitivity, specificity
    """
    num_classes = y_pred.shape[1]
    if num_classes == 1:
        # 二值化预测结果
        y_pred = torch.sigmoid(y_pred)#(B, 1, H, W), (B, 1, D, H, W)
        y_pred = (y_pred > 0.5).float()#(B, 1, H, W), (B, 1, D, H, W)
        y_pred = y_pred.squeeze(1) # (B, H, W), (B, D, H, W)
        y_true = y_true.float() # (B, H, W), (B, D, H, W)
        
        # 计算 TP, FN, TN, FP
        TP = torch.sum(y_pred * y_true)      # True Positive
        FN = torch.sum((1 - y_pred) * y_true) # False Negative
        TN = torch.sum((1 - y_pred) * (1 - y_true)) # True Negative
        FP = torch.sum(y_pred * (1 - y_true)) # False Positive
        
    else:
        y_pred = torch.softmax(y_pred, dim = 1)#(B, C, H, W), (B, C, D, H, W)
        y_pred = torch.argmax(y_pred, dim = 1)#(B, H, W), (B, D, H, W)
        y_true = y_true.long()#(B, H, W), (B, D, H, W)
        if y_pred.dim() == 3:
            y_pred = F.one_hot(y_pred, num_classes).permute(0, 3, 1, 2).float()#B,C,H,W
            y_true = F.one_hot(y_true, num_classes).permute(0, 3, 1, 2).float()#B,C,H,W
 
            # 计算 TP, FN, TN, FP
            TP = torch.sum(y_pred * y_true, dim=(2, 3))      # True Positive
            FN = torch.sum((1 - y_pred) * y_true, dim=(2, 3)) # False Negative
            TN = torch.sum((1 - y_pred) * (1 - y_true), dim=(2, 3)) # True Negative
            FP = torch.sum(y_pred * (1 - y_true), dim=(2, 3)) # False Positive
        
        if y_pred.dim() == 4:
            y_pred = F.one_hot(y_pred, num_classes).permute(0, 4, 1, 2, 3).float()
            y_true = F.one_hot(y_true, num_classes).permute(0, 4, 1, 2, 3).float()
            
            # 计算 TP, FN, TN, FP
            TP = torch.sum(y_pred * y_true, dim=(2, 3, 4))      # True Positive
            FN = torch.sum((1 - y_pred) * y_true, dim=(2, 3, 4)) # False Negative
            TN = torch.sum((1 - y_pred) * (1 - y_true), dim=(2, 3, 4)) # True Negative
            FP = torch.sum(y_pred * (1 - y_true), dim=(2, 3, 4)) # False Positive
            
    # 避免除以零
    sensitivity = TP / (TP + FN + 1e-6)
    sensitivity = sensitivity.sum()/num_classes
    specificity = TN / (TN + FP + 1e-6)
    specificity = specificity.sum()/num_classes

    return sensitivity, specificity


#########################Scaffold##############################
from torch.optim import Optimizer

class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(ScaffoldOptimizer, self).__init__(params, defaults)

    def step(self, server_controls, client_controls, closure=None):

        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            for p, c, ci in zip(group['params'], server_controls.values(), client_controls.values()):
                if p.grad is None:
                    continue
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp.data * group['lr']

        return loss
    
class FedRDNTransform:
    """
    Federated Random Data Normalization Transform
    用于特征分布偏斜的联邦学习数据增强
    """
    
    def __init__(self, 
                 local_stats,
                 global_stats,
                 mode: str = 'train',
                 p: float = 1.0):
        """
        初始化FedRDN变换
        
        Args:
            local_stats: 本地统计信息 (mean, std)
            global_stats: 全局统计信息列表 [(mean1, std1), (mean2, std2), ...]
            mode: 'train' 或 'test'
            p: 应用变换的概率
        """
        self.local_mean, self.local_std = local_stats
        self.global_means = [s[0] for s in global_stats]
        self.global_stds = [s[1] for s in global_stats]
        self.mode = mode
        self.p = p
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用FedRDN变换
        
        Args:
            x: 输入图像 [C, H, W]
            
        Returns:
            变换后的图像
        """
        if random.random() > self.p:
            return x
            
        if self.mode == 'train':
            # 训练阶段：随机选择全局统计信息
            idx = random.randint(0, len(self.global_means) - 1)
            mean = self.global_means[idx]
            std = self.global_stds[idx]
        else:
            # 测试阶段：使用本地统计信息
            mean = self.local_mean
            std = self.local_std
            
        # 归一化
        x_normalized = (x - mean.view(-1, 1, 1)) / (std.view(-1, 1, 1) + 1e-8)
        
        return x_normalized