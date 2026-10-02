from cProfile import label
import numpy as np
import torch
import torch.nn.functional as F
from utils import validate, model_parameter_vector, freeze_layers, set_params
import copy
from nodes import Node
from utils import dice_fn as dice_fn
from utils import iou_fn as iou_fn
import torch.nn as nn

from loss import DiceBCELoss, BinaryDistillationLoss

def receive_server_model(args, client_nodes, central_node):

    for idx in range(len(client_nodes)):
        client_nodes[idx].model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))

    return client_nodes

def Client_update(args, client_nodes, central_node, select_list):
    '''
    client update functions
    '''
    '''
    if central_node is not None:
        client_nodes = receive_server_model(args, client_nodes, central_node)
    '''
    if args.client_method == 'local_train':
        client_losses = []
        client_accs = []
        for i in select_list:
            epoch_losses = []
            epoch_accs = []
            for epoch in range(args.E):
                loss, acc = client_localTrain(args, client_nodes[i])
                epoch_losses.append(loss)
                epoch_accs.append(acc)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            client_accs.append(sum(epoch_accs)/len(epoch_accs))
    elif args.client_method == 'fedrdn':
        client_losses = []
        client_accs = []
        for i in select_list:
            epoch_losses = []
            epoch_accs = []
            for epoch in range(args.E):
                loss, acc = client_fedrdn(args, client_nodes[i])
                epoch_losses.append(loss)
                epoch_accs.append(acc)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            client_accs.append(sum(epoch_accs)/len(epoch_accs))

    elif args.client_method == 'fedprox':
        client_losses = []
        client_accs = []
        for i in select_list:
            epoch_losses = []
            epoch_accs = []
            for epoch in range(args.E):
                loss, acc = client_fedprox(args, client_nodes[i], central_node)
                epoch_losses.append(loss)
                epoch_accs.append(acc)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            client_accs.append(sum(epoch_accs)/len(epoch_accs))

    elif 'fedrod' in args.client_method:
        client_losses = []
        client_accs = []
        for i in select_list:
            epoch_losses = []
            epoch_accs = []
            for epoch in range(args.E):
                loss, acc = client_fedrod(args, client_nodes[i])
                epoch_losses.append(loss)
                epoch_accs.append(acc)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            client_accs.append(sum(epoch_accs)/len(epoch_accs))
    elif args.client_method == 'feddyn':
        global_model_vector = copy.deepcopy(model_parameter_vector(args, central_node.model).detach().clone())
        client_losses = []
        client_accs = []
        for i in select_list:
            epoch_losses = []
            epoch_accs = []
            for epoch in range(args.E):
                loss, acc  = client_feddyn(global_model_vector, args, client_nodes[i])
                epoch_losses.append(loss)
                epoch_accs.append(acc)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            client_accs.append(sum(epoch_accs)/len(epoch_accs))
            # update old grad
            v1 = model_parameter_vector(args, client_nodes[i].model).detach()
            client_nodes[i].old_grad = client_nodes[i].old_grad - args.mu * (v1 - global_model_vector)
    #########################Scaffold##############################
    elif 'scaffold' in args.client_method:
        client_losses = []
        client_accs = []
        for i in select_list:
            epoch_losses = []
            epoch_accs = []
            x = copy.deepcopy(client_nodes[i].model)
            #x_control = copy.deepcopy(client_nodes[i].control)
            for epoch in range(args.E):
                loss, acc = client_scaffold(args, client_nodes[i], central_node)
                epoch_losses.append(loss)
                epoch_accs.append(acc)
            ann = copy.deepcopy(client_nodes[i].model)
            temp = {}
            for k, v in ann.named_parameters():
                temp[k] = v.data.clone()
            for k, v in x.named_parameters():
                local_steps = args.E * len(client_nodes[i].train_loader)
                client_nodes[i].control[k] = client_nodes[i].control[k] - central_node.control[k] + (v.data - temp[k]) / (local_steps * args.lr)
                client_nodes[i].delta_y[k] = temp[k] - v.data
                client_nodes[i].delta_control[k] = client_nodes[i].control[k] - client_nodes[i].control[k]
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            client_accs.append(sum(epoch_accs)/len(epoch_accs))
    elif args.client_method == 'ditto':
        client_losses = []
        client_accs = []
        for i in select_list:
            # peronalized training
            epoch_losses = []
            epoch_accs = []
            for epoch in range(args.E):
                loss, acc = client_ditto(args, client_nodes[i], central_node)
                epoch_losses.append(loss)
                epoch_accs.append(acc)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            client_accs.append(sum(epoch_accs)/len(epoch_accs))
            # global model training
            for epoch in range(args.E):
                _, _ = client_localTrain(args, client_nodes[i])
    elif args.client_method == 'moon':
        client_losses = []
        client_accs = []
        for i in select_list:
            epoch_losses = []
            epoch_accs = []
            for epoch in range(args.E):
                loss, acc = client_moon(args, client_nodes[i], central_node)
                epoch_losses.append(loss)
                epoch_accs.append(acc)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            client_accs.append(sum(epoch_accs)/len(epoch_accs))
    else:
        raise ValueError('Undefined client method...')

    return client_nodes, client_losses, client_accs


def client_localTrain(args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    num_data = 0
    correct = 0
    train_loader = node.train_loader  # iid
    if args.dataset == 'Digit':
        criterion = torch.nn.CrossEntropyLoss()
    if args.dataset == 'FeTS2022' or args.dataset == 'Fundus' or args.dataset == 'Prostate' or args.dataset == 'Meibomian_Gland' or args.dataset == 'Pancreas' or args.dataset == 'Polyp' or args.dataset == 'Pathology_COSAS2024' or args.dataset == 'FL_Ultrasound'  or args.dataset == 'FL_Breast_Ultrasound' or args.dataset == 'FL_Skin' or args.dataset == 'KiTS19' or args.dataset == 'MMS':
        if args.loss == 'bce':
            criterion = torch.nn.BCEWithLogitsLoss()
        elif args.loss == 'dice_bce':
            criterion = DiceBCELoss()
        else:
            criterion = FocalDiceloss()
        
    for idx, batch in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        
        data = batch['img']
        target = batch['mask']
        
        data, target = data.cuda(), target.cuda()
        logits, _  = node.model(data)
        loss_local =  criterion(logits, target)
        loss_local.backward()
        loss = loss + loss_local.item()
        node.optimizer.step()
        
        dice = dice_fn(logits, target)
        correct += dice.item()
        num_data += data.size(0)
        
    return loss/num_data, correct/num_data

def client_fedrdn(args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    num_data = 0
    correct = 0
    train_loader = node.train_loader  # iid
    if args.dataset == 'Digit':
        criterion = torch.nn.CrossEntropyLoss()
    if args.dataset == 'FeTS2022' or args.dataset == 'Fundus' or args.dataset == 'Prostate' or args.dataset == 'Meibomian_Gland' or args.dataset == 'Pancreas' or args.dataset == 'Polyp' or args.dataset == 'Pathology_COSAS2024' or args.dataset == 'FL_Ultrasound'  or args.dataset == 'FL_Breast_Ultrasound' or args.dataset == 'FL_Skin' or args.dataset == 'KiTS19' or args.dataset == 'MMS':
        if args.loss == 'bce':
            criterion = torch.nn.BCEWithLogitsLoss()
        elif args.loss == 'dice_bce':
            criterion = DiceBCELoss()
        else:
            criterion = FocalDiceloss()
        
    for idx, batch in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        
        data = batch['img']
        target = batch['mask']
        
        data, target = data.cuda(), target.cuda()
        
        data = node.FedRDNTransform_train(data)
        
        logits, _  = node.model(data)
        loss_local =  criterion(logits, target)
        loss_local.backward()
        loss = loss + loss_local.item()
        node.optimizer.step()
        
        dice = dice_fn(logits, target)
        correct += dice.item()
        num_data += data.size(0)
        
    return loss/num_data, correct/num_data

def client_moon(args, node, central_node, loss = 0.0):
    node.model.train()
    
    node.pre_model.eval()
    central_node.model.eval()
    cos=torch.nn.CosineSimilarity(dim=-1)
    
    loss = 0.0
    num_data = 0
    correct = 0
    train_loader = node.train_loader  # iid
    if args.dataset == 'Digit':
        criterion = torch.nn.CrossEntropyLoss()
    if args.dataset == 'FeTS2022' or args.dataset == 'Fundus' or args.dataset == 'Prostate' or args.dataset == 'Meibomian_Gland' or args.dataset == 'Pancreas' or args.dataset == 'Polyp' or args.dataset == 'Pathology_COSAS2024' or args.dataset == 'FL_Ultrasound' or args.dataset == 'FL_Breast_Ultrasound' or args.dataset == 'FL_Skin' or args.dataset == 'KiTS19' or args.dataset == 'MMS':
        if args.loss == 'bce':
            criterion = torch.nn.BCEWithLogitsLoss()
        elif args.loss == 'dice_bce':
            criterion = DiceBCELoss()
        else:
            criterion = FocalDiceloss()
    
    for idx, batch in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        
        data = batch['img']
        target = batch['mask']
        
        data, target = data.cuda(), target.cuda()
        outputs, fea_current  = node.model(data)
        
        with torch.no_grad():
            _, fea_global = central_node.model(data)
            fea_global = fea_global.view(data.size(0), -1)
            _, fea_pre = node.pre_model(data)
            fea_pre = fea_pre.view(data.size(0), -1)
        fea_current = fea_current.view(data.size(0), -1)
        pos_sim = cos(fea_current, fea_global).unsqueeze(-1) / 0.5
        neg_sim = cos(fea_current, fea_pre).unsqueeze(-1) / 0.5
        numerator = torch.exp(pos_sim)
        denominator = numerator + torch.exp(neg_sim)
        con_loss = -torch.log(numerator / denominator).mean()
        
        loss_local =  criterion(outputs, target) + args.mu * con_loss
        loss_local.backward()
        loss = loss + loss_local.item()
        node.optimizer.step()
        
        dice = dice_fn(outputs, target)
        correct += dice.item()
        num_data += data.size(0)
    
    central_node.model.train()
        
    return loss/num_data, correct/num_data


def client_fedprox(args, node, central_node, loss = 0.0):
    node.model.train()

    loss = 0.0
    num_data = 0
    correct = 0
    train_loader = node.train_loader  # iid
    
    if args.dataset == 'Digit':
        criterion = torch.nn.CrossEntropyLoss()
    if args.dataset == 'FeTS2022' or args.dataset == 'Fundus' or args.dataset == 'Prostate' or args.dataset == 'Meibomian_Gland' or args.dataset == 'Pancreas' or args.dataset == 'Polyp' or args.dataset == 'Pathology_COSAS2024' or args.dataset == 'FL_Ultrasound' or args.dataset == 'FL_Breast_Ultrasound' or args.dataset == 'FL_Skin' or args.dataset == 'KiTS19' or args.dataset == 'MMS':
        if args.loss == 'bce':
            criterion = torch.nn.BCEWithLogitsLoss()
        elif args.loss == 'dice_bce':
            criterion = DiceBCELoss()
        else:
            criterion = FocalDiceloss()
            
    for idx, batch in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        
        data = batch['img']
        target = batch['mask']
        
        data, target = data.cuda(), target.cuda()
        logits, _  = node.model(data)

        loss_local =  criterion(logits, target)
        
        #########################we implement FedProx Here###########################
        # referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819
        if idx>0:
            w_diff = torch.tensor(0.).cuda()
            for w, w_t in zip(central_node.model.parameters(), node.model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            loss_local += args.mu / 2. * w_diff
        #############################################################################
        
        loss_local.backward()
        loss = loss + loss_local.item()
        node.optimizer.step()
        
        dice = dice_fn(logits, target)
        correct += dice.item()
        num_data += data.size(0)
        
    return loss/num_data, correct/num_data


def client_ditto(args, node, central_node, loss = 0.0):
    node.p_model.train()

    loss = 0.0
    num_data = 0
    correct = 0
    train_loader = node.train_loader  # iid
    
    if args.dataset == 'Digit':
        criterion = torch.nn.CrossEntropyLoss()
    if args.dataset == 'FeTS2022' or args.dataset == 'Fundus' or args.dataset == 'Prostate' or args.dataset == 'Meibomian_Gland' or args.dataset == 'Pancreas' or args.dataset == 'Polyp' or args.dataset == 'Pathology_COSAS2024' or args.dataset == 'FL_Ultrasound' or args.dataset == 'FL_Breast_Ultrasound' or args.dataset == 'FL_Skin' or args.dataset == 'KiTS19' or args.dataset == 'MMS':
        if args.loss == 'bce':
            criterion = torch.nn.BCEWithLogitsLoss()
        elif args.loss == 'dice_bce':
            criterion = DiceBCELoss()
        else:
            criterion = FocalDiceloss()
            
    for idx, batch in enumerate(train_loader):
        # zero_grad
        node.p_optimizer.zero_grad()
        # train model
        
        data = batch['img']
        target = batch['mask']
        
        data, target = data.cuda(), target.cuda()
        logits, _  = node.p_model(data)

        loss_local =  criterion(logits, target)
        
        #########################we implement FedProx Here###########################
        # referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819
        if idx>0:
            w_diff = torch.tensor(0.).cuda()
            for w, w_t in zip(central_node.model.parameters(), node.p_model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            loss_local += args.mu / 2. * w_diff
        #############################################################################
        
        loss_local.backward()
        loss = loss + loss_local.item()
        node.p_optimizer.step()
        
        dice = dice_fn(logits, target)
        correct += dice.item()
        num_data += data.size(0)
        
    return loss/num_data, correct/num_data


def client_fedrod(args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    num_data = 0
    correct = 0
    train_loader = node.train_loader  # iid
    if args.dataset == 'Digit':
        criterion = torch.nn.CrossEntropyLoss()
    if args.dataset == 'FeTS2022' or args.dataset == 'Fundus' or args.dataset == 'Prostate' or args.dataset == 'Meibomian_Gland' or args.dataset == 'Pancreas' or args.dataset == 'Polyp' or args.dataset == 'Pathology_COSAS2024' or args.dataset == 'FL_Ultrasound' or args.dataset == 'FL_Breast_Ultrasound' or args.dataset == 'FL_Skin' or args.dataset == 'KiTS19' or args.dataset == 'MMS':
        if args.loss == 'bce':
            criterion = torch.nn.BCEWithLogitsLoss()
        elif args.loss == 'dice_bce':
            criterion = DiceBCELoss()
        else:
            criterion = FocalDiceloss()

    # initialize the optimizer of p_head
    p_head_optimizer = torch.optim.SGD(node.p_head.parameters(), lr=args.lr)

    for idx, batch in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        p_head_optimizer.zero_grad()

        # train model
        data = batch['img']
        target = batch['mask']
        
        data, target = data.cuda(), target.cuda()
        logit_g, feature = node.model(data)

        # balanced loss for base and g_head
        loss_local =  criterion(logit_g, target)
        loss_local.backward()
        loss = loss + loss_local.item()
        node.optimizer.step()

        # ce loss for p_head
        logit_p = node.p_head(feature.detach())
        logits = logit_g.detach() + logit_p
        #loss_p =  F.cross_entropy(logits, target.long())
        loss_p =  criterion(logits, target)
        loss_p.backward()
        p_head_optimizer.step()
        
        
        dice = dice_fn(logits, target)
        correct += dice.item()
        num_data += data.size(0)

    return loss/num_data, correct/num_data

def client_feddyn(global_model_vector, args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    num_data = 0
    correct = 0
    train_loader = node.train_loader  # iid
    if args.dataset == 'Digit':
        criterion = torch.nn.CrossEntropyLoss()
    if args.dataset == 'FeTS2022' or args.dataset == 'Fundus' or args.dataset == 'Prostate' or args.dataset == 'Meibomian_Gland' or args.dataset == 'Pancreas' or args.dataset == 'Polyp' or args.dataset == 'Pathology_COSAS2024' or args.dataset == 'FL_Ultrasound' or args.dataset == 'FL_Breast_Ultrasound' or args.dataset == 'FL_Skin' or args.dataset == 'KiTS19' or args.dataset == 'MMS':
        if args.loss == 'bce':
            criterion = torch.nn.BCEWithLogitsLoss()
        elif args.loss == 'dice_bce':
            criterion = DiceBCELoss()
        else:
            criterion = FocalDiceloss()
            
    for idx, batch in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        data = batch['img']
        target = batch['mask']
        
        data, target = data.cuda(), target.cuda()
        logits, _  = node.model(data)

        loss_local =  criterion(logits, target)
        loss = loss + loss_local.item()

        # feddyn update
        v1 = model_parameter_vector(args, node.model)
        loss_local += args.mu/2 * torch.norm(v1 - global_model_vector, 2)
        loss_local -= torch.dot(v1, node.old_grad)

        loss_local.backward()
        node.optimizer.step()

        dice = dice_fn(logits, target)
        correct += dice.item()
        num_data += data.size(0)

    return loss/num_data, correct/num_data


#########################Scaffold##############################
def client_scaffold(args, node, central_node, loss = 0.0):
    node.model.train()

    loss = 0.0
    num_data = 0
    correct = 0
    train_loader = node.train_loader  # iid
    if args.dataset == 'Digit':
        criterion = torch.nn.CrossEntropyLoss()
    if args.dataset == 'FeTS2022' or args.dataset == 'Fundus' or args.dataset == 'Prostate' or args.dataset == 'Meibomian_Gland' or args.dataset == 'Pancreas' or args.dataset == 'Polyp' or args.dataset == 'Pathology_COSAS2024' or args.dataset == 'FL_Ultrasound' or args.dataset == 'FL_Breast_Ultrasound' or args.dataset == 'FL_Skin' or args.dataset == 'KiTS19' or args.dataset == 'MMS':
        if args.loss == 'bce':
            criterion = torch.nn.BCEWithLogitsLoss()
        elif args.loss == 'dice_bce':
            criterion = DiceBCELoss()
        else:
            criterion = FocalDiceloss()
            
    for idx, batch in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        data = batch['img']
        target = batch['mask']
        
        data, target = data.cuda(), target.cuda()
        logits, _  = node.model(data)

        loss_local =  criterion(logits, target)
        loss = loss + loss_local.item()

        loss_local.backward()
        node.optimizer.step(central_node.control, node.control)

        dice = dice_fn(logits, target)
        correct += dice.item()
        num_data += data.size(0)

    return loss/num_data, correct/num_data


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - p) ** self.gamma
        w_neg = p ** self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)

        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        intersection = torch.sum(p * mask)
        union = torch.sum(p) + torch.sum(mask)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_loss


class MaskIoULoss(nn.Module):

    def __init__(self, ):
        super(MaskIoULoss, self).__init__()

    def forward(self, pred_mask, ground_truth_mask, pred_iou):
        """
        pred_mask: [B, 1, H, W]
        ground_truth_mask: [B, 1, H, W]
        pred_iou: [B, 1]
        """
        assert pred_mask.shape == ground_truth_mask.shape, "pred_mask and ground_truth_mask should have the same shape."

        p = torch.sigmoid(pred_mask)
        intersection = torch.sum(p * ground_truth_mask)
        union = torch.sum(p) + torch.sum(ground_truth_mask) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_loss = torch.mean((iou - pred_iou) ** 2)
        return iou_loss


class FocalDiceloss_IoULoss(nn.Module):
    
    def __init__(self, weight=20.0, iou_scale=1.0):
        super(FocalDiceloss_IoULoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.maskiou_loss = MaskIoULoss()

    def forward(self, pred, mask, pred_iou):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss =self.dice_loss(pred, mask)
        loss1 = self.weight * focal_loss + dice_loss
        loss2 = self.maskiou_loss(pred, mask, pred_iou)
        loss = loss1 + loss2 * self.iou_scale
        return loss, loss1, loss2

class FocalDiceloss(nn.Module):
    
    def __init__(self, weight=20.0):
        super(FocalDiceloss, self).__init__()
        self.weight = weight
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss =self.dice_loss(pred, mask)
        loss = self.weight * focal_loss + dice_loss
        return loss#, focal_loss, dice_loss
    
    