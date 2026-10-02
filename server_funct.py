import numpy as np
import torch
import torch.nn.functional as F
import os
import random
from torch.backends import cudnn
from random import sample
import math
import torch.optim as optim
import torch.nn as nn
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import init_model, freeze_layers, set_params, dice_fn
from torch.autograd import Variable
from loss import DiceBCELoss, FocalDiceloss

def get_server_gradient_flat(args, central_node, g_client_nodes):
    """
    计算服务器端（验证集/测试集）的梯度方向，并返回展平的一维向量。
    注意：这里计算的是 Loss 的梯度（Gradient），也就是上升方向。
    实际更新时需要取反，或者是直接计算 Descent Direction。
    """
    central_node.model.train()
    
    # 定义 Loss (与 client_funct_seg 保持一致)
    if args.dataset == 'Digit':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.dataset in ['FeTS2022', 'Fundus', 'Prostate', 'Meibomian_Gland', 'Pancreas', 'Polyp', 'Pathology_COSAS2024', 'FL_Ultrasound', 'FL_Breast_Ultrasound', 'FL_Skin', 'KiTS19', 'MMS']:
        if args.loss == 'bce':
            criterion = torch.nn.BCEWithLogitsLoss()
        elif args.loss == 'dice_bce':
            criterion = DiceBCELoss()
        else:
            criterion = FocalDiceloss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # 初始化梯度累加器
    total_grads = {}
    
    # 初始化
    for name, param in central_node.model.named_parameters():
        if param.requires_grad:
            total_grads[name] = torch.zeros_like(param.data)

    count = 0
    limit_batches = 10
    
    for node_idx, node in g_client_nodes.items():
        loader = node.val_loader if node.val_loader else node.test_loader
        if loader is None:
            continue

        for idx, batch in enumerate(loader):
            if idx >= limit_batches:
                break
                
            data = batch['img']
            target = batch['mask']
            data, target = data.cuda(), target.cuda()
            
            central_node.model.zero_grad()
            logits, _ = central_node.model(data)
            loss = criterion(logits, target)
            loss.backward()
            
            for name, param in central_node.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    total_grads[name] += param.grad.data
            
            count += 1
            
    if count == 0:
        return None

    # 平均并展平
    vec = []
    for name, param in central_node.model.named_parameters():
        if param.requires_grad:
            avg_grad = total_grads[name] / count
            vec.append(avg_grad.view(-1))
            
    return torch.cat(vec)


def get_server_gradient(args, central_node, g_client_nodes):
    """
    计算服务器端（验证集/测试集）的梯度方向。
    注意：这里计算的是 Loss 的梯度（Gradient），也就是上升方向。
    实际更新时需要取反，或者是直接计算 Descent Direction。
    """
    central_node.model.train()
    
    # 定义 Loss (与 client_funct_seg 保持一致)
    if args.dataset == 'Digit':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.dataset in ['FeTS2022', 'Fundus', 'Prostate', 'Meibomian_Gland', 'Pancreas', 'Polyp', 'Pathology_COSAS2024', 'FL_Ultrasound', 'FL_Breast_Ultrasound', 'FL_Skin', 'KiTS19', 'MMS']:
        if args.loss == 'bce':
            criterion = torch.nn.BCEWithLogitsLoss()
        elif args.loss == 'dice_bce':
            criterion = DiceBCELoss()
        else:
            criterion = FocalDiceloss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    server_grad_dict = {}
    
    # 初始化梯度
    for name, param in central_node.model.named_parameters():
        if param.requires_grad:
            server_grad_dict[name] = torch.zeros_like(param.data)

    count = 0
    # 为了获得稳定的梯度方向，建议多计算几个 Batch，这里设为 10
    limit_batches = 10 
    
    # 遍历所有服务器端节点（g_client_names）
    for node_idx, node in g_client_nodes.items():
        loader = node.val_loader if node.val_loader else node.test_loader
        if loader is None:
            continue

        for idx, batch in enumerate(loader):
            if idx >= limit_batches:
                break
                
            data = batch['img']
            target = batch['mask']
            data, target = data.cuda(), target.cuda()
            
            # Forward
            central_node.model.zero_grad()
            logits, _ = central_node.model(data)
            loss = criterion(logits, target)
            
            # Backward
            loss.backward()
            
            # 累积梯度
            for name, param in central_node.model.named_parameters():
                if param.grad is not None:
                    server_grad_dict[name] += param.grad.data
            
            count += 1
            
    # 平均梯度
    if count > 0:
        for name in server_grad_dict:
            server_grad_dict[name] /= count
            
    return server_grad_dict

def get_trainable_params_flat(model):
    """
    只获取可训练参数（requires_grad=True）并展平。
    排除 running_mean, running_var, num_batches_tracked 等不可训练参数。
    """
    vec = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            vec.append(param.view(-1))
    return torch.cat(vec)

def flatten_params(param_dict):
    """将参数字典展平为一维向量"""
    vec = []
    for key in sorted(param_dict.keys()):
        vec.append(param_dict[key].view(-1))
    return torch.cat(vec)

def unflatten_params(flat_vec, template_dict):
    """将一维向量恢复为参数字典"""
    param_dict = {}
    pointer = 0
    for key in sorted(template_dict.keys()):
        param = template_dict[key]
        num_param = param.numel()
        param_dict[key] = flat_vec[pointer:pointer + num_param].view_as(param)
        pointer += num_param
    return param_dict


global_T_weight = None

def Server_update(args, central_node, client_nodes, g_client_nodes, select_list, size_agg_weights, epoch = 0):
    '''
    Original server update functions for baselines
    '''
    # receive the local models from clients
    #agg_weights, client_params, local_protos_list = receive_client_models(args, client_nodes, select_list, size_weights)

    agg_weights = [size_agg_weights[idx] for idx in select_list]
    agg_weights = [w/sum(agg_weights) for w in agg_weights]
    
    # update the global model
    if args.server_method == 'fedavg':
        #avg_global_param = fedavg(client_params, agg_weights)
        #central_node.model.load_state_dict(avg_global_param)
        for key in central_node.model.state_dict().keys():
            # num_batches_tracked is a non trainable LongTensor and
            # num_batches_tracked are the same for all clients for the given datasets
            if 'num_batches_tracked' in key:
                central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key])
            else:
                temp = torch.zeros_like(central_node.model.state_dict()[key])
                for i, client_idx in enumerate(select_list):
                    temp += agg_weights[i] * client_nodes[client_idx].model.state_dict()[key]
                central_node.model.state_dict()[key].data.copy_(temp)
                for i, client_idx in enumerate(select_list):
                    client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])

    elif args.server_method == 'fednova':
        fed_avg_freqs = agg_weights
        client_step = [(len(client_nodes[i].train_loader)) for i in select_list]
        tao_eff = 0.0
        for j in range(len(fed_avg_freqs)):
            tao_eff += fed_avg_freqs[j] * client_step[j]
        correct_term = 0.0
        for j in range(len(fed_avg_freqs)):
            correct_term += fed_avg_freqs[j] / client_step[j] * tao_eff
        
        # Model aggregation
        for key in central_node.model.state_dict().keys():
            # num_batches_tracked is a non trainable LongTensor and
            # num_batches_tracked are the same for all clients for the given datasets
            if 'num_batches_tracked' in key:
                central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key])
            else:
                #temp = torch.zeros_like(central_node.model.state_dict()[key])
                temp = (1.0 - correct_term)*central_node.model.state_dict()[key]
                for i, client_idx in enumerate(select_list):
                    temp += fed_avg_freqs[client_idx] / client_step[client_idx] * tao_eff * client_nodes[client_idx].model.state_dict()[key]
                central_node.model.state_dict()[key].data.copy_(temp)
                for i, client_idx in enumerate(select_list):
                    client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])
    elif args.server_method == 'fedper':
        if args.local_model != 'UNet':
            for key in central_node.model.state_dict().keys():
                if 'seg1' not in key:
                    temp = torch.zeros_like(central_node.model.state_dict()[key], dtype=torch.float32)
                    for i, client_idx in enumerate(select_list):
                        temp += agg_weights[i] * client_nodes[client_idx].model.state_dict()[key]
                    central_node.model.state_dict()[key].data.copy_(temp)
                    for i, client_idx in enumerate(select_list):
                        client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])
        else:
            for key in central_node.model.state_dict().keys():
                if 'final' not in key:
                    temp = torch.zeros_like(central_node.model.state_dict()[key], dtype=torch.float32)
                    for i, client_idx in enumerate(select_list):
                        temp += agg_weights[i] * client_nodes[client_idx].model.state_dict()[key]
                    central_node.model.state_dict()[key].data.copy_(temp)
                    for i, client_idx in enumerate(select_list):
                        client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])

    elif args.server_method == 'feddyn':
        # update server's state
        uploaded_models = []
        for i in select_list:
            uploaded_models.append(copy.deepcopy(client_nodes[i].model))
    
        model_delta = copy.deepcopy(uploaded_models[0])
        for param in model_delta.parameters():
            param.data = torch.zeros_like(param.data)
    
        for idx, client_model in enumerate(uploaded_models):
            for server_param, client_param, delta_param in zip(central_node.model.parameters(), client_model.parameters(), model_delta.parameters()):
                delta_param.data += (client_param - server_param) * agg_weights[idx]
    
        for state_param, delta_param in zip(central_node.server_state.parameters(), model_delta.parameters()):
            state_param.data -= args.mu * delta_param
    
        # aggregation
        central_node.model = copy.deepcopy(uploaded_models[0])
        for param in central_node.model.parameters():
            param.data = torch.zeros_like(param.data)
            
        for idx, client_model in enumerate(uploaded_models):
            for server_param, client_param in zip(central_node.model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * agg_weights[idx]
    
        for server_param, state_param in zip(central_node.model.parameters(), central_node.server_state.parameters()):
            server_param.data -= (1/args.mu) * state_param
            
        for key in central_node.model.state_dict().keys():
            if 'num_batches_tracked' in key:
                central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key])
            else:
                for i, client_idx in enumerate(select_list):
                    client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])

    elif args.server_method == 'fedawa':
        
        global global_T_weight
        
        if epoch==0:
            global_T_weight = agg_weights

        def get_flat_weights(node):
            
            weights_flat = []
        
            for key in node.model.state_dict().keys():
                if ('num_batches_tracked' not in key) and ('running_mean' not in key) and ('running_var' not in key):
                    p = node.model.state_dict()[key]
                    weights_flat.append(p.clone().detach().reshape(-1))
            with torch.no_grad():
                weights_flat = torch.cat(weights_flat, 0)
            
            return weights_flat


        flat_w_list = []
        for _, client_idx in enumerate(select_list):
            c_weights_flat = get_flat_weights(client_nodes[client_idx])
            flat_w_list.append(c_weights_flat)
        local_param_list = torch.stack(flat_w_list)#C, N
        
        global_weights_flat = get_flat_weights(central_node)
        
        
        T_weights = torch.tensor(global_T_weight, dtype=torch.float32).cuda()
        T_weights = Variable(T_weights, requires_grad=True)
        
        if args.server_optimizer=='sgd':##########
            Attoptimizer = torch.optim.SGD([T_weights], lr=0.01, momentum=0.9, weight_decay=5e-4)
        elif args.server_optimizer=='adam':
            Attoptimizer = optim.Adam([T_weights], lr=0.001, betas=(0.5, 0.999))
        
        #print("T_weights_before update:",torch.nn.functional.softmax(T_weights, dim=0).data.cpu())
        
        
        def _cost_matrix(x, y, dis, p=2):
            d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)
        
            x_col = x.unsqueeze(-2)
            y_lin = y.unsqueeze(-3)
            if dis == 'cos':
                # print('cos_dis')
                C = 1-d_cosine(x_col, y_lin)
            elif dis == 'euc':
                # print('euc_dis')
                C= torch.mean((torch.abs(x_col - y_lin)) ** p, -1)
            return C
        
        for i in range(args.server_epochs):###########
            #print("server weight update:",i)
            probability_train = torch.nn.functional.softmax(T_weights, dim=0)
            
            #########
            C = _cost_matrix(global_weights_flat.detach().unsqueeze(0), local_param_list.detach(), args.reg_distance)
            
            reg_loss = torch.sum(probability_train* C, dim=(-2, -1))
            #print("reg_loss:",reg_loss.data.cpu())
            
            client_grad=local_param_list-global_weights_flat
            
            column_sum=torch.matmul(probability_train.unsqueeze(0),client_grad) #weighted sum
            l2_distance = torch.norm(client_grad.unsqueeze(0) - column_sum.unsqueeze(1), p=2, dim=2)
            
            #print("L2_distance:",l2_distance.data.cpu())
            sim_loss=(torch.sum(probability_train*l2_distance, dim=(-2, -1)))
    
            #print("Sim_loss:",sim_loss.data.cpu())
            
            Loss=sim_loss+reg_loss
            Attoptimizer.zero_grad()
            Loss.backward()
            Attoptimizer.step()
            #print("step "+str(i)+" Loss:"+str(Loss))
            
        global_T_weight = T_weights.data
        #print("T_weights_after update:",global_T_weight)
        #print("probability_train_after update:",probability_train)

        for key in central_node.model.state_dict().keys():
            # num_batches_tracked is a non trainable LongTensor and
            # num_batches_tracked are the same for all clients for the given datasets
            if 'num_batches_tracked' in key:
                central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key])
            else:
                temp = torch.zeros_like(central_node.model.state_dict()[key])
                for i, client_idx in enumerate(select_list):
                    temp += probability_train[i] * client_nodes[client_idx].model.state_dict()[key]
                central_node.model.state_dict()[key].data.copy_(temp/sum(probability_train))
                for i, client_idx in enumerate(select_list):
                    client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])


    elif args.server_method == 'fedlws':
        #https://github.com/ChanglongShi/FedLWS/blob/main/server_funct.py
        client_params = []
        for idx in select_list:
            client_params.append(copy.deepcopy(client_nodes[idx].model.state_dict()))
        
        param=central_node.model.state_dict()
        global_params = copy.deepcopy(param)
        fedavg_global_params = copy.deepcopy(client_params[0])
        
        for name_param in client_params[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(client_params,agg_weights):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param)# / sum(list_nums_local_data)
            fedavg_global_params[name_param] = value_global_param
        
        cur_w=[]
        last_w=[]
        l=torch.tensor([]).cuda()
        l_last=torch.tensor([]).cuda()
        for name_param in client_params[0]:
            # print(name_param)
            l=torch.cat((l,fedavg_global_params[name_param].reshape(-1)))
            l_last=torch.cat((l_last,global_params[name_param].reshape(-1)))
            # a=layer_cossim[i][name_param]
            if "bias" in name_param:
                cur_w.append(l)
                last_w.append(l_last)
                # print(name_param)
                # print(l.shape)
                # a=torch.tensor(0).cuda().float()
                l=torch.tensor([]).cuda()
                l_last=torch.tensor([]).cuda()
        clients_w=[]
        for i in range(len(client_params)):
            client_w=[]
            l_client=torch.tensor([]).cuda()
            for name_param in client_params[0]:
                l_client=torch.cat((l_client,client_params[i][name_param].reshape(-1)))
                # a=layer_cossim[i][name_param]
                if "bias" in name_param:
                    client_w.append(l_client)
                    # a=torch.tensor(0).cuda().float()
                    l_client=torch.tensor([]).cuda()
            clients_w.append(client_w)
            
            
        layer_gammas=[]
        ######### layer_tau ##############
        taus=[]
        for i in range(len(last_w)):
            # print(cur_w[i].shape)
            grad=torch.norm(cur_w[i]-last_w[i],p=2)
            layer_grad=[]
            for k in range(len(client_params)):
                layer_grad.append(clients_w[k][i]-last_w[i])
            global_grad=torch.mean(torch.stack(layer_grad),dim=0)
            l2_norms = [torch.norm(tensor - global_grad, p=2) for tensor in layer_grad]
            l2_norm_average = sum(l2_norms) / len(l2_norms)
            tau=args.beta*(l2_norm_average)
            tau = torch.clamp(tau, min=args.min_tau, max=args.max_tau)
            taus.append(tau)
            gamma=torch.norm(last_w[i],p=2)/(torch.norm(last_w[i],p=2)+tau*grad)
            layer_gammas.append(gamma)
        #print("taus:",taus)
        #print("layer_gammas:",layer_gammas)
        cur_layer=0
        for name_param in client_params[0]:
            # if name_param[-6:]=="weight":
            #     k+=1
            fedavg_global_params[name_param] = fedavg_global_params[name_param]*layer_gammas[cur_layer]#+fedavg_global_params[name_param]*(1-d0)
            if "bias" in name_param:
                cur_layer+=1

        ##added by meilu
        central_node.model.load_state_dict(fedavg_global_params)
        for key in central_node.model.state_dict().keys():
            if 'num_batches_tracked' in key:
                central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key])
            else:
                for i, client_idx in enumerate(select_list):
                    client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])


    elif args.server_method == 'scaffold':
        
        fedavg_global_params = {}
        for k, v in client_nodes[0].model.state_dict().items():
            list_values_param = []
            ck_list_values_param = []#########
            for j, num_local_data in zip(select_list, agg_weights):
                if k in client_nodes[0].delta_y.keys():
                    list_values_param.append(client_nodes[j].delta_y[k]/len(select_list))
                    ck_list_values_param.append(client_nodes[j].delta_control[k]/len(select_list))#########
                else:
                    list_values_param.append(client_nodes[j].model.state_dict()[k]/len(select_list))
            value_global_param = sum(list_values_param)# / sum(agg_weights)
            if k in client_nodes[0].delta_y.keys():
                fedavg_global_params[k] = value_global_param + central_node.model.state_dict()[k]
                ck_list_values_param = sum(ck_list_values_param)# / sum(agg_weights)#########
                central_node.control[k].data += ck_list_values_param * (len(select_list) / args.node_num)#########
            else:
                fedavg_global_params[k] = value_global_param
        central_node.model.load_state_dict(fedavg_global_params)

        for key in central_node.model.state_dict().keys():
            if 'num_batches_tracked' in key:
                central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key])
            else:
                for i, client_idx in enumerate(select_list):
                    client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])

    elif args.server_method == 'fedbn':
        if args.local_model != 'UNet':
            for key in central_node.model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(central_node.model.state_dict()[key], dtype=torch.float32)
                    for i, client_idx in enumerate(select_list):
                        temp += agg_weights[i] * client_nodes[client_idx].model.state_dict()[key]
                    central_node.model.state_dict()[key].data.copy_(temp)
                    for i, client_idx in enumerate(select_list):
                        client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])
        else:
            for key in central_node.model.state_dict().keys():
                if 'enc1.1' not in key and 'enc2.1' not in key and 'enc3.1' not in key and 'enc4.1' not in key and 'dec4.1' not in key and 'dec3.1' not in key and 'dec2.1' not in key: # 'enc1.1'
                    temp = torch.zeros_like(central_node.model.state_dict()[key], dtype=torch.float32)
                    for i, client_idx in enumerate(select_list):
                        temp += agg_weights[i] * client_nodes[client_idx].model.state_dict()[key]
                    central_node.model.state_dict()[key].data.copy_(temp)
                    for i, client_idx in enumerate(select_list):
                        client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])

    elif args.server_method == 'siobn':
        if args.local_model != 'UNet':
            for key in central_node.model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(central_node.model.state_dict()[key], dtype=torch.float32)
                    for i, client_idx in enumerate(select_list):
                        temp += agg_weights[i] * client_nodes[client_idx].model.state_dict()[key]
                    central_node.model.state_dict()[key].data.copy_(temp)
                    for i, client_idx in enumerate(select_list):
                        client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])
                else:
                    if 'num_batches_tracked' in key:
                        central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key])
                    if 'weight' in key or 'bias' in key: # ignore  running_mean, running_var
                        temp = torch.zeros_like(central_node.model.state_dict()[key], dtype=torch.float32)
                        for i, client_idx in enumerate(select_list):
                            temp += agg_weights[i] * client_nodes[client_idx].model.state_dict()[key]
                        central_node.model.state_dict()[key].data.copy_(temp)
                        for i, client_idx in enumerate(select_list):
                            client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])
        else:
            for key in central_node.model.state_dict().keys():
                if 'enc1.1' not in key and 'enc2.1' not in key and 'enc3.1' not in key and 'enc4.1' not in key and 'dec4.1' not in key and 'dec3.1' not in key and 'dec2.1' not in key: # 'enc1.1'
                    temp = torch.zeros_like(central_node.model.state_dict()[key], dtype=torch.float32)
                    for i, client_idx in enumerate(select_list):
                        temp += agg_weights[i] * client_nodes[client_idx].model.state_dict()[key]
                    central_node.model.state_dict()[key].data.copy_(temp)
                    for i, client_idx in enumerate(select_list):
                        client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])
                else:
                    if 'num_batches_tracked' in key:
                        central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key])
                    if 'weight' in key or 'bias' in key: # ignore  running_mean, running_var
                        temp = torch.zeros_like(central_node.model.state_dict()[key], dtype=torch.float32)
                        for i, client_idx in enumerate(select_list):
                            temp += agg_weights[i] * client_nodes[client_idx].model.state_dict()[key]
                        central_node.model.state_dict()[key].data.copy_(temp)
                        for i, client_idx in enumerate(select_list):
                            client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])

    elif args.server_method == 'fedgm':
         # 1. 获取服务端梯度 (直接获取展平后的 trainable 梯度) 
         server_grad_flat = get_server_gradient_flat(args, central_node, g_client_nodes) 
         
         if server_grad_flat is None: 
              print("Warning: Server gradient is empty. Fallback to FedAvg weights.") 
              final_weights = agg_weights 
         else: 
             server_grad_flat = server_grad_flat.cuda() 
             
             # 2. 准备客户端的"伪梯度" 
             flat_client_diffs = [] 
             
             # 获取全局模型的可训练参数展平向量 
             global_params_flat = get_trainable_params_flat(central_node.model) 
             
             for i, client_idx in enumerate(select_list): 
                 # 获取客户端模型的可训练参数展平向量 
                 client_params_flat = get_trainable_params_flat(client_nodes[client_idx].model) 
                 
                 # 计算差异 (global - client) 
                 # 确保都在同一个设备上 
                 diff = global_params_flat.cuda() - client_params_flat.cuda() 
                 flat_client_diffs.append(diff) 
             
             # Stack [N_clients, N_params] 
             client_diffs_matrix = torch.stack(flat_client_diffs) 
             
             # 验证形状是否一致 
             if server_grad_flat.shape[0] != client_diffs_matrix.shape[1]: 
                 raise RuntimeError(f"Shape mismatch: Server grad {server_grad_flat.shape} vs Client diff {client_diffs_matrix.shape}") 
 
             # 3. 初始化可学习的权重参数 
             # 使用 logit 初始化，经过 softmax 后初始为均匀分布或基于样本量的分布 
             # 这里我们基于样本量权重进行初始化 
             initial_logits = torch.log(torch.tensor(agg_weights) + 1e-9).cuda() 
             learnable_weights_logits = Variable(initial_logits, requires_grad=True) 
             
             # 定义优化器 
             optimizer_gm = torch.optim.Adam([learnable_weights_logits], lr=args.gm_lr) # 学习率可根据需要调整 
             
             # 4. 优化循环 
             central_node.model.train() # 保持训练模式 
             cosine_loss_func = torch.nn.CosineSimilarity(dim=0) 
             
             print(f"FedGM: Optimizing weights for {args.server_epochs} epochs...") 
             
             for epoch_gm in range(args.server_epochs): 
                 optimizer_gm.zero_grad() 
                 
                 # 计算当前的归一化权重 
                 softmax_weights = torch.nn.functional.softmax(learnable_weights_logits, dim=0) 
                 
                 # 计算聚合梯度： weighted_sum(client_diffs) 
                 # [N_clients] * [N_clients, N_params] -> broadcating -> sum -> [N_params] 
                 # 为了高效，使用矩阵乘法: [1, N_clients] @ [N_clients, N_params] -> [1, N_params] 
                 weighted_client_grad = torch.matmul(softmax_weights.unsqueeze(0), client_diffs_matrix).squeeze(0) 
                 
                 # 计算 Loss: 1 - CosineSimilarity(server_grad, aggregated_client_grad) 
                 # 我们希望相似度越高越好，即 Loss 越小越好 
                 cos_sim = cosine_loss_func(server_grad_flat, weighted_client_grad) 
                 loss = 1.0 - cos_sim 
                 
                 loss.backward() 
                 optimizer_gm.step() 
                 
                 if epoch_gm % 10 == 0: 
                     print(f"  Epoch {epoch_gm}: Loss {loss.item():.4f}, CosSim {cos_sim.item():.4f}") 
 
             # 获取最终优化后的权重 
             final_weights = torch.nn.functional.softmax(learnable_weights_logits, dim=0).detach().cpu().numpy() 
             print(f"FedGM: Final weights: {final_weights}") 
 
         # 5. 使用优化后的权重更新全局模型 
         for key in central_node.model.state_dict().keys(): 
             if 'num_batches_tracked' in key: 
                 central_node.model.state_dict()[key].data.copy_(client_nodes[0].model.state_dict()[key]) 
             else: 
                 temp = torch.zeros_like(central_node.model.state_dict()[key]) 
                 for i, client_idx in enumerate(select_list): 
                     temp += final_weights[i] * client_nodes[client_idx].model.state_dict()[key] 
                 central_node.model.state_dict()[key].data.copy_(temp) # 注意：这里不需要再除以sum，因为softmax后和为1 
                 
                 # 同步回客户端 
                 for i, client_idx in enumerate(select_list): 
                     client_nodes[client_idx].model.state_dict()[key].data.copy_(central_node.model.state_dict()[key])

    elif args.server_method == 'singleset':
        pass
    else:
        raise ValueError('Undefined server method...')
        
    return central_node, client_nodes


def receive_client_models(args, client_nodes, select_list, size_weights):
    client_params = []
    local_protos_list = {}

    for idx in select_list:
        client_params.append(copy.deepcopy(client_nodes[idx].model.state_dict()))

    agg_weights = [size_weights[idx] for idx in select_list]
    agg_weights = [w/sum(agg_weights) for w in agg_weights]

    return agg_weights, client_params, local_protos_list