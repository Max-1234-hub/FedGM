
import time
import torch
import numpy as np
import os
import copy
import gc
import pprint
import argparse
import warnings
from datasets import Data
from nodes import Node
from server_funct import Server_update
from client_funct_seg import Client_update
from utils import setup_seed, set_server_method, lr_scheduler, validate_seg, FedRDNTransform
from nodes import Seed_Averager_seg
import matplotlib.pyplot as plt

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# 如果使用了原子操作
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
torch.use_deterministic_algorithms(True)

warnings.filterwarnings('ignore')
np.set_printoptions(precision=7, suppress=True)


_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_size', type=float, default=1.0,
                        help="data_size")

    parser.add_argument('--save_path', type=str, default='./save/',
                        help="save_path")

    # Data
    parser.add_argument('--batchsize', type=int, default=8, 
                        help="batchsize")
    parser.add_argument('--num_classes', type=int, default=1, 
                        help="num_classes")
    
    # System
    parser.add_argument('--device', type=str, default='0',
                        help="CUDA_VISIBLE_DEVICES index (CUDA required)")
    parser.add_argument('--T', type=int, default=200, 
                        help="Number of communication rounds")
    parser.add_argument('--E', type=int, default=2, 
                        help="Number of local epochs: E")
    parser.add_argument('--dataset', type=str, default='Fundus',
                        help="Type of dataset") 
    parser.add_argument('--data_path', type=str, default='./',
                        help="data_path") 
    parser.add_argument('--local_model', type=str, default='UNet2D',
                        help='model architecture (UNet2D in this release)')

    # Server function
    parser.add_argument('--gm_lr', type=float, default=0.001,
                        help="FedGM aggregation-weight optimization learning rate")


    # Client function
    parser.add_argument('--optimizer', type=str, default='adam',
                        help="optimizer: {sgd, adam}")
    parser.add_argument('--lr', type=float, default=0.001,  
                        help='learning rate')
    parser.add_argument('--local_wd_rate', type=float, default=5e-4,
                        help='clients local wd rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')

    parser.add_argument('--method', type=str, default='FedGM',
                        choices=['FedGM', 'FedAvg', 'FedBN', 'FedProx', 'SioBN', 'SingleSet',
                                 'FedRoD', 'FedDYN', 'Scaffold', 'Ditto', 'MOON', 'FedNova',
                                 'FedPer', 'PN', 'FedRDN', 'FedLWS', 'FedAWA'],
                        help='federated method')
 
    parser.add_argument('--mu', type=float, default=0.01,
                    help="FedProx mu")

    parser.add_argument('--loss', type=str, default='dice_bce',
                    help="segmentation loss: bce, dice_bce, or focal dice")
    
    ####personalization step
    parser.add_argument('--lr_per', type=float, default=0.0001,
                    help="lr for personalization")

    #FedLWS
    parser.add_argument('--beta', type=float, default=0.03, 
                        help="beta")
    parser.add_argument('--min_tau', type=float, default=0.01,
                        help="min of tau")
    parser.add_argument('--max_tau', type=float, default=0.2,
                        help="max of tau")

    ### for FedAWA
    parser.add_argument('--server_optimizer', type=str, default='adam',
                        help="server_optimizer")
    parser.add_argument('--server_epochs', type=int, default=1,
                        help="optimizer epochs on server")
    parser.add_argument('--reg_distance', type=str, default='cos',
                        help="cos or euc")
    
    parser.add_argument('--val_percent', type=float, default=0.3,
                        help="val_percent")
    
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                        help='random seeds (default: 0 1 2)')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    if not torch.cuda.is_available():
        parser.error('A CUDA GPU is required by this implementation')
    if not 0 < args.data_size <= 1:
        parser.error('--data_size must be in (0, 1]')
    if args.T < 1 or args.E < 1 or args.batchsize < 1 or args.server_epochs < 1:
        parser.error('--T, --E, --batchsize, and --server_epochs must be positive')
    if args.num_classes != 1:
        parser.error('This release supports binary segmentation (--num_classes 1)')
    

    if args.dataset == 'Fundus': 
        random_seeds = args.seeds #
        #args.client_names = ['ARIA', 'ChaseDB', 'DR-Hagis', 'DRIVE', 'HRF', 'IOSTAR','LES-AV','ORVS']
        args.client_names = ['ChaseDB', 'DR-Hagis', 'DRIVE', 'HRF','LES-AV','ORVS']
        args.g_client_names = ['IOSTAR']
        args.node_num = len(args.client_names)
        args.select_ratio = 1.0
        args.stepsize = max(1, args.T//4)
        
    if args.dataset == 'Prostate': 
        random_seeds = args.seeds #
        #BIDMC  BMC  HK  I2CVB  RUNMC  UCL
        args.client_names = ['BIDMC', 'BMC', 'HK', 'I2CVB','RUNMC','UCL']
        args.g_client_names = ['MSD']
        args.node_num = len(args.client_names)
        args.select_ratio = 1.0
        args.num_classes = 1
        args.stepsize = max(1, args.T//4)
        
    if args.dataset == 'Polyp': 
        random_seeds = args.seeds #
        args.client_names = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'EndoTect', 'ETISLaribPolypDB'] # , 
        args.g_client_names = ['Kvasir-SEG']
        args.node_num = len(args.client_names)
        args.select_ratio = 1.0
        args.stepsize = max(1, args.T//4)
        
    if args.dataset == 'FL_Breast_Ultrasound': 
        random_seeds = args.seeds #
        args.client_names = ['BUID', 'BUSI','BUS_UC', 'BUS_UCLM']
        args.g_client_names = ['BUS']
        args.node_num = len(args.client_names)
        args.select_ratio = 1.0
        args.stepsize = max(1, args.T//4)
        
        
    if args.dataset not in {'Fundus', 'Polyp', 'Prostate', 'FL_Breast_Ultrasound'}:
        parser.error('Unsupported dataset for this release')
    if args.local_model != 'UNet2D':
        parser.error('This release supports --local_model UNet2D')

    if args.method == 'FedProx':
        args.mu = 0.01
        
    if args.method in ['MOON', 'FedDYN'] :
        args.mu = 0.01
    elif args.method in ['FedProx', 'Ditto']:
        args.mu = 0.001
    
    lr = args.lr 
    
    best_averagers = []

    for i, client_name in enumerate(args.g_client_names):   
        best_averagers.append(Seed_Averager_seg(i, client_name))
        
    logs_save_path = os.path.join(args.save_path, 'logs')
    if not os.path.exists(logs_save_path):
        os.makedirs(logs_save_path)
        
    for random_seed in random_seeds:
        g_best_val_dice = [0 for _ in args.g_client_names]
        gc.collect()
        torch.cuda.empty_cache()
        args.random_seed = random_seed
        args.lr = lr
        print('starting run seed', args.random_seed)
        setup_seed(random_seed)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        print('The starting time ：{}'.format(now), flush=True)
        args = set_server_method(args)
        
        # 确保保存路径存在
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
            print(f"Created save directory: {args.save_path}")
        
        pprint(vars(args))
    
        select_list_recorder = [[i for i in range(args.node_num)] for _ in range(args.T)]

        setting_name =  args.method + '_' + args.dataset + '_' + args.local_model + '_nodenum' + str(args.node_num) +'_E'+ str(args.E) \
        + '_seed' + str(args.random_seed)
    
        data = Data(args)

        sample_size = []
        for i in range(args.node_num): 
            sample_size.append(len(data.train_loaders[i]))
        size_weights = [i/sum(sample_size) for i in sample_size]
        
        #size_weights = [1.0/args.node_num for i in range(args.node_num)]
        print('size-based weights',size_weights, flush=True)

        central_node = Node(args,-1 , 'Server', train_loader = None, val_loader=None, test_loader=None)
        # initialize the client nodes
        client_nodes = {}
        for i in range(args.node_num): 
            client_nodes[i] = Node(args, i, args.client_names[i] , train_loader=data.train_loaders[i], val_loader=data.val_loaders[i], test_loader=data.test_loaders[i]) 
            client_nodes[i].model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))
            #########################Scaffold##############################
            if args.method == 'Scaffold':
                client_nodes[i].control = copy.deepcopy(central_node.control)
                client_nodes[i].delta_control = copy.deepcopy(central_node.delta_control)
                client_nodes[i].delta_y = copy.deepcopy(central_node.delta_y)

        g_client_nodes = {}
        for i in range(len(args.g_client_names)):
            g_client_nodes[i] = Node(args, i, args.g_client_names[i] , train_loader=None, val_loader=data.g_val_loaders[i], test_loader=data.g_test_loaders[i]) 

        if args.method == 'FedRDN':
            global_stats = [] 
            for i in range(args.node_num): 
                global_stats.append(client_nodes[i].local_stats) 
            for i in range(args.node_num):
                client_nodes[i].FedRDNTransform_train = FedRDNTransform(
                    local_stats=client_nodes[i].local_stats,
                    global_stats=global_stats,
                    mode='train')
                client_nodes[i].FedRDNTransform_test = FedRDNTransform(
                    local_stats=client_nodes[i].local_stats,
                    global_stats=global_stats,
                    mode='test')
    
        print(setting_name, flush=True)
        
        training_logs = []
        for rounds in range(0, args.T):
            
            round_logs = {}
            round_logs['rounds'] = rounds
            
            print('===============Stage 1 The {:d}-th round==============='.format(rounds + 1), flush=True)
            
            if args.dataset == 'Fundus' or args.dataset == 'Meibomian_Gland': #args.dataset == 'FL_Skin' or args.dataset == 'FL_Breast_Ultrasound' or args.dataset == 'Pathology_COSAS2024' args.dataset == 'FL_Ultrasound'
                args = lr_scheduler(rounds, client_nodes, args)
            
            # Client selection
            select_list = select_list_recorder[rounds]
            # Local update
            client_nodes, client_train_losses, client_train_dices = Client_update(args, client_nodes, central_node, select_list)
            round_logs['client_train_losses']=client_train_losses
            round_logs['client_train_dices']=client_train_dices
            
            for i in select_list: 
                print('Train {:<12.12}, loss:{:.5f}, Dice:{:.5f}'.format(args.client_names[i],client_train_losses[i], client_train_dices[i] ), flush=True)
            print()
            
            if args.method == 'MOON':
                for i in select_list:
                    client_nodes[i].pre_model = copy.deepcopy(client_nodes[i].model)
            
            # Server aggregation
            # MODIFIED: Passed g_client_nodes to allow server-side gradient calculation and alignment
            central_node, client_nodes = Server_update(args, central_node, client_nodes, g_client_nodes, select_list, size_weights,  rounds)
            
           
            
            client_val_losses = []
            client_val_dices = []

            for i in range(len(args.g_client_names)):
                val_loss, val_dice, val_iou, val_sensitivity, val_specificity  = validate_seg(args, central_node, g_client_nodes[i].val_loader)
                client_val_losses.append(val_loss)
                client_val_dices.append(val_dice)
                print('Val   {:<12.12}, loss:{:.5f}, Dice:{:.3f}, IOU:{:.3f}, Sen:{:.3f}, Spe:{:.3f}'.format(args.g_client_names[i],val_loss, val_dice, val_iou, val_sensitivity, val_specificity), flush=True)
                
                if val_dice>g_best_val_dice[i]: # and rounds>args.T//3
                    g_best_val_dice[i] = val_dice
                    # [关键代码] 保存最佳模型
                    best_model_name = f'best_model_{args.g_client_names[i]}_seed{args.random_seed}_{args.method}.pth'
                    torch.save(central_node.model.state_dict(), os.path.join(args.save_path, best_model_name))
                    print(f"  [Saved Best Model] {best_model_name} (Dice: {val_dice:.4f})", flush=True)
                    
                    if args.dataset != 'Prostate3D':
                        test_loss, test_dice, test_iou, test_sensitivity, test_specificity  = validate_seg(args, central_node, g_client_nodes[i].test_loader)
                        g_client_nodes[i].recorder.update(rounds, test_loss, test_dice, test_iou, test_sensitivity, test_specificity)
                    else:
                        g_client_nodes[i].recorder.update(rounds, val_loss, val_dice, val_iou, val_sensitivity, val_specificity)
            round_logs['client_val_losses']=client_val_losses
            round_logs['client_val_dices']=client_val_dices
            training_logs.append(round_logs)
            
            

           
            if rounds==args.T-1:  
                plt.figure(figsize=(10, 6))
                for i, client_name in enumerate(args.g_client_names):
                    # 提取历史 Dice 数据
                    dices = [log['client_val_dices'][i] for log in training_logs]
                    plt.plot(range(1, len(dices) + 1), dices, label=f'{client_name} (Max: {max(dices):.4f})')
                
                plt.xlabel('Rounds')
                plt.ylabel('Validation Dice')
                plt.title(f'Dice Curve (Seed {args.random_seed}, Round {rounds})')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(args.save_path, f'dice_rounds_seed{args.random_seed}_{args.method}_round{rounds}.png'), dpi=300, bbox_inches='tight')
                plt.close()  # 关闭画布防止内存泄漏
            
            print()
            
            for i in range(len(args.g_client_names)):
                g_client_nodes[i].recorder.log(is_log = True)
            print()

 
        for i in range(len(args.g_client_names)): 
            loss, dice, iou, sensitivity, specificity, epoch = g_client_nodes[i].recorder.log(is_log = False)
            best_averagers[i].update(dice, iou, sensitivity, specificity)

        torch.save(training_logs, os.path.join(logs_save_path, setting_name+'_log.pt'))
    
        # [关键代码] 训练结束后绘制最终的 Dice 曲线图
        plt.figure(figsize=(10, 6))
        for i, client_name in enumerate(args.g_client_names):
            # 提取历史 Dice 数据
            dices = [log['client_val_dices'][i] for log in training_logs]
            plt.plot(range(1, len(dices) + 1), dices, label=f'{client_name} (Max: {max(dices):.4f})', linewidth=2)
    
        plt.xlabel('Rounds', fontsize=12)
        plt.ylabel('Validation Dice', fontsize=12)
        plt.title(f'Final Dice Curve (Seed {args.random_seed}, Method: {args.method})', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(args.save_path, f'final_dice_curve_seed{args.random_seed}_{args.method}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    end = time.strftime("%Y-%m-%d %H:%M:%S")
    print('The ending time ：{}'.format(end))
    
    
    
    print('==========================best=================================')
    for i in range(len(args.g_client_names)):
        best_averagers[i].log(details = True)
    print()
    
    print('==========================best=================================')
    for i in range(len(args.g_client_names)):
        best_averagers[i].log(is_log = True)
    print()
    
    print('==========================average=================================')
    total_num = 0.
    total_dice = 0.
    total_iou = 0.
    total_sensitivity = 0.
    total_specificity = 0.
    
    dices = []
    ious = []
    sensitivities = []
    specificities = []
    
    for i in range(len(args.g_client_names)):
        mean_dice, _, mean_iou, _, mean_sensitivity, _, mean_specificity, _ = best_averagers[i].log(is_log = False, details = False)
        if args.dataset != 'Prostate3D':
            local_num = g_client_nodes[i].test_loader.dataset.__len__()
        else:
            local_num = g_client_nodes[i].val_loader.dataset.__len__()
        total_num += local_num
        total_dice += mean_dice*local_num
        total_iou += mean_iou*local_num
        total_sensitivity += mean_sensitivity*local_num
        total_specificity += mean_specificity*local_num
        
        dices.append(mean_dice)
        ious.append(mean_iou)
        sensitivities.append(mean_sensitivity)
        specificities.append(mean_specificity)
    print('Test Average, Dice:{:.3f}, IOU:{:.3f}, Sen:{:.3f}, Spe:{:.3f}'.format(total_dice/total_num,total_iou/total_num,total_sensitivity/total_num,total_specificity/total_num), flush=True)

    mean_dice = np.mean(dices)
    std_dice = np.std(dices)
    
    mean_iou = np.mean(ious)
    std_iou = np.std(ious)
    
    mean_sensitivity = np.mean(sensitivities)
    std_sensitivity = np.std(sensitivities)
    
    mean_specificity = np.mean(specificities)
    std_specificity = np.std(specificities)
    
    print('Test Average, Dice:{:.3f}[{:.3f}], IOU:{:.3f}[{:.3f}], Sen:{:.3f}[{:.3f}], Spe:{:.3f}[{:.3f}]'.format(mean_dice, std_dice, mean_iou, std_iou, mean_sensitivity, std_sensitivity, mean_specificity, std_specificity), flush=True)
    
    print()