from sys import path
import pickle
import os

import yaml
from easydict import EasyDict
from sklearn.model_selection import train_test_split
import numpy as np

import paddle

import seaborn as sns
from tqdm.auto import tqdm
from termcolor import colored

import pathlib
from argparse import Namespace
import argparse

import time

from lira_trigger_generation import create_config_parser, create_paths, create_models, get_train_test_loaders, get_target_transform, loss_fn
from utils.dataloader import get_dataloader, PostTensorTransform

def get_model(args, model_only=False):
    if args.clsmodel == 'vgg11':
        from paddle.vision.models import vgg11
        def create_net():
            return vgg11(num_classes=args.num_classes)

    elif args.clsmodel == 'resnet18':
        from paddle.vision.models import resnet18
        def create_net():
            return resnet18(num_classes=args.num_classes)
    
    elif args.clsmodel == 'mnist_cnn':
        from classifier_models.cnn import NetC_MNIST
        def create_net():
            return NetC_MNIST()
        
    else:
        raise Exception(f'Invalid clsmodel {args.clsmodel}')
        
    netC = create_net()

    if model_only:
        return netC
    else:

        # Scheduler
        schedulerC = paddle.optimizer.lr.MultiStepDecay(learning_rate=args.test_lr, 
                                                        milestones=args.schedulerC_milestones, 
                                                        gamma=args.schedulerC_lambda)
        # Optimizer
        optimizerC = paddle.optimizer.Momentum(parameters=netC.parameters(), 
                                               learning_rate=schedulerC, momentum=0.9, weight_decay=5e-4)

        return netC, optimizerC, schedulerC

def final_test(args, test_model_path, atkmodel, netC, target_transform, train_loader, test_loader, 
         trainepoch, writer, alpha=0.5, optimizerC=None, 
         schedulerC=None, log_prefix='Internal', epochs_per_test=1, data_transforms=None, start_epoch=1,
         clip_image=None):
    test_loss = 0
    correct = 0
    
    clean_accs, poison_accs = [], []
    
    correct_transform = 0
    test_transform_loss = 0
    
    best_clean_acc, best_poison_acc = 0, 0
    
    atkmodel.eval()
    
    if optimizerC is None:
        print('No optimizer, creating default SGD...')  
        optimizerC = optim.SGD(netC.parameters(), lr=args.test_lr)
    if schedulerC is None:
        print('No scheduler, creating default 100,200,300,400...')  
        schedulerC = optim.lr_scheduler.MultiStepLR(optimizerC, [100, 200, 300, 400], args.test_lr)
        
    for cepoch in range(start_epoch, trainepoch+1):
        netC.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, target) in pbar:
            if data_transforms is not None:
                data = data_transforms(data)                                    
            optimizerC.clear_grad()
            
            output = netC(data)
            loss_clean = loss_fn(output, target)
            
            if alpha < 1:
                with paddle.no_grad():
                    noise = atkmodel(data) * args.test_eps
                    if clip_image is None:
                        atkdata = paddle.clip(data + noise, 0, 1)
                    else:
                        atkdata = clip_image(data + noise)
                atkoutput = netC(atkdata)
                loss_poison = loss_fn(atkoutput, target_transform(target))
            else:
                loss_poison = paddle.tensor(0.0)
            
            loss = alpha * loss_clean + (1-alpha) * loss_poison
            
            loss.backward()
            optimizerC.step()
            
            if batch_idx % 10 == 0 or batch_idx == (len(train_loader)-1):
                pbar.set_description(
                    'Train-{} Loss: Clean {:.5f}  Poison {:.5f}  Total {:.5f} LR={:.6f}'.format(
                        cepoch,
                        loss_clean.item(),
                        loss_poison.item(),
                        loss.item(), schedulerC.get_lr()
                    ))
        schedulerC.step()
        if cepoch % epochs_per_test == 0 or cepoch == trainepoch-1:
            with paddle.no_grad():
                for data, target in tqdm(test_loader, desc=f'Evaluation {cepoch}'):
                    if len(target.shape) == 1:
                        target = target.reshape([data.shape[0], 1])
                    output = netC(data)
                    test_loss += paddle.nn.functional.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                    correct += paddle.metric.accuracy(output, target).item() * len(target)

                    noise = atkmodel(data) * args.test_eps
                    if clip_image is None:
                        atkdata = paddle.clip(data + noise, 0, 1)
                    else:
                        atkdata = clip_image(data + noise)
                    atkoutput = netC(atkdata)
                    test_transform_loss += paddle.nn.functional.cross_entropy(
                        atkoutput, target_transform(target), reduction='sum').item()  # sum up batch loss
                    correct_transform += paddle.metric.accuracy(atkoutput, target_transform(target)).item() * len(target)

            test_loss /= len(test_loader.dataset)
            test_transform_loss /= len(test_loader.dataset)

            correct /= len(test_loader.dataset)
            correct_transform /= len(test_loader.dataset)
            
            clean_accs.append(correct)
            poison_accs.append(correct_transform)
            
            print('\n{}-Test [{}]: Loss: clean {:.4f} poison {:.4f}, '
                  'Accuracy: clean {:.4f} (best {:.4f}) poison {:.4f} (best {:.4f})'.format(
                    log_prefix, cepoch, 
                    test_loss, test_transform_loss,
                    correct, best_clean_acc, correct_transform, best_poison_acc
                ))
            if correct > best_clean_acc or (correct > best_clean_acc-0.02 and correct_transform > best_poison_acc):
                best_clean_acc = correct
                best_poison_acc = correct_transform
                
                print(f'Saving current best model in {test_model_path}')
                paddle.save({
                    'atkmodel': atkmodel.state_dict(),
                    'netC': netC.state_dict(), 
                    'optimizerC': optimizerC.state_dict(), 
                    'clean_schedulerC': schedulerC,
                    'best_clean_acc': best_clean_acc, 
                    'best_poison_acc': best_poison_acc
                }, test_model_path)

        
    return clean_accs, poison_accs

def main():
    parser = create_config_parser()
    #parser.add_argument('--test_eps', default=None, type=float)
    #parser.add_argument('--test_alpha', default=None, type=float)
    parser.add_argument('--test_epochs', default=500, type=int)
    parser.add_argument('--test_lr', default=None, type=float)
    parser.add_argument('--schedulerC_lambda', default=0.05, type=float)
    parser.add_argument('--schedulerC_milestones', default='30,60,90,150')
    parser.add_argument('--test_n_size', default=10)
    parser.add_argument('--test_optimizer', default='sgd')
    parser.add_argument('--test_use_train_best', default=False, action='store_true')
    parser.add_argument('--test_use_train_last', default=False, action='store_true')
    parser.add_argument('--use_data_parallel', default=False, action='store_true')
    
    args = parser.parse_args()
    
    if args.test_alpha is None:
        print(f'Defaulting test_alpha to train alpha of {args.alpha}')
        args.test_alpha = args.alpha
        
    if args.test_lr is None:
        print(f'Defaulting test_lr to train lr {args.lr}')
        args.test_lr = args.lr
        
    if args.test_eps is None:
        print(f'Defaulting test_eps to train eps {args.test_eps}')
        args.test_eps = args.eps
    
    args.schedulerC_milestones = [int(e) for e in args.schedulerC_milestones.split(',')]
    
    print('====> ARGS')
    print(args)
    
    paddle.set_device("gpu" if paddle.device.is_compiled_with_cuda() else "cpu")    
    
    args.basepath, args.checkpoint_path, args.bestmodel_path = basepath, checkpoint_path, bestmodel_path = create_paths(args)
    test_model_path = os.path.join(
        basepath, f'poisoned_classifier_{args.test_alpha}_{args.test_eps}_{args.test_optimizer}.ph')
    print(f'Will save test model at {test_model_path}')
    
    train_loader, test_loader, clip_image = get_train_test_loaders(args)
    post_transforms = PostTensorTransform(args)

    
    atkmodel, tgtmodel, tgtoptimizer, _, create_net = create_models(args)
    netC, optimizerC, schedulerC = get_model(args)
    #clsmodel = create_net().to(args.device)
    
    if args.test_use_train_best:
        checkpoint = paddle.load(f'{bestmodel_path}')
        print('Load atkmodel and classifier states from best training: {}'.format(bestmodel_path))
        netC.load_dict(checkpoint['clsmodel'], use_structured_name=True)
        atk_checkpoint = checkpoint['atkmodel']
    elif args.test_use_train_last:
        checkpoint = paddle.load(f'{checkpoint_path}')
        print('Load atkmodel and classifier states from last training: {}'.format(checkpoint_path))
        netC.load_dict(checkpoint['clsmodel'], use_structured_name=True)
        atk_checkpoint = checkpoint['atkmodel']
    else: #also use this model for a new classifier model
        checkpoint = paddle.load(f'{bestmodel_path}')
        if 'atkmodel' in checkpoint:
            atk_checkpoint = checkpoint['atkmodel'] #this is for the new changes when we save both cls and atk
        else:
            atk_checkpoint = checkpoint
        print('Use scratch clsmodel. Load atkmodel state from best training: {}'.format(bestmodel_path))
        
    target_transform = get_target_transform(args)
    
    if args.test_alpha != 1.0:
        print(f'Loading best model from {bestmodel_path}')
        atkmodel.load_dict(atk_checkpoint, use_structured_name=True)
    else:
        print(f'Skip loading best atk model since test_alpha=1')
    
    if args.test_optimizer == 'adam':
        print('Change optimizer to adam')
        schedulerC = paddle.optimizer.lr.MultiStepDecay(learning_rate=args.test_lr, 
                                                        milestones=args.schedulerC_milestones, 
                                                        gamma=args.schedulerC_lambda)
        # Optimizer
        optimizerC = paddle.optimizer.Adam(parameters=netC.parameters(), 
                                           learning_rate=schedulerC, momentum=0.9, weight_decay=5e-4)

    elif args.test_optimizer == 'sgdo':
        schedulerC = paddle.optimizer.lr.MultiStepDecay(learning_rate=args.test_lr, 
                                                        milestones=args.schedulerC_milestones, 
                                                        gamma=args.schedulerC_lambda)
        # Optimizer
        optimizerC = paddle.optimizer.SGD(parameters=netC.parameters(), 
                                               learning_rate=schedulerC, weight_decay=5e-4)
        
    print(netC)
    print(optimizerC)
    print(schedulerC)
    
    data_transforms = PostTensorTransform(args)
    print('====> Post tensor transform')
    print(data_transforms)
    
    clean_accs, poison_accs = final_test(
        args, test_model_path, atkmodel, netC, target_transform, 
        train_loader, test_loader, trainepoch=args.test_epochs,
        writer=None, log_prefix='POISON', alpha=args.test_alpha, epochs_per_test=1, 
        optimizerC=optimizerC, schedulerC=schedulerC, data_transforms=data_transforms, clip_image=clip_image)
    
if __name__ == '__main__':
    main()
    #CUDA_VISIBLE_DEVICES=6 python paddle/lira_backdoor_injection.py --dataset mnist --clsmodel mnist_cnn --path experiments/ --epochs 50 --train-epoch 1 --mode all2one --target_label 0 --epochs_per_external_eval 10 --cls_test_epochs 5 --verbose 2 --batch-size 128 --alpha 0.5 --eps 0.01 --test_eps 0.01 --test_alpha 0.5 --test_epochs 50 --test_lr 0.01 --schedulerC_lambda 0.1 --schedulerC_milestones 10,20,30,40
    #CUDA_VISIBLE_DEVICES=6 python paddle/lira_backdoor_injection.py --dataset cifar10 --clsmodel vgg11 --path experiments/ --epochs 50 --train-epoch 1 --mode all2one --target_label 0 --epochs_per_external_eval 10 --cls_test_epochs 5 --verbose 2 --batch-size 128 --alpha 0.5 --eps 0.01 --test_eps 0.01 --test_alpha 0.5 --test_epochs 50 --test_lr 0.01 --schedulerC_lambda 0.1 --schedulerC_milestones 10,20,30,40

