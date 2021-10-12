import sys
import os
import pickle
import pathlib
import argparse


from paddle import nn
import paddle

import yaml
from easydict import EasyDict
from sklearn.model_selection import train_test_split
import numpy as np

import seaborn as sns
from tqdm.auto import tqdm
from termcolor import colored

import time

from utils.dataloader import get_dataloader, PostTensorTransform

loss_fn = nn.CrossEntropyLoss()

def all2one_target_transform(x, attack_target=1):
    return paddle.ones_like(x) * attack_target

def all2all_target_transform(x, num_classes):
    return (x + 1) % num_classes

def create_attack_model(dataset, attack_model=None):
    if dataset == 'cifar10':
        from attack_models.unet import UNet
        
        atkmodel = UNet(3)
    elif dataset == 'mnist':
        from attack_models.autoencoders import MNISTAutoencoder as Autoencoder
        atkmodel = Autoencoder()

    elif dataset == 'tiny-imagenet' or dataset == 'tiny-imagenet32' or dataset == 'gtsrb':
        if attack_model is None:
            from attack_models.autoencoders import Autoencoder
            atkmodel = Autoencoder()
        elif attack_model == 'unet':
            from attack_models.unet import UNet
            atkmodel = UNet(3)
    else:
        raise Exception(f'Invalid atk model {dataset}')
        
    return atkmodel

def create_models(args):
    """DONE
    """
    if args.dataset == 'cifar10':
        if args.attack_model is None or args.attack_model == 'autoencoder':
            from attack_models.autoencoders import Autoencoder
            atkmodel = Autoencoder(args.input_channel)
            # Copy of attack model
            tgtmodel = Autoencoder(args.input_channel)
        elif args.attack_model == 'unet':
            from attack_models.unet import UNet
            atkmodel = UNet(args.input_channel)        
            # Copy of attack model
            tgtmodel = UNet(args.input_channel)
    elif args.dataset == 'mnist':
        from attack_models.autoencoders import Autoencoder
        atkmodel = Autoencoder(args.input_channel)
        
        # Copy of attack model
        tgtmodel = Autoencoder(args.input_channel)
    else:
        raise Exception(f'Invalid atk model {args.dataset}')

    # Classifier
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
    
    clsmodel = create_net()
    
    # Optimizer
    tgtoptimizer = paddle.optimizer.Adam(parameters=tgtmodel.parameters(), learning_rate=args.lr_atk)

    return atkmodel, tgtmodel, tgtoptimizer, clsmodel, create_net

def test(args, atkmodel, scratchmodel, target_transform, 
         train_loader, test_loader, epoch, trainepoch, clip_image, 
         testoptimizer=None, log_prefix='Internal', epochs_per_test=5):
    #default phase 2 parameters to phase 1 
    if args.test_alpha is None:
        args.test_alpha = args.alpha
    if args.test_eps is None:
        args.test_eps = args.eps
    
    test_loss = 0
    correct = 0
    
    correct_transform = 0
    test_transform_loss = 0
    
    atkmodel.eval()
    if testoptimizer is None:
        testoptimizer = paddle.optimizer.SGD(parameters=scratchmodel.parameters(), learning_rate=args.lr)
        
    for cepoch in range(trainepoch):
        scratchmodel.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
        for batch_idx, (data, target) in pbar:
            testoptimizer.clear_grad()
            with paddle.no_grad():
                noise = atkmodel(data) * args.test_eps
                atkdata = clip_image(data + noise)

            atkoutput = scratchmodel(atkdata)
            output = scratchmodel(data)
            
            loss_clean = loss_fn(output, target)
            loss_poison = loss_fn(atkoutput, target_transform(target))
            
            loss = args.alpha * loss_clean + (1-args.test_alpha) * loss_poison
            
            loss.backward()
            testoptimizer.step()
            
            if batch_idx % 10 == 0 or batch_idx == (len(train_loader)-1):
                pbar.set_description(
                    'Test [{}-{}] Loss: Clean {:.4f} Poison {:.4f} Total {:.5f}'.format(
                        epoch, cepoch,
                        loss_clean.item(),
                        loss_poison.item(),
                        loss.item()
                    ))
        if cepoch % epochs_per_test == 0 or cepoch == trainepoch-1:
            scratchmodel.eval()
            with paddle.no_grad():
                for data, target in test_loader:
                    if len(target.shape) == 1:
                        target = target.reshape([data.shape[0], 1])
                    output = scratchmodel(data)
                    test_loss += paddle.nn.functional.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                    correct += paddle.metric.accuracy(output, target).item() * len(target)

                    noise = atkmodel(data) * args.test_eps
                    atkdata = clip_image(data + noise)
                    atkoutput = scratchmodel(atkdata)
                    test_transform_loss += paddle.nn.functional.cross_entropy(
                        atkoutput, target_transform(target), reduction='sum').item()  # sum up batch loss
                    correct_transform += paddle.metric.accuracy(atkoutput, target_transform(target)).item() * len(target)

            test_loss /= len(test_loader.dataset)
            test_transform_loss /= len(test_loader.dataset)

            correct /= len(test_loader.dataset)
            correct_transform /= len(test_loader.dataset)

            print(
                '\n{}-Test set [{}]: Loss: clean {:.4f} poison {:.4f}, Accuracy: clean {:.2f} poison {:.2f}'.format(
                    log_prefix, cepoch, 
                    test_loss, test_transform_loss,
                    correct, correct_transform
                ))
       
    return correct, correct_transform

def train(args, atkmodel, tgtmodel, clsmodel, tgtoptimizer, clsoptimizer, target_transform, 
          train_loader, epoch, train_epoch, create_net, clip_image, post_transforms=None):
    clsmodel.train()
    atkmodel.eval()
    tgtmodel.train()
    losslist = []
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
    for batch_idx, (data, target) in pbar:
        if post_transforms is not None:
            data = post_transforms(data)
        
        ########################################
        #### Update Transformation Function ####
        ########################################
        noise = tgtmodel(data) * args.eps
        atkdata = clip_image(data + noise)
        
        # Calculate loss
        atkoutput = clsmodel(atkdata)
        loss_poison = loss_fn(atkoutput, target_transform(target))
        loss1 = loss_poison
        
        losslist.append(loss1.item())
        clsoptimizer.clear_grad()
        tgtoptimizer.clear_grad()
        loss1.backward()
        tgtoptimizer.step() #this is the slowest step

        ###############################
        #### Update the classifier ####
        ###############################
        noise = atkmodel(data) * args.eps
        atkdata = clip_image(data + noise)
        output = clsmodel(data)
        atkoutput = clsmodel(atkdata)
        loss_clean = loss_fn(output, target)
        loss_poison = loss_fn(atkoutput, target_transform(target))
        loss2 = loss_clean * args.alpha + (1-args.alpha) * loss_poison
        clsoptimizer.clear_grad()
        loss2.backward()
        clsoptimizer.step()

        if batch_idx % 10 == 0 or batch_idx == (len(train_loader)-1):
            pbar.set_description('Train [{}] Loss: clean {:.4f} poison {:.4f} CLS {:.4f} ATK:{:.4f}'.format(
                epoch, loss_clean.item(), loss_poison.item(), loss1.item(), loss2.item()))
    pbar.close()
    atkloss = sum(losslist) / len(losslist)

    return atkloss

def create_paths(args):
    if args.mode == 'all2one': 
        basepath = os.path.join(args.path, f'{args.mode}_{args.target_label}', args.dataset, args.clsmodel)
    else:
        basepath = os.path.join(args.path, args.mode, args.dataset, args.clsmodel)
   
    basepath = os.path.join(basepath, f'lr{args.lr}-lratk{args.lr_atk}-eps{args.eps}-alpha{args.alpha}-clsepoch{args.train_epoch}-atkmodel{args.attack_model}')

    if not os.path.exists(basepath):
        print(f'Creating new model training in {basepath}')
        os.makedirs(basepath)
    checkpoint_path = os.path.join(basepath, 'checkpoint.ckpt')
    bestmodel_path = os.path.join(basepath, 'bestmodel.ckpt')
    return basepath, checkpoint_path, bestmodel_path

def get_target_transform(args):
    """DONE
    """
    if args.mode == 'all2one':
        target_transform = lambda x: all2one_target_transform(x, args.target_label)
    elif args.mode == 'all2all':
        target_transform = lambda x: all2all_target_transform(x, args.num_classes)
    else:
        raise Exception(f'Invalid mode {args.mode}')
    return target_transform

def get_train_test_loaders(args):
    """DONE
    """
    if args.dataset == "cifar10":
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
        args.num_classes = 10
    elif args.dataset == "gtsrb":
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
        args.num_classes = 43
    elif args.dataset == "mnist":
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 1
        args.num_classes = 10
    elif args.dataset in ['tiny-imagenet32']:
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3
        args.num_classes = 200
    else:
        raise Exception("Invalid Dataset")
        
    train_loader = get_dataloader(args, True, args.pretensor_transform)
    test_loader = get_dataloader(args, False, args.pretensor_transform)
    if args.dataset in ['tiny-imagenet', 'tiny-imagenet32']:
        xmin, xmax = -2.1179039478302, 2.640000104904175
        def clip_image(x):
            return paddle.clip(x, xmin, xmax)
    elif args.dataset == 'cifar10':
        def clip_image(x):
            return x #no clipping
    elif args.dataset == 'mnist':
        def clip_image(x):
            return paddle.clip(x, -1.0, 1.0)
    elif args.dataset == 'gtsrb':
        def clip_image(x):
            return paddle.clip(x, 0.0, 1.0)
    else:
        raise Exception(f'Invalid dataset: {args.dataset}')
    return train_loader, test_loader, clip_image 

def main(args):
    paddle.seed(args.seed)
    np.random.seed(args.seed)
    
    args.device = paddle.set_device("gpu" if paddle.device.is_compiled_with_cuda() else "cpu")    
    if args.verbose >= 1:
        print('========== ARGS ==========')
        print(args)
    
    train_loader, test_loader, clip_image = get_train_test_loaders(args)
    post_transforms = PostTensorTransform(args)
    
    print('========== DATA ==========')
    print('Loaders: Train {} examples/{} iters, Test {} examples/{} iters'.format(
        len(train_loader.dataset), len(train_loader),  len(test_loader.dataset), len(test_loader)))
    
    atkmodel, tgtmodel, tgtoptimizer, clsmodel, create_net = create_models(args)
    if args.verbose >= 2:
        print('========== MODELS ==========')
        print(atkmodel)
        print(clsmodel)
    
    target_transform = get_target_transform(args)
    basepath, checkpoint_path, bestmodel_path = create_paths(args)
    
    print('========== PATHS ==========')
    print(f'Basepath: {basepath}')
    print(f'Checkpoint Model: {checkpoint_path}')
    print(f'Best Model: {bestmodel_path}')

    if os.path.exists(checkpoint_path):
        #Load previously saved models
        checkpoint = paddle.load(checkpoint_path)
        print(colored('Load existing attack model from path {}'.format(checkpoint_path), 'red'))
        atkmodel.load_dict(checkpoint['atkmodel'], use_structured_name=True)
        clsmodel.load_dict(checkpoint['clsmodel'], use_structured_name=True)
        trainlosses = checkpoint['trainlosses']
        best_acc_clean = checkpoint['best_acc_clean']
        best_acc_poison = checkpoint['best_acc_poison']
        start_epoch = checkpoint['epoch']
        tgtoptimizer.load_dict(checkpoint['tgtoptimizer'])
    else:
        #Create new model
        print(colored('Create new model from {}'.format(checkpoint_path), 'blue'))
        best_acc_clean = 0
        best_acc_poison = 0
        trainlosses = []
        start_epoch = 1
        
    #Initialize the tgtmodel
    tgtmodel.load_dict(atkmodel.state_dict(), use_structured_name=True)

    print('============================')
    print('============================')
        
    print('BEGIN TRAINING >>>>>>')

    clsoptimizer = paddle.optimizer.Momentum(parameters=clsmodel.parameters(), learning_rate=args.lr, momentum=0.9)
    for epoch in range(start_epoch, args.epochs + 1):
        for i in range(args.train_epoch):
            print(f'===== EPOCH: {epoch}/{args.epochs + 1} CLS {i+1}/{args.train_epoch} =====')
            if not args.avoid_clsmodel_reinitialization:
                clsoptimizer = paddle.optimizer.SGD(parameters=clsmodel.parameters(), learning_rate=args.lr)
            trainloss = train(args, atkmodel, tgtmodel, clsmodel, tgtoptimizer, clsoptimizer, target_transform, train_loader,
                               epoch, i, create_net, clip_image,
                              post_transforms=post_transforms)
            trainlosses.append(trainloss)
        atkmodel.load_dict(tgtmodel.state_dict())
        if args.avoid_clsmodel_reinitialization:
            scratchmodel = create_net()
            scratchmodel.load_dict(clsmodel.state_dict()) #transfer from cls to scratch for testing
        else:
            clsmodel = create_net()
            scratchmodel = create_net()

        if epoch % args.epochs_per_external_eval == 0 or epoch == args.epochs: 
            acc_clean, acc_poison = test(args, atkmodel, scratchmodel, target_transform, 
                   train_loader, test_loader, epoch, args.cls_test_epochs, clip_image, 
                   log_prefix='External')
        else:
            acc_clean, acc_poison = test(args, atkmodel, scratchmodel, target_transform, 
                   train_loader, test_loader, epoch, args.train_epoch, clip_image,
                   log_prefix='Internal')

        if acc_clean > best_acc_clean or (acc_clean > (best_acc_clean-args.best_threshold) and best_acc_poison < acc_poison):
            best_acc_poison = acc_poison
            best_acc_clean = acc_clean
            paddle.save({'atkmodel': atkmodel.state_dict(), 'clsmodel': clsmodel.state_dict()}, bestmodel_path)
            
        paddle.save({
            'atkmodel': atkmodel.state_dict(),
            'clsmodel': clsmodel.state_dict(),
            'tgtoptimizer': tgtoptimizer.state_dict(),
            'best_acc_clean': best_acc_clean,
            'best_acc_poison': best_acc_poison,
            'trainlosses': trainlosses,
            'epoch': epoch
        }, checkpoint_path)   


def create_config_parser():
    parser = argparse.ArgumentParser(description='PaddlePaddle LIRA Phase 1')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data_root', type=str, default='data/')
    parser.add_argument("--random_rotation", type=int, default=10)
    parser.add_argument("--random_crop", type=int, default=5)
    parser.add_argument("--pretensor_transform", action='store_true', default=False)
    
    
    parser.add_argument('--num-workers', type=int, default=2, help='dataloader workers')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--lr-atk', type=float, default=0.0001, help='learning rate for attack model')
    parser.add_argument('--seed', type=int, default=999, help='random seed (default: 999)')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--train-epoch', type=int, default=1, help='training epochs for victim model')
    

    parser.add_argument('--target_label', type=int, default=1) #only in effect if it's all2one
    parser.add_argument('--eps', type=float, default=0.3, help='epsilon for data poisoning')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--clsmodel', type=str, default='vgg11')
    parser.add_argument('--attack_model', type=str, default='autoencoder')
    parser.add_argument('--mode', type=str, default='all2one')
    parser.add_argument('--epochs_per_external_eval', type=int, default=50)
    parser.add_argument('--cls_test_epochs', type=int, default=20)
    parser.add_argument('--path', type=str, default='', help='resume from checkpoint')
    parser.add_argument('--best_threshold', type=float, default=0.1)
    parser.add_argument('--verbose', type=int, default=1, help='verbosity')
    parser.add_argument('--avoid_clsmodel_reinitialization', action='store_true', 
                        default=False, help='whether test the poisoned model from scratch')
    
    parser.add_argument('--test_eps', default=None, type=float)
    parser.add_argument('--test_alpha', default=None, type=float)
    
    return parser
if __name__ == '__main__':
    parser = create_config_parser()
    args = parser.parse_args()
    
    main(args)       