import sys
import os
import pickle
import pathlib
import argparse


from torch import nn
import torch

import yaml
from easydict import EasyDict
from sklearn.model_selection import train_test_split
import numpy as np

import seaborn as sns
from tqdm.auto import tqdm
from termcolor import colored

from tensorboardX import SummaryWriter
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
import time

from utils.dataloader import get_dataloader, PostTensorTransform

def clear_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

def all2one_target_transform(x, attack_target=1):
    return torch.ones_like(x) * attack_target

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
        from attack_models.unet import UNet
        atkmodel = UNet(3).to(args.device)
        
        # Copy of attack model
        tgtmodel = UNet(3).to(args.device)
    elif args.dataset == 'mnist':
        from attack_models.autoencoders import MNISTAutoencoder as Autoencoder
        atkmodel = Autoencoder().to(args.device)
        
        # Copy of attack model
        tgtmodel = Autoencoder().to(args.device)
    elif args.dataset == 'tiny-imagenet' or args.dataset == 'tiny-imagenet32' or args.dataset == 'gtsrb':
        if args.attack_model == 'autoencoder':
            from attack_models.autoencoders import Autoencoder
            atkmodel = Autoencoder().to(args.device)
            tgtmodel = Autoencoder().to(args.device)
        elif args.attack_model == 'unet':
            from attack_models.unet import UNet
            atkmodel = UNet(3).to(args.device)
            tgtmodel =UNet(3).to(args.device)
        else:
            raise Exception(f'Invalid generator model {args.attack_model}')
    else:
        raise Exception(f'Invalid atk model {args.dataset}')
    
    # Classifier
    if args.clsmodel == 'vgg':
        from classifier_models import vgg
        def create_net():
            if args.dataset == 'tiny-imagenet':
                return vgg.VGG('VGG11', num_classes=args.num_classes, feature_dim=2048)
            else:
                return vgg.VGG('VGG11', num_classes=args.num_classes)
        
    elif args.clsmodel == 'mnist_cnn':
        from networks.models import NetC_MNIST
        def create_net():
            return NetC_MNIST()
        
    elif args.clsmodel == 'PreActResNet18':
        from classifier_models import PreActResNet18
        def create_net():
            return PreActResNet18(num_classes=args.num_classes)
        
    elif args.clsmodel == 'ResNet18':
        from classifier_models import ResNet18
        def create_net():
            return ResNet18()
        
    elif args.clsmodel == 'ResNet18TinyImagenet':
        from classifier_models import ResNet18TinyImagenet
        def create_net():
            return ResNet18TinyImagenet()
        
    else:
        raise Exception(f'Invalid clsmodel {args.clsmodel}')
    
    clsmodel = create_net().to(args.device)

    # Optimizer
    tgtoptimizer = optim.Adam(tgtmodel.parameters(), lr=args.lr_atk)

    return atkmodel, tgtmodel, tgtoptimizer, clsmodel, create_net

def test(args, atkmodel, scratchmodel, target_transform, 
         train_loader, test_loader, epoch, trainepoch, writer, clip_image, 
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
        testoptimizer = optim.SGD(scratchmodel.parameters(), lr=args.lr)
    for cepoch in range(trainepoch):
        scratchmodel.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
        for batch_idx, (data, target) in pbar:
            data, target = data.to(args.device), target.to(args.device)
            testoptimizer.zero_grad()
            with torch.no_grad():
                noise = atkmodel(data) * args.test_eps
                atkdata = clip_image(data + noise)

            atkoutput = scratchmodel(atkdata)
            output = scratchmodel(data)
            
            loss_clean = F.cross_entropy(output, target)
            loss_poison = F.cross_entropy(atkoutput, target_transform(target))
            
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
            #scratchmodel.eval()
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(args.device), target.to(args.device)
                    output = scratchmodel(data)
                    test_loss += F.cross_entropy(output, target,
                                                 reduction='sum').item()  # sum up batch loss
                    pred = output.max(1, keepdim=True)[
                        1]  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

                    noise = atkmodel(data) * args.test_eps
                    atkdata = clip_image(data + noise)
                    atkoutput = scratchmodel(atkdata)
                    test_transform_loss += F.cross_entropy(atkoutput,
                                                           target_transform(target),
                                                           reduction='sum').item()  # sum up batch loss
                    atkpred = atkoutput.max(1, keepdim=True)[
                        1]  # get the index of the max log-probability
                    correct_transform += atkpred.eq(
                        target_transform(target).view_as(atkpred)).sum().item()

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
    
    if writer is not None:
        writer.add_scalar(f'{log_prefix}-acc(clean)', correct,
                          global_step=epoch-1)
        writer.add_scalar(f'{log_prefix}-acc(poison)',
                          correct_transform,
                          global_step=epoch-1)

        batch_img = torch.cat(
                  [data[:16].clone().cpu(), noise[:16].clone().cpu(), atkdata[:16].clone().cpu()], 0)
        batch_img = F.upsample(batch_img, scale_factor=(4, 4))
        grid = torchvision.utils.make_grid(batch_img, normalize=True)
        writer.add_image(f"{log_prefix}-Test Images", grid, global_step=(epoch-1))

    return correct, correct_transform

def train(args, atkmodel, tgtmodel, clsmodel, tgtoptimizer, clsoptimizer, target_transform, 
          train_loader, epoch, train_epoch, create_net, writer, clip_image, post_transforms=None):
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
        tmpmodel = create_net().to(args.device)
        data, target = data.to(args.device), target.to(args.device)
        noise = tgtmodel(data) * args.eps
        atkdata = clip_image(data + noise)
        
        # Calculate loss
        output = clsmodel(data)
        atkoutput = clsmodel(atkdata)
        loss_clean = F.cross_entropy(output, target)
        loss_poison = F.cross_entropy(atkoutput, target_transform(target))
        loss = loss_clean * args.alpha + (1-args.alpha) * loss_poison
        
        losslist.append(loss.item())
        clear_grad(clsmodel)
        paragrads = torch.autograd.grad(loss, clsmodel.parameters(),
                                        create_graph=True)
        for i, (layername, layer) in enumerate(clsmodel.named_parameters()):
            modulenames, weightname = \
                layername.split('.')[:-1], layername.split('.')[-1]
            module = tmpmodel._modules[modulenames[0]]
            # TODO: could be potentially faster if we save the intermediate mappings
            for name in modulenames[1:]:
                module = module._modules[name]
            #print(layername, modulenames, weightname)
            module._parameters[weightname] = \
                layer - clsoptimizer.param_groups[0]['lr'] * paragrads[i]        
        tgtoptimizer.zero_grad()
        
        noise = tgtmodel(data) * args.eps
        atkdata = clip_image(data + noise)
        tgtloss = F.cross_entropy(tmpmodel(atkdata), target_transform(target))
        
        tgtloss.backward()
        tgtoptimizer.step() #this is the slowest step

        ###############################
        #### Update the classifier ####
        ###############################
        noise = atkmodel(data) * args.eps
        atkdata = clip_image(data + noise)
        output = clsmodel(atkdata)
        loss = F.cross_entropy(output, target)
        clsoptimizer.zero_grad()
        loss.backward()
        clsoptimizer.step()

        if batch_idx % 10 == 0 or batch_idx == (len(train_loader)-1):
            pbar.set_description('Train [{}] Loss: clean {:.4f} poison {:.4f} total {:.4f} Tgt:{:.4f}'.format(
                epoch, loss_clean.item(), loss_poison.item(), loss.item(), tgtloss.item()))
    pbar.close()
    atkloss = sum(losslist) / len(losslist)
    writer.add_scalar('train/loss(atk)', atkloss,
                      global_step=(epoch-1)*args.train_epoch + train_epoch)
    
    batch_img = torch.cat(
              [data[:16].clone().cpu(), noise[:16].clone().cpu(), atkdata[:16].clone().cpu()], 0)
    batch_img = F.upsample(batch_img, scale_factor=(4, 4))
    grid = torchvision.utils.make_grid(batch_img, normalize=True)
    writer.add_image("Train Images", grid, global_step=(epoch-1)*args.train_epoch+train_epoch)
    
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
        args.input_height = 28
        args.input_width = 28
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
            return torch.clamp(x, xmin, xmax)
    elif args.dataset == 'cifar10':
        def clip_image(x):
            return x #no clipping
    elif args.dataset == 'mnist':
        def clip_image(x):
            return torch.clamp(x, -1.0, 1.0)
    elif args.dataset == 'gtsrb':
        def clip_image(x):
            return torch.clamp(x, 0.0, 1.0)
    else:
        raise Exception(f'Invalid dataset: {args.dataset}')
    return train_loader, test_loader, clip_image 

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    if args.verbose >= 1:
        print('========== ARGS ==========')
        print(args)
    
    train_loader, test_loader, clip_image = get_train_test_loaders(args)
    post_transforms = PostTensorTransform(args).to(args.device)
    
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

    writer = SummaryWriter(log_dir=basepath)
    if os.path.exists(checkpoint_path):
        #Load previously saved models
        checkpoint = torch.load(checkpoint_path)
        print(colored('Load existing attack model from path {}'.format(checkpoint_path), 'red'))
        atkmodel.load_state_dict(checkpoint['atkmodel'], strict=True)
        clsmodel.load_state_dict(checkpoint['clsmodel'], strict=True)
        trainlosses = checkpoint['trainlosses']
        best_acc_clean = checkpoint['best_acc_clean']
        best_acc_poison = checkpoint['best_acc_poison']
        start_epoch = checkpoint['epoch']
        tgtoptimizer.load_state_dict(checkpoint['tgtoptimizer'])
    else:
        #Create new model
        print(colored('Create new model from {}'.format(checkpoint_path), 'blue'))
        best_acc_clean = 0
        best_acc_poison = 0
        trainlosses = []
        start_epoch = 1
        
    #Initialize the tgtmodel
    tgtmodel.load_state_dict(atkmodel.state_dict(), strict=True)

    print('============================')
    print('============================')
        
    print('BEGIN TRAINING >>>>>>')

    clsoptimizer = optim.SGD(clsmodel.parameters(), lr=args.lr, momentum=0.9)
    for epoch in range(start_epoch, args.epochs + 1):
        for i in range(args.train_epoch):
            print(f'===== EPOCH: {epoch}/{args.epochs + 1} CLS {i+1}/{args.train_epoch} =====')
            #if not args.avoid_clsmodel_reinitialization:
            #    clsoptimizer = optim.SGD(clsmodel.parameters(), lr=args.lr)
            trainloss = train(args, atkmodel, tgtmodel, clsmodel, tgtoptimizer, clsoptimizer, target_transform, train_loader,
                               epoch, i, create_net, writer, clip_image,
                              post_transforms=post_transforms)
            trainlosses.append(trainloss)
        atkmodel.load_state_dict(tgtmodel.state_dict())
#         if not args.avoid_clsmodel_reinitialization:
#             clsmodel = create_net().to(args.device)
#             scratchmodel = create_net().to(args.device)
#         else:
        scratchmodel = create_net().to(args.device)
        scratchmodel.load_state_dict(clsmodel.state_dict()) #transfer from cls to scratch for testing
        

        if epoch % args.epochs_per_external_eval == 0 or epoch == args.epochs: 
            acc_clean, acc_poison = test(args, atkmodel, scratchmodel, target_transform, 
                   train_loader, test_loader, epoch, args.cls_test_epochs, writer, clip_image, 
                   log_prefix='External')
        else:
            acc_clean, acc_poison = test(args, atkmodel, scratchmodel, target_transform, 
                   train_loader, test_loader, epoch, args.train_epoch, writer, clip_image,
                   log_prefix='Internal')

        if acc_clean > best_acc_clean or (acc_clean > (best_acc_clean-args.best_threshold) and best_acc_poison < acc_poison):
            best_acc_poison = acc_poison
            best_acc_clean = acc_clean
            torch.save({'atkmodel': atkmodel.state_dict(), 'clsmodel': clsmodel.state_dict()}, bestmodel_path)
            
        torch.save({
            'atkmodel': atkmodel.state_dict(),
            'clsmodel': clsmodel.state_dict(),
            'tgtoptimizer': tgtoptimizer.state_dict(),
            'best_acc_clean': best_acc_clean,
            'best_acc_poison': best_acc_poison,
            'trainlosses': trainlosses,
            'epoch': epoch
        }, checkpoint_path)   


def create_config_parser():
    parser = argparse.ArgumentParser(description='PyTorch LIRA Phase 1')
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
    parser.add_argument('--clsmodel', type=str, default='vgg')
    parser.add_argument('--attack_model', type=str, default='autoencoder')
    parser.add_argument('--mode', type=str, default='all2one')
    parser.add_argument('--epochs_per_external_eval', type=int, default=50)
    parser.add_argument('--cls_test_epochs', type=int, default=20)
    parser.add_argument('--path', type=str, default='', help='resume from checkpoint')
    parser.add_argument('--best_threshold', type=float, default=0.1)
    parser.add_argument('--verbose', type=int, default=1, help='verbosity')
    
    parser.add_argument('--test_eps', default=None, type=float)
    parser.add_argument('--test_alpha', default=None, type=float)
    
    return parser
if __name__ == '__main__':
    parser = create_config_parser()
    args = parser.parse_args()
    
    main(args)
    
    #python python/lira_trigger_generation.py --dataset mnist --clsmodel vgg --path experiments/ --epochs 50 --train-epoch 1 --mode all2one --target_label 0 --epochs_per_external_eval 10 --cls_test_epochs 5 --verbose 2 --batch-size 128 --alpha 0.5 --eps 0.01
    