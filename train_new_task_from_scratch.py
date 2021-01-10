import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import models
import os
import time
import argparse
import numpy as np
import general_dataset_loader
from torch.autograd import Variable
from copy import copy
from jigsaw_model import JigsawModel
import utils_pytorch
import sys
from lib.LinearAverage import LinearAverage



def full_training(args):
    if not os.path.isdir(args.expdir):
        os.makedirs(args.expdir) 
    elif os.path.exists(args.expdir + '/results.npy'):
        return

    if 'ae' in args.task:
        os.mkdir(args.expdir + '/figs/')

    train_batch_size = args.train_batch_size // 4 if args.task == 'rot' else args.train_batch_size
    test_batch_size = args.test_batch_size // 4 if args.task == 'rot' else args.test_batch_size
    yield_indices = (args.task == 'inst_disc')
    datadir = args.datadir + args.dataset
    trainloader, valloader, num_classes = general_dataset_loader.prepare_data_loaders(datadir, image_dim=args.image_dim,
                                              yield_indices=yield_indices,
                                              train_batch_size=train_batch_size,
                                              test_batch_size=test_batch_size,
                                              train_on_10_percent=args.train_on_10,
                                              train_on_half_classes=args.train_on_half
                                              )
    _, testloader, _ = general_dataset_loader.prepare_data_loaders(datadir, image_dim=args.image_dim,
                                              yield_indices=yield_indices,
                                              train_batch_size=train_batch_size,
                                              test_batch_size=test_batch_size,
                                              )

    args.num_classes = num_classes
    if args.task=='rot':
        num_classes = 4
    elif args.task == 'inst_disc':
        num_classes = args.low_dim

    if args.task == 'ae':
        net = models.AE([args.code_dim], image_dim=args.image_dim)
    elif args.task == 'jigsaw':
        net = JigsawModel(num_perms=args.num_perms, code_dim=args.code_dim, gray_prob=args.gray_prob, image_dim=args.image_dim)
    else:
        net = models.resnet26(num_classes, mlp_depth=args.mlp_depth,
                              normalize=(args.task=='inst_disc'))
    if args.task == 'inst_disc':
        train_lemniscate = LinearAverage(args.low_dim, trainloader.dataset.__len__(), args.nce_t, args.nce_m)
        train_lemniscate.cuda()
        args.train_lemniscate = train_lemniscate
        test_lemniscate = LinearAverage(args.low_dim, valloader.dataset.__len__(), args.nce_t, args.nce_m)
        test_lemniscate.cuda()
        args.test_lemniscate = test_lemniscate
    if args.source:
        try:
            old_net = torch.load(args.source)
        except:
            print("Falling back encoding")
            from functools import partial
            import pickle
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            old_net = torch.load(args.source, map_location=lambda storage, loc: storage, pickle_module=pickle)
        
        # net.load_state_dict(old_net['net'].state_dict())
        old_net = old_net['net']
        if hasattr(old_net, "module"):
            old_net = old_net.module
        old_state_dict = old_net.state_dict()
        new_state_dict = net.state_dict()
        for key,weight in old_state_dict.items():
            if 'linear' not in key:
                new_state_dict[key] = weight
            elif key == 'linears.0.weight' and weight.shape[0]==num_classes:
                new_state_dict['linears.0.0.weight'] = weight
            elif key == 'linears.0.bias' and weight.shape[0]==num_classes:
                new_state_dict['linears.0.0.bias'] = weight
        net.load_state_dict(new_state_dict)


        del old_net
    net = torch.nn.DataParallel(net).cuda()
    start_epoch = 0
    if args.task in ['ae', 'inst_disc']:
        best_acc = np.inf
    else:
        best_acc = -1
    results = np.zeros((4,start_epoch+args.nb_epochs))

    net.cuda()
    cudnn.benchmark = True

    if args.task in  ['ae']:
        args.criterion = nn.MSELoss()
    else:
        args.criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.wd)


    print("Start training")
    train_func = eval('utils_pytorch.train_' + args.task)
    test_func = eval('utils_pytorch.test_' + args.task)
    if args.test_first: 
        with torch.no_grad():
            test_func(0,valloader, net, best_acc, args, optimizer)
    for epoch in range(start_epoch, start_epoch+args.nb_epochs):
        utils_pytorch.adjust_learning_rate(optimizer, epoch, args)
        st_time = time.time()
        
        # Training and validation
        train_acc, train_loss = train_func(epoch, trainloader, net, args, optimizer)
        test_acc, test_loss, best_acc = test_func(epoch,valloader, net, best_acc, args, optimizer)
        
        # Record statistics
        results[0:2,epoch] = [train_loss,train_acc]
        results[2:4,epoch] = [test_loss,test_acc]
        np.save(args.expdir + '/results.npy', results)
        print('Epoch lasted {0}'.format(time.time()-st_time))
        sys.stdout.flush()
        if (args.task=='rot') and (train_acc >= 98) and args.early_stopping: break
    if args.task == 'inst_disc':
        args.train_lemniscate = None
        args.test_lemniscate = None
    else:
        best_net = torch.load(args.expdir + 'checkpoint.t7')['net']
        if args.task in ['ae', 'inst_disc']:
            best_acc = np.inf
        else:
            best_acc = -1
        final_acc, final_loss, _ = test_func(0, testloader, best_net, best_acc, args, None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar100', nargs='+', help='Task(s) to be trained')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--nb-epochs', default=120, type=int, help='nb epochs')
    parser.add_argument('--step1', default=80, type=int, help='nb epochs before first lr decrease')
    parser.add_argument('--step2', default=100, type=int, help='nb epochs before second lr decrease')
    parser.add_argument('--expdir', default='tmp/', help='Save folder')
    parser.add_argument('--datadir', default='/scratch/datasets/bw462/deca/my_deca/', help='folder containing data folder')
    parser.add_argument('--source', default='', type=str, help='Network source basedir')
    parser.add_argument('--train-batch-size', default=512, type=int, help='samples per train batch')
    parser.add_argument('--test-batch-size', default=100, type=int, help='samples per test batch')
    parser.add_argument('--image-dim', default=64, type=int, help='width/height of image')
    parser.add_argument('--test-first', action='store_true', help='Start w/ eval')
    parser.add_argument('--task', default='supervised', nargs='+', help='pretext task to learn')
    parser.add_argument('--ngf', default=64, type=int, help='ngf for Generator')
    parser.add_argument('--gray-prob', default=0.3, type=float, help='probability of tile being grayed')
    parser.add_argument('--num-perms', default=2000, type=int, help='Number perms to have, currently 100 and 701 implemented')
    parser.add_argument('--code-dim', default=256, type=int, help='AE bottleneck dim or Jigsaw siamese rep dim')
    parser.add_argument('--low-dim', default=128, type=int, help='ID or exemp rep dim')
    parser.add_argument('--mlp-depth', default=1, type=int, help='Depth of mlp at end, minimum/default 1 is just linear classification')
    parser.add_argument('--nce-t', default=0.1, type=float,
                        metavar='T', help='temperature parameter for softmax')
    parser.add_argument('--nce-m', default=0.5, type=float,
                        metavar='M', help='momentum for non-parametric updates')
    parser.add_argument('--always-save', action='store_true', help='Always save instead of early stopping')
    parser.add_argument('--early-stopping', action='store_true', help='whether to early stop')
    parser.add_argument('--train-on-10', action='store_true', help='Only use 10% of training labels')
    parser.add_argument('--train-on-half', action='store_true', help='Only use half of classes')
    parser.add_argument('--use-nuc-loss', action='store_true', help='Use nuc loss on features in supervision')
    parser.add_argument('--nuc-weight', type=float, default=0.05, help='Weight for nuc loss')
    args = parser.parse_args()


    dataset_list = copy(args.dataset)
    if type(dataset_list)==str: dataset_list = [dataset_list]
    task_list = copy(args.task)
    if type(task_list)==str: task_list = [task_list]

    base_expdir = args.expdir
    print(dataset_list)
    for dataset in dataset_list:
        for task in task_list:
            args.dataset = dataset
            args.task = task
            args.expdir = '/'.join([base_expdir, args.task, dataset, ''])
            full_training(args)

if __name__ == '__main__':
    main()


