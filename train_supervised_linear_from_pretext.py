import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import general_dataset_loader
import models
import os
import sys
import time
import argparse
import numpy as np
from copy import copy

from torch.autograd import Variable

from jigsaw_model import JigsawModel
import utils_pytorch


def full_training(args):
    if not os.path.isdir(args.expdir):
        os.makedirs(args.expdir)
    elif os.path.exists(args.expdir + '/results.npy'):
        return

    datadir = args.datadir + args.dataset
    trainloader, valloader, num_classes = general_dataset_loader.prepare_data_loaders(datadir, image_dim=args.image_dim,
                                                                                          train_batch_size=args.train_batch_size,
                                                                                          test_batch_size=args.test_batch_size,
                                                                                          train_on_10_percent=args.train_on_10,
                                                                                          no_flip=args.no_flip)
    _, testloader, _ = general_dataset_loader.prepare_data_loaders(datadir, image_dim=args.image_dim,
                                                                                          train_batch_size=args.train_batch_size,
                                                                                          test_batch_size=args.test_batch_size, test=True)

    if args.random_labels:
        trainloader.dataset.shuffle_labels()
        valloader = trainloader
        testloader = trainloader
    args.num_classes = num_classes

    # Load checkpoint and initialize the networks with the weights of a pretrained network
    print('==> Resuming from checkpoint..')
    if args.source:
        try:
            checkpoint = torch.load(args.source)
        except:
            print("Falling back encoding")
            from functools import partial
            import pickle
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            checkpoint = torch.load(args.source, map_location=lambda storage, loc: storage, pickle_module=pickle)


    if args.from_ae:
        if hasattr(checkpoint['net'], "module"): checkpoint['net'] = checkpoint['net'].module
        net_old = checkpoint['net'].encoder
    else:
        net_old = checkpoint['net']

    if args.from_jigsaw:
        mode = 'jigsaw'
    elif args.from_ae:
        mode = 'ae'
    else:
        mode = ''

    if hasattr(net_old, "module"):
        net_old = net_old.module
    if args.from_jigsaw:
        net_old.shuffle = False
    net = models.LinearFromRepModel(net_old, num_classes=num_classes, return_layer=args.return_layer, mode=mode, defined_rep_dim=args.rep_dim, mlp_depth=args.mlp_depth)
    # Freeze all but linear
    for param in net.feature_extractor.parameters():
        param.requires_grad = False

    net = torch.nn.DataParallel(net).cuda()

    start_epoch = 0
    best_acc = -1  # best test accuracy
    results = np.zeros((4,start_epoch+args.nb_epochs))

    net.cuda()
    cudnn.benchmark = True

    args.criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.wd)


    print("Start training")
    for epoch in range(start_epoch, start_epoch+args.nb_epochs):
        utils_pytorch.adjust_learning_rate(optimizer, epoch, args)
        st_time = time.time()
        
        # Training and validation
        train_acc, train_loss = utils_pytorch.train_supervised(epoch, trainloader, net, args, optimizer)
        with torch.no_grad(): test_acc, test_loss, best_acc = utils_pytorch.test_supervised(epoch,valloader, net, best_acc, args, optimizer)
            
        # Record statistics
        results[0:2,epoch] = [train_loss,train_acc]
        results[2:4,epoch] = [test_loss,test_acc]
        np.save(args.expdir+'/results.npy',results)
        print('Epoch lasted {0}'.format(time.time()-st_time))
        sys.stdout.flush()
    best_net = torch.load(args.expdir + 'checkpoint.t7')['net']
    best_acc = -1
    print(args.expdir)
    final_acc, final_loss, _ = utils_pytorch.test_supervised(0, testloader, best_net, best_acc, args, optimizer)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar100', nargs='+', help='Task(s) to be trained')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--wd', default=1., type=float, help='weight decay for the classification layer')
    parser.add_argument('--nb-epochs', default=120, type=int, help='nb epochs')
    parser.add_argument('--step1', default=80, type=int, help='nb epochs before first lr decrease')
    parser.add_argument('--step2', default=100, type=int, help='nb epochs before second lr decrease')
    parser.add_argument('--expdir', default='./tmp/linear', help='Save folder')
    parser.add_argument('--datadir', default='/data/bw462/benchmark_datasets/', help='folder containing data folder')
    parser.add_argument('--source', type=str, help='Network source (use for all)')
    parser.add_argument('--source-dir', type=str, help='Network source (directory with dataset-specific networks)')
    parser.add_argument('--train-batch-size', default=512, type=int, help='samples per train batch')
    parser.add_argument('--test-batch-size', default=100, type=int, help='samples per test batch')
    parser.add_argument('--image-dim', default=64, type=int, help='width/height of image')
    parser.add_argument('--mlp-depth', default=1, type=int, help='Depth of mlp at end, minimum/default 1 is just linear classification')
    parser.add_argument('--from-ae', action='store_true', help='Get net from AE model not classifier')
    parser.add_argument('--from-jigsaw', action='store_true', help='Get net from jigsaw model not classifier')
    parser.add_argument('--rep-dim', nargs='+', default=256, type=int, help='Dimension of representation, 0 = automatically figures it out')
    parser.add_argument('--num-perms', default=100, type=int, help='# of jigsaw permutations')
    parser.add_argument('--return-layer', default=3, type=int, help='After which block to extract features')
    parser.add_argument('--always-save', action='store_true', help='Always save instead of early stopping')
    parser.add_argument('--train-on-10', action='store_true', help='Only use 10% of training labels')
    parser.add_argument('--no-flip', action='store_true', help='Do not augment w/ horiz flips')
    parser.add_argument('--random-labels', action='store_true', help='Randomly reassign training labels and use throughout')
    args = parser.parse_args()

    dataset_list = copy(args.dataset)
    if type(dataset_list)==str: dataset_list = [dataset_list]
    rep_dim_list = copy(args.rep_dim)
    if type(rep_dim_list)==int: rep_dim_list = [rep_dim_list]
    base_expdir = args.expdir
    for dataset in dataset_list:
        for rep_dim in rep_dim_list:
            args.dataset = dataset
            args.rep_dim = rep_dim
            if args.source_dir:
                args.source = '/'.join([args.source_dir, dataset, 'checkpoint.t7'])
                if args.return_layer == 4:
                    args.expdir = '/'.join(args.source.split('/')[:-1]) + '/linear/final_output/'
                else:
                    args.expdir = '/'.join(args.source.split('/')[:-1]) + '/linear/rep_dim_%d/' %args.rep_dim
                if args.train_on_10:
                    args.expdir = args.expdir.replace(dataset, dataset + '10')
                if args.random_labels:
                    args.expdir = args.expdir.replace('linear', 'linear_random')
                args.from_jigsaw = ('jigsaw' in args.source)
                # args.expdir = '/'.join([base_expdir, dataset, ''])
            else:
                args.expdir = base_expdir + '/rep_dim_%d/' %args.rep_dim
            print(args.source, args.expdir, args.dataset, args.rep_dim)
            full_training(args)


if __name__ == '__main__':
    main()
