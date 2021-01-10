import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import time
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import sys

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every X epochs"""
    if epoch >= args.step2: 
        lr = args.lr * 0.01
    elif epoch >= args.step1:
        lr = args.lr * 0.1
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Return training classes
    return range(len(args.dataset))

# Training
def train_supervised(epoch, trainloader, net, args, optimizer,list_criterion=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
        
    for batch_idx, (inputs,targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = inputs.cuda(async=True), targets.cuda(async=True)
        optimizer.zero_grad()
        outputs = net(inputs)
        # measure accuracy and record loss
        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(targets.data).cpu().sum()
        top1.update(correct*100./targets.size(0), targets.size(0))     
        # apply gradients   
        nuc_loss = 0
        if hasattr(args, 'use_nuc_loss') and args.use_nuc_loss:
            features = net(inputs, return_layer=3).squeeze()
            norm = features.pow(2).sum(1, keepdim=True).pow(1./2)
            features = features.div(norm)
            factor = min(targets.size(0), (targets.size(0) * features.size(1)) ** 0.5)
            nuc_loss = -1 * args.nuc_weight * torch.norm(features, p="nuc") / factor
        loss = args.criterion(outputs, targets) + nuc_loss
        loss.backward()
        losses.update(loss.item(), targets.size(0))
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_idx % 50 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                   epoch, batch_idx, len(trainloader), batch_time=batch_time,
                   data_time=data_time))
            print('Task {0} : Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {top1.val:.3f} ({top1.avg:.3f})'.format(len(trainloader), loss=losses, top1=top1))
            sys.stdout.flush()
    return top1.avg, losses.avg




def test_supervised(epoch, testloader, net, best_acc, args, optimizer):
    net.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    print('Epoch: [{0}]'.format(epoch))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = args.criterion(outputs, targets)
        
        losses.update(loss.item(), targets.size(0))
        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(targets.data).cpu().sum()
        top1.update(correct*100./targets.size(0), targets.size(0))
    
    print('Task: Test Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Test Acc {top1.val:.3f} ({top1.avg:.3f})'.format(loss=losses, top1=top1))
    # Save checkpoint.
    acc = top1.avg
    if acc > best_acc or args.always_save:
        print('Saving..')
        state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, args.expdir +'checkpoint.t7')
        best_acc = acc
    
    return top1.avg, losses.avg, best_acc


def train_inst_disc(epoch, trainloader, net, args, optimizer,list_criterion=None):
    print('\nEpoch: %d' % epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    # switch to train mode
    net.train()
    end = time.time()
    for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
        data_time.update(time.time() - end)
        inputs, indexes = inputs.cuda(), indexes.cuda()
        optimizer.zero_grad()

        features = net(inputs)
        outputs = args.train_lemniscate(features, indexes)
        loss = args.criterion(outputs, indexes)

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}]'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
                  epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss))
    return train_loss.avg, train_loss.avg

def test_inst_disc(epoch, testloader, net, best_loss, args, optimizer):
    print('\nEpoch: %d' % epoch)
    test_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    # switch to test mode
    net.eval()
    end = time.time()
    for batch_idx, (inputs, _, indexes) in enumerate(testloader):
        data_time.update(time.time() - end)
        inputs, indexes = inputs.cuda(), indexes.cuda()

        features = net(inputs)
        outputs = args.test_lemniscate(features, indexes)
        loss = args.criterion(outputs, indexes)
        loss.backward()
        test_loss.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch: [{}][{}/{}]'
          'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
          'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
          'Loss: {test_loss.val:.4f} ({test_loss.avg:.4f})'.format(
          epoch, batch_idx, len(testloader), batch_time=batch_time, data_time=data_time, test_loss=test_loss))
    current_loss = test_loss.avg
    if current_loss < best_loss:
        print('Saving..')
        state = {
            'net': net,
            'loss':current_loss,
            'epoch': epoch,
        }
        torch.save(state, args.expdir +'checkpoint.t7')
        best_loss = current_loss
    return test_loss.avg, test_loss.avg, best_loss


def rotated_imgs_and_targets(imgs):
    num_samples = imgs.shape[0]
    imgs_90 = imgs.transpose(2, 3).flip(2)
    imgs_180 = imgs.flip(2).flip(3)
    imgs_270 = imgs.flip(2).transpose(2,3)
    all_imgs = torch.cat([imgs, imgs_90, imgs_180, imgs_270])
    labels = torch.LongTensor([0]*num_samples + [1]*num_samples + [2]*num_samples + [3]*num_samples)
    return all_imgs, labels


def train_rot(epoch, trainloader, net, args, optimizer,list_criterion=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, _) in enumerate(trainloader):
        data_time.update(time.time() - end)
        inputs, targets = rotated_imgs_and_targets(inputs)
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = args.criterion(outputs, targets)
        # measure accuracy and record loss
        losses.update(loss.item(), targets.size(0))
        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(targets.data).cpu().sum()
        top1.update(correct*100./targets.size(0), targets.size(0))
        # apply gradients
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_idx % 50 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                   epoch, batch_idx, len(trainloader), batch_time=batch_time,
                   data_time=data_time))
            print('Task: Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {top1.val:.3f} ({top1.avg:.3f})'.format(loss=losses, top1=top1))
            sys.stdout.flush()
    return top1.avg, losses.avg

def test_rot(epoch, testloader, net, best_acc, args, optimizer):
    net.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    for batch_idx, (inputs, _) in enumerate(testloader):
        inputs, targets = rotated_imgs_and_targets(inputs)
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = args.criterion(outputs, targets)
        
        losses.update(loss.item(), targets.size(0))
        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(targets.data).cpu().sum()
        top1.update(correct*100./targets.size(0), targets.size(0))
    
    print('Task: Test Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Test Acc {top1.val:.3f} ({top1.avg:.3f})'.format(loss=losses, top1=top1))
    # Save checkpoint.
    acc = top1.avg
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, args.expdir +'checkpoint.t7')
        best_acc = acc
    
    return top1.avg, losses.avg, best_acc



def train_ae(epoch, trainloader, net, args, optimizer,list_criterion=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, _) in enumerate(trainloader):
        data_time.update(time.time() - end)
        inputs = inputs.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = args.criterion(outputs, inputs)
        # measure accuracy and record loss
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(loss.data.item(), inputs.size(0))
        # apply gradients
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 50 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                   epoch, batch_idx, len(trainloader), batch_time=batch_time,
                   data_time=data_time))
            print('Task: Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {top1.val:.3f} ({top1.avg:.3f})'.format(loss=losses, top1=top1))
            sys.stdout.flush()
    return top1.avg, losses.avg



def test_ae(epoch, testloader, net, best_loss, args, optimizer):
    net.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    print('Epoch: [{0}]'.format(epoch))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.cuda()
        outputs = net(inputs)
        loss = args.criterion(outputs, inputs)

        losses.update(loss.data.item(), inputs.size(0))
        top1.update(loss.data.item(), inputs.size(0))

    print('Task: Test Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Test Acc {top1.val:.3f} ({top1.avg:.3f})'.format(loss=losses, top1=top1))
    # Save checkpoint.
    current_loss = losses.avg
    if current_loss < best_loss:
        print('Saving..')
        state = {
            'net': net,
            'loss':current_loss,
            'epoch': epoch,
        }
        torch.save(state, args.expdir +'checkpoint.t7')
        best_loss = current_loss
    chosen_index = epoch
    image_index = min(epoch%10, inputs.size(0)-1)
    combined_tensor = torch.cat((inputs[image_index], outputs[image_index]), dim=2)
    combined_image = 0.5 * combined_tensor.permute(1, 2, 0).detach().cpu().numpy() + 0.5
    plt.figure()
    plt.imshow(combined_image)
    plt.savefig('%s/figs/%d.png' %(args.expdir,epoch))
    plt.close()
    return top1.avg, losses.avg, best_loss

def train_jigsaw(epoch, trainloader, net, args, optimizer,list_criterion=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    net.shuffle = True
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, _) in enumerate(trainloader):
        data_time.update(time.time() - end)
        inputs = inputs.cuda()
        optimizer.zero_grad()
        outputs, targets = net(inputs)
        targets = targets.cuda()
        loss = args.criterion(outputs, targets)
        losses.update(loss.item(), targets.size(0))
        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(targets.data).cpu().sum()
        top1.update(correct*100./targets.size(0), targets.size(0))
        # apply gradients•••
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 50 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                   epoch, batch_idx, len(trainloader), batch_time=batch_time,
                   data_time=data_time))
            print('Task: Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {top1.val:.3f} ({top1.avg:.3f})'.format(loss=losses, top1=top1))
            sys.stdout.flush()

    return top1.avg, losses.avg

def test_jigsaw(epoch, testloader, net, best_acc, args, optimizer):
    net.eval()
    net.shuffle = False
    losses = AverageMeter()
    top1 = AverageMeter()
    for batch_idx, (inputs, _) in enumerate(testloader):
        inputs = inputs.cuda()
        outputs, targets = net(inputs)
        loss = args.criterion(outputs, targets)

        losses.update(loss.item(), targets.size(0))
        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(targets.data).cpu().sum()
        top1.update(correct*100./targets.size(0), targets.size(0))

    print('Task: Test Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Test Acc {top1.val:.3f} ({top1.avg:.3f})'.format(loss=losses, top1=top1))
    # Save checkpoint.
    acc = top1.avg
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, args.expdir +'checkpoint.t7')
        best_acc = acc

    return top1.avg, losses.avg, best_acc
