import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import resnet26

image_dim_to_params = {
                        64 :{"edge_length":18, "split_length":21, "jitter":1},
                      }
for image_dim in image_dim_to_params:
    image_dim_to_params[image_dim]["gap"] = image_dim_to_params[image_dim]["split_length"] - image_dim_to_params[image_dim]["edge_length"]

def SiameseNet(code_dim=256):
    return resnet26(num_classes=[code_dim])

class ShufflerWithTargets(nn.Module):
    
    def __init__(self, num_perms=100, gray_prob=0.3, image_dim=64):
        super(ShufflerWithTargets, self).__init__()
        if num_perms < 1000:
            self.perms = np.load('npy/naroozi_perms_%d_patches_9_max.npy' %num_perms)
        else:
            self.perms = np.load('npy/hamming_perms_%d_patches_9_max_avg.npy' %num_perms)
        self.gray_prob = gray_prob
        self.gap = image_dim_to_params[image_dim]["gap"]
        self.split_length = image_dim_to_params[image_dim]["split_length"]
        self.edge_length = image_dim_to_params[image_dim]["edge_length"]
    
    def extract_random_patches(self, x, i):
        tile_list = []
        jitter = image_dim_to_params[image_dim]["jitter"]
        for j in range(3):
            for k in range(3):
                base_horiz_offset = np.random.randint(self.gap - jitter)
                base_vert_offset = np.random.randint(self.gap - jitter)
                channel_tiles = []
                for channel_i in range(3):
                    horiz_jitter, vert_jitter = np.random.randint(2*jitter +1, size=2)
                    horiz_offset = base_horiz_offset + horiz_jitter
                    vert_offset = base_vert_offset + vert_jitter
                    channel_tile = x[i, channel_i, j * self.split_length + horiz_offset : j * self.split_length + horiz_offset + self.edge_length, \
                                     k * self.split_length + vert_offset : k * self.split_length + vert_offset + self.edge_length]
                    channel_tiles.append(channel_tile)
                tile = torch.stack(channel_tiles)
                tile_list.append(tile)
        return torch.stack(tile_list)

    def forward(self, x):
        # Assume x is n x 3 x 64 x 64
        targets = []
        tiles = []
        for i in range(x.size(0)):
            t_list = self.extract_random_patches(x, i)
            for tile_i in range(9):
                if torch.rand(1).item() < self.gray_prob:
                    # print(t_list[tile_i][0][0][0])
                    t_list[tile_i] = t_list[tile_i].mean(0).repeat(3, 1, 1)
                    # print(t_list[tile_i][0][0][0])
                    # print(t_list[tile_i].min(), t_list[tile_i].max())
                t_list[tile_i] = (t_list[tile_i] - t_list[tile_i].mean()) / (t_list[tile_i].std()+ 1e-5)
                if t_list[tile_i].abs().max() > 1e10:
                    print(t_list[tile_i].min(), t_list[tile_i].max())
                    asdfasdfasdf
            perm_i = np.random.randint(self.perms.shape[0])
            perm = self.perms[perm_i] - 1
            t_list = t_list[perm]
            targets.append(perm_i)
            tiles.append(t_list)
            
        targets = torch.LongTensor(targets)
        tiles = torch.stack(tiles)
        if not (tiles==tiles).all(): asdf
        return tiles, targets

class Tiler(nn.Module):
    
    def __init__(self, image_dim=64):
        super(Tiler, self).__init__()
        self.gap = image_dim_to_params[image_dim]["gap"]
        self.split_length = image_dim_to_params[image_dim]["split_length"]
        self.edge_length = image_dim_to_params[image_dim]["edge_length"]

    def extract_set_patches(self, x):
        tile_list = []
        horiz_offset = self.gap // 2
        vert_offset = self.gap // 2
        for j in range(3):
            for k in range(3):
                tile = x[:, :, j * self.split_length + horiz_offset : j * self.split_length + horiz_offset + self.edge_length, \
                            k * self.split_length + vert_offset : k * self.split_length + vert_offset + self.edge_length]
                tile_list.append(tile)
        return torch.stack(tile_list)

    def forward(self, x):
        # Assume x is n x 3 x 64 x 64
        # All of these are n x 3 x edge_length x edge_length
        x = self.extract_set_patches(x)
        return x

class JigsawModel(nn.Module):
    
    def __init__(self, code_dim=256, num_perms=100, shuffle=True, gray_prob=0.3, image_dim=64):
        super(JigsawModel, self).__init__()
        self.code_dim = code_dim
        if image_dim==64:
            self.conv = SiameseNet(code_dim=code_dim)
        self.fc1 = torch.nn.Sequential(
                        torch.nn.Linear(9 * code_dim, 1024),
                        torch.nn.ReLU(inplace=True))
        self.fc2 = torch.nn.Linear(1024, num_perms)
        self.tiler = Tiler(image_dim=image_dim)
        self.shuffler = ShufflerWithTargets(num_perms=num_perms, gray_prob=gray_prob, image_dim=image_dim)
        self.shuffle = shuffle
        self.num_perms = num_perms

    def forward(self, x, return_layer=7, rep_dim=0):
        if return_layer <= 4:
            return self.conv(x, return_layer=return_layer, rep_dim=rep_dim)
        if self.shuffle:
            x, targets = self.shuffler(x)
            x = x.transpose(0, 1)
            assert return_layer==7
        else:
            x = self.tiler(x)
        num_tiles, num_samples, c, h, w = x.size()
        x_list = []
        for i in range(num_tiles):
            dummy_return_layer = 4 if return_layer >= 4 else return_layer
            z = self.conv(x[i], return_layer=dummy_return_layer)
            z = z.view([num_samples, -1])
            x_list.append(z)
        x = torch.cat(x_list, 1)
        if return_layer==5: return x.cuda()
        x = self.fc1(x)
        if return_layer==6: return x.cuda()
        x = self.fc2(x)
        if self.shuffle:
            return x.cuda(), targets.cuda()
        else:
            return x.cuda()

