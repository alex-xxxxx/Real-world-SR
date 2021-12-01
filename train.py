import os, argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import Img_Dataset
import dataloader

import utils
from EDSR.edsr import EDSR
from modules import DSN
from adaptive_gridsampler.gridsampler import Downsampler
from skimage.color import rgb2ycbcr

import dataloader

parser = argparse.ArgumentParser(description='Content Adaptive Resampler for Image downscaling')
parser.add_argument('--model_dir', type=str, default='./models', help='path to the pre-trained model')
parser.add_argument('--img_dir', type=str, help='path to the HR images to be downscaled')
parser.add_argument('--mode', type=str, default='DIV2K_train', help='dataset mode')
parser.add_argument('--scale', type=int, default=4, help='downscale factor')
parser.add_argument('--output_dir', type=str, help='path to store results')
parser.add_argument('--benchmark', type=bool, default=False, help='report benchmark results')
parser.add_argument('--trainimg_noise', type=str, help='path to training noise images')
parser.add_argument('--trainimg_clean', type=str, help='path to training HR images')
parser.add_argument('--valimg_noise', type=str, help='path to validation noise images')
parser.add_argument('--valimg_clean', type=str, help='path to validation HR images')
parser.add_argument('--savemodel_dir', type = str, help='save models to path')
parser.add_argument('--saveval_img', type = str, help='save validation images to path')
parser.add_argument('--batch_size', type = int, default = 10,help='number of batches loaded')
parser.add_argument('--epochs', type = int, default = 10,help='number of training epochs')
args = parser.parse_args()

dataset_opt = {}
dataset_opt['phase'] = 'train'

val_dataset_opt = {}
val_dataset_opt['phase'] = 'val'


saveval_img = args.saveval_img
SCALE = args.scale
KSIZE = 3 * SCALE + 1
OFFSET_UNIT = SCALE
BENCHMARK = args.benchmark
lr = 1e-4

kernel_generation_net = DSN(k_size=KSIZE, scale=SCALE).cuda()
downsampler_net = Downsampler(SCALE, KSIZE).cuda()
upscale_net = EDSR(32, 256, scale=SCALE).cuda()
#l1_loss = nn.MSELoss().cuda()
#opt_para = list(kernel_generation_net.parameters()) + list(downsampler_net.parameters()) + list(upscale_net.parameters())
#optim = torch.optim.Adam(opt_para, lr=lr, weight_decay=1e-6, betas=(0.9, 0.999))


kernel_generation_net =  nn.DataParallel(kernel_generation_net, [0])
downsampler_net = nn.DataParallel(downsampler_net, [0]) 
upscale_net = nn.DataParallel(upscale_net, [0]) 
torch.set_grad_enabled(True)
# kernel_generation_net.load_state_dict(torch.load(os.path.join(args.model_dir, '{0}x'.format(SCALE), 'kgn.pth')))
# upscale_net.load_state_dict(torch.load(os.path.join(args.model_dir, '{0}x'.format(SCALE), 'usn.pth')))
#torch.set_grad_enabled(True)

kernel_generation_net.load_state_dict(torch.load(os.path.join(args.model_dir, '{0}x'.format(SCALE), 'kgn.pth')))
upscale_net.load_state_dict(torch.load(os.path.join(args.model_dir, '{0}x'.format(SCALE), 'usn.pth')))

torch.set_grad_enabled(True)

opt_para = list(kernel_generation_net.parameters()) + list(downsampler_net.parameters()) + list(upscale_net.parameters())
optim = torch.optim.Adam(opt_para, lr=lr, weight_decay=1e-6, betas=(0.9, 0.999))

class QuantForBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = torch.clamp(input, 0, 1)
        output = (input * 255.).round() / 255.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        pi = np.pi
        const = 0.5
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = grad_input * (1 - const * torch.cos(2 * pi * input))
        return grad_input


class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input):
        return QuantForBackward.apply(input)


class ReconstructionLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super(ReconstructionLoss, self).__init__()
        self.eps = eps

    def forward(self, x, target):
        return torch.mean(torch.sqrt((x - target) * (x - target) + self.eps))


def validation(noise_img, gt_img, epoch, img_cnt):
    kernel_generation_net.eval()
    downsampler_net.eval()
    upscale_net.eval()
    
    kernels, offsets_h, offsets_v = kernel_generation_net(noise_img)
    downscaled_img = downsampler_net(noise_img, kernels, torch.zeros(offsets_h.shape), torch.zeros(offsets_v.shape),
                                     OFFSET_UNIT)
    downscaled_img = torch.clamp(downscaled_img, 0, 1)
    downscaled_img = torch.round(downscaled_img * 255)

    reconstructed_img = upscale_net(downscaled_img / 255.0)

    img = gt_img * 255
    img = img.data.cpu().numpy().transpose(0, 2, 3, 1)
    img = np.uint8(img)

    reconstructed_img = torch.clamp(reconstructed_img, 0, 1) * 255
    reconstructed_img = reconstructed_img.data.cpu().numpy().transpose(0, 2, 3, 1)
    reconstructed_img = np.uint8(reconstructed_img)

    orig_img = img[0, ...].squeeze()
    recon_img = reconstructed_img[0, ...].squeeze()

    img = Image.fromarray(recon_img)
    img.save(os.path.join(saveval_img, 'e_'+str(epoch)+'_'+str(img_cnt)+'_recon.png'))
    psnr = utils.cal_psnr(orig_img[SCALE:-SCALE, SCALE:-SCALE, ...], recon_img[SCALE:-SCALE, SCALE:-SCALE, ...],
                          benchmark=BENCHMARK)

    return psnr

l1_loss = ReconstructionLoss().cuda()
Quant = Quantization()


def train(img_noise, img_gt, save_imgs=False, save_dir=None):
    optim.zero_grad()
    # generate kernels
    kernels, offsets_h, offsets_v = kernel_generation_net(img_noise)
    
    # downscale
    downscaled_img = downsampler_net(img_noise, kernels, offsets_h, offsets_v, OFFSET_UNIT)
    downscaled_img = Quant(downscaled_img)

    # upscale
    reconstructed_img = upscale_net(downscaled_img)

    l1 = l1_loss(reconstructed_img, img_gt)
    l1.backward()
    optim.step()
    return l1


if __name__ == '__main__':

    #train_set = create_dataset(dataset_opt)
    trainpath_noise = args.trainimg_noise
    trainpath_gt = args.trainimg_clean
    valpath_noise = args.valimg_noise
    valpath_gt = args.valimg_clean
    savepth = args.savemodel_dir
    train_set = Img_Dataset.ImageDataset(trainpath_noise, trainpath_gt)
    train_loader = dataloader.create_dataloader(train_set, 'train', args.batch_size, 24)
    val_set = Img_Dataset.ImageDataset(valpath_noise, valpath_gt)
    val_loader = dataloader.create_dataloader(val_set, 'val', args.batch_size, 24)
    pre_avg_psnr = 0
    for i in range(args.epochs):
        for data in train_loader:
            loss = train(data['noise'].cuda(), data['GT'].cuda())
        if i % 1 == 0:
            print(i, loss, lr)
            psnr_list = []
            val_cnt = 0
            for val in val_loader:
                psnr = validation(val['noise'], val['GT'], i, val_cnt)
                val_cnt += 1
                psnr_list.append(psnr)
            avg_psnr = np.mean(psnr_list)
            print('Mean PSNR(goal>28.34): {0:.2f}'.format(avg_psnr))

            if i % 4000 == 0 and i != 0:
                
                for param_group in optim.param_groups:
                    lr = lr * 0.5
                    param_group['lr'] = lr
                pre_avg_psnr = avg_psnr

            kernel_generation_net.train()
            downsampler_net.train()
            upscale_net.train()
        
        torch.save(kernel_generation_net.state_dict(), savepth + str(i) + '_' + 'kgn.pth')
        torch.save(downsampler_net.state_dict(), savepth + str(i) + '_' + 'dsn.pth')
        torch.save(upscale_net.state_dict(), savepth + str(i) + '_' + 'usn.pth')
