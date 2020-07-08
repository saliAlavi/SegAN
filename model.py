import torch.nn.functional as F
import torch.nn as nn
import torch

LEAKAGE = 0.01


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, (kernel, kernel), stride=stride)
        self.batchnorm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        residual = self.conv(x)
        residual = self.batchnorm(residual)
        residual = F.leaky_relu(residual, LEAKAGE)
        residual += x
        return residual


class DeconvBlock(nn.Module):
    def __init__(self, d_in, d_out, kernel, stride):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(d_in, d_out, (kernel, kernel), stride=stride)
        self.batchnorm = nn.BatchNorm2d(d_out)

    def forward(self, x):
        residual = self.deconv(x)
        residual = self.batchnorm(residual)
        residual += F.leaky_relu(residual, LEAKAGE)
        residual += x
        return residual


class Segmentor(nn.Module):
    def __init__(self, d_in, d_out):
        super(Segmentor, self).__init__()
        self.conv_1 = nn.Conv2d(3, 64, (4, 4), stride=2)
        self.conv_layer_1 = ConvBlock(64, 128, 4, 2)
        self.conv_layer_2 = ConvBlock(128, 256, 4, 2)
        self.conv_layer_3 = ConvBlock(256, 512, 4, 2)
        self.conv_layer_4 = ConvBlock(512, 256, 3, 1)
        self.conv_layer_5 = ConvBlock(256, 128, 3, 1)
        self.conv_layer_5 = ConvBlock(128, 64, 3, 1)
        self.conv_2 = nn.Conv2d(128, 3, 3, 1)

    def forward(self, x):
        res_0 = self.conv_1(x)
        res_0 = F.leaky_relu(res_0, LEAKAGE)
        res_1 = self.conv_layer_1(res_0)
        res_2 = self.conv_layer_2(res_1)
        res_3 = self.conv_layer_3(res_2)
        res_4 = self.conv_layer_4(res_3)
        res_4 = F.upsample_bilinear(res_4, scale_factor=2)
        res_4 += res_2
        res_5 = self.conv_layer_4(res_4)
        res_5 = F.upsample_bilinear(res_5, scale_factor=2)
        res_5 += res_1
        res_6 = self.conv_layer_4(res_5)
        res_6 = F.upsample_bilinear(res_6, scale_factor=2)
        res_6 += res_0
        output = self.conv_2(res_6)
        output = F.upsample_bilinear(output, scale_factor=2)
        return output


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv_1_prd = nn.Conv2d(3, 64, (4, 4))
        self.conv_1_gt = nn.Conv2d(3, 64, (4, 4))

        self.conv_layer_1_prd = ConvBlock(3,64,4,2)
        self.conv_layer_2_prd = ConvBlock(64, 128, 4, 2)
        self.conv_layer_3_prd = ConvBlock(128, 256, 4, 2)

        self.conv_layer_1_gt = ConvBlock(3, 64, 4, 2)
        self.conv_layer_2_gt  = ConvBlock(64, 128, 4, 2)
        self.conv_layer_3_gt  = ConvBlock(128, 256, 4, 2)

    def forward(self, image, segmentor_result, gt):
        pred_mask = image * segmentor_result
        gt_mask = image * gt

        res_0_prd = self.conv_1_prd(pred_mask)
        res_1_prd =self.conv_layer_1_prd(res_0_prd)
        res_2_prd =self.conv_layer_2_prd(res_1_prd)
        res_3_prd =self.conv_layer_3_prd(res_2_prd)

        res_0_gt = self.conv_1_prd(gt_mask)
        res_1_gt =self.conv_layer_1_prd(res_0_gt)
        res_2_gt =self.conv_layer_2_prd(res_1_gt)
        res_3_gt =self.conv_layer_3_prd(res_2_gt)

        out_prd =torch.cat((res_0_prd, res_1_prd), 1)
        out_prd = torch.cat((out_prd, res_2_prd), 1)
        out_prd = torch.cat((out_prd, res_3_prd), 1)

        out_gt = torch.cat((res_0_gt, res_1_gt), 1)
        out_gt = torch.cat((out_gt, res_2_gt), 1)
        out_gt = torch.cat((out_gt, res_3_gt), 1)

        return out_prd, out_gt
