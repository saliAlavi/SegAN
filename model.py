import torch.nn.functional as F
import torch.nn as nn
import torch
from math import sqrt

LEAKAGE = 0.001


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, (kernel, kernel), stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        residual = self.conv(x)
        residual = self.batchnorm(residual)
        residual = F.leaky_relu(residual, LEAKAGE)
        return residual
    # def __init__(self, in_dim, out_dim, kernel, stride):
    #     super(ConvBlock, self).__init__()
    #     self.conv = nn.Conv2d(in_dim, out_dim, (kernel, kernel), stride=stride)
    #     self.batchnorm = nn.BatchNorm2d(out_dim)
    #
    # def forward(self, x):
    #     residual = self.conv(x)
    #     residual = self.batchnorm(residual)
    #     residual = F.leaky_relu(residual, LEAKAGE)
    #     residual += x
    #     return residual


class DeconvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride, padding=1, scale_factor=2):
        super(DeconvBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode = 'bilinear')
        self.conv = nn.Conv2d(in_dim, out_dim, (kernel, kernel), stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_dim)
    # def __init__(self, in_dim, out_dim, kernel, stride, padding=1, scale_factor=2):
    #     super(DeconvBlock, self).__init__()
    #     self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
    #     self.conv = nn.Conv2d(in_dim, out_dim, (kernel, kernel), stride=stride, padding=padding)
    #     self.batchnorm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        residual = self.upsample(x)
        residual = self.conv(residual)
        residual = self.batchnorm(residual)
        residual = F.relu(residual)
        return residual
    # def __init__(self, d_in, d_out, kernel, stride):
    #     super(DeconvBlock, self).__init__()
    #     self.deconv = nn.ConvTranspose2d(d_in, d_out, (kernel, kernel), stride=stride)
    #     self.batchnorm = nn.BatchNorm2d(d_out)
    #
    # def forward(self, x):
    #     residual = self.deconv(x)
    #     residual = self.batchnorm(residual)
    #     residual += F.leaky_relu(residual, LEAKAGE)
    #     residual += x
    #     return residual


class Segmentor(nn.Module):
    def __init__(self, d_in, d_out):
        super(Segmentor, self).__init__()

        # Encoder
        self.conv_1 = nn.Conv2d(d_in, 64, kernel_size=4, stride=2, padding=1)
        self.conv_2 = ConvBlock(64, 128, 4, 2, 1)
        self.conv_3 = ConvBlock(128, 256, 4, 2, 1)
        self.conv_4 = ConvBlock(256, 512, 4, 2, 1)

        # Decoder
        self.deconv_1 = DeconvBlock(512, 256, 3, 1, padding=1, scale_factor=2)
        self.deconv_2 = DeconvBlock(256, 128, 3, 1, padding=1, scale_factor=2)
        self.deconv_3 = DeconvBlock(128, 64, 3, 1, padding=1, scale_factor=2)
        self.deconv_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, d_out, 3, 1, padding=1)
        )

        # self.conv_layer_4 = ConvBlock(512, 256, 3, 1)
        # self.conv_layer_5 = ConvBlock(256, 128, 3, 1)
        # self.conv_layer_5 = ConvBlock(128, 64, 3, 1)
        # self.conv_2 = nn.Conv2d(128, 3, 3, 1)

    def forward(self, x):
        res_1 = self.conv_1(x)
        res_1 = F.leaky_relu(res_1)
        res_2 = self.conv_2(res_1)
        res_3 = self.conv_3(res_2)
        res_4 = self.conv_4(res_3)
        res_5 = self.deconv_1(res_4)
        # The "add" connection between corresponding encoder and decoder part
        res_5 = res_5 + res_3
        res_6 = self.deconv_2(res_5)
        # The "add" connection between corresponding encoder and decoder part
        res_6 = res_6 + res_2
        res_7 = self.deconv_3(res_6)
        # The "add" connection between corresponding encoder and decoder part
        res_7 = res_7 + res_1
        res_8 = self.deconv_4(res_7)
        return res_8


# class Critic(nn.Module):
#     def __init__(self, d_in):
#         '''
#         d_in: number of channels in the first convolutional layer.
#         '''
#         super(Critic, self).__init__()
#         self.conv_1 = nn.Conv2d(d_in, 64, kernel_size=4, stride=2, padding=1)
#         self.conv_2 = ConvBlock(64, 128, 4, 2, 1)
#         self.conv_3 = ConvBlock(128, 256, 4, 2, 1)
#     # def __init__(self,d_in):
#     #     super(Critic, self).__init__()
#     #     super(Critic, self).__init__()
#     #     self.conv_1 = nn.Conv2d(d_in, 64, kernel_size=4, stride=2, padding=1)
#     #     self.conv_2 = ConvBlock(64, 128, 4, 2, 1)
#     #     self.conv_3 = ConvBlock(128, 256, 4, 2, 1)
#         # self.conv_1_prd = nn.Conv2d(3, 64, (4, 4))
#         # self.conv_1_gt = nn.Conv2d(3, 64, (4, 4))
#         #
#         # self.conv_layer_1_prd = ConvBlock(3,64,4,2)
#         # self.conv_layer_2_prd = ConvBlock(64, 128, 4, 2)
#         # self.conv_layer_3_prd = ConvBlock(128, 256, 4, 2)
#         #
#         # self.conv_layer_1_gt = ConvBlock(3, 64, 4, 2)
#         # self.conv_layer_2_gt  = ConvBlock(64, 128, 4, 2)
#         # self.conv_layer_3_gt  = ConvBlock(128, 256, 4, 2)
#
#     def forward(self, predicted_seg, gt_seg, input_img):
#         gt_masked_imgs = gt_seg.unsqueeze(1) * input_img
#         predicted_masked_imgs = predicted_seg.unsqueeze(1) * input_img
#
#         # pass the predicted_masked_imgs through the net --> requires gradient
#         res_1 = predicted_masked_imgs
#         res_2 = self.conv_1(res_1)
#         res_3 = self.conv_2(res_2)
#         res_4 = self.conv_3(res_3)
#
#         # Reshaping the resulted images for concatenation
#         res_1 = res_1.reshape((64,-1))
#         res_2 = res_2.reshape((64,-1))
#         res_3 = res_3.reshape((64,-1))
#         res_4 = res_4.reshape((64,-1))
#         # Perform concatenation
#         res_concat = torch.cat((res_1, res_2, res_3, res_4), dim=1)
#
#         # Pass ground truth masked images forward through the network
#         res_gt_1 = gt_masked_imgs
#         res_gt_2 = self.conv_1(res_gt_1)
#         res_gt_3 = self.conv_2(res_gt_2)
#         res_gt_4 = self.conv_3(res_gt_3)
#
#         res_gt_1 = res_gt_1.reshape((64,-1))
#         res_gt_2 = res_gt_2.reshape((64,-1))
#         res_gt_3 = res_gt_3.reshape((64,-1))
#         res_gt_4 = res_gt_4.reshape((64,-1))
#         # Perform concatenation
#         res_gt_concat = torch.cat((res_gt_1, res_gt_2, res_gt_3, res_gt_4), dim=1)
#
#
#         return torch.abs(res_concat - res_gt_concat)


        # pred_mask = image * segmentor_result
        # gt_mask = image * gt
        #
        # res_0_prd = self.conv_1_prd(pred_mask)
        # res_1_prd =self.conv_layer_1_prd(res_0_prd)
        # res_2_prd =self.conv_layer_2_prd(res_1_prd)
        # res_3_prd =self.conv_layer_3_prd(res_2_prd)
        #
        # res_0_gt = self.conv_1_prd(gt_mask)
        # res_1_gt =self.conv_layer_1_prd(res_0_gt)
        # res_2_gt =self.conv_layer_2_prd(res_1_gt)
        # res_3_gt =self.conv_layer_3_prd(res_2_gt)
        #
        # out_prd =torch.cat((res_0_prd, res_1_prd), 1)
        # out_prd = torch.cat((out_prd, res_2_prd), 1)
        # out_prd = torch.cat((out_prd, res_3_prd), 1)
        #
        # out_gt = torch.cat((res_0_gt, res_1_gt), 1)
        # out_gt = torch.cat((out_gt, res_2_gt), 1)
        # out_gt = torch.cat((out_gt, res_3_gt), 1)
        #
        # return out_prd, out_gt

class GlobalConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(GlobalConvBlock, self).__init__()
        pad0 = (kernel_size[0] - 1) / 2
        pad1 = (kernel_size[1] - 1) / 2

        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        #combine two paths
        x = x_l + x_r
        return x


class Critic(nn.Module):
    def __init__(self, d_in):
        super(Critic, self).__init__()
        ndf=64
        self.convblock1 = nn.Sequential(
            # input is (channel_dim) x 128 x 128
            nn.Conv2d(d_in, ndf, 7, 2, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf) x 64 x 64
        )
        self.convblock1_1 = nn.Sequential(
            # state size. (ndf) x 64 x 64
            GlobalConvBlock(ndf, ndf * 2, (13, 13)),
            # nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*2) x 64 x 64
        )
        self.convblock2 = nn.Sequential(
            # state size. (ndf * 2) x 64 x 64
            nn.Conv2d(ndf * 1, ndf * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*2) x 32 x 32
        )
        self.convblock2_1 = nn.Sequential(
            # input is (ndf*2) x 32 x 32
            GlobalConvBlock(ndf * 2, ndf * 4, (11, 11)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*4) x 32 x 32
        )
        self.convblock3 = nn.Sequential(
            # state size. (ndf * 4) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*4) x 16 x 16
        )
        self.convblock3_1 = nn.Sequential(
            # input is (ndf*4) x 16 x 16
            GlobalConvBlock(ndf * 4, ndf * 8, (9, 9)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf * 8) x 16 x 16
        )
        self.convblock4 = nn.Sequential(
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*8) x 8 x 8
        )
        self.convblock4_1 = nn.Sequential(
            # input is (ndf*8) x 8 x 8
            GlobalConvBlock(ndf * 8, ndf * 16, (7, 7)),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*16) x 8 x 8
        )
        self.convblock5 = nn.Sequential(
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*16) x 4 x 4
        )
        self.convblock5_1 = nn.Sequential(
            # input is (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, ndf * 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*32) x 4 x 4
        )
        self.convblock6 = nn.Sequential(
            # state size. (ndf*32) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*32) x 2 x 2
        )
        # self._initialize_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, input):

        batchsize = input.size()[0]
        out1 = self.convblock1(input)
        # out1 = self.convblock1_1(out1)
        out2 = self.convblock2(out1)
        # out2 = self.convblock2_1(out2)
        out3 = self.convblock3(out2)
        # out3 = self.convblock3_1(out3)
        out4 = self.convblock4(out3)
        # out4 = self.convblock4_1(out4)
        out5 = self.convblock5(out4)
        # out5 = self.convblock5_1(out5)
        out6 = self.convblock6(out5)
        # out6 = self.convblock6_1(out6) + out6
        output = torch.cat((input.view(batchsize,-1),1*out1.view(batchsize,-1),
                            2*out2.view(batchsize,-1),2*out3.view(batchsize,-1),
                            2*out4.view(batchsize,-1),2*out5.view(batchsize,-1),
                            4*out6.view(batchsize,-1)),1)

        return output