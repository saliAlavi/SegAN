import torch

def multi_scale_L1_loss(c_output, number_of_critic_scales=4):
  '''
  The critic_output is the difference between the concatenated vectors(Multi-scale) of the generated segmentation passed through the critic and the ground truth segmentation passed through it.
  the result will be a vector of size: (Batch_size*_)
  '''

  c_output = torch.abs(c_output)
  c_output = torch.sum(c_output, dim=1, keepdim=True)
  c_output = (1/number_of_critic_scales) * c_output
  c_output = torch.mean(c_output)

  return c_output


# def loss_dice(source, target):
#     assert source.size() == target.size()
#     assert source.dim() == 4
#
#     cross = source * target
#     cross = torch.sum(cross, dim=3)
#     cross = torch.sum(cross, dim=2)
#
#     auto_1 = source * source
#     auto_1 = torch.sum(auto_1, dim=3)
#     auto_1 = torch.sum(auto_1, dim=2)
#
#     auto_2 = source * source
#     auto_2 = torch.sum(auto_1, dim=3)
#     auto_2 = torch.sum(auto_1, dim=2)
#
#     loss = cross / (auto_1 + auto_2)
#     loss /= loss.size(0)
#
#     return loss
#
#
# def dice_loss(s_output, s_target):
#     '''
#     s_output: Segmentor's output
#     s_target: Segmentor's target --> the desired segmentation
#     '''
#
#     num = s_output * s_target
#     den1 = s_output * s_output
#     den2 = s_target * s_target
#
#     dice = (2 * torch.sum(num)) / (torch.sum(den1) + torch.sum(den2))
#
#     # divide by batch size
#     dice = 1 / (s_output.size()[0]) * dice
#
#     return 1 - dice

def dice_loss(input,target):
    num=input*target
    num=torch.sum(num,dim=2)
    num=torch.sum(num,dim=2)

    den1=input*input
    den1=torch.sum(den1,dim=2)
    den1=torch.sum(den1,dim=2)

    den2=target*target
    den2=torch.sum(den2,dim=2)
    den2=torch.sum(den2,dim=2)

    dice=2*(num/(den1+den2))

    dice_total=1-1*torch.sum(dice)/dice.size(0)#divide by batchsize

    return dice_total