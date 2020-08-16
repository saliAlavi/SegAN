import torch

def multi_scale_L1_loss(c_output, number_of_critic_scales=4):
  c_output = torch.abs(c_output)
  c_output = torch.sum(c_output, dim=1, keepdim=True)
  c_output = (1/number_of_critic_scales) * c_output
  c_output = torch.mean(c_output)

  return c_output


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

    dice_total=1-1*torch.sum(dice)/dice.size(0)

    return dice_total