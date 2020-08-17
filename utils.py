import torch
import os


def segmentor_evaluation(s_output, s_target):
    """
    This function creates the batch_averaged FP(flase positive), TP(true positive), FN(flase negative) and TN(true negative) values for the segmented output
    THE ELEMENTS OF THE INPUTS ARE ASSUMED TO BE 0 OR 1.
    """
    batch_size = s_output.size()[0]
    s_output_inversed = custom_replace(s_output, 1, 0)
    s_target_inversed = custom_replace(s_target, 1, 0)
    # The true_positive value where s_output and s_target are both 1
    TP = (1) / (batch_size) * (torch.sum(s_output * s_target))

    # The false_positive value  is where s_output=1 and s_target=0(i.e.  s_target_inversed=1)
    FP = (1) / (batch_size) * torch.sum(s_output * s_target_inversed)

    # The true_negative value is where s_output=0 (i.e. s_output_inversed=1) and s_target=0 (i.e. s_target_inversed=1)
    TN = (1) / (batch_size) * torch.sum(s_output_inversed * s_target_inversed)

    # The false_negative is where s_output=0 (i.e. s_output_inversed=1) and s_target=1
    FN = (1) / (batch_size) * torch.sum(s_output_inversed * s_target)

    return TP, FP, TN, FN


def recall(TP, FN):
    '''
    Computes the recall value of the model
    '''
    recall = (TP) / (TP + FN)
    return recall


def precision(TP, FP):
    '''
    Computes the precision value of the model
    '''
    precision = (TP) / (TP + FP)
    return precision


def false_positive_rate(FP, TN):
    '''
    Computes the false positive rate of the model
    '''
    fpr = (FP) / (FP + TN)
    return fpr


def accuracy(TP, FP, TN, FN):
    '''
    Computes the accuracy of the model
    '''
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    return accuracy


def custom_replace(tensor, on_zero, on_non_zero):
    # we create a copy of the original tensor,
    # because of the way we are replacing them.
    res = tensor.clone()
    res[tensor == 0] = on_zero
    res[tensor != 0] = on_non_zero
    return res


def make_dir(path):
    if (not os.path.isdir(path)):
        os.makedirs(path)
