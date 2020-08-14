import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from model import *
from create_dataloader import *
import torch.optim as optim
from utils import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from loss import *

if __name__ == "__main__":
    args = None

    dl_BraT_train = None
    dl_BraT_val = None
    d_in = 3
    d_out = 1
    learning_rate = 0.00002
    threshold = 0.5
    train_gpu = True
    batch_size = 2
    epochs = 10
    lr = 0.02
    beta1 = 0.5
    writer = SummaryWriter(log_dir="./runs")

    # NetS = Segmentor()
    # NetC = Critic()

    torch.manual_seed(100)
    # Instantiate the Segmentor(3 output channels) and 3 Critics (1 input channel)
    segmentor = Segmentor(3, 3)
    critic_0 = Critic(3)
    critic_1 = Critic(3)
    critic_2 = Critic(3)

    torch.manual_seed(42)
    # Define the optimizers
    s_parameters = segmentor.parameters()
    # optimizer_s = torch.optim.RMSprop(s_parameters, lr=learning_rate)
    optimizer_s = optim.Adam(s_parameters, lr=lr, betas=(beta1, 0.999))

    c_parameters = list(critic_0.parameters()) + list(critic_1.parameters()) + list(critic_2.parameters())
    # optimizer_c = torch.optim.RMSprop(c_parameters, lr=learning_rate)
    optimizer_c = optimizer_s = optim.Adam(c_parameters, lr=lr, betas=(beta1, 0.999))
    # if torch.cuda.is_available():
    if train_gpu:
        segmentor = segmentor.cuda()
        critic_0 = critic_0.cuda()
        critic_1 = critic_1.cuda()
        critic_2 = critic_2.cuda()

    # for i in range(epochs):
    # Set the model to train mode
    segmentor.train()
    critic_0.train()
    critic_1.train()
    critic_2.train()

    flair_imgs_dir = './dataset/flair'
    t1ce_imgs_dir = './dataset/t1ce'
    t2_imgs_dir = './dataset/t2'
    gt_imgs_dir = './dataset/segmentation'

    train_tuple, validation_tuple, test_tuple = get_data(flair_imgs_dir=flair_imgs_dir, t1ce_imgs_dir=t1ce_imgs_dir,
                                                         t2_imgs_dir=t2_imgs_dir, gt_imgs_dir=gt_imgs_dir)

    torch.manual_seed(42)
    # Instanciating train, validation and test datasets
    train_dataset = Train_Dataset(train_tuple[0], train_tuple[1], train_tuple[2], train_tuple[3])
    val_dataset = vld_tst_Dataset(validation_tuple[0], validation_tuple[1], validation_tuple[2], validation_tuple[3])
    test_dataset = vld_tst_Dataset(test_tuple[0], test_tuple[1], test_tuple[2], test_tuple[3])

    torch.manual_seed(100)
    # Define the Dataloader for each dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)
    for i in range(epochs):
        # Set the model to train mode
        segmentor.train()
        critic_0.train()
        critic_1.train()
        critic_2.train()
        for j, (input_img, gt_3d) in enumerate(train_loader):
            # print(input_img.shape)
            # print(gt_3d.shape)

            # Transfer loaded data to GPU if cuda is available
            # if torch.cuda.is_available:
            if train_gpu:
                input_img = input_img.cuda()
                gt_3d = gt_3d.cuda()

            # "TRAIN C NET"
            output = segmentor(input_img)
            # output = F.sigmoid(output*k)
            output = F.sigmoid(output)

            output = output.detach()
            output_masked = input_img.clone()
            input_mask = input_img.clone()
            # detach G from the network
            for d in range(3):
                output_masked[:, d, :, :] = input_mask[:, d, :, :] * output[:, d, :, :]
            if train_gpu:
                output_masked = output_masked.cuda()

            target_masked = input_img.clone()
            for d in range(3):
                target_masked[:, d, :, :] = input_mask[:, d, :, :] * gt_3d[:, d, :, :]
            if train_gpu:
                target_masked = target_masked.cuda()

            result_C_0 = critic_0(output_masked)
            result_C_1 = critic_1(output_masked)
            result_C_2 = critic_2(output_masked)

            target_C_0 = critic_0(target_masked)
            target_C_1 = critic_1(target_masked)
            target_C_2 = critic_2(target_masked)

            loss_C = - torch.mean(torch.abs(result_C_0 - target_C_0) + torch.abs(result_C_1 - target_C_1) + torch.abs(
                result_C_2 - target_C_2))
            loss_C.backward()
            optimizer_c.step()

            # clip parameters in D
            for p in critic_0.parameters():
                p.data.clamp_(-0.05, 0.05)

            for p in critic_1.parameters():
                p.data.clamp_(-0.05, 0.05)

            for p in critic_2.parameters():
                p.data.clamp_(-0.05, 0.05)

            # train G
            segmentor.zero_grad()
            output = segmentor(input_img)
            output = F.sigmoid(output)

            output_masked = input_img.clone()
            for d in range(3):
                output_masked[:, d, :, :] = input_mask[:, d, :, :] * output[:, d, :, :]
            if train_gpu:
                output_masked = output_masked.cuda()

            for d in range(3):
                target_masked[:, d, :, :] = input_mask[:, d, :, :] * gt_3d[:, d, :, :]
            if train_gpu:
                target_masked = target_masked.cuda()

            result_C_0 = critic_0(output_masked)
            result_C_1 = critic_1(output_masked)
            result_C_2 = critic_2(output_masked)

            target_C_0 = critic_0(target_masked)
            target_C_1 = critic_1(target_masked)
            target_C_2 = critic_2(target_masked)

            loss_dice_0 = dice_loss(output, gt_3d)
            loss_dice_1 = dice_loss(output, gt_3d)
            loss_dice_2 = dice_loss(output, gt_3d)

            loss_C_0 = torch.mean(torch.abs(result_C_0 - target_C_0))
            loss_C_1 = torch.mean(torch.abs(result_C_1 - target_C_1))
            loss_C_2 = torch.mean(torch.abs(result_C_2 - target_C_2))

            loss_S_joint = torch.mean(loss_C_0 + loss_C_1 + loss_C_2) / 3 + torch.mean(loss_dice_0 + loss_dice_1 + loss_dice_2) / 3
            loss_S_joint.backward()
            optimizer_s.step()
            print(f'Segmentor Loss: {loss_S_joint}, Critic Loss: {loss_C}')
            # optimizer_c.zero_grad()
            # # Pass input images forward through the segmentor
            # output_s = segmentor.forward(input_img)
            #
            # # For Critic_1
            # output_1 = output_s[:, 0, :, :]
            # target_1 = gt_3d[:, 0, :, :]
            # # For Critic_2
            # output_2 = output_s[:, 1, :, :]
            # target_2 = gt_3d[:, 1, :, :]
            # # For Critic_3
            # output_3 = output_s[:, 2, :, :]
            # target_3 = gt_3d[:, 2, :, :]
            #
            # # Forward pass through critic nets
            # output_c1 = critic_1.forward(predicted_seg=output_1.detach(), gt_seg=target_1.detach(),
            #                              input_img=input_img.clone())
            # output_c2 = critic_2.forward(predicted_seg=output_2.detach(), gt_seg=target_2.detach(),
            #                              input_img=input_img.clone())
            # output_c3 = critic_3.forward(predicted_seg=output_3.detach(), gt_seg=target_3.detach(),
            #                              input_img=input_img.clone())
            #
            # # Compute loss due to each critic net
            # loss1 = multi_scale_L1_loss(c_output=output_c1)
            # loss2 = multi_scale_L1_loss(c_output=output_c2)
            # loss3 = multi_scale_L1_loss(c_output=output_c3)
            #
            # # Compute average Loss
            # # Multiply by (-1) because we want the critic to MAXIMIZE the loss during training
            # loss_c = (-1) * (loss1 + loss2 + loss3) / 3
            #
            # # Backprob & update critic parameters
            # loss_c.backward()
            # optimizer_c.step()
            #
            # # "TRAIN S NET"
            # optimizer_s.zero_grad()
            # # The same input_img is required to be passed through the same net --> forward pass is not necessary and we can use output_s from before
            #
            # # For Critic_1
            # output_1 = output_s[:, 0, :, :]
            # target_1 = gt_3d[:, 0, :, :]
            # # For Critic_2
            # output_2 = output_s[:, 1, :, :]
            # target_2 = gt_3d[:, 1, :, :]
            # # For Critic_3
            # output_3 = output_s[:, 2, :, :]
            # target_3 = gt_3d[:, 2, :, :]
            #
            # # Forward pass through critic nets --> this time we should not use detach
            # output_c1 = critic_1.forward(predicted_seg=output_1, gt_seg=target_1, input_img=input_img.clone())
            # output_c2 = critic_2.forward(predicted_seg=output_2, gt_seg=target_2, input_img=input_img.clone())
            # output_c3 = critic_3.forward(predicted_seg=output_3, gt_seg=target_3, input_img=input_img.clone())
            #
            # # Compute loss due to each critic net
            # loss1 = multi_scale_L1_loss(c_output=output_c1)
            # loss2 = multi_scale_L1_loss(c_output=output_c2)
            # loss3 = multi_scale_L1_loss(c_output=output_c3)
            #
            # # Compute dice loss
            # loss_dice = dice_loss(output_s, gt_3d)
            #
            # # Compute average Loss
            # loss_s = (loss1 + loss2 + loss3) / 3 + loss_dice
            # print(loss_s)
            # # Backprob & update segmentor parameters
            # loss_s.backward()
            # optimizer_s.step()

            # Compute evaluation metrics and save values to tensorboard in the second to last epoch
            if j == len(train_loader) - 2:
                # Compute metric values

                TP_trn, FP_trn, TN_trn, FN_trn = segmentor_evaluation(s_output=output_s, s_target=gt_3d)
                recall_trn = recall(TP=TP_trn, FN=FN_trn)
                precision_trn = precision(TP=TP_trn, FP=FP_trn)
                false_positive_rate_trn = false_positive_rate(FP=FP_trn, TN=TN_trn)
                accuracy_trn = accuracy(TP=TP_trn, FP=FP_trn, TN=TN_trn, FN=FN_trn)

                # Save computed metrics and losses
                writer.add_scalar('Recall/train', recall_trn, i + 1)
                writer.add_scalar('Precision/train', precision_trn, i + 1)
                writer.add_scalar('False Positive Rate/train', false_positive_rate_trn, i + 1)
                writer.add_scalar('Accuracy/train', accuracy_trn, i + 1)
                writer.add_scalar('Dice Loss/train', loss_dice, i + 1)
                writer.add_scalar('Critic Loss/train', loss_c, i + 1)
                writer.add_scalar('Segmentor Loss/train', loss_s, i + 1)

        ####################
        # Evaluation process
        ####################

        # After each epoch we evaluate the model and save evaluation metrics
        with torch.no_grad():
            # Set models on evaluation mode
            segmentor.eval()
            critic_0.eval()
            critic_1.eval()
            critic_2.eval()

            recall_val_avg = 0
            precision_val_avg = 0
            false_positive_rate_val_avg = 0
            accuracy_val_avg = 0
            loss_dice_val_avg = 0
            loss_c_val_avg = 0
            loss_s_val_avg = 0

            for j, (input_img_val, gt_3d_val) in enumerate(validation_loader):
                # Transfer loaded data to GPU if cuda is available
                # if torch.cuda.is_available:
                if train_gpu:
                    input_img_val = input_img_val.cuda()
                    gt_3d_val = gt_3d_val.cuda()

                # Pass input_img forward through the segmentor
                output_s_val = segmentor.forward(input_img_val)

                # We binarize the output of the segmentor to determine it's decision for every pixel of the input images
                output_s_val[output_s_val >= threshold] = 1.0
                output_s_val[output_s_val < threshold] = 0.0

                # Now the output of the segmentor can be used to compute the recall, precision, false_positive_rate and accuracy of the segmentor
                TP_val, FP_val, TN_val, FN_val = segmentor_evaluation(s_output=output_s_val, s_target=gt_3d_val)
                recall_val = recall(TP=TP_val, FN=FN_val)
                precision_val = precision(TP=TP_val, FP=FP_val)
                false_positive_rate_val = false_positive_rate(FP=FP_val, TN=TN_val)
                accuracy_val = accuracy(TP=TP_val, FP=FP_val, TN=TN_val, FN=FN_val)

                # For Critic_1
                output_1_val = output_s_val[:, 0, :, :]
                target_1_val = gt_3d_val[:, 0, :, :]
                # For Critic_2
                output_2_val = output_s_val[:, 1, :, :]
                target_2_val = gt_3d_val[:, 1, :, :]
                # For Critic_3
                output_3_val = output_s_val[:, 2, :, :]
                target_3_val = gt_3d_val[:, 2, :, :]

                # Forward pass through critic nets --> this time we should not use detach
                output_c1_val = critic_0.forward(predicted_seg=output_1_val, gt_seg=target_1_val,
                                                 input_img=input_img_val.clone())
                output_c2_val = critic_1.forward(predicted_seg=output_2_val, gt_seg=target_2_val,
                                                 input_img=input_img_val.clone())
                output_c3_val = critic_2.forward(predicted_seg=output_3_val, gt_seg=target_3_val,
                                                 input_img=input_img_val.clone())

                # Compute the loss_s, loss_c and loss_dice for the validation data
                loss_dice_val = dice_loss(output_s_val, gt_3d_val)

                # Compute loss due to each critic net
                loss1_val = multi_scale_L1_loss(c_output=output_c1_val)
                loss2_val = multi_scale_L1_loss(c_output=output_c2_val)
                loss3_val = multi_scale_L1_loss(c_output=output_c3_val)
                # Compute the loss_s, loss_c for the validation data
                loss_c_val = (-1) * (loss1_val + loss2_val + loss3_val) / 3
                loss_s_val = (loss1_val + loss2_val + loss3_val) / 3 + loss_dice_val

                # Save the Computed parameters --> to be averaged
                recall_val_avg += recall_val
                precision_val_avg += precision_val
                false_positive_rate_val_avg += false_positive_rate_val
                accuracy_val_avg += accuracy_val
                loss_dice_val_avg += loss_dice_val
                loss_c_val_avg += loss_c_val
                loss_s_val_avg += loss_s_val

            # After all the validation data has been passed through the network, compute the average evaluation metrics
            recall_val_avg = recall_val_avg / len(validation_loader)
            precision_val_avg = precision_val_avg / len(validation_loader)
            false_positive_rate_val_avg = false_positive_rate_val_avg / len(validation_loader)
            accuracy_val_avg = accuracy_val_avg / len(validation_loader)
            loss_dice_val_avg = loss_dice_val_avg / len(validation_loader)
            loss_c_val_avg = loss_c_val_avg / len(validation_loader)
            loss_s_val_avg = loss_s_val_avg / len(validation_loader)

            # Save computed losses and metric values via tensorboard
            writer.add_scalar('Recall/validation', recall_val_avg, i + 1)
            writer.add_scalar('Precision/validation', precision_val_avg, i + 1)
            writer.add_scalar('False Positive Rate/validation', false_positive_rate_val_avg, i + 1)
            writer.add_scalar('Accuracy/validation', accuracy_val_avg, i + 1)
            writer.add_scalar('Dice Loss/validation', loss_dice_val_avg, i + 1)
            writer.add_scalar('Critic Loss/validation', loss_c_val_avg, i + 1)
            writer.add_scalar('Segmentor Loss/validation', loss_s_val_avg, i + 1)

    # Print results at the end of earch epoch
    print(
        f"==> EPOCH {i + 1}({i + 1}/{epochs}) Train Dice Loss: {loss_dice:.8f}   Validation Dice Loss: {loss_dice_val:.8f}")
    print(f"==> EPOCH {i + 1}({i + 1}/{epochs}) Segmentor Loss:{loss_s:.4f}")
    print(f"==> EPOCH {i + 1}({i + 1}/{epochs}) Critic Loss:{loss_c:.4f} \n")

#     optimizer_S = optim.Adam(NetS.params(), lr=args.lr, betas=(args.beta1, args.beta2))
#     optimizer_C = optim.Adam(NetC.params(), lr=args.lr, betas=(args.beta1, args.beta2))
#
#     if args.gpu:
#         device = torch.device('cuda')
#         NetS = NetS.to(device)
#         NetC = NetC.to(device)
# for epoch in range(1, args.epochs + 1):
#     NetS.train()
#     for j, data in enumerate(dl_BraT_train, 1):
#         optimizer_C.zero_grad()
#         optimizer_S.zero_grad()
#
#         image, gt = data
#
#         seg_out = NetS(image)
#         seg_out = F.softmax(seg_out)
#         seg_out = seg_out.detach()
#         critic_prd, critic_gt = NetC(image, seg_out, gt)
#
#         loss_C = 1 - torch.mean(torch.abs(critic_prd - critic_gt))
#
#         loss_C.backward()
#         optimizer_C.step()
#
#         for p in NetC.params():
#             p.data._clamp(-0.01, 0.01)
#
#         seg_out = NetS(image)
#         seg_out = F.softmax(seg_out)
#         loss_S_target = torch.mean(torch.abs(seg_out, gt))
#         critic_prd, critic_gt = NetC(image, seg_out, gt)
#
#         loss_S_dice = loss_dice(critic_prd, critic_gt)
#         loss_S = args.alpha * loss_S_dice + loss_S_target
#
#         loss_S.backward()
#         optimizer_S.step()
#     NetS.eval()
#     for j, data in enumerate(dl_BraT_val, 1):
#         image, gt = data
#
#         seg_out = NetS(image)
#         seg_out = F.softmax(seg_out)
#         loss_S_target = torch.mean(torch.abs(seg_out, gt))
