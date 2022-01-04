import argparse
import numpy as np
from tqdm import tqdm
from earlystopping import EarlyStopping
import sys
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn
import torch
from data.seqgen import ClimateData
from data.seqgen_multi_channel import ClimateData_MC
from net_params import convlstm_encoder_params, convlstm_decoder_params
from net_params_large import convlstm_encoder_params_large, convlstm_decoder_params_large
from model import ED
from decoder import Decoder
from encoder import Encoder
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# TIMESTAMP = str(datetime.datetime.now()).replace(" ", "")

parser = argparse.ArgumentParser()
parser.add_argument('-clstm',
                    '--convlstm',
                    help='use convlstm as base cell',
                    action='store_true')
parser.add_argument('--batch_size',
                    default=1,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=0.00001, type=float, help='G learning rate')
parser.add_argument('-frames_input',
                    default=7,
                    type=int,
                    help='sum of input frames')
parser.add_argument('-frames_output',
                    default=1,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('-epochs', default=0, type=int, help='sum of epochs')
args = parser.parse_args()

random_seed = 1996
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# print("x")


def train(runtype, target, data_loc, epochsss, small):

    TIMESTAMP = runtype + str(target)

    save_dir = './save_model/' + TIMESTAMP

    if small:
        trainFolder = ClimateData(is_train=True,
                                  root=data_loc,
                                  n_frames_input=args.frames_input,
                                  n_frames_output=args.frames_output, target=target)
        validFolder = ClimateData(is_train=False,
                                  root=data_loc,
                                  n_frames_input=args.frames_input,
                                  n_frames_output=args.frames_output, target=target)
    else:
        trainFolder = ClimateData_MC(is_train=True,
                                     root=data_loc,
                                     n_frames_input=args.frames_input,
                                     n_frames_output=args.frames_output, target=target)
        validFolder = ClimateData_MC(is_train=False,
                                     root=data_loc,
                                     n_frames_input=args.frames_input,
                                     n_frames_output=args.frames_output, target=target)
    trainLoader = torch.utils.data.DataLoader(trainFolder,
                                              batch_size=args.batch_size,
                                              shuffle=False)
    validLoader = torch.utils.data.DataLoader(validFolder,
                                              batch_size=args.batch_size,
                                              shuffle=False)

    if small:
        encoder_params = convlstm_encoder_params
        decoder_params = convlstm_decoder_params
    else:
        encoder_params = convlstm_encoder_params_large
        decoder_params = convlstm_decoder_params_large

    '''
    main function to run the training
    '''
    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    net = ED(encoder, decoder)
    net = net.double()
    run_dir = './runs/' + TIMESTAMP
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=20, verbose=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # print("xt")

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(os.path.join(save_dir, 'checkpoin.pth.tar'))
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0
    lossfunction = nn.MSELoss().cuda()
    optimizer = optim.AdamW(net.parameters(), lr=args.lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=0.5,
                                                      patience=4,
                                                      verbose=True)
    # print("xt")

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    train_iter_losses = []
    valid_iter_losses = []
    # mini_val_loss = np.inf
    for epoch in range(cur_epoch, epochsss + 1):
        # for epoch in range(cur_epoch, args.epochs + 1):
        ###################
        # train the model #
        ###################
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            inputs = inputVar.to(device)  # B,S,C,H,W
            label = targetVar.to(device)  # B,S,C,H,W
            # print("xtl")
            optimizer.zero_grad()
            net.train()
            pred = net(inputs)  # B,S,C,H,W
            loss = lossfunction(pred, label)
            loss_aver = loss.item() / args.batch_size
            train_losses.append(loss_aver)
            train_iter_losses.append(loss_aver)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
            # print("xtl")
            optimizer.step()
            t.set_postfix({
                'trainloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
        tb.add_scalar('TrainLoss', loss_aver, epoch)
        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
                if i == 3000:
                    break
                inputs = inputVar.to(device)
                label = targetVar.to(device)
                pred = net(inputs)
                # print("xvl")
                loss = lossfunction(pred, label)
                loss_aver = loss.item() / args.batch_size
                # record validation loss
                valid_losses.append(loss_aver)
                valid_iter_losses.append(loss_aver)
                #print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
                t.set_postfix({
                    'validloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch)
                })

        tb.add_scalar('ValidLoss', loss_aver, epoch)
        torch.cuda.empty_cache()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(args.epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f}')

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        pla_lr_scheduler.step(valid_loss)  # lr_scheduler
        model_dict = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        early_stopping(valid_loss.item(), model_dict, epoch, save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    with open("avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)

    np.savetxt("./save_model/" + TIMESTAMP +
               "/trainloss.txt", train_iter_losses)
    np.savetxt("./save_model/" + TIMESTAMP +
               "/validloss.txt", valid_iter_losses)


if __name__ == "__main__":

    # for i in range(5):
    #     print('testwowind_adamw', i)
    #     train('testwowind_adamw', i, './data/fin_data_ww.npz')

    # for i in range(5):
    #     print('wowind_adamw', i)
    #     train('wowind_adamw', i, './data/fin_data_ww.npz')

    for i in range(2, 5):
        print('wwind_adamw_lg', i)
        train('wwind_adamw_lg', i, './data/fin_data_ww.npz', 0, True)

    # print('MultiChannel wowwind adamw')
    # train('mc_wowind_adamw', 0, './data/fin_data_ww.npz', 2, False)
