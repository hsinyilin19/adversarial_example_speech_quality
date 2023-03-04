import argparse
import os
import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from datasets import Perturbation_Data
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import *
from onnx2torch import convert
from tensorboardX import SummaryWriter


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def evaluate(g, g_pretrained, device, loader, optimizer, pbar, stage, **best_args):
    epoch_loss1 = AverageMeter()   # record attack loss
    epoch_loss2 = AverageMeter()   # record forgetting loss
    epoch_loss3 = AverageMeter()   # record pretrained loss
    epoch_total_loss = AverageMeter()  # sum of losses

    # record test results
    y_attacks = []
    y_originals = []
    y_attack_pretraineds = []

    g = g.to(device)
    g_pretrained = g_pretrained.to(device)
    g_pretrained.eval()

    for batch, (X_original, X_attack, y_original, files) in enumerate(loader):
        X_original = X_original.to(device)   # original audio
        X_attack = X_attack.to(device)       # attacked audio
        y_original = y_original.to(device)   # original scores

        if stage == 'train':
            g.train()
            y_attack = g(X_attack)   # attacked audio prediction
            y_original_pred = g(X_original)   # original audio prediction

            loss1 = nn.MSELoss()(y_attack, y_original)          # to correct attacked scores
            loss2 = nn.MSELoss()(y_original_pred, y_original)   # to avoid forgetting
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                y_attack_pretrained = g_pretrained(X_attack)  # used for comparison only
                loss3 = nn.MSELoss()(y_attack_pretrained, y_original)

        else:  # eval or test
            g.eval()
            with torch.no_grad():
                y_attack = g(X_attack)           # attacked audio prediction
                y_original_pred = g(X_original)  # original audio prediction
                y_attack_pretrained = g_pretrained(X_attack)  # used for comparison only

                loss1 = nn.MSELoss()(y_attack, y_original)             # to correct attacked scores
                loss2 = nn.MSELoss()(y_original_pred, y_original)      # to avoid forgetting
                loss3 = nn.MSELoss()(y_attack_pretrained, y_original)  # used for comparison only
                loss = loss1 + loss2

        epoch_loss1.update(loss1.item(), len(X_attack))
        epoch_loss2.update(loss2.item(), len(X_attack))
        epoch_loss3.update(loss3.item(), len(X_attack))
        epoch_total_loss.update(loss.item(), len(X_attack))

        improved = loss3.item() > loss1.item()  # we expected to see this!

        pbar.set_postfix({'stage': stage, 'Attack-enhanced loss': f'{epoch_loss1.avg:.4f}', 'Forgetting loss': f'{epoch_loss2.avg:.4f}', 'Improved': f'{improved}'})
        pbar.update(len(X_attack))

        if stage == 'test':
            # test stage has no loss to compute, only record scores
            y_attacks.append(y_attack)
            y_originals.append(y_original)
            y_attack_pretraineds.append(y_attack_pretrained)

    if stage == 'eval':
        if epoch_total_loss.avg < best_args['best_loss']:
            best_args['best_epoch'] = epoch
            best_args['best_loss'] = epoch_total_loss.avg
            best_args['best_weights'] = copy.deepcopy(g.state_dict())

            print(f"best epoch: {best_args['best_epoch']}, loss: {best_args['best_loss']}")
            save_name = f"new_DNSMOS_epoch{epoch}_loss{best_args['best_loss']:.3f}.pth"
            torch.save(best_args['best_weights'], os.path.join(args.save_model_path, save_name))

    if stage == 'test':
        y_attacks = torch.vstack(y_attacks).detach().cpu()
        y_originals = torch.vstack(y_originals).detach().cpu()
        y_attack_pretraineds = torch.vstack(y_attack_pretraineds).detach().cpu()

        # save
        torch.save(y_attacks, os.path.join(args.save_model_path, 'y_attacks_test.pth'))
        torch.save(y_originals, os.path.join(args.save_model_path, 'y_originals_test.pth'))
        torch.save(y_attack_pretraineds, os.path.join(args.save_model_path, 'y_attack_pretraineds_test.pth'))

    return epoch_loss1.avg, epoch_loss2.avg, epoch_loss3.avg, epoch_total_loss.avg, best_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx-model-path', type=str, default='./DNSMOS/sig_bak_ovr.onnx')
    parser.add_argument('--data-path', type=str, default="./noisy-vctk-16k/")
    parser.add_argument('--log-path', type=str, default='./logs/')
    parser.add_argument('--save-model-path', type=str, default='./new_DNSMOS_models/')
    parser.add_argument('--test-model-path', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=60)
    parser.add_argument('--model-name', type=str, default='new_DNSMOS_model')
    parser.add_argument('--device', type=str, default='cuda')  # 'cpu' 'cuda:0' or ...
    parser.add_argument('--num-epochs', type=int, default=15)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--N-samples', type=int, default=None, help='use partial samples')  # put a number or None
    args = parser.parse_args()

    # check / create output folders
    Path(args.save_model_path).mkdir(parents=True, exist_ok=True)
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    Path(args.log_path).mkdir(parents=True, exist_ok=True)

    # tensorboard
    writer = SummaryWriter(args.log_path)

    # train, test data path
    data_paths = {'train_noisy': os.path.join(args.data_path, 'noisy_trainset_28spk_wav_16k'),
                  'test_noisy': os.path.join(args.data_path, 'noisy_testset_wav_16k')}

    # path to pt perturbations
    perturb_paths = ['./perturbation_folder1',
                     './perturbation_folder2']

    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    num_GPUs = torch.cuda.device_count()
    print(f"[using {num_GPUs} GPUs]")

    INPUT_LENGTH, fs = 9.01, 16000
    transform = get_feature(sr=fs)

    original_DNSMOS = convert(args.onnx_model_path).to(args.device)
    original_DNSMOS.eval()  # fix DNSMOS network

    # split train, eval
    train_noisy_paths, eval_noisy_paths, train_noisy_files, eval_noisy_files = train_test_split(get_filepaths(data_paths['train_noisy'])[0], get_filepaths(data_paths['train_noisy'])[1], test_size=0.1, random_state=999)

    train_paths = train_noisy_paths
    train_files = train_noisy_files
    eval_paths = eval_noisy_paths
    eval_files = eval_noisy_files

    # total test = test (noisy + clean)
    test_paths = get_filepaths(data_paths['test_noisy'])[0]
    test_files = get_filepaths(data_paths['test_noisy'])[1]

    # data loaders
    train_loader_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': args.num_workers, 'drop_last': False, 'pin_memory': True}
    test_loader_params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': args.num_workers, 'drop_last': False, 'pin_memory': True}

    train_set = [Perturbation_Data(input_files=train_files, audio_paths=train_paths, perturb_path=perturb_paths[i], max_audio_length=int(INPUT_LENGTH * fs), eval_metric=original_DNSMOS, N=args.N_samples, test=False) for i in range(len(perturb_paths))]
    eval_set = [Perturbation_Data(input_files=eval_files, audio_paths=eval_paths, perturb_path=perturb_paths[i], max_audio_length=int(INPUT_LENGTH * fs), eval_metric=original_DNSMOS, N=args.N_samples, test=False) for i in range(len(perturb_paths))]
    test_set = [Perturbation_Data(input_files=test_files, audio_paths=test_paths, perturb_path=perturb_paths[i], max_audio_length=int(INPUT_LENGTH * fs), eval_metric=original_DNSMOS, N=None, test=True) for i in range(len(perturb_paths))]

    total_train_set = torch.utils.data.ConcatDataset(train_set)
    total_eval_set = torch.utils.data.ConcatDataset(eval_set)
    total_test_set = torch.utils.data.ConcatDataset(test_set)

    train_loader = DataLoader(dataset=total_train_set, **train_loader_params)
    eval_loader = DataLoader(dataset=total_eval_set, **train_loader_params)
    test_loader = DataLoader(dataset=total_test_set, **test_loader_params)

    # preserve an original DNSMOS for score comparison
    pretrained_DNSMOS = copy.deepcopy(original_DNSMOS)
    pretrained_DNSMOS.eval()

    # make a new copy to refine DNSMOS
    new_DNSMOS = copy.deepcopy(original_DNSMOS)
    optimizer = torch.optim.Adam(new_DNSMOS.parameters(), lr=args.lr)

    # record best status
    best_eval_args = {'best_epoch': 0, 'best_loss': 1000000.0, 'best_weights': copy.deepcopy(new_DNSMOS.state_dict())}

    if args.test_model_path is None:
        for epoch in range(args.num_epochs):
            # train
            with tqdm(total=(len(total_train_set))) as t:
                train_loss1, train_loss2, train_loss3, train_total_loss, _ = evaluate(g=new_DNSMOS, g_pretrained=pretrained_DNSMOS, device=args.device, loader=train_loader, optimizer=optimizer, pbar=t, stage='train', **best_eval_args)
                writer.add_scalar("Train_loss1", train_loss1, epoch)
                writer.add_scalar("Train_loss2", train_loss2, epoch)
                writer.add_scalar("Train_loss3", train_loss3, epoch)
                writer.add_scalar("Train_total_loss", train_total_loss, epoch)

            # evaluation to save model
            with tqdm(total=(len(total_eval_set))) as t:
                eval_loss1, eval_loss2, eval_loss3, eval_total_loss, best_eval_args = evaluate(g=new_DNSMOS, g_pretrained=pretrained_DNSMOS, device=args.device, loader=eval_loader, optimizer=optimizer, pbar=t, stage='eval', **best_eval_args)
                writer.add_scalar("Eval_loss1", eval_loss1, epoch)
                writer.add_scalar("Eval_loss2", eval_loss2, epoch)
                writer.add_scalar("Eval_loss3", eval_loss3, epoch)
                writer.add_scalar("Eval_total_loss", eval_total_loss, epoch)
        # test
        with tqdm(total=(len(total_test_set))) as t:
            test_loss1, test_loss2, test_loss3, test_total_loss, best_eval_args = evaluate(g=new_DNSMOS, g_pretrained=pretrained_DNSMOS, device=args.device, loader=test_loader, optimizer=optimizer, pbar=t, stage='test', **best_eval_args)
            writer.add_scalar("Test_loss1", test_loss1)
            writer.add_scalar("Test_loss2", test_loss2)
            writer.add_scalar("Test_loss3", test_loss3)
            writer.add_scalar("Test_total_loss", test_total_loss)

    # direct test
    if args.test_model_path:
        new_DNSMOS.load_state_dict(torch.load(args.test_model_path))
        print('test DNSMOS model loaded!')
        with tqdm(total=(len(total_test_set))) as t:
            test_loss1, test_loss2, test_loss3, test_total_loss, best_eval_args = evaluate(g=new_DNSMOS, g_pretrained=pretrained_DNSMOS, device=args.device, loader=test_loader, optimizer=optimizer, pbar=t, stage='test', **best_eval_args)
