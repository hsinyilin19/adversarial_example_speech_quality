import argparse
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from attack_modules import Attack, AttackModule
from utils import *
from onnx2torch import convert
from tensorboardX import SummaryWriter
from pathlib import Path
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def perturb_transform(z):
    z = 0.03 * nn.Tanh()(z)
    return z

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-index', type=int, default=0)
    parser.add_argument('--onnx-model-path', type=str, default='./DNSMOS/sig_bak_ovr.onnx')
    parser.add_argument('--data-path', type=str, default="./noisy-vctk-16k/")
    parser.add_argument('--output-path', type=str, default='./DNS_audio_output/')
    parser.add_argument('--save-model-path', type=str, default='./perturbation_models/')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--c', type=float, default=10)
    parser.add_argument('--max-iter', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')  # 'cpu' 'cuda:0' or ...
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    # check / create output folders
    Path(args.save_model_path).mkdir(parents=True, exist_ok=True)
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    # train, test data path
    data_paths = {'train_noisy': os.path.join(args.data_path, 'noisy_trainset_28spk_wav_16k'),
                  'test_noisy': os.path.join(args.data_path, 'noisy_testset_wav_16k')}

    # tensorboard writer
    log_path = f'./log'
    writer = SummaryWriter(log_path)

    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    num_GPUs = torch.cuda.device_count()
    print(f"[using {num_GPUs} GPUs]")

    INPUT_LENGTH, fs = 9.01, 16000
    transform = get_feature(sr=fs)

    # # segment audios into uniform pieces
    noisy_paths, files = get_filepaths(data_paths['train_noisy'])   # train_noisy or test_noisy

    # use partial samples
    if (args.batch_index + 1) * args.batch_size <= len(files):
        files = files[(args.batch_index * args.batch_size): ((args.batch_index + 1) * args.batch_size)]
        noisy_paths = noisy_paths[(args.batch_index * args.batch_size): ((args.batch_index + 1) * args.batch_size)]
    else:
        files = files[(args.batch_index * args.batch_size):]
        noisy_paths = noisy_paths[(args.batch_index * args.batch_size):]

    # find audio lengths
    audio_inputs = {}
    for file, noisy_path in zip(files, noisy_paths):
        file_path = os.path.join(noisy_path, file)
        wav, sr = torchaudio.load(file_path)
        wav = wav / torch.max(torch.abs(wav))
        wav = wav.squeeze()
        audio_inputs[file] = wav.to(args.device)

    # set Attack as model
    attack_module = AttackModule(audio_inputs, perturb_transform=perturb_transform).to(args.device)
    attack_optimizer = optim.Adam(attack_module.parameters(), lr=args.lr)

    DNSMOS = convert(args.onnx_model_path).to(args.device)
    setting = vars(args)

    for epoch in range(args.num_epochs):
        attack_module.train()
        DNSMOS.eval()         # fix DNSMOS network
        epoch_train_loss = AverageMeter()

        # generate attack wav-form
        setting.update({'INPUT_LENGTH': INPUT_LENGTH, 'fs': fs, 'batch': 0, 'epoch': epoch, 'num_GPUS': num_GPUs})
        X_attack, attacked_DNSMOS, X_noisy, original_DNSMOS, best_δ_model = Attack(f=attack_module, optimizer=attack_optimizer, g=DNSMOS, audio_inputs=audio_inputs, writer=writer, **setting)

        # save attacked filenames and corresponding perturbations
        best_δ = list(map(lambda x: perturb_transform(x.detach().cpu()), list(best_δ_model.values())))  # detach model perturbations

        for i, audio_file in enumerate(audio_inputs.keys()):
            print(f'saving audio: {audio_file}')
            save_pt_file = os.path.join(args.save_model_path, f'δ_{os.path.splitext(audio_file)[0]}.pt')
            torch.save(best_δ[i], save_pt_file)

        # record audio
        record_audio(input=X_noisy, filenames=list(audio_inputs.keys()), scores=original_DNSMOS, score_types={'original_DNSMOS': ['SIG', 'BAK', 'OVR']}, output_path=args.output_path, fs=fs)
        record_audio(input=X_attack, filenames=list(audio_inputs.keys()), scores=attacked_DNSMOS, score_types={'attacked_DNSMOS': ['SIG', 'BAK', 'OVR']}, output_path=args.output_path, fs=fs)
