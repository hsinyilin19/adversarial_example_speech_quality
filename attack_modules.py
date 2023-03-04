from utils import *
import torchaudio
from torch import nn
import copy

class AttackModule(torch.nn.Module):
    def __init__(self, audio_inputs, perturb_transform):
        super(AttackModule, self).__init__()

        lengths = list(map(lambda x: len(x), list(audio_inputs.values())))
        self.δ = nn.ParameterList([torch.nn.Parameter(torch.zeros(_)) for _ in lengths])
        self.num_params = sum(lengths)  # total number of parameters in all perturbations
        self.perturb_transform = perturb_transform

        # STFT transform
        n_fft = 512
        win_length = 512
        hop_length = n_fft // 4
        self.stft_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                                                center=True, normalized=False, onesided=True, pad_mode='reflect', window_fn=torch.hann_window)

    def forward(self, audio_inputs, fix_length):
        x_attack_repeated = []
        x_orignal_repeated = []

        for i, (δ, x) in enumerate(zip(list(self.δ), list(audio_inputs.values()))):
            x_attack = x + self.perturb_transform(δ)
            aud_len = x_attack.shape[-1]

            # use STFT to measure similarity
            x_STFT = self.stft_transform(x)
            x_attack_STFT = self.stft_transform(x_attack)

            if aud_len <= fix_length:
                cycle = fix_length // aud_len
                x = x.repeat(cycle)                 # original audio
                x_attack = x_attack.repeat(cycle)   # attacked audio

                # concat with the last small piece
                rest = fix_length - cycle * aud_len
                x = torch.concat((x, x[:rest]))                       # original audio
                x_attack = torch.concat((x_attack, x_attack[:rest]))  # attacked audio
            else:
                # if original audio longer than fix_length, random crop
                random_start = torch.randint(0, (aud_len - fix_length), (1,))
                x = x[random_start: (random_start + fix_length)]                # original audio
                x_attack = x_attack[random_start: (random_start + fix_length)]  # attacked audio

            x_attack_repeated.append(x_attack)
            x_orignal_repeated.append(x)

        x_attack_repeated = torch.stack(x_attack_repeated, dim=0)
        x_orignal_repeated = torch.stack(x_orignal_repeated, dim=0)

        return x_attack_repeated, x_orignal_repeated, x_attack_STFT, x_STFT


def score_transform(input):
    result = input
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            if input[i, j] > 3:
                result[i, j] = 1
            else:
                result[i, j] = 5
    return result


def Attack(f, optimizer, g, audio_inputs, writer, **args):
    ''' f: attack model, g: evaluation net (eg. DNSMOS) '''

    best_loss = 1e10
    f.train()   # train perturbations
    g.eval()    # fix evaluation network

    with tqdm(total=args['max_iter'], desc=f"Epoch {args['epoch'] + 1}/{args['num_epochs']} | Batch {args['batch'] + 1}", unit='step') as t:
        iter_losses = AverageMeter()

        for iteration in range(args['max_iter']):

            # find perturbation to attack
            X_attack, X_noisy, X_attack_STFT, X_STFT = f(audio_inputs, fix_length=int(args['INPUT_LENGTH'] * args['fs']))

            # compute loss
            similarity_loss = nn.L1Loss(reduction='sum')(X_attack_STFT, X_STFT)  # measure on STFT

            # set target score depending on the original score
            target_scores = score_transform(g(X_noisy).detach()).to(args['device'])

            false_target_loss = (target_scores - g(X_attack)).abs().sum()         # on wav (sum over all samples)

            loss = similarity_loss + args['c'] * false_target_loss
            iter_losses.update(loss.item(), 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # display in tqdm
            t.set_description(f"Progress: {(iteration + 1) / args['max_iter'] * 100:.2f} % | similarity loss: {similarity_loss:.3f}, target loss: {false_target_loss:.3f}")
            t.update(1)

            # record loss
            writer.add_scalars(
                'Similarity-loss_batch{}_c{}'.format(args['batch_size'], args['c']),
                {'train': similarity_loss.item()}, iteration)
            writer.add_scalars(
                'Target-loss_batch{}_c{}'.format(args['batch_size'], args['c']),
                {'train': false_target_loss.item()}, iteration)
            writer.add_scalars(
                'Total-loss_batch{}_c{}'.format(args['batch_size'], args['c']),
                {'train': loss.item()}, iteration)

            if iter_losses.avg < best_loss:
                best_loss = iter_losses.avg
                best_δ = copy.deepcopy(f.state_dict())  # best perturbation

            # Early Stop when loss does not converge.
            if iteration % (args['max_iter'] // 10) == 0:

                if loss.item() > best_loss:
                    print(f'Attack Stopped at {iteration} due to CONVERGENCE....')
                    return X_attack.detach(), g(X_attack).detach(), X_noisy.detach(), g(X_noisy).detach(), best_δ


    return X_attack.detach(), g(X_attack).detach(), X_noisy.detach(), g(X_noisy).detach(), best_δ
