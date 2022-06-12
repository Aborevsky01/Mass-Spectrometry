import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from random import shuffle
import difflib
from matplotlib.colors import NoNorm




class BCELossWeight(torch.nn.BCELoss):
    """
    Make a convolution layer with untrainable center explicitly.
    """

    def __init__(self, pos_weight):
        super(torch.nn.BCELoss, self).__init__()
        # self.__class__.__name__ += '_'+"_pos_weight"
        self.pos_weight = pos_weight
        self.tiny = 1e-20

    def forward(self, estimated, target, penalty=0):
        cost = 5 * self.pos_weight * target * torch.log(estimated + self.tiny) + (1 - target) * torch.log(
            1 - estimated + self.tiny)

        return -cost.mean()


class DeepConv1(torch.nn.Module):
    """
    Make a convolution layer with untrainable center explicitly.
    """

    def __init__(self, window_width, kernel_num, activation_fnc=torch.nn.functional.relu,
                 spectrum_size=None, torch_device=None, torch_type=None):
        super(DeepConv1, self).__init__()
        self.indicate = torch.nn.Conv1d(1, 20, 359, padding=400)
        self.intervals = {}
        self.kernels = torch.empty(size=(359, 0))
        self.__class__.__name__ += '_' + activation_fnc.__class__.__name__
        self.window_width = window_width
        self.kernel_num = kernel_num
        bias = True
        self.sigmoid = torch.nn.Sigmoid()
        self.indic_guess = [[], []]

        in_channels = 1

        self.layer1 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features=1),
            torch.nn.Linear(1999, 1999),
            torch.nn.MaxPool1d(5, padding=2, stride=1),
            torch.nn.Conv1d(in_channels, 10, self.window_width * 2 + 1, padding=self.window_width,
                            bias=bias),
            torch.nn.BatchNorm1d(num_features=10),
            torch.nn.Sigmoid(),
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features=1),
            torch.nn.MaxPool1d(5, padding=2, stride=1),
            torch.nn.Conv1d(in_channels, 10, self.window_width * 2 + 1, padding=self.window_width,
                            bias=bias),
            torch.nn.BatchNorm1d(num_features=10),
            torch.nn.Sigmoid(),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(10, 1, 1, padding=0, bias=bias),
            torch.nn.Sigmoid(),
            torch.nn.BatchNorm1d(num_features=1),
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv1d(3, 1, 1, padding=0, bias=bias),
            torch.nn.Sigmoid(),
            torch.nn.BatchNorm1d(num_features=1),
        )

        self.bias = torch.nn.Parameter(torch.tensor(5.0, requires_grad=True))  # nn.Parameters(torch.zeros(1))
        self.offset = torch.tensor(0.0 / self.kernel_num)

    def forward(self, x, graph=False, positions=None, acids=None):
        if graph:
            tmp = x[0, 0:].clone().detach()
            sns.heatmap(tmp.detach().numpy(), yticklabels=['Experimental spectra'], vmin=0.0, vmax=1.0, cmap='magma')
            plt.xticks(ticks=positions, labels=acids)
            plt.xlabel('m/z')
            plt.show()
        out = self.sigmoid(self.layer2(self.layer1(x)) - self.offset - self.bias)
        out_x = self.sigmoid(self.layer2(self.layer4(x)) - self.offset - self.bias)
        y = torch.cat([out, out_x, x], dim=1)
        if graph:
            tmp = y[0, 0:].clone().detach()
            sns.heatmap(tmp.detach().numpy(), yticklabels=['OC', 'SlS', 'Input'], vmin=0.0, vmax=1.0, cmap='magma')
            plt.xticks(ticks=positions, labels=acids)
            plt.xlabel('m/z')
            plt.show()
        out = self.sigmoid(self.layer3(y) - self.offset - self.bias)
        if graph:
            tmp = out[0, 0:].clone().detach()
            sns.heatmap(tmp.detach().numpy(), vmin=0.0, vmax=1.0, yticklabels=['Un'], cmap='magma')
            plt.xticks(ticks=positions, labels=acids)
            plt.xlabel('m/z')
            plt.show()
        return out

    def clip_grads(self, clip_value=0.0001):
        # for layer in self.layers:
        self.layer1[0].weight.grad.data.clamp_(min=-clip_value, max=clip_value)
        self.layer2[0].weight.grad.data.clamp_(min=-clip_value, max=clip_value)

    def identify(self, target_spectra, target_labels):
        amino_acids = []
        for peptide in target_labels:
            if len(amino_acids) == 20:
                break
            for letter in peptide:
                if letter not in amino_acids and letter.isalpha():
                    amino_acids.append(letter)

        self.intervals = {key: [] for key in amino_acids}
        self.intervals['rubbish'] = []

        clean = re.compile('[^a-zA-Z]')
        count = 0
        for runner in range(len(target_spectra)):
            spectrum = torch.as_tensor(target_spectra[runner])
            peptide = clean.sub('', target_labels[runner])
            peaks = torch.nonzero(spectrum, as_tuple=False)
            for i, peak in enumerate(peaks[:-1]):
                acid = peptide[i + 1]
                if peak + 359 > spectrum.shape[0]:
                    tmp = torch.concat([spectrum[peak:len(spectrum)], torch.zeros((peak + 359 - len(spectrum)))])
                    self.intervals[acid].append(tmp.reshape(1, -1))
                else:
                    self.intervals[acid].append(spectrum[peak:peak + 359].reshape(1, -1))
                    if i % 10 == 0:
                        shift = np.random.randint(1, 1500)
                        while spectrum[shift] == 1:
                            shift = np.random.randint(1000, 1500)
                        self.intervals['rubbish'].append(spectrum[shift:shift + 359].reshape(1, -1))
                        count += 1
                count += 1

        xent = BCELossWeight(pos_weight=torch.tensor(20, dtype=torch.float))
        for i, acid in enumerate(amino_acids):
            false = [block for letter in self.intervals.keys() if letter == 'rubbish' for block in self.intervals[letter]]
            shuffle(false)
            target = torch.concat([torch.ones(len(self.intervals[acid])), torch.zeros(len(self.intervals['rubbish']))])
            true_false = self.intervals[acid] + false
            train_set = torch.Tensor((len(true_false), false[0].shape[0]))
            torch.cat(true_false, out=train_set, dim=0)

            spectrum_num = len(target)
            indicies = np.arange(spectrum_num)
            kernel = torch.ones((359, 1), requires_grad=True)
            optimizer = torch.optim.Adam([kernel], weight_decay=1e-1)

            for epoch in range(40):
                partial_loss = 0
                batch_cnt = 0
                spectrum_id = 0
                flg = 0
                while spectrum_num > 32 + spectrum_id or flg == 0:
                    if not spectrum_num > 32 + spectrum_id:
                        batch_idx = indicies[spectrum_id:]
                        flg = 1
                    batch_idx = indicies[spectrum_id:spectrum_id + 32]
                    spectrum_id += 32
                    spectrum_batch = train_set[batch_idx].float()
                    output = self.sigmoid(torch.mm(spectrum_batch, kernel))
                    loss = xent(output, target[batch_idx].float())
                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()
                    partial_loss += loss
                    batch_cnt += 1
                    if flg: break
            if i == 0:
                self.kernels = kernel
            else:
                self.kernels = torch.cat((self.kernels, kernel), dim=1)
        self.kernels = torch.transpose(self.kernels, 0, 1)
        self.indicate.weight = torch.nn.Parameter(self.kernels[:, None, :], requires_grad=False)

    def amino_acids(self, peptide, labels=None, positions=None, plot=False):

        def func(x):
            x[np.argwhere(x != x.max())] = 0
            x[np.argwhere(x == x.max())] = x.max() * (x.max() > 3)
            return x

        clean = re.compile('[^a-zA-Z]')
        labels = clean.sub('', labels)
        peptide = self.forward(peptide, graph=plot, acids=list(labels)[:len(positions)], positions=positions)
        tmp = self.indicate(torch.as_tensor(peptide))[0, 0:].detach().numpy()
        ttmp = np.apply_along_axis(func, 0, tmp[:, 400:].copy())

        winners = np.argmax(ttmp[:, positions[:-1]], axis=0).astype(int)
        predict = np.array(list(self.intervals.keys()))[winners]
        mod_lab = np.array(list(labels)[1:len(positions)])
        self.indic_guess[0].append(difflib.SequenceMatcher(None, mod_lab, predict).ratio() * 100)
        predict[np.where(predict == 'L')] = 'I'
        mod_lab[np.where(mod_lab == 'L')] = 'I'
        self.indic_guess[1].append(difflib.SequenceMatcher(None, mod_lab, predict).ratio() * 100)
        # print(predict.tolist())
        # print(list(labels)[1:len(positions)])
        # print()

        if plot:
            fig, ax = plt.subplots()
            ax.set_yticks(ticks=np.arange(-.5, 20, 1), minor=True)
            ax.set_yticklabels(labels=list(self.intervals.keys())[0:20])
            ax.set_xticks(ticks=positions[:-1])
            ax.set_xticklabels(labels=list(labels)[1:len(positions)])
            ax.set_yticks(np.arange(0, 20))
            ax.grid(alpha=0.3, color='red', linestyle='--', axis='x')
            ax.grid(alpha=0.2, which='minor', color='white', linestyle='-', axis='y')
            plt.imshow(ttmp, aspect='auto', cmap='inferno', norm=NoNorm(), vmax=3)
            plt.title('Amino acids identification')
            plt.ylabel('Indicators')
            plt.xlabel('Peptide')
            plt.savefig('saving-a-seaborn-plot-as-pdf-file-300dpi.pdf', dpi=500)
            plt.show()


class Slider(torch.nn.Module):
    """
    Make a convolution layer with untrainable center explicitly.
    """

    def __init__(self, window_width, kernel_num, activation_fnc=torch.nn.functional.relu, spectrum_size=None,
                 torch_device=None, torch_type=None):
        super(Slider, self).__init__()
        self.__class__.__name__ += '_' + activation_fnc.__class__.__name__
        self.window_width = window_width
        self.kernel_num = kernel_num+1
        bias = True
        self.sigmoid = torch.nn.Sigmoid()

        in_channels = 1

        self.layer1 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features=1),
            torch.nn.Conv1d(in_channels, self.kernel_num, 359, padding=179,
                            bias=bias),
            torch.nn.BatchNorm1d(num_features=self.kernel_num),
            torch.nn.Sigmoid(),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(self.kernel_num, 1, 1, padding=0, bias=bias),
            # torch.nn.MaxPool2d((self.kernel_num, 1)),
            torch.nn.BatchNorm1d(num_features=1),
        )
        self.bias = torch.nn.Parameter(torch.tensor(5.0, requires_grad=True))  # nn.Parameters(torch.zeros(1))
        self.offset = torch.tensor(0.0 / self.kernel_num)

    def forward(self, x):
        out = self.layer1(x) - self.offset  # shape: [batch_size, channel, spectrum_bin]
        sfm = torch.nn.Softmax(dim=1)
        out = sfm(out)
        out = self.layer2(out) - self.offset - self.bias
        return self.sigmoid(out)

    def clip_grads(self, clip_value=0.0001):
        # for layer in self.layers:
        self.layer1[0].weight.grad.data.clamp_(min=-clip_value, max=clip_value)
        self.layer2[0].weight.grad.data.clamp_(min=-clip_value, max=clip_value)

    def check(self, labels, peptide, positions, graph):
        def func(x):
            x[np.argwhere(x != x.max())] = 0
            x[np.argwhere(x == x.max())] = x.max() * (x.max() > 3)
            return x

        clean = re.compile('[^a-zA-Z]')
        labels = clean.sub('', labels)
        peptide = self.layer1(peptide)
        tmp = peptide[0, 0:].clone().detach()
        sns.heatmap(tmp.detach().numpy(), vmin=0.0, vmax=1.0, yticklabels=['Un'], cmap='magma')
        plt.xticks(ticks=positions, labels=list(labels)[:len(positions)])
        plt.xlabel('m/z')
        plt.show()
        mp = torch.nn.MaxPool2d((self.kernel_num, 1), return_indices=True)
        tmp, indices = mp(torch.as_tensor(peptide))
        tmp = tmp[0, 0:].clone().detach()
        sns.heatmap(tmp.detach().numpy(), vmin=0.0, vmax=1.0, yticklabels=['Un'], cmap='magma')
        plt.xticks(ticks=positions, labels=list(labels)[:len(positions)])
        plt.xlabel('m/z')
        plt.show()
