import time
import datetime
import torch
import torch.optim as optim
import numpy as np
import source.spytrometer as spytrometer
import source.models as models
import source.parameters as params
import source.utils as utils
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import _pickle as pickle
except ModuleNotFoundError:
    import pickle

torch.set_num_threads(4)
torch_type = params.types['float_type']

PATH = '../../Downloads/'

log_softmax = torch.nn.LogSoftmax(dim=0)


def Info(text, end='\n', flush=True, show=True):
    if show:
        print(text, end=end, flush=flush)


def main(parameters=None):
    if GPU_ID == "-1":
        torch_device = torch.device("cpu")
    else:
        torch_device = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")

    if parameters is None:
        training_parameters = params.training
    else:
        training_parameters = parameters

    filename_stem = '{}/{}-{}'.format(PATH + 'results', DATASET, RES)

    param_string = params.params_to_string(training_parameters)

    filename = '{}-{}'.format(filename_stem, param_string)

    total_time = time.time()
    result = open('result_cross_entropy.txt', 'a')
    result.write("DATASET: " + DATASET + '\n')
    result.write("RES: " + RES + '\n')
    print('XENT', DATASET, RES)
    start_time = time.time()
    print_info = training_parameters['print_info']
    printing_tick = training_parameters['printing_tick']
    start_time = time.time()

    with open('pls_work.pkl', 'rb') as f:
        data = pickle.load(f)
    dataset = torch.stack([torch.as_tensor(data[0]), torch.as_tensor(data[1])], dim=1)

    Info("Done. Time: {} sec.".format(round(time.time() - start_time, 2)), show=print_info)

    Info("Training starts", end='\n', flush=True, show=print_info)
    window_width = int(training_parameters['window_width'] / params.dbsearch[RES]['bin_width'])
    nEpochs = training_parameters['epochs']
    batch_size = training_parameters['batch_size']
    kernel_num = training_parameters['kernel_num']
    clip_value = training_parameters['clip_value']
    spectrum_size = len(dataset[0][0])

    xent = models.BCELossWeight(
        pos_weight=torch.tensor(params.spectrum[DATASET]['pos_weight'], dtype=torch.float, device=torch_device))

    Info("Training filter parameters...", end='\n', flush=True, show=print_info)

    spectrum_num = len(dataset)
    indicies = np.arange(spectrum_num)

    for mode in range(1, 2):
        if mode == 1:
            conv_net = models.DeepConv1(window_width=window_width, kernel_num=kernel_num, spectrum_size=spectrum_size,
                                        torch_device=torch_device, torch_type=torch_type)
            print(sum(p.numel() for p in conv_net.parameters() if p.requires_grad))
            start_time = time.time()
            conv_net.identify(target_spectra=data[1], target_labels=data[2])
            print('Kernels creation time:', round(time.time() - start_time, 2))
        else:
            conv_net = models.Slider(window_width=window_width, kernel_num=kernel_num, spectrum_size=spectrum_size,
                                        torch_device=torch_device, torch_type=torch_type)
        conv_net = conv_net.to(torch_device)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, conv_net.parameters()),
                                lr=training_parameters['learning_rate'],
                                weight_decay=1e-6)
        learning_curve = []
        time_curve = []
        start_time = time.time()
        for epoch in range(10):
            partial_loss = 0
            batch_cnt = 0
            spectrum_id = 0
            flg = 0
            while spectrum_num > batch_size + spectrum_id or flg == 0:
                if not spectrum_num > batch_size + spectrum_id:
                    batch_idx = indicies[spectrum_id:]
                    flg = 1
                batch_idx = indicies[spectrum_id:spectrum_id + batch_size]
                spectrum_id += batch_size
                if (epoch+1) % 3 == 0:
                    spectrum_batch = dataset[batch_idx][:, 0:1, :].gt(0.35).float()
                else:
                    spectrum_batch = dataset[batch_idx][:, 0:1, :].float()
                target = dataset[batch_idx][:, 1:, :].float()
                spectrum_batch = spectrum_batch.to(torch_device)
                target = target.to(torch_device)

                output = conv_net(spectrum_batch)
                loss = xent(output, target)
                loss.backward()

                conv_net.clip_grads(clip_value=clip_value)
                optimizer.step()
                optimizer.zero_grad()
                partial_loss += loss
                batch_cnt += 1
                if flg: break
            epoch_error = (partial_loss / batch_cnt).data.cpu().numpy()
            if epoch == 3:
                conv_net.layer1[1].weight.requires_grad_(False)
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, conv_net.parameters()),
                                       lr=training_parameters['learning_rate'],
                                       weight_decay=1e-6)

            time_curve.append(round(time.time() - start_time, 2))
            learning_curve.append(epoch_error)
            if (epoch + 1) % printing_tick == 0 or epoch == 0:
                print("Epoch: {}/{}. Time: {}, loss:{}".format(epoch + 1, nEpochs, round(time.time() - start_time, 2),
                                                               epoch_error))

        Info("Learning done. Time: {} sec.".format(round(time.time() - start_time, 2)), show=print_info)

        start_time = time.time()
        for runner in range(0, len(data[1])):
            pos = np.nonzero(data[1][runner])[0].tolist()
            if runner == 0:
                conv_net.amino_acids(torch.tensor(data[0][runner].reshape(1, 1, -1)).float(),
                                 positions=pos, labels=data[2][runner], plot=True)
            
            else:
                conv_net.amino_acids(torch.tensor(data[0][runner].reshape(1, 1, -1)).float(),
                                    positions=pos, labels=data[2][runner])
        print('Identification time:', round(time.time() - start_time, 2))
        print(np.mean(conv_net.indic_guess[0]))
        print(np.mean(conv_net.indic_guess[1]))

    return filename
