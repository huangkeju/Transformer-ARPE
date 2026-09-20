
import sys

import numpy as np
import pickle
import datetime
import torch.nn as nn
import torch
from TrainHelper import train_model
from sklearn.model_selection import train_test_split
from model import make_cls_model
import os
import matplotlib.pyplot as plt
import logging

# Set GPUID in here
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

modulationTypes = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK", "BFM", "DSBAM"]

label_dict = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4, 'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7,
              'PAM4': 8, 'QPSK': 9}

TrainParams = {}
DatasetParams = {}
L = 128
# Training done for the following batch sizes
BS_range = [256]
LR = 10 ** -3
rand_scale = 0.5
initialize_type = 'Xavier'  # ['Xavier','Orthogonal']
TrainParams['num_epochs'] = 200
TrainParams['n_epochs_earlystop'] = 16
TrainParams['test_bactchsize'] = 1024
TrainParams['LRScheduler_stepsize'] = 8  # These many iterations are done for every step
TrainParams['LRSchedulerGamma'] = 0.1  # Every r itertaions, the LR is reduced by a multiplicative factor of LRSchedulerDecay
weight_decay = TrainParams['weight_decay'] = 5e-4
TrainParams['optimizer_type'] = 'Adam'
TrainParams['validation_size'] = 0.2
TrainParams['clip'] = 5

DatasetParams['SNRrange'] = np.arange(-20, 21, 2)
DatasetParams['Modulationtypes'] = ['QAM16', 'QAM64', '8PSK', 'WBFM', 'BPSK', 'CPFSK', 'AM-DSB', 'GFSK', 'PAM4', 'QPSK']
DatasetParams['NumClasses'] = len(DatasetParams['Modulationtypes'])
DatasetParams['datatype'] = 'RML22_sps=8'
DatasetParams['NumFrames'] = 2000
TrainParams['criterion'] = nn.CrossEntropyLoss()
TrainParams['computing_device'] = torch.device("cuda")

Savemodelfile_location = ''
datafilelocation = '../autodl-tmp/'
# datafilelocation = 'E:/Data/RML22/'
datafilename = datafilelocation + DatasetParams['datatype']

            
for BS in BS_range:
        TrainParams['initialize_type'] = initialize_type
        TrainParams['L'] = L
        TrainParams['BS'] = BS
        TrainParams['LR'] = LR

        Savemodelfilename = DatasetParams['datatype']

        '''
        Load dataset from the pickle file. The data is in a dictionary format with keys corresponding 
        to modulation and SNR. Every dict item in the dictionary contains X items per mod per SNR.
        '''
        f = open(datafilename, 'rb')
        dataset = pickle.load(f, encoding='latin1')
        f.close()
        print("Dataset loading completed")
        # read the keys - snrs and mods.
        snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], dataset.keys())))), [1, 0])
        X = []
        lbl = []
        for mod in mods:
            if mod in DatasetParams['Modulationtypes']:
                for snr in snrs:
                    if snr in DatasetParams['SNRrange']:
                        X.append(dataset[(mod, snr)][0:DatasetParams['NumFrames']])
                        for i in range(DatasetParams['NumFrames']):  lbl.append((mod, snr))


        X = np.vstack(X)
        label_val = list(map(lambda x: lbl[x][0], range(len(lbl))))
        label = list(map(lambda x: label_dict[x], label_val))
        label = np.array(label)
        data = X[:, :, 0:L]
        del dataset, X  # deleting large arrays to free up space in RAM.


        def loadSplitTrain(Savemodelfile_location, Savemodelfilename, data, label, TrainParams):

            model = make_cls_model(2, 10, L, rand_scale)
            x_train, x_test, y_train, y_test = train_test_split(data, label,
                                                                test_size=TrainParams['validation_size'] \
                                                                , random_state=1)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                              test_size=TrainParams['validation_size'],
                                                              random_state=1)
            train_set = {'data': torch.tensor(x_train).float(), 'labels': torch.tensor(y_train).float()}
            val_set = {'data': torch.tensor(x_val).float(), 'labels': torch.tensor(y_val).float()}
            del data

            ############ ############### Train Model ########################## ##########
            ############ ############ ############ ############ ############ ############
            model_file = Savemodelfile_location + Savemodelfilename + '_model.pt'
            model1, Loss, Accuracy = train_model(model, model_file, train_set, val_set, TrainParams)
            # Save Loss and accuracy plots
        
            plt.figure(1)
            epochs = [i for i in range(len(Loss['train']))]
            plt.plot(epochs, Loss['train'])
            plt.plot(epochs, Loss['valid'])
            plt.title('')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['Training', 'Validation'])
            plt.savefig(Savemodelfile_location + Savemodelfilename + 'model_loss.png')
            # plt.show()
            # plt.clf()

            plt.figure(2)
            plt.plot(epochs, Accuracy['train'])
            plt.plot(epochs, Accuracy['valid'])
            plt.title('')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(['Training', 'Validation'])
            plt.savefig(Savemodelfile_location + Savemodelfilename + 'model_acc.png')
            # plt.show()
            # plt.clf()


        loadSplitTrain(Savemodelfile_location, Savemodelfilename, data, label, TrainParams)
f.close()