# extract speaker embeddings from test clips
# F:\4th_DNSChallenge\Diagnose_DevTestset\RawNet-master
# /mnt/f/4th_DNSChallenge/Diagnose_DevTestset/RawNet-master
from tqdm import tqdm
from collections import OrderedDict

import os
import argparse
import json
import numpy as np
import glob
import pickle

import torch
import torch.nn as nn
from torch.utils import data
import glob
import sys
sys.path.insert(0,'./python/RawNet2')
from dataloader import *
from model_RawNet2 import RawNet2
from parser import get_args
from trainer import *
from utils import *
from model_RawNet2_original_code import *
from pydub import AudioSegment

import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()
    #dir
    parser.add_argument('-name', type = str, required = True)
    parser.add_argument('-save_dir', type = str, default = 'DNNs/')
    parser.add_argument('-DB', type = str, default = 'DB/VoxCeleb1/')
    parser.add_argument('-DB_vox2', type = str, default = 'DB/VoxCeleb2/')
    parser.add_argument('-dev_wav', type = str, default = 'wav/')
    parser.add_argument('-val_wav', type = str, default = 'dev_wav/')
    parser.add_argument('-eval_wav', type = str, default = 'eval_wav/')
    
    #hyper-params
    parser.add_argument('-bs', type = int, default = 100)
    parser.add_argument('-lr', type = float, default = 0.001)
    parser.add_argument('-nb_samp', type = int, default = 59049)
    parser.add_argument('-window_size', type = int, default = 11810)
    
    parser.add_argument('-wd', type = float, default = 0.0001)
    parser.add_argument('-epoch', type = int, default = 80)
    parser.add_argument('-optimizer', type = str, default = 'Adam')
    parser.add_argument('-nb_worker', type = int, default = 8)
    parser.add_argument('-temp', type = float, default = .5)
    parser.add_argument('-seed', type = int, default = 1234) 
    parser.add_argument('-nb_val_trial', type = int, default = 40000) 
    parser.add_argument('-lr_decay', type = str, default = 'keras')
    parser.add_argument('-load_model_dir', type = str, default = '')
    parser.add_argument('-load_model_opt_dir', type = str, default = '')

    #DNN args
    parser.add_argument('-m_first_conv', type = int, default = 251)
    parser.add_argument('-m_in_channels', type = int, default = 1)
    parser.add_argument('-m_filts', type = list, default = [128, [128,128], [128,256], [256,256]])
    parser.add_argument('-m_blocks', type = list, default = [2, 4])
    parser.add_argument('-m_nb_fc_att_node', type = list, default = [1])
    parser.add_argument('-m_nb_fc_node', type = int, default = 1024)
    parser.add_argument('-m_gru_node', type = int, default = 1024)
    parser.add_argument('-m_nb_gru_layer', type = int, default = 1)
    parser.add_argument('-m_nb_samp', type = int, default = 59049)
    
    #flag
    parser.add_argument('-amsgrad', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-make_val_trial', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-debug', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-comet_disable', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-save_best_only', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-do_lr_decay', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-mg', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-load_model', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-reproducible', type = str2bool, nargs='?', const=True, default = True)


    args = parser.parse_args()
    args.model = {}
    for k, v in vars(args).items():
        if k[:2] == 'm_':
            print(k, v)
            args.model[k[2:]] = v
    return args

def cos_sim(a,b) :
    return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))

def read_wav_and_get_clip_tensor(test_wav_path, nb_samp, window_size, wav_file = True):
    
    if not wav_file:
        X = AudioSegment.from_file(test_wav_path)
        X = X.get_array_of_samples()
        X = np.array(X)
    else:
        X, _ = sf.read(test_wav_path)
    X = X.astype(np.float64)
    X = _normalize_scale(X).astype(np.float32)
    X = X.reshape(1,-1)
    
    nb_time = X.shape[1]
    list_X = []
    nb_time = X.shape[1]
    if nb_time < nb_samp:
        nb_dup = int(nb_samp / nb_time) + 1
        list_X.append(np.tile(X, (1, nb_dup))[:, :nb_samp][0])
    elif nb_time > nb_samp:
        step = nb_samp - window_size
        iteration = int( (nb_time - window_size) / step ) + 1
        for i in range(iteration):
            if i == 0:
                list_X.append(X[:, :nb_samp][0])
            elif i < iteration - 1:
                list_X.append(X[:, i*step : i*step + nb_samp][0])
            else:
                list_X.append(X[:, -nb_samp:][0])
    else :
        list_X.append(X[0])
    return torch.from_numpy(np.asarray(list_X))

def get_embedding_from_clip_tensor(clip_tensor, model, device):
    model.eval()
    
    with torch.set_grad_enabled(False):
        #1st, extract speaker embeddings.
        l_embeddings = []
        l_code = []
        mbatch = clip_tensor
        mbatch = mbatch.unsqueeze(1)
#         print("Batch size = {}".format(mbatch.size()))
        for batch in mbatch:
            batch = batch.to(device)
            code = model(x = batch, is_test=True)
#             print("Code size = {}".format(code.size()))
            l_code.extend(code.cpu().numpy())
        embedding = np.mean(l_code, axis=0)
#         print("Embedding shape = {}".format(embedding.shape))
        return embedding

def _normalize_scale(x):
    '''
    Normalize sample scale alike SincNet.
    '''
    return x/np.max(np.abs(x))

def main_test():
    #parse arguments
    #args = get_args()

    wav_path = args.wav_path
    save_path = args.sav_path
    direc_level = args.root #args.direc_level
    wav_file = True if args.wav_file==1 else False
    load_model_dir = args.load_model_dir
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #device setting
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print('Device: {}'.format(device))
    
    model = RawNet(args.model, device)
    #.to(device)

    if device.type=='cpu':
        model.load_state_dict(torch.load(load_model_dir, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(load_model_dir))

    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])

    nb_samp = args.model["nb_samp"]
    window_size = args.window_size
    print('nb_params: {}'.format(nb_params))
    
    X1 = read_wav_and_get_clip_tensor(test_wav_path3, nb_samp, window_size, wav_file)
    emb_X1 = get_embedding_from_clip_tensor(X1, model, device)
    
    X2 = read_wav_and_get_clip_tensor(test_wav_path4, nb_samp, window_size, wav_file)
    emb_X2 = get_embedding_from_clip_tensor(X2, model, device)
    
    sim_score = cos_sim(emb_X1, emb_X2)
    print("Similarity = {}".format(sim_score))

if __name__ == '__main__':
    import argparse
    from argparse import ArgumentParser
    import pathlib
    import sys
    sys.path.insert(0,'python/RawNet2')

    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = ArgumentParser( description='detec duplicate wavs' )
    parser.add_argument('--root', help='Dir to search.', default="/mnt/f/4th_DNSChallenge/Diagnose_DevTestset/personalized_dev_testset/noisy_testclips_noduplicate")
    parser.add_argument('--load_model_dir', action='store_true', default="Pre-trained_model/rawnet2_best_weights.pt") # store_true
    parser.add_argument('--sav_path', default="/mnt/f/4th_DNSChallenge/Diagnose_DevTestset/personalized_dev_testset/noisy_testclips_noduplicate_embeddings") # store_true
    parser.add_argument('-remove', action='store_false',help='Delete duplicate files.' ) # store_true
    parser.add_argument('--model',help='model') # store_true

    #dir
    parser.add_argument('-name', type = str, required = False)
    parser.add_argument('-save_dir', type = str, default = 'DNNs/')
    parser.add_argument('-DB', type = str, default = 'DB/VoxCeleb1/')
    parser.add_argument('-DB_vox2', type = str, default = 'DB/VoxCeleb2/')
    parser.add_argument('-dev_wav', type = str, default = 'wav/')
    parser.add_argument('-val_wav', type = str, default = 'dev_wav/')
    parser.add_argument('-eval_wav', type = str, default = 'eval_wav/')
    
    #hyper-params
    parser.add_argument('-bs', type = int, default = 100)
    parser.add_argument('-lr', type = float, default = 0.001)
    parser.add_argument('-nb_samp', type = int, default = 59049)
    parser.add_argument('-window_size', type = int, default = 11810)
    
    parser.add_argument('-wd', type = float, default = 0.0001)
    parser.add_argument('-epoch', type = int, default = 80)
    parser.add_argument('-optimizer', type = str, default = 'Adam')
    parser.add_argument('-nb_worker', type = int, default = 8)
    parser.add_argument('-temp', type = float, default = .5)
    parser.add_argument('-seed', type = int, default = 1234) 
    parser.add_argument('-nb_val_trial', type = int, default = 40000) 
    parser.add_argument('-lr_decay', type = str, default = 'keras')
    parser.add_argument('-load_model_dir', type = str, default = '')
    parser.add_argument('-load_model_opt_dir', type = str, default = '')

    #DNN args
    parser.add_argument('-m_first_conv', type = int, default = 251)
    parser.add_argument('-m_in_channels', type = int, default = 1)
    parser.add_argument('-m_filts', type = list, default = [128, [128,128], [128,256], [256,256]])
    parser.add_argument('-m_blocks', type = list, default = [2, 4])
    parser.add_argument('-m_nb_fc_att_node', type = list, default = [1])
    parser.add_argument('-m_nb_fc_node', type = int, default = 1024)
    parser.add_argument('-m_gru_node', type = int, default = 1024)
    parser.add_argument('-m_nb_gru_layer', type = int, default = 1)
    parser.add_argument('-m_nb_samp', type = int, default = 59049)
    
    #flag
    parser.add_argument('-amsgrad', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-make_val_trial', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-debug', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-comet_disable', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-save_best_only', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-do_lr_decay', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-mg', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-load_model', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-reproducible', type = str2bool, nargs='?', const=True, default = True)

    args = parser.parse_args()
    args.model = {}
    for k, v in vars(args).items():
        if k[:2] == 'm_':
            print(k, v)
            args.model[k[2:]] = v
    return args
    #args = parser.parse_args()
#    return args

    args.wav_file = 1
    args.model['nb_classes'] = 6112 

    wavlist= glob.glob(os.path.join(args.root,'*.wav'))
    args.wav_path = wavlist
    save_path = args.sav_path
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    test_wav_path1 = wavlist[0]
    test_wav_path2 = wavlist[1]

    ## Number of speakers in VoxCeleb2 dataset. 
    ## Not used in computing embeddings but should still be there. 
    ## Do not comment this.

    main_test()