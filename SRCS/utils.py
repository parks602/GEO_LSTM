import random
import os
import torch
import psutil
import numpy as np
from argparse import ArgumentParser

def get_argument_parser():
    parser = ArgumentParser(description="Geo-LSTM")
    modes  = ['train', 'eval']
    # Other parameters
    parser.add_argument('--mode', choices=modes, required=True, help='Select Mode[train, valid, test]')
    # parameters
    parser.add_argument('--var', type=str, help='Variable Name',required=True)
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
    parser.add_argument('--gpus', default=None, help='gpus')
    # Output or Model Saved Path
    parser.add_argument('--opath', default='./DAOU', help='folder to output and save model checkpoints')
    parser.add_argument('--model', default='./DAOU', help='trained model filename')
    return parser

def set_seed(value):
    print("Random Seed: ", value)
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    np.random.seed(value)

def create_folder(folder):
    try:
        os.makedirs(folder)
    except OSError:
        pass

def check_memory(mesg):
    print("===== %s" %(mesg))
    ### General RAM Usage
    memory_usage_dict = dict(psutil.virtual_memory()._asdict())
    memory_usage_percent = memory_usage_dict['percent']
    ### RAM Usage
    ram_total = int(psutil.virtual_memory().total) / 1024 / 1024
    ram_usage = int(psutil.virtual_memory().total - psutil.virtual_memory().available) / 1024 / 1024
    print(f"RAM total: {ram_total: 9.3f} MB")
    print(f"RAM usage: {ram_usage: 9.3f} MB / {memory_usage_percent}%")
    print("="*20)

def isExists(fname):
    if not os.path.exists(fname):
       print("Can't find : %s" %(fname))
       return False
    else:
       return True


