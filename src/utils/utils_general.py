import os
import random
import shutil

import click
import numpy as np
import torch




def do_nothing(*args, **kwargs):
    return args, kwargs

def print_CUDA_info():

    print("\n")
    print("".center(100, '|'))
    print(" CUDA GPUs REPORT ".center(100, '|'))
    print("".center(100, '|'))
    print("1) Number of GPUs devices: ", torch.cuda.device_count())
    print('2) CUDNN VERSION:', torch.backends.cudnn.version())
    print('3) Nvidia SMI terminal command: \n \n', )
    os.system('nvidia-smi')

    for device in range(torch.cuda.device_count()):
        print("|  DEVICE NUMBER : {%d} |".center(100, '-') % (device))
        print('| 1) CUDA Device Name:', torch.cuda.get_device_name(device))
        print('| 2) CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(device).total_memory / 1e9)
        print("|")

    print("".center(100, '|'))
    print(" GPU REPORT END ".center(100, '|'))
    print("".center(100, '|'))
    print('\n \n')
    return


class ConvertStrToList(click.Option):
    """Convert the list inside a yaml configuration file into a list
    """
    def type_cast_value(self, ctx, value):
        try:
            value = str(value)
            assert value.count('[') == 1 and value.count(']') == 1
            return list(int(x) for x in value.replace('"', "'").split('[')[1].split(']')[0].split(','))
        except Exception:
            raise click.BadParameter(value)


def mkdirs(paths: list):
    """create empty paths if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path: str):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def del_dir(path: str):
    """delete all the folders after the defined path

       Parameters:
           path (str) -- a single directory path
       """
    if os.path.exists(path):
        shutil.rmtree(path)


def seed_all(seed=None):  # for deterministic behaviour
    if seed is None:
        seed = 42
    print("Using Seed : ", seed)


    # This part is necessary to have reproducible behavior. It affects the CuDNN algorithm.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.empty_cache()
    torch.manual_seed(seed)   # Set torch pseudo-random generator at a fixed value
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)   # Set numpy pseudo-random generator at a fixed value
    random.seed(seed)   # Set python built-in pseudo-random generator at a fixed value
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed

def is_debug():
    import sys

    gettrace = getattr(sys, 'gettrace', None)

    if gettrace is None:
        return False
    else:
        v = gettrace()
        if v is None:
            return False
        else:
            return True

def debugging_only():
    """ This function is called only if in the DEBUG modality of Pycharm """

    print("".center(100, '°'))
    print(" DEBUG MODALITY ".center(100, '°'))
    print("".center(100, '°'))

def running():
    print("".center(100, '*'))
    print(" RUNNING CODE ".center(100, '*'))
    print("".center(100, '*'))