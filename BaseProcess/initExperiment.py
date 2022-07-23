import os
import sys
import shutil
import logging
import torch
from utils.log_function import print_options

def init_experiment(opts, parser):
    if opts.isTrain:
        opts.result_dir = os.path.join(opts.result_root, opts.result_dir, opts.exp_name)
        if not os.path.exists(opts.result_dir):
            os.makedirs(opts.result_dir)
            code_dir = os.path.abspath(os.path.dirname(os.getcwd()))
            print_options(parser, opts)
            shutil.copytree(code_dir, os.path.join(opts.result_dir, 'code'))
        else:
            sys.exit("result_dir exists: "+opts.result_dir)



    """ device configuration """
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.visible_devices
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        logging.warning('cuda is unavailable!')
    return device


def init_experiment_2(opts, parser):
    if opts.isTrain:
        opts.result_dir = os.path.join(opts.result_root, opts.result_dir, opts.exp_name)
        if not os.path.exists(opts.result_dir):
            os.makedirs(opts.result_dir)
            code_dir = os.path.abspath(os.path.dirname(os.getcwd()))
            print_options(parser, opts)
            shutil.copytree(code_dir, os.path.join(opts.result_dir, 'code'))
        else:
            sys.exit("result_dir exists: "+opts.result_dir)



    """ device configuration """
    if opts.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        logging.warning('cuda is unavailable!')
    return device