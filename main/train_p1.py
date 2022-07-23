# -*- coding:utf-8 -*-
import argparse
import os, time
import sys
import torch
import tqdm
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
from BaseProcess import datasetting
from BaseProcess.initExperiment import init_experiment
from datasets.dataset_for_3d import Opt3D_multi_cat_Dataset
from datasets.dataset_for_concat import ConcatDataset
from utils.log_function import  print_network
from trainer_p1 import PMFAN_Trainer
from utils.util_for_3d import gen_patch_data_list, get_train_pos, gen_half_local_data
""" set flags / seeds """
torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    """ Hpyer parameters """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--exp_name', type=str, default='required')
    parser.add_argument('--visible_devices', type=str, default='0')
    # training option
    parser.add_argument('--isTrain', type=bool, default=True)
    parser.add_argument('--use_amp', type=bool, default=True)
    parser.add_argument('--num_iters', type=int, default=150000)
    parser.add_argument('--write_iters', type=int, default=1000)
    parser.add_argument('--save_iters', type=int, default=10000)  # 20000
    parser.add_argument('--cube_size', default=(400,640,400))
    parser.add_argument('--global_size', default=(160,640,160))
    parser.add_argument('--local_size', default=(80,320,80))
    parser.add_argument('--per_sample_time', type=int, default=20)  # 1000
    # data option
    datasetting.get_data_setting(parser, device_id=0)
    parser.add_argument('--result_dir', type=str, default='required')
    opts = parser.parse_args()
    device = init_experiment(opts, parser)

    # datasets and dataloader
    A_root = opts.train_A_root
    B_root = opts.train_B_root
    train_datasetA = Opt3D_multi_cat_Dataset([A_root], [], total_cube=True, with_aug=False, with_path=False, cube_size=opts.cube_size)
    train_datasetB = Opt3D_multi_cat_Dataset([B_root], [], total_cube=True, with_aug=False, with_path=False, cube_size=opts.cube_size)
    train_dataset = ConcatDataset([train_datasetA, train_datasetB], align=False)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=opts.num_workers)

    # instantiate network and loss function
    model = PMFAN_Trainer(opts, device)

    print_network(model, opts)

    # training part
    print('start_train')
    pbar = tqdm.tqdm(total=opts.num_iters)
    global_iter = 0

    model = model.to(device)
    model.train()
    while global_iter < opts.num_iters:
        start_time = time.time()
        for _, (dataA, dataB) in enumerate(BackgroundGenerator(train_loader)):
            real_A, = dataA
            real_B, = dataB
            real_A, real_B = real_A.to(device), real_B.to(device)
            for i in range(opts.per_sample_time):
                real_A_patch, = gen_patch_data_list([real_A], opts.cube_size,
                                                                 opts.global_size)
                real_B_patch, = gen_patch_data_list([real_B], opts.cube_size, opts.global_size)
                z_size, x_size, y_size = opts.global_size
                pos_A = (get_train_pos(z_size), get_train_pos(x_size), get_train_pos(y_size))
                real_A_patch_half, real_A_patch_local = gen_half_local_data(real_A_patch, pos_A)
                z_size, x_size, y_size = opts.global_size
                pos_B = (get_train_pos(z_size), get_train_pos(x_size), get_train_pos(y_size))
                real_B_patch_half, real_B_patch_local = gen_half_local_data(real_B_patch, pos_B)

                prepare_time = time.time() - start_time

                model.gen_dis_update_full(real_A_patch_local, real_A_patch_half, pos_A, real_B_patch_local, real_B_patch_half, pos_B, global_iter)
                model.update_learning_rate()

                global_iter += 1
                if global_iter % opts.save_iters == 0:
                    ckpt_dir = os.path.join(opts.result_dir, 'checkpoints')
                    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
                    model.save(ckpt_dir, global_iter)

                if global_iter >= opts.num_iters:

                    sys.exit('Finish training')

                pbar.update(1)
                # compute computation time and *compute_efficiency*
                process_time = time.time() - start_time - prepare_time
                pbar.set_description("Compute efficiency: {:.2f}, iter: process{:.2f}/prepare{:.2f}:".format(
                    process_time / (process_time + prepare_time), process_time, prepare_time))
                start_time = time.time()
