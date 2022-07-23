# -*- coding:utf-8 -*-
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from MSUNIT_v3_noada_trainer import MSUNIT_Trainer
from BaseProcess import datasetting
from BaseProcess.initExperiment import init_experiment
from datasets.dataset_for_3d import Opt3D_multi_cat_Dataset
from utils.util_for_3d import test_cubes

""" set flags / seeds """
torch.backends.cudnn.benchmark = True
class ModelTest(nn.Module):
    def __init__(self, gen_a, gen_b):
        super(ModelTest, self).__init__()
        self.gen_a = gen_a
        self.gen_b = gen_b
    def forward(self, input):
        x_a = input
        content = self.gen_a.encode(x_a, None)
        x_ab = self.gen_b.decode(content)
        output = x_ab
        return output

if __name__ == "__main__":
    """ Hpyer parameters """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model_save_dir', type=str, default='xxx/checkpoints')
    parser.add_argument('--test_iter', type=int, default=100000)
    parser.add_argument('--visible_devices', type=str, default='2')
    parser.add_argument('--isTrain', type=bool, default=False)
    parser.add_argument('--test_A2O', type=bool, default=False)
    parser.add_argument('--test_A2B2O', type=bool, default=False)
    parser.add_argument('--test_B2O', type=bool, default=True)
    parser.add_argument('--test_B2O_2', type=bool, default=False)
    parser.add_argument('--cube_size', default=(400,640,400))
    parser.add_argument('--test_per_size', default=(80,320,80))
    parser.add_argument('--center_size', default=(40,160,40))
    datasetting.get_data_setting(parser, device_id=0)
    opts = parser.parse_args()

    device = init_experiment(opts, parser)

    gen_name = os.path.join(opts.model_save_dir, 'gen_%08d.pt' % (opts.test_iter))
    model = MSUNIT_Trainer(opts=opts, device=device)
    model.gen_a.load_state_dict(torch.load(gen_name)['a'])
    model.decoder.load_state_dict(torch.load(gen_name)['dec'])
    model.gen_b.load_state_dict(torch.load(gen_name)['b'])


    if opts.test_A2O:
        data_root = opts.test_A2_root
        save_dir = os.path.join(opts.model_save_dir, str(opts.test_iter), 'test_A2O_2')
        model_test = ModelTest(model.gen_a, model.decoder)
        test_dataset = Opt3D_multi_cat_Dataset([data_root], [], total_cube=True, with_path=True, with_aug=False, cube_size=opts.cube_size)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opts.num_workers)
        print('test_dataset len:', len(test_loader))
        test_cubes(model_test, test_loader, save_dir, opts, device)

    if opts.test_A2B2O:
        data_root = '/home/Data2/huangkun/DATA/result/MSUNIT_v3/MSUNIT_v3_cycle_recon_ada_p1_amp/checkpoints/100000/test_A2B_2'
        save_dir = os.path.join(opts.model_save_dir, str(opts.test_iter), 'test_A2B2O_2')
        model_test = ModelTest(model.gen_a, model.decoder)
        test_dataset = Opt3D_multi_cat_Dataset([data_root], [], total_cube=True, with_path=True, with_aug=False, cube_size=opts.cube_size)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opts.num_workers)
        print('test_dataset len:', len(test_loader))
        test_cubes(model_test, test_loader, save_dir, opts, device)

    if opts.test_B2O_2:
        data_root = opts.test_B2_root
        save_dir = os.path.join(opts.model_save_dir, str(opts.test_iter), 'test_B2O_2')
        model_test = ModelTest(model.gen_b, model.decoder)
        test_dataset = Opt3D_multi_cat_Dataset([data_root], [], total_cube=True, with_path=True, with_aug=False, cube_size=opts.cube_size)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opts.num_workers)
        print('test_dataset len:', len(test_loader))
        test_cubes(model_test, test_loader, save_dir, opts, device)

    if opts.test_B2O:
        data_root = opts.test_B_root
        save_dir = os.path.join(opts.model_save_dir, str(opts.test_iter), 'test_B2O')
        model_test = ModelTest(model.gen_b, model.decoder)
        test_dataset = Opt3D_multi_cat_Dataset([data_root], [], total_cube=True, with_path=True, with_aug=False, cube_size=opts.cube_size)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opts.num_workers)
        print('test_dataset len:', len(test_loader))
        test_cubes(model_test, test_loader, save_dir, opts, device)