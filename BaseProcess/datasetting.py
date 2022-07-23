# -*- coding:utf-8 -*-
def get_data_setting(parser, device_id = 0):
    if device_id == 0:
        return device0_config(parser)

def device0_config(parser):
    parser.add_argument('--train_A_root', type=str, default='source domain OCT cube dir')
    parser.add_argument('--train_C_root', type=str,default='source domain OCTA cube dir')
    parser.add_argument('--train_B_root', type=str, default='target domain OCT cube dir')
    parser.add_argument('--result_root', type=str, default='result save dir')
    parser.add_argument('--tensorboard_log_dir', type=str, default='required')
    parser.add_argument('--num_workers', type=int, default=4)