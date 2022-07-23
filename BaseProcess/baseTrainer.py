import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
class BaseTrainer(nn.Module):
    def __init__(self, opts, device):
        super(BaseTrainer, self).__init__()
        self.opts = opts
        self.device = device

        if opts.isTrain:
            log_dir = opts.tensorboard_log_dir
            if log_dir == 'runs':
                log_dir = os.path.join(opts.result_dir, 'runs')
            self.writer = SummaryWriter(log_dir)
            print("track view:", log_dir)

            self.snapshot_dir = os.path.join(opts.result_dir, 'checkpoints')
            if not os.path.exists(self.snapshot_dir): os.makedirs(self.snapshot_dir)


    def write_loss(self, global_iter):
        members = [attr for attr in dir(self) \
                   if not callable(getattr(self, attr)) and not attr.startswith("__") and (
                           'loss' in attr or 'grad' in attr or 'nwd' in attr)]
        for m in members:
            self.writer.add_scalar(m, getattr(self, m), global_iter)