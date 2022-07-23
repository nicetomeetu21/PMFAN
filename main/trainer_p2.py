import os
import torch
from networks.mynet_parts.init_weights import weights_init_UNIT
from networks.mynet_parts.scheduler import get_scheduler_UNIT
from BaseProcess.baseTrainer import BaseTrainer
from networks.unet.guided_unet_3d_v2 import GuidedUNet, UNet3_dec
from torch.cuda.amp import autocast,GradScaler
import torch.nn.functional as F


class PMFAN_Trainer(BaseTrainer):
    def __init__(self, opts, device):
        super(PMFAN_Trainer, self).__init__(opts, device)

        self.gen_a = GuidedUNet(opts.local_size, is_ret_global=True).to(device)  # auto-encoder for domain a
        self.gen_b = GuidedUNet(opts.local_size, is_ret_global=True).to(device)  # auto-encoder for domain b
        self.decoder = UNet3_dec().to(device)

        if opts.isTrain:

            """ optimizer and scheduler """
            gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())+ list(self.decoder.parameters())
            self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                            lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)
            hyperparameters_scheduler = dict(lr_policy='step', step_size=30000, gamma=0.5)

            self.gen_scheduler = get_scheduler_UNIT(self.gen_opt, hyperparameters_scheduler)
            self.apply(weights_init_UNIT('kaiming'))

            self.hyperparameters_w = dict(su_w = 100)
            self.mse_criterion = torch.nn.MSELoss()
            pretrained_path = 'required; stage-1 checkpoint'
            state = torch.load(pretrained_path)
            self.gen_a.load_state_dict(state['a'])
            self.gen_b.load_state_dict(state['b'])
            print('load pretrained from', pretrained_path)
            if opts.use_amp:
                print('use amp')
                self.scalar = GradScaler()

    def gen_dis_update_full(self,  x_a, x_a_global, x_a_local_pos, y_a_real, x_ab, x_ab_global, global_iter):
        if self.opts.use_amp:
            with autocast():
                h_a, _ = self.gen_a.encode(global_img = x_a_global, local_img = x_a, local_pos=x_a_local_pos)
                h_ab, _ = self.gen_b.encode(global_img = x_ab_global, local_img = x_ab, local_pos=x_a_local_pos)

                y_a = self.decoder.decode(h_a)
                y_ab = self.decoder.decode(h_ab)

                self.loss_f_su = self.mse_criterion(y_ab, y_a_real)+self.mse_criterion(y_a, y_a_real)
                self.loss_gen_total = self.hyperparameters_w['su_w'] * self.loss_f_su

            self.gen_opt.zero_grad()
            self.scalar.scale(self.loss_gen_total).backward()
            self.scalar.step(self.gen_opt)
            self.scalar.update()

        if global_iter % self.opts.write_iters == 0:
            self.write_loss(global_iter)

            size_z, size_x, size_y = self.opts.local_size
            pos_z, pos_x, pos_y = x_a_local_pos
            x_a_global_upsampled_stricted = F.interpolate(x_a_global, scale_factor=2)[:,:,pos_z:pos_z+ size_z,pos_x:pos_x+size_x,pos_y:pos_y+size_y]
            x_ab_global_upsampled_stricted = F.interpolate(x_ab_global, scale_factor=2)[:,:,pos_z:pos_z+ size_z,pos_x:pos_x+size_x,pos_y:pos_y+size_y]

            imgs = self.add_imgs([x_a, x_ab,x_a_global_upsampled_stricted, x_ab_global_upsampled_stricted, y_a_real, y_a, y_ab])
            self.writer.add_images('train_A_imgs', torch.cat(imgs, dim=0), global_iter)


    def add_imgs(self,imgs):
        ret = []
        for img in imgs:
            for i in range(4):
                ret.append(img[:, :, i, :, :])
        return ret
    def update_learning_rate(self):
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations))
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict(), 'dec': self.decoder.state_dict()}, gen_name)
        env_name = os.path.join(snapshot_dir, 'train_env.pt' )
        torch.save({'global_iters': iterations,
                    'gen_opt': self.gen_opt.state_dict(),
                    'gen_scheduler': self.gen_scheduler.state_dict()
                    },
                   env_name)



