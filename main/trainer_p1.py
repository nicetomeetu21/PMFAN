import os
import torch
import networks.networks_for_UNIT3D as networks
from networks.mynet_parts.init_weights import weights_init_UNIT
from networks.mynet_parts.scheduler import get_scheduler_UNIT
from BaseProcess.baseTrainer import BaseTrainer
from networks.unet.guided_unet_3d_v2 import GuidedUNet
from torch.cuda.amp import autocast,GradScaler
import torch.nn.functional as F


class PMFAN_Trainer(BaseTrainer):
    def __init__(self, opts, device):
        super(PMFAN_Trainer, self).__init__(opts, device)

        self.gen_a = GuidedUNet(opts.local_size, is_ret_global=True).to(device)  # auto-encoder for domain a
        self.gen_b = GuidedUNet(opts.local_size, is_ret_global=True).to(device)  # auto-encoder for domain b

        if opts.isTrain:
            hyperparameters_dis = dict(dim=32, norm='none', activ='lrelu', n_layer=3, gan_type='lsgan', num_scales=2,
                                       pad_type='reflect')
            self.dis_a = networks.MsImageDis(1, hyperparameters_dis).to(device)  # discriminator for domain a
            self.dis_b = networks.MsImageDis(1, hyperparameters_dis).to(device)  # discriminator for domain b

            """ optimizer and scheduler """
            dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
            gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
            self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                            lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)
            self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                            lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)
            hyperparameters_scheduler = dict(lr_policy='step', step_size=30000, gamma=0.5)

            self.dis_scheduler = get_scheduler_UNIT(self.dis_opt, hyperparameters_scheduler)
            self.gen_scheduler = get_scheduler_UNIT(self.gen_opt, hyperparameters_scheduler)
            self.apply(weights_init_UNIT('kaiming'))
            self.dis_a.apply(weights_init_UNIT('gaussian'))
            self.dis_b.apply(weights_init_UNIT('gaussian'))

            self.hyperparameters_w = dict(gan_w=1, recon_x_w=10, recon_x_cyc_w=10, su_w = 100)
            self.recon_criterion = torch.nn.L1Loss()
            self.mse_criterion = torch.nn.MSELoss()

            if opts.use_amp:
                print('use amp')
                self.scalar = GradScaler()

    def gen_dis_update_full(self,  x_a, x_a_global, x_a_local_pos, y_a_real, x_b, x_b_global, x_b_local_pos, global_iter):
        if self.opts.use_amp:
            with autocast():
                h_a, x_ab_global = self.gen_a.encode(global_img = x_a_global, local_img = x_a, local_pos=x_a_local_pos)
                h_b, x_ba_global = self.gen_b.encode(global_img = x_b_global, local_img = x_b, local_pos=x_b_local_pos)
                x_ba = self.gen_a.decode(h_b)
                x_ab = self.gen_b.decode(h_a)
                loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
                loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
                self.loss_dis_total = self.hyperparameters_w['gan_w'] * (loss_dis_a + loss_dis_b)

                x_a_recon = self.gen_a.decode(h_a)
                x_b_recon = self.gen_b.decode(h_b)
                # encode again
                h_ab, x_aba_global = self.gen_b.encode(global_img = x_ab_global, local_img = x_ab, local_pos = x_a_local_pos)
                h_ba, x_bab_global = self.gen_a.encode(global_img = x_ba_global, local_img = x_ba, local_pos = x_b_local_pos)
                # decode again (if needed)
                x_aba = self.gen_a.decode(h_ab)
                x_bab = self.gen_b.decode(h_ba)

                self.loss_gen_recon_x = self.recon_criterion(x_a_recon, x_a) + self.recon_criterion(x_b_recon, x_b)
                self.loss_gen_cyc_x = self.recon_criterion(x_aba, x_a)+self.recon_criterion(x_bab, x_b)
                self.loss_gen_cyc_h = self.recon_criterion2(h_a, h_ab)+self.recon_criterion2(h_b, h_ba) if self.hyperparameters_w[
                    'recon_x_cyc_w'] else 0
                self.loss_gen_adv = self.dis_a.calc_gen_loss(x_ba) + self.dis_b.calc_gen_loss(x_ab)

                self.loss_gen_total = self.hyperparameters_w['gan_w'] * self.loss_gen_adv + \
                                      self.hyperparameters_w['recon_x_w'] * self.loss_gen_recon_x + \
                                      self.hyperparameters_w['recon_x_cyc_w'] * self.loss_gen_cyc_x + \
                                      self.hyperparameters_w['recon_x_cyc_w'] * self.loss_gen_cyc_h

            self.dis_opt.zero_grad()
            self.scalar.scale(self.loss_dis_total).backward()
            self.scalar.step(self.dis_opt)

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
            x_aba_global_upsampled_stricted = F.interpolate(x_aba_global, scale_factor=2)[:,:,pos_z:pos_z+ size_z,pos_x:pos_x+size_x,pos_y:pos_y+size_y]
            pos_z, pos_x, pos_y = x_b_local_pos
            x_b_global_upsampled_stricted = F.interpolate(x_b_global, scale_factor=2)[:,:,pos_z:pos_z+ size_z,pos_x:pos_x+size_x,pos_y:pos_y+size_y]
            x_ba_global_upsampled_stricted = F.interpolate(x_ba_global, scale_factor=2)[:,:,pos_z:pos_z+ size_z,pos_x:pos_x+size_x,pos_y:pos_y+size_y]
            x_bab_global_upsampled_stricted = F.interpolate(x_bab_global, scale_factor=2)[:,:,pos_z:pos_z+ size_z,pos_x:pos_x+size_x,pos_y:pos_y+size_y]

            imgs = self.add_imgs([x_a,x_a_recon, x_ab,x_aba,x_a_global_upsampled_stricted, x_ab_global_upsampled_stricted, x_aba_global_upsampled_stricted])
            self.writer.add_images('train_A_imgs', torch.cat(imgs, dim=0), global_iter)
            imgs = self.add_imgs([x_b,x_b_recon, x_ba, x_bab, x_b_global_upsampled_stricted, x_ba_global_upsampled_stricted, x_bab_global_upsampled_stricted])
            self.writer.add_images('train_B_imgs', torch.cat(imgs, dim=0), global_iter)

    def add_imgs(self,imgs):
        ret = []
        for img in imgs:
            for i in range(4):
                ret.append(img[:, :, i, :, :])
        return ret
    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations))
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        env_name = os.path.join(snapshot_dir, 'train_env.pt' )
        torch.save({'global_iters': iterations,
                    'gen_opt': self.gen_opt.state_dict(),
                    'gen_scheduler': self.gen_scheduler.state_dict(),
                    'dis_opt': self.dis_opt.state_dict(),
                    'dis_scheduler': self.dis_scheduler.state_dict(),
                    'dis_a':self.dis_a.state_dict(),
                    'dis_b':self.dis_b.state_dict()
                    },
                   env_name)
