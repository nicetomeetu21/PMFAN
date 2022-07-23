import torch
import torch.nn.functional as F
import os
import tqdm
from torchvision.utils import save_image
import numpy as np

# for coarse-to-fine train
def get_train_pos(size):
    #torch.randint(0, z_size - z_size // 2 + 1, (1,))
    pos = torch.randint(0,3,(1,))*size//4
    pos = pos.item()
    return pos

def get_train_pos_fix(size):
    #torch.randint(0, z_size - z_size // 2 + 1, (1,))
    pos = 1*size//4
    pos = pos
    return pos

def get_train_pos_random(size):

    pos = torch.randint(0, size - size // 2 + 1, (1,))
    pos = pos.item()
    return pos

def gen_half_local_data(data, local_pos):
    _,_,z,x,y = data.size()
    data_half = F.interpolate(data, size=(z//2,x//2,y//2), mode='trilinear', align_corners=True)
    pos_z, pos_x, pos_y = local_pos
    data_center = data[:,:, pos_z:pos_z+z//2,pos_x:pos_x+x//2, pos_y:pos_y+y//2]
    return data_half, data_center

def gen_patch_data_list(data_list, cube_size, patch_size):
    patch_pos = []
    for i in range(3):
        patch_pos.append(torch.randint(0, cube_size[i] - patch_size[i] + 1, (1,)))
        # print(cube_size[i], patch_size[i], patch_pos[i])
    patchs = []
    for data in data_list:
        # print(data.shape)
        patch = data[:, :, patch_pos[0]:patch_pos[0] + patch_size[0], patch_pos[1]:patch_pos[1] + patch_size[1],
                       patch_pos[2]:patch_pos[2] + patch_size[2]]
        patch = patch.contiguous()
        patchs.append(patch)
    return patchs

def cal_region_bound(pos, center, border, cube):
    if pos-border == 0:
        input_l = pos-border
        crop_l = pos-border
        crop_r = center+border*2
        save_l = pos-border
        save_r = pos+center+border
    else:
        input_l = pos-border
        crop_l = border
        crop_r = center+border*2
        save_l = pos
        save_r = pos+center+border
    if pos+center+border > cube:
        input_r = cube
    else:
        input_r = pos+center+border
    return input_l, input_r, crop_l, crop_r, save_l, save_r

def cal_region_boundv2(pos, center, border_l, border_r, cube):
    if pos-border_l == 0:
        input_l = pos-border_l
        crop_l = pos-border_l
        crop_r = pos+center+border_r
        save_l = pos-border_l
        save_r = pos+center+border_r
    else:
        input_l = pos-border_l
        crop_l = border_l
        crop_r = center+border_l+border_r
        save_l = pos
        save_r = pos+center+border_r
    if pos+center+border_r >= cube:
        input_r = cube
    else:
        input_r = pos+center+border_r
    return input_l, input_r, crop_l, crop_r, save_l, save_r


def size2str(x):
    return '%sx%sx%s' % (str(x[0]), str(x[1]), str(x[2]))
def splited_test( real_A, opts, model, outshape):
    z_cube, x_cube, y_cube = opts.cube_size
    z_per, x_per, y_per = opts.test_per_size
    fake_A = torch.zeros(real_A.shape)[:,:1,:,:,:]
    for x in range(0, x_cube, x_per):
        for y in range(0, y_cube, y_per):
            for z in range(0, z_cube, z_per):
                real_patch = real_A[:, :, z:z+z_per, x:x+x_per, y:y+y_per]
                real_patch = real_patch.cuda()
                fake_patch =model(real_patch)
                fake_A[:, :, z:z+z_per, x:x+x_per, y:y+y_per] = fake_patch
    return fake_A
# splited_test_overlap
def splited_test_overlap(real_A, opts,model, outshape=None):
    z_cube, x_cube, y_cube = opts.cube_size
    z_per, x_per, y_per = opts.test_per_size
    z_center, x_center, y_center = opts.center_size

    x_border_l = (x_per - x_center) // 2
    x_border_r = x_per-x_center-x_border_l
    y_border_l = (y_per - y_center) // 2
    y_border_r = y_per-y_center-y_border_l
    z_border_l = (z_per - z_center) // 2
    z_border_r = z_per-z_center-z_border_l

    if outshape is None:
        result = torch.zeros_like(real_A)
    else:
        result = torch.zeros(outshape)
        result = result.to(real_A.device)
    for x in range(x_border_l, x_cube - x_border_r, x_center):
        for y in range(y_border_l, y_cube - y_border_r, y_center):
            for z in range(z_border_l, z_cube - z_border_r, z_center):
                x_in_l, x_in_r, x_cr_l, x_cr_r, x_save_l, x_save_r = cal_region_boundv2(x, x_center, x_border_l,x_border_r, x_cube)
                y_in_l, y_in_r, y_cr_l, y_cr_r, y_save_l, y_save_r = cal_region_boundv2(y, y_center, y_border_l, y_border_r, y_cube)
                z_in_l, z_in_r, z_cr_l, z_cr_r, z_save_l, z_save_r = cal_region_boundv2(z, z_center, z_border_l, z_border_r, z_cube)

                input = real_A[:, :, z_in_l:z_in_r, x_in_l:x_in_r, y_in_l:y_in_r]
                output = model(input)
                result[:, :, z_save_l:z_save_r, x_save_l:x_save_r, y_save_l:y_save_r] = output[:, :, z_cr_l:z_cr_r,
                                                                                        x_cr_l:x_cr_r,
                                                                                        y_cr_l:y_cr_r]
    return result

def test_cubes(model, test_loader, test_img_save_dir, opts, device, outshape=None):
    if not os.path.exists(test_img_save_dir):
        os.makedirs(test_img_save_dir)
    model.to(device)
    model.eval()
    pbar  = tqdm.tqdm(total=len(test_loader))
    with torch.no_grad():
        for i, (data) in enumerate(test_loader):
            (real_A,), rela_dir= data

            real_A = real_A.to(device)
            fake_B = splited_test_overlap(real_A, opts, model, outshape=outshape)

            rela_dir = str(rela_dir[0])
            img_dir = os.path.join(test_img_save_dir, rela_dir)
            if not os.path.exists(img_dir): os.makedirs(img_dir)
            for j in range(fake_B.shape[2]):
                img_path = os.path.join(img_dir, str(j + 1) + '.png')
                save_image(fake_B[:, :, j, :, :], img_path)
            pbar.update(1)

    pbar.close()
def test_cubes_unoverlap(model, test_loader, test_img_save_dir, opts, device, outshape=None):
    if not os.path.exists(test_img_save_dir):
        os.makedirs(test_img_save_dir)
    model.to(device)
    model.eval()
    pbar  = tqdm.tqdm(total=len(test_loader))
    with torch.no_grad():
        for i, (data) in enumerate(test_loader):
            (real_A,), rela_dir= data

            real_A = real_A.to(device)
            fake_B = splited_test(real_A, opts, model, outshape=outshape)

            rela_dir = str(rela_dir[0])
            img_dir = os.path.join(test_img_save_dir, rela_dir)
            if not os.path.exists(img_dir): os.makedirs(img_dir)
            for j in range(fake_B.shape[2]):
                img_path = os.path.join(img_dir, str(j + 1) + '.png')
                save_image(fake_B[:, :, j, :, :], img_path)
            pbar.update(1)

    pbar.close()


# splited_test_overlap
def splited_test_overlap_2(real_A, opts, model):
    z_in_cube, x_in_cube, y_in_cube = opts.in_cube_size
    z_in_per, x_in_per, y_in_per = opts.in_per_size
    z_in_center, x_in_center, y_in_center = opts.in_center_size
    x_in_border_l = (x_in_per - x_in_center) // 2
    x_in_border_r = x_in_per-x_in_center-x_in_border_l
    y_in_border_l = (y_in_per - y_in_center) // 2
    y_in_border_r = y_in_per-y_in_center-y_in_border_l
    z_in_border_l = (z_in_per - z_in_center) // 2
    z_in_border_r = z_in_per-z_in_center-z_in_border_l

    z_out_cube, x_out_cube, y_out_cube = opts.out_cube_size
    z_out_per, x_out_per, y_out_per = opts.out_per_size
    z_out_center, x_out_center, y_out_center = opts.out_center_size
    x_out_border_l = (x_out_per - x_out_center) // 2
    x_out_border_r = x_out_per-x_out_center-x_out_border_l
    y_out_border_l = (y_out_per - y_out_center) // 2
    y_out_border_r = y_out_per-y_out_center-y_out_border_l
    z_out_border_l = (z_out_per - z_out_center) // 2
    z_out_border_r = z_out_per-z_out_center-z_out_border_l

    shape = (1,1)+opts.out_cube_size
    result = torch.zeros(shape)
    result = result.to(real_A.device)

    for x_in in range(x_in_border_l, x_in_cube - x_in_border_r, x_in_center):
        for y_in in range(y_in_border_l, y_in_cube - y_in_border_r, y_in_center):
            for z_in in range(z_in_border_l, z_in_cube - z_in_border_r, z_in_center):
                x_in_l, x_in_r, _, _, _, _ = cal_region_boundv2(x_in, x_in_center, x_in_border_l, x_in_border_r, x_in_cube)
                y_in_l, y_in_r, _, _, _, _ = cal_region_boundv2(y_in, y_in_center, y_in_border_l, y_in_border_r, y_in_cube)
                z_in_l, z_in_r, _, _, _, _ = cal_region_boundv2(z_in, z_in_center, z_in_border_l, z_in_border_r, z_in_cube)

                input = real_A[:, :, z_in_l:z_in_r, x_in_l:x_in_r, y_in_l:y_in_r]
                output = model(input)
                x_out = x_out_cube * x_in // x_in_cube
                y_out = y_out_cube * y_in // y_in_cube
                z_out = z_out_cube * z_in // z_in_cube
                _, _, x_cr_l, x_cr_r, x_save_l, x_save_r = cal_region_boundv2(x_out, x_out_center, x_out_border_l,
                                                                                        x_out_border_r, x_out_cube)
                _, _, y_cr_l, y_cr_r, y_save_l, y_save_r = cal_region_boundv2(y_out, y_out_center, y_out_border_l,
                                                                                        y_out_border_r, y_out_cube)
                _, _, z_cr_l, z_cr_r, z_save_l, z_save_r = cal_region_boundv2(z_out, z_out_center, z_out_border_l,
                                                                                        z_out_border_r, z_out_cube)

                result[:, :, z_save_l:z_save_r, x_save_l:x_save_r, y_save_l:y_save_r] = output[:, :, z_cr_l:z_cr_r,
                                                                                        x_cr_l:x_cr_r,
                                                                                        y_cr_l:y_cr_r]
    return result

def test_cubes_2(model, test_loader, test_img_save_dir, opts, device):
    if not os.path.exists(test_img_save_dir):
        os.makedirs(test_img_save_dir)
    model.to(device)
    model.eval()
    pbar  = tqdm.tqdm(total=len(test_loader))
    with torch.no_grad():
        for i, (data) in enumerate(test_loader):
            (real_A,), rela_dir= data

            real_A = real_A.to(device)
            fake_B = splited_test_overlap_2(real_A, opts, model)

            rela_dir = str(rela_dir[0])
            img_dir = os.path.join(test_img_save_dir, rela_dir)
            if not os.path.exists(img_dir): os.makedirs(img_dir)
            for j in range(fake_B.shape[2]):
                img_path = os.path.join(img_dir, str(j + 1) + '.png')
                save_image(fake_B[:, :, j, :, :], img_path)
            pbar.update(1)

    pbar.close()



def test_cubes_resize(model, test_loader, test_img_save_dir, opts, device):
    if not os.path.exists(test_img_save_dir):
        os.makedirs(test_img_save_dir)
    model.to(device)
    model.eval()
    pbar  = tqdm.tqdm(total=len(test_loader))
    with torch.no_grad():
        for i, (data) in enumerate(test_loader):
            (real_A,), rela_dir= data

            real_A = real_A.to(device)
            real_A = F.interpolate(real_A, size=(128,640,400), mode='trilinear', align_corners=True)
            real_A = F.interpolate(real_A, size=(400,640,400), mode='trilinear', align_corners=True)
            fake_B = splited_test_overlap(real_A, opts, model)

            rela_dir = str(rela_dir[0])
            img_dir = os.path.join(test_img_save_dir, rela_dir)
            if not os.path.exists(img_dir): os.makedirs(img_dir)
            for j in range(fake_B.shape[2]):
                img_path = os.path.join(img_dir, str(j + 1) + '.png')
                save_image(fake_B[:, :, j, :, :], img_path)
            pbar.update(1)

    pbar.close()

def cal(pos, center, border, cube):
    if pos-border == 0:
        input_l = pos-border
        crop_l = pos-border
        crop_r = center+border*2
        save_l = pos-border
        save_r = pos+center+border
    else:
        input_l = pos-border
        crop_l = border
        crop_r = center+border*2
        save_l = pos
        save_r = pos+center+border
    if pos+center+border > cube:
        input_r = cube
    else:
        input_r = pos+center+border
    return input_l, input_r, crop_l, crop_r, save_l, save_r
class Tester_3d():
    def __init__(self, opts):
        super(Tester_3d, self).__init__()
        self.opts = opts

    def dataset_test(self, model, test_loader, test_img_save_dir, cube_test_process):
        opts = self.opts
        if not os.path.exists(test_img_save_dir): os.makedirs(test_img_save_dir)
        model.cuda()
        model.eval()
        with tqdm.tqdm(total=len(test_loader)) as pbar:
            with torch.no_grad():
                for i, (real_A, rela_dir) in enumerate(test_loader):
                    rela_dir = str(rela_dir[0])
                    pbar.update(1)
                    fake_A = cube_test_process(real_A, model)

                    cubedir = os.path.join(test_img_save_dir, rela_dir)
                    if not os.path.exists(cubedir):
                        os.makedirs(cubedir)
                    for j in range(fake_A.shape[2]):
                        img_path = os.path.join(cubedir, str(j + 1) + '.png')
                        save_image(fake_A[:, :, j, :, :], img_path)
                    if opts.is_sample and i >= 10: break

    def dataset_test_mix(self, model, test_loader, test_img_save_dir, cube_test_process):
        opts = self.opts
        if not os.path.exists(test_img_save_dir): os.makedirs(test_img_save_dir)
        model.cuda()
        model.eval()
        with tqdm.tqdm(total=len(test_loader)) as pbar:
            with torch.no_grad():
                for i, (data1, data2) in enumerate(test_loader):
                    # print(len(data1), len(data2))
                    real_A, rela_dir = data1
                    real_A2, _ = data2
                    rela_dir = str(rela_dir[0])
                    pbar.update(1)
                    real_A = torch.cat([real_A, real_A2], dim=1)
                    fake_A = cube_test_process(real_A, model)
                    # save_image(real_A[:,0,0,:,:],'1.png')
                    # save_image(real_A[:,1,0,:,:],'2.png')
                    # save_image(fake_A[:,0,0,:,:], '3.png')
                    # exit()
                    # print(fake_A.shape)
                    cubedir = os.path.join(test_img_save_dir, rela_dir)
                    if not os.path.exists(cubedir):
                        os.makedirs(cubedir)
                    for j in range(fake_A.shape[2]):
                        img_path = os.path.join(cubedir, str(j + 1) + '.png')
                        save_image(fake_A[:, :, j, :, :], img_path)
                    if opts.is_sample and i >= 10: break
    def multi_test(self, model, test_loader_a, test_id, cube_test_type, is_mix=False):
        opts = self.opts

        if cube_test_type == 'overlap':
            cube_test_process = self.splited_test_overlap

            dir_name = str(opts.test_iter) + '_per' + size2str(opts.per_size)
            test_img_save_dir = os.path.join(opts.model_save_dir, dir_name,
                                             test_id + '_' + cube_test_type + '_' + size2str(opts.center_size),
                                             'cubes')

        elif cube_test_type == 'unoverlap':
            cube_test_process = self.splited_test

            dir_name = str(opts.test_iter)+ '_per' + size2str(opts.per_size)
            test_img_save_dir = os.path.join(opts.model_save_dir, dir_name,
                                             test_id + '_' + cube_test_type,
                                             'cubes')

        elif cube_test_type == 'overlap_local':
            cube_test_process = self.splited_test_overlap_for_local
            dir_name = str(opts.test_iter) + '_local_' + size2str(opts.per_size)+ '_global_' + size2str(opts.per_size_global)
            test_img_save_dir = os.path.join(opts.model_save_dir, dir_name,
                                             test_id + '_' + cube_test_type + '_' + size2str(opts.center_size),
                                             'cubes')

        elif cube_test_type == 'unoverlap_local':
            cube_test_process = self.splited_test_for_local

            dir_name = str(opts.test_iter) + '_local_' + size2str(opts.per_size)+ '_global_' + size2str(opts.per_size_global)
            test_img_save_dir = os.path.join(opts.model_save_dir, dir_name,
                                             test_id + '_' + cube_test_type,
                                             'cubes')
        else:
            assert False, cube_test_type + ' is invalid'

        print('test_img_save_dir: ', test_img_save_dir)
        if not is_mix:
            self.dataset_test(model, test_loader_a, test_img_save_dir, cube_test_process)
        else:
            self.dataset_test_mix(model, test_loader_a, test_img_save_dir, cube_test_process)

    def splited_test_overlap_for_local(self, real_A, model):
        opts = self.opts

        z_cube, x_cube, y_cube = opts.cube_size
        z_per, x_per, y_per = opts.per_size
        z_center, x_center,y_center = opts.center_size

        x_border = (x_per-x_center)//2
        y_border = (y_per-y_center)//2
        z_border = (z_per-z_center)//2

        result = torch.zeros(real_A.shape)[:,:1,:,:,:]
        for x in range(x_border,x_cube-x_border, x_center):
            for y in range(y_border, y_cube-y_border, y_center):
                for z in range(z_border, z_cube-z_border, z_center):
                    x_in_l, x_in_r, x_cr_l, x_cr_r, x_save_l, x_save_r = cal(x, x_center, x_border, x_cube)
                    y_in_l, y_in_r, y_cr_l, y_cr_r, y_save_l, y_save_r = cal(y, y_center, y_border, y_cube)
                    z_in_l, z_in_r, z_cr_l, z_cr_r, z_save_l, z_save_r = cal(z, z_center, z_border, z_cube)

                    local_patch, global_patch, local_pos = get_test_patch_pos(real_A, (z_in_l, x_in_l, y_in_l), opts.cube_size, opts.per_size,
                                                                              opts.per_size_global)
                    local_patch = local_patch.cuda()
                    global_patch = global_patch.cuda()
                    output = model((local_patch, global_patch, local_pos))
                    result[:, :, z_save_l:z_save_r, x_save_l:x_save_r, y_save_l:y_save_r] = output[:,:,z_cr_l:z_cr_r, x_cr_l:x_cr_r, y_cr_l:y_cr_r]
        return result

    def splited_test_for_local(self, real_A, model):
        opts = self.opts
        z_cube, x_cube, y_cube = opts.cube_size
        z_per, x_per, y_per = opts.per_size
        fake_A = torch.zeros(real_A.shape)[:,:1,:,:,:]
        for x in range(0, x_cube, x_per):
            for y in range(0, y_cube, y_per):
                for z in range(0, z_cube, z_per):
                    local_patch, global_patch, local_pos = get_test_patch_pos(real_A, (z, x, y), opts.cube_size, opts.per_size, opts.per_size_global)
                    # real_patch = real_A[:, :, z:z+z_per, x:x+x_per, y:y+y_per]
                    local_patch = local_patch.cuda()
                    global_patch = global_patch.cuda()
                    # print(local_patch.shape, global_patch.shape, local_pos)
                    fake_patch = model((local_patch, global_patch, local_pos))
                    fake_A[:, :, z:z + z_per, x:x + x_per, y:y + y_per] = fake_patch
        return fake_A


    def splited_test_overlap(self, real_A, model):
        opts = self.opts
        z_cube, x_cube, y_cube = opts.cube_size
        z_per, x_per, y_per = opts.per_size
        z_center, x_center,y_center = opts.center_size

        x_border = (x_per-x_center)//2
        y_border = (y_per-y_center)//2
        z_border = (z_per-z_center)//2

        result = torch.zeros(real_A.shape)[:,:1,:,:,:]
        for x in range(x_border,x_cube-x_border, x_center):
            for y in range(y_border, y_cube-y_border, y_center):
                for z in range(z_border, z_cube-z_border, z_center):
                    x_in_l, x_in_r, x_cr_l, x_cr_r, x_save_l, x_save_r = cal(x, x_center, x_border, x_cube)
                    y_in_l, y_in_r, y_cr_l, y_cr_r, y_save_l, y_save_r = cal(y, y_center, y_border, y_cube)
                    z_in_l, z_in_r, z_cr_l, z_cr_r, z_save_l, z_save_r = cal(z, z_center, z_border, z_cube)

                    input = real_A[:, :, z_in_l:z_in_r, x_in_l:x_in_r, y_in_l:y_in_r]
                    input = input.cuda()
                    output = model(input)
                    result[:, :, z_save_l:z_save_r, x_save_l:x_save_r, y_save_l:y_save_r] = output[:,:,z_cr_l:z_cr_r, x_cr_l:x_cr_r, y_cr_l:y_cr_r]
        return result

    def splited_test(self, real_A, model):
        opts = self.opts
        z_cube, x_cube, y_cube = opts.cube_size
        z_per, x_per, y_per = opts.per_size
        fake_A = torch.zeros(real_A.shape)[:,:1,:,:,:]
        for x in range(0, x_cube, x_per):
            for y in range(0, y_cube, y_per):
                for z in range(0, z_cube, z_per):
                    real_patch = real_A[:, :, z:z+z_per, x:x+x_per, y:y+y_per]
                    real_patch = real_patch.cuda()
                    fake_patch =model(real_patch)
                    fake_A[:, :, z:z+z_per, x:x+x_per, y:y+y_per] = fake_patch
        return fake_A

