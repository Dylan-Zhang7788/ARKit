import os
import pickle
import glob
import traceback

import scipy.io
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import face_alignment
import nvdiffrast.torch as dr
from utils.inference import parse_param, crop_img, param2vert, param_pack, ParamsPack
from utils.params import ParamsPackLarge
from face_parsing.test import Predictor

# torch.multiprocessing.set_start_method('spawn')
class Shader(object):

    @staticmethod
    def _lambertian_attentuation():
        """ constant weight from sfsnet matlab """
        return np.pi * np.array([1, 2.0 / 3, 1.0 / 4])

    @staticmethod
    def _harmonics(ver_norm, order=2):
        """compute the spherical harmonics function for 3D vertices.
        :param:
            ver_norm: [batch, N, 3], vertex normal

        :return:
            H: [batch, 9], 2-order harmonic basis
        """
        lam_attn = Shader._lambertian_attentuation()

        x, y, z = torch.chunk(ver_norm, 3, -1)
        x2 = torch.square(x)
        y2 = torch.square(y)
        z2 = torch.square(z)
        xy = x * y
        yz = y * z
        xz = x * z
        PI = np.pi

        l0 = np.sqrt(1.0 / (4 * PI)) * torch.ones_like(x)
        l1x = np.sqrt(3.0 / (4 * PI)) * x
        l1y = np.sqrt(3.0 / (4 * PI)) * y
        l1z = np.sqrt(3.0 / (4 * PI)) * z
        l2xy = np.sqrt(15.0 / (4 * PI)) * xy
        l2yz = np.sqrt(15.0 / (4 * PI)) * yz
        l2xz = np.sqrt(15.0 / (4 * PI)) * xz
        l2z2 = np.sqrt(5.0 / (16 * PI)) * (3 * z2 - 1)
        l2x2_y2 = np.sqrt(15.0 / (16 * PI)) * (x2 - y2)
        H = torch.concat(
            [l0, l1z, l1x, l1y, l2z2, l2xz, l2yz, l2x2_y2, l2xy], -1)
        if order == 3:
            b9 = 1.0 / 4.0 * np.sqrt(35.0 / (2 * PI)) * (3 * x2 - z2) * z
            b10 = 1.0 / 2.0 * np.sqrt(105.0 / PI) * x * yz
            b11 = 1.0 / 4 * np.sqrt(21.0 / (2 * PI)) * z * (4 * y2 - x2 - z2)
            b12 = 1.0 / 4 * np.sqrt(7.0 / PI) * y * (2 * y2 - 3 * x2 - 3 * z2)
            b13 = 1.0 / 4 * np.sqrt(21.0 / (2 * PI)) * x * (4 * y2 - x2 - z2)
            b14 = 1.0 / 4 * np.sqrt(105.0 / PI) * (x2 - z2) * y
            b15 = 1.0 / 4 * np.sqrt(35.0 / (2 * PI)) * (x2 - 3 * z2) * x
            H = torch.concat(
                [H, b9, b10, b11, b12, b13, b14, b15], -1)
        batch_size, img_height, img_width, _ = ver_norm.shape
        return H

    @staticmethod
    def sh_shader(normals, alphas, background_images, sh_coefficients, diffuse_colors):
        """
        render mesh into image space and return all intermediate results.
        :param:
            normals: [batch,300,300,3], vertex normals in image space
            alphas: [batch,H,W,1], alpha channels
            background_images: [batch,H,W,3], background images for rendering results
            sh_coefficient: [batch,27], 2-order SH coefficient
            diffuse_colors: [batch,H,W,3], vertex colors in image space

        sh_coefficient: [batch_size, 27] spherical harmonics coefficients.
        """
        batch_size, image_height, image_width = [s for s in normals.shape[:-1]]

        sh_coef_count = sh_coefficients.shape[-1]

        if sh_coef_count == 27:
            init_para_illum = torch.Tensor([1] + [0] * 8)
            init_para_illum = torch.reshape(init_para_illum, [1,9])

            init_para_illum = torch.concat(
                [init_para_illum] * 3, dim=1)

            sh_coefficients = sh_coefficients + init_para_illum.to(sh_coefficients.device)  # batch x 27
            order = 2
        else:

            init_para_illum = torch.Tensor([1.0]*2 + [0] * 14)
            init_para_illum = torch.reshape(init_para_illum, [1, 16])
            init_para_illum = torch.concat(
                [init_para_illum] * 3, dim=1)
            sh_coefficients = sh_coefficients + init_para_illum.to(sh_coefficients.device)
            sh_coefficients = sh_coefficients.view([-1, 1, 1, 3, 16])
            sh_coefficients  = sh_coefficients.repeat([1, image_height, image_width, 1, 1])
            order = 3
        batch_size = diffuse_colors.shape[0]
        sh_kernels = torch.chunk(sh_coefficients, batch_size, 0)
        harmonic_output = Shader._harmonics(normals, order).to(sh_coefficients.device)
        harmonic_output_list = torch.chunk(harmonic_output, batch_size, axis=0)

        results = []
        for ho, shk in zip(harmonic_output_list, sh_kernels):
            shk = shk.view([3, 9]).t().view([1, 1, 9, 3])
            res = F.conv2d(ho.permute(0,3,1,2), shk.permute(3, 2, 0, 1), stride=1, padding='same').permute(0,2,3,1)
            results.append(res)
        shading = torch.concat(results, 0)

        rgb_images = shading * diffuse_colors

        alpha_images = alphas.view([-1, image_height, image_width, 1])
        valid_rgb_values = torch.concat(3 * [alpha_images > 0.5], 3)
        rgb_images = torch.where(valid_rgb_values, rgb_images, background_images)

        return rgb_images, shading

    @staticmethod
    def remove_shading(images, image_normals, sh_coefficients):

        init_para_illum = torch.Tensor([1] + [0] * 8)
        init_para_illum = torch.concat([init_para_illum] * 3, 1)
        sh_coefficients = sh_coefficients + init_para_illum

        _, image_height, image_width = [s.value for s in image_normals.shape[:-1]]
        sh_coefficients = sh_coefficients.view([-1, 1, 1, 3, 9])
        sh_coefficients = sh_coefficients.repeat([1, image_height, image_width, 1, 1])
        harmonic_output = Shader._harmonics(image_normals).unsqueeze(-1)
        shading = torch.squeeze(torch.matmul(sh_coefficients, harmonic_output))
        diffuse_maps = images / (shading + 1e-18)
        return diffuse_maps

class ParamPackTorch(ParamsPack):
    def __init__(self, device):
        super().__init__()
        self.u_base = torch.from_numpy(self.u_base).to(device).float()
        self.w_shp_base = torch.from_numpy(self.w_shp_base).to(device).float()
        self.w_exp_base = torch.from_numpy(self.w_exp_base).to(device).float()
        self.u = torch.from_numpy(self.u).to(device).float()
        self.w_shp = torch.from_numpy(self.w_shp).to(device).float()
        self.w_exp = torch.from_numpy(self.w_exp).to(device).float()
        self.w_tex = torch.from_numpy(np.load('3dmm_data/w_tex_sim.npy')).to(device).float().permute(1, 0)
        self.param_std = torch.from_numpy(self.param_std).to(device).float()
        self.param_mean = torch.from_numpy(self.param_mean).to(device).float()

class ParamsPackLargeTorch(ParamsPackLarge):
    def __init__(self, device):
        super().__init__()
        self.u = torch.from_numpy(self.u).to(device).float()
        self.tex_mean = torch.from_numpy(self.tex_mean).to(device).float().contiguous()
        self.w_shp = torch.from_numpy(self.w_shp).to(device).float()
        self.w_exp = torch.from_numpy(self.w_exp).to(device).float()
        self.w_tex = torch.from_numpy(self.w_tex).to(device).float().permute(1, 0)
        self.keypoints = torch.from_numpy(self.keypoints).to(device).int()
        self.triangles = torch.from_numpy(self.triangles).to(device).int()
        self.param_std = torch.from_numpy(self.param_std).to(device).float()
        self.param_mean = torch.from_numpy(self.param_mean).to(device).float()


class ImageMimic(object):
    def __init__(self, device):
        self.device = device
        self.param_pack_torch = ParamPackTorch(device)
        self.param_pack_torch_large = ParamsPackLargeTorch(device)
        self.triangles = scipy.io.loadmat('./3dmm_data/tri.mat')['tri'] - 1
        self.triangles = torch.from_numpy(self.triangles.astype(np.int32)).to(device).permute(1, 0).contiguous()
        self.face_points = set(np.load('./3dmm_data/keptInd.npy'))
        self.face_triangles = torch.tensor([tri for tri in self.triangles.tolist() if all([p in self.face_points for p in tri])]).int().to(device).contiguous()
        self.macro_glctx = dr.RasterizeCudaContext(device=self.device)
        self.image_path = "/home/momo/liuchengyu/landmark/synth_data/imgs/020400.png"
        self.tex_mean = torch.from_numpy(np.load('3dmm_data/u_tex.npy').reshape(1, -1, 3)[..., ::-1].copy()).to(device).contiguous()
        self.face_parsing = Predictor("./face_parsing/res/cp/79999_iter.pth")
        self.face_align = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)
        self.train_aug_data = pickle.load(open('3dmm_data/param_all_norm_v201.pkl', 'rb'))
        self.train_aug_list = [i.strip() for i in open('3dmm_data/train_aug_120x120.list.train')]
        self.train_aug_dict = {k: v for k, v in zip(self.train_aug_list, self.train_aug_data)}

    @staticmethod
    def ralign(X, Y):
        """
        Rigid alignment between two pointsets using Umeyama algorithm

        X           (n x m) points (i.e., m=3 or 2)
        Y           (n x m) points
        """

        X = X.T
        Y = Y.T

        m, n = X.shape

        mx = X.mean(1)
        my = Y.mean(1)
        Xc = X - np.tile(mx, (n, 1)).T
        Yc = Y - np.tile(my, (n, 1)).T

        sx = np.mean(np.sum(Xc*Xc, 0))

        Sxy = np.dot(Yc, Xc.T) / n

        U, D, V = np.linalg.svd(Sxy, full_matrices=True, compute_uv=True)
        V = V.T.copy()

        S = np.eye(m)

        R = np.dot(np.dot(U, S), V.T)

        s = np.trace(np.dot(np.diag(D), S)) / sx
        t = my - s * np.dot(R, mx)

        return R, s, t

    def param2vert(self, param, dense=False, transform=True, param_pack=param_pack):
        if param.shape[0] == 62:
            param_ = param * param_pack.param_std[:62] + param_pack.param_mean[:62]
        else:
            raise RuntimeError('length of params mismatch')

        p, offset, alpha_shp, alpha_exp = parse_param(param_)

        if dense:
            if isinstance(param_pack, ParamPackTorch):
                vertex = p @ (param_pack.u + param_pack.w_shp @ alpha_shp + param_pack.w_exp @ alpha_exp).reshape(-1, 3).permute(1, 0) + offset
            else:
                vertex = p @ (param_pack.u + param_pack.w_shp @ alpha_shp + param_pack.w_exp @ alpha_exp).reshape(3, -1, order='F') + offset
            if transform:
                # transform to image coordinate space
                vertex[1, :] = param_pack.std_size + 1 - vertex[1, :]

        else:
            if isinstance(param_pack, ParamPackTorch):
                vertex = p @ (param_pack.u_base + param_pack.w_shp_base @ alpha_shp + param_pack.w_exp_base @ alpha_exp).reshape(-1, 3).permute(1, 0) + offset
            else:
                vertex = p @ (param_pack.u_base + param_pack.w_shp_base @ alpha_shp + param_pack.w_exp_base @ alpha_exp).reshape(3, -1, order='F') + offset
            if transform:
                """Work for only tensor"""
                # transform to image coordinate space
                vertex[1, :] = param_pack.std_size + 1 - vertex[1, :]

        return vertex

    def parse_param_batch(self, param):
        p_ = param[:, :12].reshape(-1, 3, 4)
        p = p_[..., :3]
        offset = p_[..., -1].reshape(-1, 3, 1)
        if param.shape[1] == 62:
            alpha_shp = param[:, 12:52].reshape(-1, 40, 1)
            alpha_exp = param[:, 52:62].reshape(-1, 10, 1)
        else:
            alpha_shp = param[:, 12:92].reshape(-1, 80, 1)
            alpha_exp = param[:, 92:].reshape(-1, 64, 1)
        return p, offset, alpha_shp, alpha_exp


    def param2vert_batch(self, params, dense=False, transform=True, param_pack=param_pack):
        # support Torch Input Only
        # assert isinstance(param_pack, ParamPackTorch)
        if params.shape[1] == 62:
            param_ = params * param_pack.param_std[None, :62] + param_pack.param_mean[None, :62]
        elif params.shape[1] == 156:
            param_ = params.clone()
            param_[:, :12] = params[:, :12] * param_pack.param_std[None, :12] + param_pack.param_mean[None, :12]
        else:
            raise RuntimeError('length of params mismatch')

        p, offset, alpha_shp, alpha_exp = self.parse_param_batch(param_)
        bs = len(params)

        if dense:
            vertex = p @ (param_pack.u[None, ...] + param_pack.w_shp[None, ...] @ alpha_shp + param_pack.w_exp[None, ...] @ alpha_exp).reshape(bs, -1, 3).permute(0, 2, 1) + offset
            if transform:
                # transform to image coordinate space
                vertex[:, 1, :] = param_pack.std_size + 1 - vertex[:, 1, :]

        else:
            vertex = p @ (param_pack.u_base[None, ...] + param_pack.w_shp_base[None, ...] @ alpha_shp + param_pack.w_exp_base[None, ...] @ alpha_exp).reshape(bs, -1, 3).permute(0, 2, 1) + offset
            if transform:
                """Work for only tensor"""
                # transform to image coordinate space
                vertex[:, 1, :] = param_pack.std_size + 1 - vertex[:, 1, :]

        return vertex

    def get_ver_norm(self, ver_xyz, tri):
        """
        CHECK PASSED
        Compute vertex normals.

        :param:
            ver_xyz: [batch, N, 3], vertex geometry
            tri: [M, 3], mesh triangles definition

        :return:
            ver_normals: [batch, N, 3], vertex normals
        """

        tri = tri.long()
        v1_idx, v2_idx, v3_idx = torch.unbind(tri, dim=-1) # (M, )
        v1 = ver_xyz[:, v1_idx, :]
        v2 = ver_xyz[:, v2_idx, :]
        v3 = ver_xyz[:, v3_idx, :]

        B = v1.shape[0]  # batch size
        EPS = 1e-8

        # compute normals of all triangles
        tri_normals = torch.cross(v2 - v1, v3 - v1) # (bs, M, 3)
        tri_normals = torch.div(
            tri_normals,
            (torch.norm(tri_normals, dim=-1, keepdim=True) + EPS)
        ) # (bs, M, 3)
        tri_normals = torch.tile(torch.unsqueeze(tri_normals, 2), (1, 1, 3, 1)) # (bs, M, 3, 3)
        tri_normals = torch.reshape(tri_normals, (B, -1, 3)) # (bs, M*3, 3)
        tri_votes = (torch.greater(tri_normals[:, :, 2:], float(0.1))).float() # (bs, M*3, 1)
        tri_cnts = torch.ones_like(tri_votes, device=self.device) # (bs, M*3, 1)

        # vertex normals conducted by mean of shared triangle norms
        tri_inds = torch.tile(tri.reshape(-1)[None, ...], (B, 1))
        ver_normals = torch.zeros_like(ver_xyz, device=self.device) #, requires_grad=True) # (bs, N, 3)
        ver_normals.index_add_(dim=1, index=tri_inds[0], source=tri_normals)
        ver_normals = ver_normals / (
            torch.norm(ver_normals, dim=2, keepdim=True) + EPS
        )

        ver_shape = ver_xyz.shape
        ver_votes = torch.zeros(size=ver_shape[:-1] + (1,), device=self.device)
        ver_cnts = torch.zeros(size=ver_shape[:-1] + (1,), device=self.device)
        ver_votes = ver_votes.index_add_(dim=1, index=tri_inds[0], source=tri_votes)
        ver_cnts = ver_cnts.index_add_(dim=1, index=tri_inds[0], source=tri_cnts)
        ver_votes = ver_votes / (ver_cnts + EPS)

        ver_votes1 = torch.less(ver_votes, float(1.0))
        ver_votes2 = torch.greater(ver_votes, float(0.0))
        ver_votes = torch.logical_and(ver_votes1, ver_votes2).float()

        return ver_normals, ver_votes

    def rasterize(self, vertices, attributes, triangles, image_width, image_height):
        batch_size = vertices.shape[0]
        vertex_count = vertices.shape[1]
        num_channels = vertices.shape[2]
        if num_channels == 3:
            homogeneous_coord = torch.ones([batch_size, vertex_count, 1], dtype=torch.float32, device=self.device)
            vertices = torch.concat([vertices, homogeneous_coord], 2)
        rast, _ = dr.rasterize(self.macro_glctx, vertices, triangles, resolution=[image_width, image_height])
        attributes, _ = dr.interpolate(attributes, rast, triangles)
        alphas = rast[..., -1:].to(torch.bool).float()
        return attributes, alphas

    def render(self, light_coefs, vertex, tex, triangles, ori_image, image_size=[256, 256]):
        vertex_norm, _ = self.get_ver_norm(vertex, triangles)
        norm_image, _ = self.rasterize(vertex, vertex_norm.contiguous(), triangles, image_size[0], image_size[1])
        rast_out, mask = self.rasterize(vertex, tex, triangles, image_size[0], image_size[1])
        rgb_image, _ = Shader.sh_shader(norm_image, mask, ori_image, light_coefs, rast_out)
        out_image = rgb_image * mask
        return out_image, mask

    def load_image(self, image_path):
        if 'train_aug' in image_path:
            cropped = cv.imread(image_path)
            h = cropped.shape[0]
            mask = self.face_parsing.predict_single(cropped)
            params = torch.from_numpy(self.train_aug_dict[image_path.split('/')[-1]]).to(self.device).float()
            ldmks = self.param2vert_batch(params[None, :62], dense=False, param_pack=self.param_pack_torch).float()
        else:
            mask_path = ''
            ldmks_path = ''
            image = cv.imread(image_path)
            if 'synth_data/imgs' in image_path:
                mask_path = image_path.replace('synth_data/imgs/', 'synth_data/mask/').replace('.png', '_seg.png')
                ldmks_path = image_path.replace('/imgs/', '/lmks/').replace('.png', '_ldmks.txt')
                synet_out = np.load(image_path.replace('synth_data/imgs', 'SynergyNet/inference_output/synergy_out').replace('.png', '_saver.npz'))
            elif 'face3d_eval_data' in image_path:
                synet_out = np.load("/mnt/data2/zhang.hongshuang/inference_output/landmarks/{}.npz".format(image_path.split('/')[-1][:-4]))
                image = cv.flip(cv.transpose(image), 1)


            else:
                synet_out = np.load('/home/momo/liuchengyu/landmark/SynergyNet/inference_output/synergy_outb/{}_saver.npz'.format(image_path.split('/')[-1].split('.')[0]))
            roi_box = synet_out['roi_box']
            params = torch.from_numpy(synet_out['params']).to(self.device)
            cropped = crop_img(image, roi_box)
            h, w = cropped.shape[:2]

            if os.path.exists(mask_path):
                mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
                mask = crop_img(mask, roi_box)
            else:
                mask = self.face_parsing.predict_single(cropped)

            if os.path.exists(ldmks_path):
                ldmks = np.loadtxt(ldmks_path)[:-2]
            else:
                ldmks = self.face_align.get_landmarks(image)[0][:, :2]
            ldmks -= np.array([roi_box[:2]])
            ldmks = torch.from_numpy(ldmks).to(self.device).float()
        # ldmks /= h/120

        image_mask = np.logical_and(mask <13, mask > 0).astype(np.int32)
        image_mask = cv.resize(image_mask.astype(np.uint8), (256, 256))[None, ...]
        image_mask = image_mask[..., None]
        compare_image = cv.resize(cropped, (256, 256))
        compare_image = torch.from_numpy(compare_image).to(self.device).float()
        image_mask = torch.from_numpy(image_mask).to(self.device).float()

        return compare_image[None, ...], image_mask, ldmks, params, h

    def batch_process(self, image_paths, steps=100):
        images = []
        masks = []
        ldmkss = []
        paramss = []
        ratios = []
        for image_path in image_paths:
            try:
                image, mask, ldmks, params, h = self.load_image(image_path)
                images.append(image)
                masks.append(mask)
                ldmkss.append(ldmks)
                paramss.append(params)
                ratios.append(h/120)
            except:
                traceback.print_exc()

        bs = len(images)
        images = torch.cat(images, dim=0)
        masks = torch.cat(masks, dim=0)
        paramss = torch.cat([param[None, ...] for param in paramss], dim=0).requires_grad_()
        ldmkss = torch.cat([ldmks[None, ...] for ldmks in ldmkss], dim=0)
        x_tex = torch.zeros((bs, 199)).to(self.device).requires_grad_()
        x_illum = torch.zeros((bs, 27)).to(self.device).requires_grad_()
        ratios = torch.tensor(ratios).to(self.device)
        optim = torch.optim.Adam([paramss, x_tex, x_illum], lr=1e-1)

        for i in range(steps):
            pred_ldmks = self.param2vert_batch(paramss, param_pack=self.param_pack_torch).permute(0, 2, 1)[..., :2] * ratios[..., None, None]
            pred_vertex = self.param2vert_batch(paramss, dense=True, param_pack=self.param_pack_torch).permute(0, 2, 1)
            pred_vertex = pred_vertex / 60 - 1
            vx, vy, vz = torch.unbind(pred_vertex, dim=-1)
            vz = vz/vz.min()
            pred_vertex_normed = torch.cat([vx[..., None], vy[..., None], vz[..., None]], dim=2)
            tex_pred = (x_tex @ self.param_pack_torch.w_tex).reshape(bs, -1, 3) + self.tex_mean
            out = self.render(x_illum, pred_vertex_normed.contiguous(), tex_pred.contiguous(), self.triangles, image, [256, 256])

            ldmks_diff = pred_ldmks - ldmkss
            ldmks_loss = torch.norm(ldmks_diff, dim=-1).sum(1)
            image_loss = (torch.square((out - image)/255) * mask).sum(1).sum(1).sum(1)
            reg = (params ** 2).sum(1) + (x_tex ** 2).sum(1) + (x_illum ** 2).sum(1)

            if i < 50:
                loss = ldmks_loss
            else:
                loss = ldmks_loss + image_loss * 10
            loss = (loss + reg * 0.1).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            print(loss.item())

        return paramss, x_tex, x_illum, image_loss, loss, reg, ldmks_loss

    def single_process(self, image_path, steps=100):
        image, mask, ldmks, params, h = self.load_image(image_path)
        # vertex = param2vert(params, dense=True).transpose()
        # pred_ldmks = param2vert(params).transpose()[..., :2]
        # pred_ldmks *= h/120
        x_tex = torch.randn((1, 199)).to(self.device).requires_grad_()
        x_illum = torch.zeros((1, 27)).to(self.device).requires_grad_()
        params = params[None, ...].requires_grad_()
        optim = torch.optim.Adam([params, x_tex, x_illum], lr=1e-2)
        ldmk_weight = torch.ones((1, 68), device=self.device)
        ldmk_weight[:, 17:] *= 5


        for i in range(steps):
            pred_ldmks = self.param2vert_batch(params, param_pack=self.param_pack_torch).permute(0, 2, 1)[..., :2] * h/120
            pred_vertex = self.param2vert_batch(params, dense=True, param_pack=self.param_pack_torch).permute(0, 2, 1)
            pred_vertex /= 60
            pred_vertex -= 1
            vx, vy, vz = torch.unbind(pred_vertex, dim=-1)
            vz = vz/vz.min()
            pred_vertex_normed = torch.cat([vx[..., None], vy[..., None], vz[..., None]], dim=2)
            # index = torch.tensor([2])
            # pred_vertex.index_copy(1, index, pred_vertex_normed)
            # pred_vertex = pred_vertex.clone()
            tex_pred = ((x_tex @ self.param_pack_torch.w_tex).reshape(1, -1, 3) + self.tex_mean)

            # homogeneous_coord = torch.ones([1, 53215, 1], dtype=torch.float32, device=device)
            # pred_vertex = torch.cat([pred_vertex_normed[None, ...], homogeneous_coord], dim=-1)
            # rast, _ = dr.rasterize(macro_glctx, pred_vertex.float(), triangles, resolution=[256, 256])
            # out, _ = dr.interpolate(tex_mean, rast, triangles)
            # out, _ = dr.interpolate(tex_pred.to(device), rast, triangles)
            out, out_mask = self.render(x_illum, pred_vertex_normed.contiguous(), tex_pred.contiguous(), self.face_triangles, image, [256, 256])

            diff = pred_ldmks - ldmks[None, ...]
            ldmk_loss = (torch.norm(diff, dim=-1) * ldmk_weight).sum()
            image_loss = (torch.square((out - image)/255) * mask).sum()
            reg = (params ** 2).sum() + (x_tex ** 2).sum() + (x_illum ** 2).sum()
            save_image = image * (1 - out_mask) + out * out_mask

            if i < 50:
                loss = ldmk_loss
            else:
                loss = ldmk_loss + image_loss * 1
            loss = loss + reg * 0.1
            optim.zero_grad()
            loss.backward()
            optim.step()
            # print(loss.item())

        return params, x_tex, x_illum, loss, image_loss, reg, ldmk_loss, torch.cat([save_image, image], dim=2)

    def init_guess(self, ldmks):
        ldmks_arr = ldmks.cpu().numpy()
        mutual_kpts = self.param_pack_torch_large.u.reshape(-1, 3)[self.param_pack_torch_large.keypoints.long()].cpu().numpy()
        R, s, T = self.ralign(mutual_kpts, np.concatenate([ldmks_arr, np.zeros((68, 1))], axis=1))
        R = R*s
        return np.concatenate([R, T[..., None]], axis=1).flatten()

    def single_person_multi_image_process_large(self, image_paths, steps=300, share_id=True, share_tex=True, share_exp=True):
        bs = len(image_paths)
        infos = [self.load_image(img_path) for img_path in image_paths]
        images = torch.cat([info[0] for info in infos], dim=0)
        mask = torch.cat([info[1] for info in infos], dim=0)
        ldmks = torch.cat([info[2][None, ...] for info in infos], dim=0)
        hs = torch.tensor([info[-1] for info in infos]).to(self.device)

        param_pose = torch.cat([info[3][None, :12] for info in infos], dim=0).requires_grad_()
        if share_id:
            param_id = torch.zeros((1, 80)).to(self.device).requires_grad_()
        else:
            param_id = torch.zeros((bs, 80)).to(self.device).requires_grad_()

        if share_exp:
            param_exp = torch.zeros((1, 64)).to(self.device).requires_grad_()
        else:
            param_exp = torch.zeros((bs, 64)).to(self.device).requires_grad_()

        if share_tex:
            x_tex = torch.randn((1, 80)).to(self.device).requires_grad_()
        else:
            x_tex = torch.randn((bs, 80)).to(self.device).requires_grad_()
        x_illum = torch.zeros((bs, 27)).to(self.device).requires_grad_()
        ldmk_weight = torch.ones((1, 68), device=self.device)
        ldmk_weight[:, 17:] *= 5
        optim = torch.optim.Adam([param_pose, param_id, param_exp, x_tex, x_illum], lr=1e-2)

        for i in range(steps):
            param_id_ = param_id.repeat(int(bs/param_id.shape[0]), 1)
            param_exp_ = param_exp.repeat(int(bs/param_exp.shape[0]), 1)
            x_tex = param_id.repeat(int(bs/x_tex.shape[0]), 1)
            params = torch.cat([param_pose, param_id_, param_exp_], dim=1)

            pred_vertex = self.param2vert_batch(params, dense=True, param_pack=self.param_pack_torch_large).permute(0, 2, 1)
            pred_ldmks = pred_vertex[:, self.param_pack_torch_large.keypoints.long(), :2] * hs[:, None, None]/120
            pred_vertex /= 60
            pred_vertex -= 1
            vx, vy, vz = torch.unbind(pred_vertex, dim=-1)
            vz = vz/vz.min()
            pred_vertex_normed = torch.cat([vx[..., None], vy[..., None], vz[..., None]], dim=2)
            tex_pred = ((x_tex @ self.param_pack_torch_large.w_tex).reshape(bs, -1, 3) + self.param_pack_torch_large.tex_mean)

            out, out_mask = self.render(x_illum, pred_vertex_normed.contiguous(), tex_pred.contiguous(), self.param_pack_torch_large.triangles.contiguous(), images, [256, 256])

            diff = pred_ldmks - ldmks.squeeze(1).permute(0, 2, 1)[..., :2]
            ldmk_loss = (torch.norm(diff, dim=-1) * ldmk_weight).sum()
            image_loss = (torch.square((out - images)/255) * mask).sum()
            reg = (params ** 2).sum() + (x_tex ** 2).sum() + (x_illum ** 2).sum()
            save_image = images * (1 - out_mask) + out * out_mask

            if i < 100:
                loss = ldmk_loss
            else:
                loss = ldmk_loss + image_loss * 1
            loss = loss + reg * 0.1
            optim.zero_grad()
            loss.backward()
            optim.step()

        return params, x_tex, x_illum, loss, image_loss, reg, ldmk_loss, torch.cat([save_image, images], dim=2)

    def single_process_large(self, image_path, steps=300):
        image, mask, ldmks, params, h = self.load_image(image_path)
        params[12:] *= 0
        # params[:12] = torch.from_numpy(init_guess(ldmks))
        # vertex = param2vert(params, dense=True).transpose()
        # pred_ldmks = param2vert(params).transpose()[..., :2]
        # pred_ldmks *= h/120
        x_tex = torch.randn((1, 80)).to(self.device).requires_grad_()
        x_illum = torch.zeros((1, 27)).to(self.device).requires_grad_()
        params = torch.cat([params, torch.zeros(94).to(params.device)])
        params = params[None, ...].requires_grad_()
        optim = torch.optim.Adam([params, x_tex, x_illum], lr=1e-2)
        ldmk_weight = torch.ones((1, 68), device=self.device)
        ldmk_weight[:, 17:] *= 5

        for i in range(steps):
            # pred_ldmks = param2vert_batch(params, param_pack=param_pack).permute(0, 2, 1)[..., :2] * h/120
            pred_vertex = self.param2vert_batch(params, dense=True, param_pack=self.param_pack_torch_large).permute(0, 2, 1)
            pred_ldmks = pred_vertex[:, self.param_pack_torch_large.keypoints.long(), :2] * h/120
            pred_vertex /= 60
            pred_vertex -= 1
            vx, vy, vz = torch.unbind(pred_vertex, dim=-1)
            vz = vz/vz.min()
            pred_vertex_normed = torch.cat([vx[..., None], vy[..., None], vz[..., None]], dim=2)
            # index = torch.tensor([2])
            # pred_vertex.index_copy(1, index, pred_vertex_normed)
            # pred_vertex = pred_vertex.clone()
            tex_pred = ((x_tex @ self.param_pack_torch_large.w_tex).reshape(1, -1, 3) + self.param_pack_torch_large.tex_mean)

            # homogeneous_coord = torch.ones([1, 53215, 1], dtype=torch.float32, device=device)
            # pred_vertex = torch.cat([pred_vertex_normed[None, ...], homogeneous_coord], dim=-1)
            # rast, _ = dr.rasterize(macro_glctx, pred_vertex.float(), triangles, resolution=[256, 256])
            # out, _ = dr.interpolate(tex_mean, rast, triangles)
            # out, _ = dr.interpolate(tex_pred.to(device), rast, triangles)
            out, out_mask = self.render(x_illum, pred_vertex_normed.contiguous(), tex_pred.contiguous(), self.param_pack_torch_large.triangles.contiguous(), image, [256, 256])

            diff = pred_ldmks - ldmks[None, ...]
            ldmk_loss = (torch.norm(diff, dim=-1) * ldmk_weight).sum()
            image_loss = (torch.square((out - image)/255) * mask).sum()
            reg = (params ** 2).sum() + (x_tex ** 2).sum() + (x_illum ** 2).sum()
            save_image = image * (1 - out_mask) + out * out_mask

            if i < 100:
                loss = ldmk_loss
            else:
                loss = ldmk_loss + image_loss * 1
            loss = loss + reg * 0.1
            optim.zero_grad()
            loss.backward()
            optim.step()
            # print(loss.item())

        return params, x_tex, x_illum, loss, image_loss, reg, ldmk_loss, torch.cat([save_image, image], dim=2)

    def render_back(self, ori_image, params, x_tex, x_illum, mask):
        assert len(params.shape) == len(x_tex.shape) == len(x_illum.shape) == 2
        tex = (x_tex @ self.param_pack_torch.w_tex).reshape(1, -1, 3) + self.tex_mean
        pred_vertex = self.param2vert_batch(params, dense=True, param_pack=self.param_pack_torch).permute(0, 2, 1)
        pred_vertex = pred_vertex / 60 - 1
        vx, vy, vz = torch.unbind(pred_vertex, dim=-1)
        vz = vz/vz.min()
        pred_vertex_normed = torch.cat([vx[..., None], vy[..., None], vz[..., None]], dim=2)

        out_image = self.render(x_illum, pred_vertex_normed, tex, self.triangles, ori_image)
        out_image = out_image * mask + ori_image * (1 - mask)

        return out_image

def mimic(device, q):
    mic = ImageMimic(device=device)
    while True:
        task = q.get()
        run_single(mic, task)

def run_single(mic, task):
    paramss, x_tex, x_illum, image_loss, loss, reg, ldmks_loss, save_image = mic.single_process_large(task)
    paramss = paramss.cpu().detach().numpy()
    x_tex = x_tex.cpu().detach().numpy()
    x_illum = x_illum.cpu().detach().numpy()
    image_loss = image_loss.item()
    loss = loss.item()
    reg = reg.item()
    ldmks_loss = ldmks_loss.item()

    for idx in range(1):
        save_path = './save/{}.npz'.format(task.split('/')[-1].split('.')[0])
        save_item = {"illumination": x_illum[idx],
                    "texture": x_tex[idx],
                    "base_param": paramss[idx],
                    "image_loss": image_loss,
                    "landmark_loss": ldmks_loss}


        print(idx, task, "loss: {}, reg: {}, image: {}, landmark: {}".format(loss, reg, image_loss, ldmks_loss))
            # print(loss.item())
        np.savez(save_path, **save_item)
        cv.imwrite("./save/{}.png".format(task.split('/')[-1].split('.')[0]),
                    save_image[0].cpu().detach().numpy().astype(np.uint8))

# from multiprocessing import Queue
# q = Queue()

if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--image_path", required=True, type=str)
    args.add_argument("--device", required=True, type=str)
    args = args.parse_args()
    # chunk_size = 16
    image_paths = glob.glob('{}/*/*.JPG'.format(args.image_path))
    mic = ImageMimic(torch.device("cuda:{}".format(args.device)))
    for image_path in image_paths:
        try:
            run_single(mic, image_path)
        except:
            traceback.print_exc()
    # for i in range(int(len(image_paths)/chunk_size)):
    #     batch_paths = image_paths[i*chunk_size: (i+1)*chunk_size]
    #     paramss, x_tex, x_illum, image_loss, loss, reg, ldmks_loss = batch_process(batch_paths)
    #     paramss = paramss.cpu().detach().numpy()
    #     x_tex = x_tex.cpu().detach().numpy()
    #     x_illum = x_illum.cpu().detach().numpy()
    #     image_loss = image_loss.cpu().detach().numpy()
    #     loss = loss.cpu().detach().numpy()
    #     reg = reg.cpu().detach().numpy()
    #     ldmks_loss = ldmks_loss.cpu().detach().numpy()

    #     for idx in range(chunk_size):
    #         image_path = batch_paths[idx]
    #         save_path = './batch_out/{}.npz'.format(image_path.split('/')[-1].split('.')[0])
    #         save_item = {"illumination": x_illum[idx],
    #                     "texture": x_tex[idx],
    #                     "base_param": paramss[idx],
    #                     "image_loss": image_loss[idx],
    #                     "landmark_loss": ldmks_loss[idx]}


    #         print(idx, image_path, "loss: {}, reg: {}, image: {}".format(loss[idx], reg[idx], image_loss[idx], ldmks_loss[idx]))
    #             # print(loss.item())
    #         np.savez(save_path, **save_item)
