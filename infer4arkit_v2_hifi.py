import torch
import scipy
from torch import optim
from torch import nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from utils.ddfa import ToTensor, Normalize
from model_building import SynergyNet
from utils.inference import crop_img, predict_sparseVert, draw_landmarks, predict_denseVert, predict_pose, draw_axis, predict_vert
import argparse
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from utils.params import ParamsPack
param_pack = ParamsPack()
import os
import pdb
import os.path as osp
import glob
from FaceBoxes import FaceBoxes
from utils.render import render
import json
from ply import save_ply
from utils.inference import parse_param

# Following 3DDFA-V2, we also use 120x120 resolution
IMG_SIZE = 120

RR = np.array([[ 0.99986391,  0.0053632,  -0.01560113],
                [-0.00416528,  0.9971122,   0.07582816],
                [ 0.01596276, -0.07575285,  0.99699884]])
ss = 0.009088776973075337
tt = np.array([-0.00193026, -0.01224346,  0.00013872])

def load_3dmm_basis(
    basis_path,
    uv_path=None,
    tri_v_path=None,
    tri_vt_path=None,
    vt_path=None,
    uv_weight_mask_path=None,
    is_train=True,
    is_whole_uv=True,
    limit_dim=-1,
):
    """load 3dmm basis and other useful files.

    :param basis_path:
        - *.mat, 3DMM basis path.
        - It contains shape/exp bases, mesh triangle definition and face vertex mask in bool.
    :param uv_path:
        - If is_whole_uv is set to true, then uv_path is a file path.
        - Otherwise, it is a directory to load regional UVs.
    :param tri_v_path:
        - contains mesh triangle definition in geometry. If it is set, it covers the definition in basis_path.
    :param tri_vt_path:
        - contains mesh triangle definition in UV space.

    """

    basis3dmm = scipy.io.loadmat(basis_path)
    basis3dmm["keypoints"] = np.squeeze(basis3dmm["keypoints"])
    basis3dmm["vt_list"] = basis3dmm["vt_list"].astype(np.float32)

    # load uv basis
    if uv_path is not None and not is_whole_uv:
        uv_region_paths = sorted(glob.glob(os.path.join(uv_path, "*_uv512.mat")))
        uv_region_bases = {}
        for region_path in uv_region_paths:
            region_name = region_path.split("/")[-1].split("_uv")[0]
            region_config = scipy.io.loadmat(region_path)
            region_config["basis"] = np.transpose(
                region_config["basis"] * region_config["sigma"]
            )
            region_config["indices"] = region_config["indices"].astype(np.int32)
            del region_config["sigma"]
            assert region_config["basis"].shape[0] < region_config["basis"].shape[1]
            uv_region_bases[region_name] = region_config
        basis3dmm["uv"] = uv_region_bases

        if not is_train:
            uv_region_paths = sorted(glob.glob(os.path.join(uv_path, "*_uv.mat")))
            uv_region_bases = {}
            for region_path in uv_region_paths:
                region_name = region_path.split("/")[-1].split("_uv")[0]
                region_config = scipy.io.loadmat(region_path)
                region_config["basis"] = np.transpose(
                    region_config["basis"] * region_config["sigma"]
                )
                region_config["indices"] = region_config["indices"].astype(np.int32)
                del region_config["sigma"]
                assert region_config["basis"].shape[0] < region_config["basis"].shape[1]
                uv_region_bases[region_name] = region_config
            basis3dmm["uv2k"] = uv_region_bases

            normal_region_paths = sorted(
                glob.glob(os.path.join(uv_path, "*_normal.mat"))
            )
            normal_region_bases = {}
            for region_path in normal_region_paths:
                region_name = region_path.split("/")[-1].split("_normal")[0]
                region_config = scipy.io.loadmat(region_path)
                region_config["basis"] = np.transpose(
                    region_config["basis"] * region_config["sigma"]
                )
                region_config["indices"] = region_config["indices"].astype(np.int32)
                del region_config["sigma"]
                assert region_config["basis"].shape[0] < region_config["basis"].shape[1]
                normal_region_bases[region_name] = region_config
            basis3dmm["normal2k"] = normal_region_bases

    if uv_path is not None and is_whole_uv:
        config = scipy.io.loadmat(uv_path)
        config["basis"] = config["basis"] * config["sigma"]
        config["indices"] = config["indices"].astype(np.int32)
        del config["sigma"]
        if config["basis"].shape[0] > config["basis"].shape[1]:
            config["basis"] = np.transpose(config["basis"])
        assert config["basis"].shape[0] < config["basis"].shape[1]
        basis3dmm["uv"] = config

        if not is_train:  # add normal
            normal_path = uv_path.replace("uv512", "norm512")
            config_norm = scipy.io.loadmat(normal_path)
            config_norm["basis"] = np.transpose(
                config_norm["basis"] * config_norm["sigma"]
            )
            config_norm["indices"] = config_norm["indices"].astype(np.int32)
            del config_norm["sigma"]
            assert config_norm["basis"].shape[0] < config_norm["basis"].shape[1]
            basis3dmm["normal"] = config_norm

    if tri_v_path is not None:
        tri_v = np.load(tri_v_path)["arr_0"].astype(np.int32)
        basis3dmm["tri"] = tri_v

    if tri_vt_path is not None:
        tri_vt = np.load(tri_vt_path)["arr_0"].astype(np.int32)
        basis3dmm["tri_vt"] = tri_vt

    if vt_path is not None:
        vt_list = np.load(vt_path)["arr_0"].astype(np.float32)
        basis3dmm["vt_list"] = vt_list

    if uv_weight_mask_path is not None:
        uv_weight_mask = cv2.imread(uv_weight_mask_path).astype(np.float32) / 255.0
        basis3dmm["uv_weight_mask"] = np.expand_dims(uv_weight_mask, 0)
        assert uv_weight_mask.shape[0] == 512

    if limit_dim > 0:
        basis3dmm["basis_shape"] = basis3dmm["basis_shape"][:limit_dim, :]

    return basis3dmm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

class Distance(nn.Module):

    def __init__(self, basis3dmm):
        super(Distance, self).__init__()
        self.basis3dmm = basis3dmm.copy()
        # uv_bases = basis3dmm['uv']

        key = np.load("array_d.npy")

        self.basis3dmm["basis_shape"] = self.basis3dmm["basis_shape"].T
        self.basis3dmm["basis_shape"] = self.basis3dmm["basis_shape"][key].T
        self.basis3dmm["basis_exp"] = self.basis3dmm["basis_exp"].T
        self.basis3dmm["basis_exp"] = self.basis3dmm["basis_exp"][key].T
        self.basis3dmm["mu_shape"] = self.basis3dmm["mu_shape"].T
        self.basis3dmm["mu_shape"] = self.basis3dmm["mu_shape"][key].T

        self.RR = nn.Parameter(torch.from_numpy(RR.copy()).to(torch.float32), requires_grad=True)
        self.ss = nn.Parameter(torch.tensor(ss).to(torch.float32), requires_grad=True)
        self.tt = nn.Parameter(torch.from_numpy(tt.copy()).to(torch.float32), requires_grad=True)

        shape = np.zeros(self.basis3dmm["basis_shape"].shape[0]) # self.basis3dmm["basis_shape"].shape[0] = 500
        exp = np.zeros(self.basis3dmm["basis_exp"].shape[0]) # self.basis3dmm["basis_exp"].shape[0] = 199
        self.shape = nn.Parameter(torch.from_numpy(shape).float().reshape((self.basis3dmm["basis_shape"].shape[0])), requires_grad=True)
        self.exp = nn.Parameter(torch.from_numpy(exp).float().reshape((self.basis3dmm["basis_exp"].shape[0])), requires_grad=True)

        self.basis_shape = nn.Parameter(torch.from_numpy(self.basis3dmm["basis_shape"]), requires_grad=False)
        self.basis_exp = nn.Parameter(torch.from_numpy(self.basis3dmm["basis_exp"]), requires_grad=False)
        self.mu_shape = nn.Parameter(torch.from_numpy(self.basis3dmm["mu_shape"]), requires_grad=False)
    
    def predict_vert(self):
        shape_inc = torch.matmul(self.shape, self.basis_shape)
        vertex = self.mu_shape + shape_inc

        exp_inc = torch.matmul(self.exp, self.basis_exp)
        vertex = vertex + exp_inc

        vertex = torch.reshape(vertex, [self.basis_shape.shape[1] // 3, 3])

        return vertex

    def return_vx(self):
        vertices = self.predict_vert().t()
        vx_ge0 = self.ss * vertices.T  @ self.RR.T + self.tt
        
        return vx_ge0.cpu().detach().numpy()

    def forward(self, target, dn1):
        vertices = self.predict_vert().t()
        vx_ge0 = self.ss * vertices.T  @ self.RR.T + self.tt
        dis_sub = (vx_ge0-target).pow(2)
        dis_sub = dis_sub.mean(-1)
        loss = dis_sub.mean()*500
        # loss = dis_sub[self.front_face].mean()*500 # + dis_sub.mean()*30 #  + (geo[self.face_42]-target_42).pow(2).mean()*50
        # loss = (geo[self.face_42]-target_42).pow(2).mean()*50
        # reg = (self.shape.pow(2) - torch.zeros_like(self.shape, device=self.shape.device)).mean()
        # print('loss: ', loss.item())
        return loss


labels = {}
f = open("0F598DC2-29E5-41F7-B374-4A3F5E27216D20230724/blendshape.txt", 'r')
for m, line in enumerate(f):
    line = json.loads(line)
    name = line["name"]
    vertices = line["vertices"]
    labels[name] = vertices
print("load labels done")

def get_idx(src,target):
        # geo = torch.mm(geo, R)
        idxs = []
        slice_num = 1.0
        len_slice = int(len(src) / slice_num)
        # src_tem = torch.zeros((len_slice, target.shape[0], 3), dtype=target.dtype, device=target.device)
        src_tem = None
        target = target.unsqueeze(0).expand(len_slice, -1, -1)
        for i in range(int(slice_num)):
            distance1 = dis(src[(i * len_slice):((i + 1) * len_slice)], target, src_tem)
            idxs.append(distance1)
        idxs = torch.cat(idxs, dim=0)
        return idxs

def dis(src, target, src_tem):
    src_tem = src.unsqueeze(1).expand(-1, target.shape[1], -1)
    dis_tem = torch.norm(src_tem - target, dim=-1)
    res = torch.argmin(dis_tem, dim=-1)
    del src_tem
    del dis_tem
    torch.cuda.empty_cache()
    return res

def main(args):
    # load pre-tained model
    if osp.isdir(args.files):
        if not args.files[-1] == '/':
            args.files = args.files + '/'
        if not args.png:
            files = sorted(glob.glob(args.files+'*.JPG'))
        else:
            files = sorted(glob.glob(args.files+'*.png'))
    else:
        files = [args.files]
    
    idx1, idx2 = 1101, 1069

    for img_fp in files:
        print("Process the image: ", img_fp)

        img_ori = cv2.imread(img_fp)
        img_ori = cv2.flip(cv2.transpose(img_ori), 1)

        name = img_fp.rsplit('/',1)[-1]
        label = np.array(labels[name])

        files_3dmm = '3dmm_data'
        basis3dmm = load_3dmm_basis(
            files_3dmm + '/AI-NEXT-Shape.mat',
            files_3dmm + '/AI-NEXT-Albedo-Global.mat',
            is_whole_uv=True)

        nmes_after = []
        vertices_lst = []
        steps = 1500
        momentum_schedule = cosine_scheduler(0.8, 1, steps, 1)
        target = torch.from_numpy(np.array(label.copy())).cuda()
        dn1 = max(1e-6, np.linalg.norm(label[idx1]-label[idx2]))
        scale = 0.1
        dis_module = Distance(basis3dmm)
        teacher = Distance(basis3dmm)
        my_optim = optim.Adam([
                        #{'params': dis_module.roll, 'lr': 1e-3* scale},
                        #{'params': dis_module.yaw, 'lr': 1e-3* scale},
                        #{'params': dis_module.pitch, 'lr': 1e-3* scale},
                        {'params': dis_module.ss, 'lr': 5e-2* scale},
                        {'params': dis_module.tt, 'lr': 1e-2* scale},
                        {'params': dis_module.RR, 'lr': 1e-2* scale},
                        {'params': dis_module.shape, 'lr': 5e-2* scale},
                        # {'params': dis_module.exp, 'lr': 5e-2* scale}
                        ], weight_decay=1e-6)
        scheduler2 = torch.optim.lr_scheduler.MultiStepLR(my_optim, milestones=[150, 300, 450, 800])
        
        dis_module = dis_module.cuda()
        teacher.load_state_dict(dis_module.state_dict())
        teacher = teacher.cuda()
        for j in range(steps):
            if j % 1 == 0:
                loss = dis_module(target.clone(), dn1)
                loss.backward()
                my_optim.step()
                my_optim.zero_grad()
                scheduler2.step()
                with torch.no_grad():
                    m = momentum_schedule[j]  # momentum parameter
                    for param_q, param_k in zip(dis_module.parameters(), teacher.parameters()):
                        if not param_k.requires_grad:
                            continue
                        # print(param_k.shape, 'param_k.shape', param_k.requires_grad)
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
                torch.cuda.synchronize()
        vx_ge0 = teacher.return_vx()
        nme = np.mean(np.linalg.norm(vx_ge0-label, axis=1))/dn1
        nmes_after.append(nme)
        print("nme_after", np.mean(nmes_after))
        save_ply(vx_ge0, label, "point_cloud_compare_after.ply", [1, 0, 0], [0, 1, 0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', default='0F598DC2-29E5-41F7-B374-4A3F5E27216D20230724', help='path to a single image or path to a folder containing multiple images')
    parser.add_argument("--png", action="store_true", help="if images are with .png extension")
    parser.add_argument('--img_size', default=120, type=int)
    parser.add_argument('-b', '--batch-size', default=1, type=int)

    args = parser.parse_args()
    main(args)
