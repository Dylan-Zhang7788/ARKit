import torch
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

RR = np.array([[ 0.99988735, 0.0146241, -0.00337969], [-0.01416047, 0.99375874, 0.11064831], [ 0.00497673, -0.11058799, 0.99385388]])
ss = 9.420123997906711e-07
tt = np.array([-0.00026256,-0.01507333, -0.04221931])

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

    def __init__(self, param):
        super(Distance, self).__init__()
        # self.basis3dmm = basis3dmm
        # uv_bases = basis3dmm['uv']
        self.RR = nn.Parameter(torch.from_numpy(RR.copy()).to(torch.float32), requires_grad=True)
        self.ss = nn.Parameter(torch.tensor(ss).to(torch.float32), requires_grad=True)
        self.tt = nn.Parameter(torch.from_numpy(tt.copy()).to(torch.float32), requires_grad=True)

        p, offset, alpha_shp, alpha_exp = parse_param(param)
        self.alpha_shp = nn.Parameter(torch.from_numpy(alpha_shp.copy()), requires_grad=True)
        self.alpha_exp = nn.Parameter(torch.from_numpy(alpha_exp.copy()), requires_grad=True)

        self.param_std = nn.Parameter(torch.from_numpy(param_pack.param_std), requires_grad=False)
        self.param_mean = nn.Parameter(torch.from_numpy(param_pack.param_mean), requires_grad=False)
        
        self.u = nn.Parameter(torch.from_numpy(param_pack.u), requires_grad=False)
        self.w_shp = nn.Parameter(torch.from_numpy(param_pack.w_shp), requires_grad=False)
        self.w_exp = nn.Parameter(torch.from_numpy(param_pack.w_exp), requires_grad=False)

        self.u_base = nn.Parameter(torch.from_numpy(param_pack.u_base), requires_grad=False)
        self.w_shp_base = nn.Parameter(torch.from_numpy(param_pack.w_shp_base), requires_grad=False)
        self.w_exp_base = nn.Parameter(torch.from_numpy(param_pack.w_exp_base).to(torch.float32), requires_grad=False)

        # self.basis_shape = nn.Parameter(torch.from_numpy(basis3dmm["basis_shape"]), requires_grad=False)
        # self.basis_exp = nn.Parameter(torch.from_numpy(basis3dmm["basis_exp"]), requires_grad=False)
        # self.mu_shape = nn.Parameter(torch.from_numpy(basis3dmm["mu_shape"]), requires_grad=False)
    
    def predict_vert(self):
        vertex = (self.u_base + self.w_shp_base @ self.alpha_shp + self.w_exp_base @ self.alpha_exp).reshape(-1, 3).t()
        return vertex

    def return_vx(self):
        vertices = self.predict_vert()
        vx_ge0 = self.ss * vertices.T  @ self.RR.T + self.tt
        
        return vx_ge0.cpu().detach().numpy()

    def forward(self, target, dn1):
        vertices = self.predict_vert()
        vx_ge0 = self.ss * vertices.T  @ self.RR.T + self.tt
        dis_sub = (vx_ge0-target).pow(2)
        dis_sub = dis_sub.mean(-1)
        loss = dis_sub.mean()
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
    checkpoint_fp = 'SynergyNet_checkpoint_epoch_100.pth.tar' 
    args.arch = 'mobilenet_v2'
    inference_output = 'test_0816'
    args.devices_id = [0]
    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    
    model = SynergyNet(args)
    model_dict = model.state_dict()
    checkpoint.pop("module.u_base")
    checkpoint.pop("module.w_shp_base")
    checkpoint.pop("module.w_exp_base")
    # because the model is trained by multiple gpus, prefix 'module' should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]

    model.load_state_dict(model_dict, strict=False)
    model = model.cuda()
    model.eval()

    # face detector
    face_boxes = FaceBoxes()

    # preparation
    transform = transforms.Compose([ToTensor(), Normalize(mean=127.5, std=128)])
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
        # crop faces
        rects = face_boxes(img_ori)
        
        # storage
        pts_res = []
        poses = []
        nmes = []
        nmes_after = []
        vertices_lst = []
        for idx, rect in enumerate(rects):
            roi_box = rect
            
            # enlarge the bbox a little and do a square crop
            HCenter = (rect[1] + rect[3])/2
            WCenter = (rect[0] + rect[2])/2
            side_len = roi_box[3]-roi_box[1]
            margin = side_len * 1.2 // 2
            roi_box[0], roi_box[1], roi_box[2], roi_box[3] = WCenter-margin, HCenter-margin, WCenter+margin, HCenter+margin
            img = crop_img(img_ori, roi_box)
            img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            # cv2.imwrite(f'validate_{idx}.png', img)
            
            input = transform(img).unsqueeze(0)
            with torch.no_grad():
                #input = torch.Tensor(np.load("input.npy"))
                input = input.cuda()
                param = model.forward_test(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
                
            # inferences
            lmks = predict_sparseVert(param, roi_box, transform=True)
            vertices = predict_vert(param)
            vx_ge0 = ss * vertices.T  @ RR.T + tt
            name = img_fp.rsplit('/',1)[-1]
            label = np.array(labels[name])
            dn1 = max(1e-6, np.linalg.norm(label[idx1]-label[idx2]))
            nme = np.mean(np.linalg.norm(vx_ge0-label, axis=1))/dn1
            pts_res.append(lmks)
            nmes.append(nme)
            print("nme_pre", np.mean(nmes))
            vertices_lst.append(vx_ge0)
            save_ply(vx_ge0, label, "point_cloud_compare_pre.ply", [1, 0, 0], [0, 1, 0])


            steps = 1500
            momentum_schedule = cosine_scheduler(0.8, 1, steps, 1)
            target = torch.from_numpy(np.array(label.copy())).cuda()
            dn2 = torch.from_numpy(np.array(dn1.copy())).cuda()
            scale = 0.1
            dis_module = Distance(param)
            teacher = Distance(param)
            my_optim = optim.Adam([
                            #{'params': dis_module.roll, 'lr': 1e-3* scale},
                            #{'params': dis_module.yaw, 'lr': 1e-3* scale},
                            #{'params': dis_module.pitch, 'lr': 1e-3* scale},
                            {'params': dis_module.ss, 'lr': 5e-2* scale},
                            {'params': dis_module.tt, 'lr': 1e-2* scale},
                            {'params': dis_module.RR, 'lr': 1e-2* scale},
                            {'params': dis_module.alpha_shp, 'lr': 5e-2* scale},
                            {'params': dis_module.alpha_exp, 'lr': 5e-2* scale}
                            ], weight_decay=1e-6)
            scheduler2 = torch.optim.lr_scheduler.MultiStepLR(my_optim, milestones=[150, 300, 450, 800])
            
            dis_module = dis_module.cuda()
            teacher.load_state_dict(dis_module.state_dict())
            teacher = teacher.cuda()
            for j in range(steps):
                if j % 1 == 0:
                    loss = dis_module(target.clone(),dn2)
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

        if not osp.exists('{}/landmarks/'.format(inference_output)):
            os.makedirs('{}/landmarks/'.format(inference_output))
        
        name = img_fp.rsplit('/',1)[-1][:-4]
        img_ori_copy = img_ori.copy()
        
        for idy in range(len(vertices_lst)):
            np.save("{}/landmarks/{}_{}.npy".format(inference_output, name, idy), vertices_lst[idy])

        # landmarks
        if len(pts_res):
            for idx  in range(pts_res[0].shape[1]):
                cv2.circle(img_ori_copy, (int(pts_res[0][0][idx]), int(pts_res[0][1][idx])), 3, color=(0, 255, 255))        
            cv2.imwrite('{}/landmarks/{}.jpg'.format(inference_output, name), img_ori_copy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', default='0F598DC2-29E5-41F7-B374-4A3F5E27216D20230724', help='path to a single image or path to a folder containing multiple images')
    parser.add_argument("--png", action="store_true", help="if images are with .png extension")
    parser.add_argument('--img_size', default=120, type=int)
    parser.add_argument('-b', '--batch-size', default=1, type=int)

    args = parser.parse_args()
    main(args)
