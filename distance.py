import os
from torch import nn
import numpy as np
import torch
from third_party.ply import write_ply, write_obj, write_obj_synergy,read_obj_synergy
from compare import read_obj, get_face_point, get_point_dist_abs, get_diff_map

is_BFM = True

def write_obj_local(name, basis3dmm, v_xyz):
    write_obj(
                    name,
                    v_xyz,
                    basis3dmm["vt_list"],
                    basis3dmm["tri"].astype(np.int32),
                    basis3dmm["tri_vt"].astype(np.int32))
class Distance(nn.Module):

    def __init__(self, basis3dmm):
        super(Distance, self).__init__()
        self.basis3dmm = basis3dmm
        # uv_bases = basis3dmm['uv']
        self.roll = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.yaw = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.pitch = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.s_roll = nn.Parameter(torch.ones(1), requires_grad=True)

        self.T = nn.Parameter(torch.zeros(3), requires_grad=True)
        shape = np.zeros(basis3dmm["basis_shape"].shape[0]) # basis3dmm["basis_shape"].shape[0] = 500
        exp = np.zeros(basis3dmm["basis_exp"].shape[0]) # basis3dmm["basis_exp"].shape[0] = 199
        # print('shape: ', shape.shape, 'exp: ', exp.shape)
        self.shape = nn.Parameter(torch.from_numpy(shape).float().reshape((basis3dmm["basis_shape"].shape[0])), requires_grad=True)
        self.exp = nn.Parameter(torch.from_numpy(exp).float().reshape((basis3dmm["basis_exp"].shape[0])), requires_grad=True)
        self.basis_shape = nn.Parameter(torch.from_numpy(basis3dmm["basis_shape"]), requires_grad=False)
        self.basis_exp = nn.Parameter(torch.from_numpy(basis3dmm["basis_exp"]), requires_grad=False)
        self.mu_shape = nn.Parameter(torch.from_numpy(basis3dmm["mu_shape"]), requires_grad=False)
        self.tensor_0 = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.tensor_1 = nn.Parameter(torch.ones(1), requires_grad=False)
        dic_idx = {}
        # print(self.basis3dmm["tri"].shape)
        for idxs in self.basis3dmm["tri"]:
            for idx in idxs:
                dic_idx[idx] = ''
        # print(len(dic_idx))
        self.first_idx = nn.Parameter(torch.from_numpy(np.array(list(dic_idx.keys()))), requires_grad=False)
        self.idxs = nn.Parameter(torch.arange(0, 9518), requires_grad=False)
        print('after init')
        self.fir = None
        key_9518 = list(dic_idx.keys())
        dic_front = self.read_front_face('/data/zhang.di/front_face.obj')
        front_face = []
        for i in range(len(dic_idx.keys())):
            if key_9518[i] in dic_front:
                front_face.append(i)
        self.front_face = nn.Parameter(torch.from_numpy(np.array(front_face)), requires_grad=False)

        # points_42 = [3731, 3659, 3750, 3735, 1187, 1110, 1190, 1182, 696, 730, 712, 715, 4099, 1554]
        points_42 = [3731, 3659, 3750, 3735, 1187, 1110, 1190, 1182]
        face_42 = [0 for _ in range(len(points_42))]
        for i in range(len(key_9518)):
            for j in range(len(points_42)):
                if key_9518[i] == points_42[j]:
                    face_42[j] = i
                    break
        # print('face_42', face_42)
        self.face_42 = nn.Parameter(torch.from_numpy(np.array(face_42)), requires_grad=False)

    def read_front_face(self, head_path):
        V, F_face = read_obj(head_path)
        face_index = get_face_point(F_face)
        dic_res = {}
        for idx in face_index:
            dic_res[idx] = ''
        return dic_res


    def first(self, A):
        unique, idx, counts = torch.unique(A, sorted=True, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0], device=A.device), cum_sum[:-1]))
        first_indicies = ind_sorted[cum_sum]
        return first_indicies
    # src: m*3
    # target: n*3
    def dis(self, src, target, src_tem):
        src_tem = src.unsqueeze(1).expand(-1, target.shape[1], -1)
        dis_tem = src_tem.sub(target[:src.shape[0], :, :]).abs().sum(-1)
        dis_tem = src_tem.sub(target[:src.shape[0], :, :]).abs().sum(-1)
        res = torch.argmin(dis_tem, dim=-1)
        del src_tem
        del dis_tem
        torch.cuda.empty_cache()
        return res

    def get_r(self):
        RX = torch.stack([
            torch.stack([self.tensor_1, self.tensor_0, self.tensor_0]),
            torch.stack([self.tensor_0, torch.cos(self.roll), -torch.sin(self.roll)]),
            torch.stack([self.tensor_0, torch.sin(self.roll), torch.cos(self.roll)])]).reshape(3, 3)

        RY = torch.stack([
            torch.stack([torch.cos(self.pitch), self.tensor_0, torch.sin(self.pitch)]),
            torch.stack([self.tensor_0, self.tensor_1, self.tensor_0]),
            torch.stack([-torch.sin(self.pitch), self.tensor_0, torch.cos(self.pitch)])]).reshape(3, 3)

        RZ = torch.stack([
            torch.stack([torch.cos(self.yaw), -torch.sin(self.yaw), self.tensor_0]),
            torch.stack([torch.sin(self.yaw), torch.cos(self.yaw), self.tensor_0]),
            torch.stack([self.tensor_0, self.tensor_0, self.tensor_1])]).reshape(3, 3)
        R = torch.matmul(RZ, RY)
        R = torch.matmul(R, RX)
        return R

    def get_idx(self, target, ori=0):
        R = self.get_r()

        geo = self.get_shape(R, self.s_roll, ori=ori)
        # geo = torch.mm(geo, R)
        firs, idxs = [], []
        slice_num = 150.0
        len_slice = int(len(geo) / slice_num) + 1
        # src_tem = torch.zeros((len_slice, target.shape[0], 3), dtype=target.dtype, device=target.device)
        src_tem = None
        target = target.unsqueeze(0).expand(len_slice, -1, -1)
        for i in range(int(slice_num)):
            distance1 = self.dis(geo[(i * len_slice):((i + 1) * len_slice)], target, src_tem)
            idxs.append(distance1)
        idxs = torch.cat(idxs, dim=0)
        return idxs
    def get_shape(self, R, S, ori=0):
        shape_inc = torch.matmul(self.shape, self.basis_shape)
        geo = self.mu_shape + shape_inc

        # exp_inc = torch.matmul(self.exp, self.basis_exp)
        # geo = geo + exp_inc

        geo = torch.reshape(geo, [self.basis_shape.shape[1] // 3, 3])
        if not os.path.exists('before.obj'):
            write_obj_local('before.obj', self.basis3dmm, geo)
        geo = torch.matmul(geo *S.pow(2).pow(0.5), torch.transpose(R, 1, 0)) + self.T
        if not os.path.exists('after.obj'):
            write_obj_local('after.obj', self.basis3dmm, geo)
        if ori==0:
            geo = geo[self.first_idx.long()][self.idxs]
        elif ori==1:
            geo = geo[self.first_idx.long()]
        return geo

    def forward(self, target, target_42):
        R = self.get_r()

        geo = self.get_shape(R, self.s_roll,ori=1)
        # geo = torch.mm(geo, R)
        dis_sub = (geo-target).pow(2)[self.fir]
        dis_sub = dis_sub.mean(-1)
        loss = dis_sub.mean()*500
        # loss = dis_sub[self.front_face].mean()*500 # + dis_sub.mean()*30 #  + (geo[self.face_42]-target_42).pow(2).mean()*50
        # loss = (geo[self.face_42]-target_42).pow(2).mean()*50
        reg = (self.shape.pow(2) - torch.zeros_like(self.shape, device=self.shape.device)).mean()
        print('loss: ', loss.item(), 'reg: ', reg.item())
        return loss + reg*0.05
    def get_res(self):
        R = self.get_r()

        geo = self.get_shape(R, self.s_roll, ori=2)
        return geo
    def get_res3(self, idxs, target):
        R = self.get_r()
        geo = self.get_shape(R, self.s_roll, ori=2)
        geo[self.first_idx.long()] = target[idxs.long()].float()
        geo[:, 1] += 30
        return geo

from utils.basis import load_3dmm_basis,load_3dmm_basis_bfm,load_3dmm_basis_synergy
from torch import optim
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
def write_find(name, teacher, basis3dmm):
    with torch.no_grad():
        idxs = teacher.get_idx(target.clone(), ori=1)
        v_xyz = teacher.get_res3(idxs, target)
    if not is_BFM:
        write_obj(
                        name,
                        v_xyz,
                        basis3dmm["vt_list"],
                        basis3dmm["tri"].astype(np.int32),
                        basis3dmm["tri_vt"].astype(np.int32))
    else:
        write_obj_synergy(name,v_xyz)
def write_target(teacher, basis3dmm):
    with torch.no_grad():
        idxs = teacher.get_idx(target.clone(), ori=1)
        v_xyz = teacher.get_res3(idxs, target)
    write_obj(
                    'longcheng/find.obj',
                    v_xyz,
                    basis3dmm["vt_list"],
                    basis3dmm["tri"].astype(np.int32),
                    basis3dmm["tri_vt"].astype(np.int32))

def compare(path_mean, path_2):
    path_result = 'diff_head.obj'
    std = 0.15906937
    V_ref, F_ref = read_obj(path_mean)
    # print(F_ref.shape)
    V2, F2 = read_obj(path_2)
    V2[:, 1] += 30
    face_index = get_face_point(F_ref)
    point_dist = get_point_dist_abs(V2, V_ref)
    get_diff_map(V_ref, F_ref, point_dist, path_result, minima=0, maxima=3 * std)

    point_dist = point_dist[face_index]
    mean = np.mean(point_dist)
    var = np.var(point_dist)
    print("compare obj mean:{:.3f}, var:{:.3f}".format(mean, var))

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    files_3dmm = '/data/zhang.di/hifi3dface/3DMM/'
    if not is_BFM:
        basis3dmm = load_3dmm_basis(
            files_3dmm + '/files/AI-NEXT-Shape.mat',
            files_3dmm + '/files//AI-NEXT-Albedo-Global.mat',
            is_whole_uv=True)
    else:
        basis3dmm = load_3dmm_basis_synergy(
            files_3dmm + '/files/BFM_model_front.mat')
    dis_module = Distance(basis3dmm)
    teacher = Distance(basis3dmm)
    f = open('/data/zhang.di/GT_rigidalign.obj', 'r')
    lines = f.readlines()
    f.close()
    steps = 500
    momentum_schedule = cosine_scheduler(0.8, 1, steps, 1)



    target = []
    for line in lines:
        if 'v ' not in line:
            continue
        sps = line[2:].split(' ')
        target.append([float(sps[0]), float(sps[1]), float(sps[2])])
    target = torch.from_numpy(np.array(target))
    # idx_42 = [186080, 198184, 192481, 192688, 175830, 187397, 181130, 176368, 297662, 197274, 214045, 253873, 673755, 231030]
    idx_42 = [186080, 198184, 192481, 192688, 175830, 187397, 181130, 176368]
    idx_42 = torch.from_numpy(np.array(idx_42))
    target_42 = target[idx_42].cuda()
    target = target.cuda()
    # print(dis_module.parameters(), 'dis.parameters()')
    scale = 10
    my_optim = optim.Adam([
                    #{'params': dis_module.roll, 'lr': 1e-3* scale},
                    #{'params': dis_module.yaw, 'lr': 1e-3* scale},
                    #{'params': dis_module.pitch, 'lr': 1e-3* scale},
                    {'params': dis_module.s_roll, 'lr': 5e-2* scale},
                    {'params': dis_module.T, 'lr': 1e-2* scale},
                    {'params': dis_module.shape, 'lr': 5e-2* scale}
                    ], weight_decay=1e-6)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(my_optim, milestones=[150, 300, 450, 800])
    dis_module = dis_module.cuda()
    teacher.load_state_dict(dis_module.state_dict())
    # print(teacher.idxs[:10], dis_module.idxs[:10], 'dis_module')
    teacher = teacher.cuda()
    for i in range(1):
        for j in range(steps):
            if j % 1 == 0:
               with torch.no_grad():
                    idxs = teacher.get_idx(target.clone(),ori=1)
                    dis_module.fir = teacher.fir
            loss = dis_module(target[idxs].clone(), target_42)
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

            # print(my_optim.param_groups[0]["lr"], dis_module.roll, dis_module.yaw, dis_module.pitch, dis_module.s_roll, dis_module.T, j, 'haha')
            if j% 10 == 0 or j == steps-1:
                write_find('/data/zhang.di/longcheng_my/' + str(j) + '.obj', teacher, basis3dmm)
                with torch.no_grad():
                    v_xyz = teacher.get_res()
                    if not is_BFM:
                        write_obj(
                        '/data/zhang.di/longcheng_my/face.obj',
                        v_xyz,
                        basis3dmm["vt_list"],
                        basis3dmm["tri"].astype(np.int32),
                        basis3dmm["tri_vt"].astype(np.int32))
                    else:
                        write_obj_synergy(
                        '/data/zhang.di/longcheng_my/face_synergy.obj',
                        v_xyz,)
                if not is_BFM:
                    compare('/data/zhang.di/longcheng_my/' + str(j) + '.obj', '/data/zhang.di/longcheng_my/face.obj')
                else:
                    compare('/data/zhang.di/longcheng_my/' + str(j) + '.obj', '/data/zhang.di/longcheng_my/face_synergy.obj')
