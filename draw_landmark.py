import numpy as np
import cv2
import os.path as osp
import pickle
import scipy.io as sio
import pdb
import os
from utils.params import ParamsPack
param_pack = ParamsPack()
import json

def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]


def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))
    elif suffix == 'mat':
        return sio.loadmat(fp)


def parse_param(param):
    p_ = param[:12].reshape(3, 4)
    p = p_[:, :3]
    offset = p_[:, -1].reshape(3, 1)
    if param.shape[0] == 62:
        alpha_shp = param[12:52].reshape(40, 1)
        alpha_exp = param[52:62].reshape(10, 1)
    else:
        alpha_shp = param[12:92].reshape(80, 1)
        alpha_exp = param[92:].reshape(64, 1)
    return p, offset, alpha_shp, alpha_exp


def predict_vertices(vertex, roi_bbox):
    sx, sy, ex, ey, _ = roi_bbox
    scale_x = (ex - sx) / 120
    scale_y = (ey - sy) / 120
    vertex[0, :] = vertex[0, :] * scale_x + sx
    vertex[1, :] = vertex[1, :] * scale_y + sy

    s = (scale_x + scale_y) / 2
    vertex[2, :] *= s
    return vertex


if __name__ == "__main__":

    save_dir = './save_0822'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #train_list = open("/home/momobot/SynergyNet/3dmm_data/train_data_0809_syn.txt", 'r')
    #for m, path in enumerate(train_list):
    f = open("{}/pos.txt".format(save_dir), 'w')
    for path in os.listdir("/data1/zhang.hongshuang/batch_out"):
        if path.endswith("npz"):
             continue
        #if 1:
        try:
            #path = path.strip()
            path = os.path.join("/data1/zhang.hongshuang/batch_out", path)
            name = path[:-3] + 'npz'
            target_param = np.load(name)
            target_param = target_param['base_param']

            img = cv2.imread(path)
            print(path, img.shape)
            if img.shape[1] == 512:
                img = img[:, 256:, :]
            #img = cv2.resize(img, (512, 512))
            param = target_param
            if param.shape[0] == 62:
                param = param * param_pack.param_std[:62] + param_pack.param_mean[:62]
            elif param.shape[0] == 156:
                param[:12] = param[:12] * param_pack.param_std[:12] + param_pack.param_mean[:12]

            p, offset, alpha_shp, alpha_exp = parse_param(param)

            vertex = p @ (param_pack.u_base + param_pack.w_shp_base @ alpha_shp +
                           param_pack.w_exp_base @ alpha_exp).reshape(3, -1, order='F') + offset

            vertex[1, :] = param_pack.std_size + 1 - vertex[1, :]
            vertex = predict_vertices(vertex, [0, 0, 120, 120, 1])
            aa = {"name":path, "vertex":vertex.tolist()}
            f.write(json.dumps(aa)+'\n') 
            #print("vertex",vertex)
            #for idx in range(vertex.shape[1]):
            #    point = [int(vertex[:, idx][0]),int(vertex[:, idx][1])]
            #    cv2.circle(img, point, 1, (0, 0, 255), -1)
            #cv2.imwrite("{}/{}".format(save_dir, path.split('/')[-1]), img)
            print("save img {}".format(path))

        except Exception as e:
            print(e)
