import numpy as np
from utils.params import ParamsPackLarge
import json
import glob
import pdb
import open3d  as o3d
import cv2
import os


IMG_SIZE = 120

RR = np.array([[ 0.99988735, 0.0146241, -0.00337969], [-0.01416047, 0.99375874, 0.11064831], [ 0.00497673, -0.11058799, 0.99385388]])
ss = 9.420123997906711e-07
tt = np.array([-0.00026256,-0.01507333, -0.04221931])

labels = {}
f = open("3dmm_data/arkit_label.txt", 'r')
for m, line in enumerate(f):
    line = json.loads(line)
    name = line["save_name"]
    vertices = line["vertices"]
    labels[name] = vertices
print("load labels done")

param_pack = ParamsPackLarge()

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


def predict_vert(param):
    if param.shape[0] == 62:
        param = param * param_pack.param_std[:62] + param_pack.param_mean[:62]
    elif param.shape[0] == 156:
        param[:12] = param[:12] * param_pack.param_std[:12] + param_pack.param_mean[:12]
    else:
        raise RuntimeError('length of params mismatch')

    p, offset, alpha_shp, alpha_exp = parse_param(param)
    vertex = (param_pack.u_base + param_pack.w_shp_base @ alpha_shp + param_pack.w_exp_base @ alpha_exp).reshape(3, -1, order='F')
    return vertex

def predict_landmarks(param):
    if param.shape[0] == 62:
        param = param * param_pack.param_std[:62] + param_pack.param_mean[:62]
    elif param.shape[0] == 156:
        param[:12] = param[:12] * param_pack.param_std[:12] + param_pack.param_mean[:12]
    else:
        raise RuntimeError('length of params mismatch')

    p, offset, alpha_shp, alpha_exp = parse_param(param)
    vertex = p @ (param_pack.u_base + param_pack.w_shp_base @ alpha_shp + param_pack.w_exp_base @ alpha_exp).reshape(3, -1, order='F') + offset
    vertex[1, :] = param_pack.std_size + 1 - vertex[1, :]
    return vertex

def predict_vertices(vertex, roi_bbox):
    sx, sy, ex, ey, _ = roi_bbox
    scale_x = (ex - sx) / 120
    scale_y = (ey - sy) / 120
    vertex[0, :] = vertex[0, :] * scale_x + sx
    vertex[1, :] = vertex[1, :] * scale_y + sy

    s = (scale_x + scale_y) / 2
    vertex[2, :] *= s
    return vertex



if __name__ == '__main__':
    image_paths = glob.glob('save/*.npz')
    nmes = []
    idx1, idx2 = 1101, 1069
    for path  in image_paths:
        #pdb.set_trace()
        param = np.load(path)["base_param"]
        vertices = predict_vert(param)
        img_name = path.split('/')[-1][:-4]+'.JPG'
        label = np.array(labels[img_name])
        vx_ge0 = ss * vertices.T  @ RR.T + tt
        dn1 = max(1e-6, np.linalg.norm(label[idx1]-label[idx2]))
        nme = np.mean(np.linalg.norm(vx_ge0-label, axis=1))/dn1
        if nme < 0.20:
            nmes.append(nme)
        print(path, nme, np.mean(nmes))
        '''
        img = cv2.imread(os.path.join('save', img_name.replace(".JPG", ".png")))
        img = img[:, img.shape[1]//2:, :]
        param = np.load(path)["base_param"]
        landmarks = predict_landmarks(param)
        landmarks = predict_vertices(landmarks, [0, 0, 256, 256, 1])
        for idx in range(landmarks.shape[1]):
            point = [int(landmarks[:, idx][0]),int(landmarks[:, idx][1])]
            cv2.circle(img, point, 1, (0, 0, 255), -1)
        cv2.imwrite("landmarks_arkit/{}_{}".format(nme, img_name), img)
        '''
