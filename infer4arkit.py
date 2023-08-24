import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from utils.ddfa import ToTensor, Normalize
from model_building import SynergyNet
from utils.inference import crop_img, predict_sparseVert, draw_landmarks, predict_denseVert, predict_pose, draw_axis, predict_vert
import argparse
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import os
import pdb
import os.path as osp
import glob
from FaceBoxes import FaceBoxes
from utils.render import render
import json
# Following 3DDFA-V2, we also use 120x120 resolution
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



def main(args):
    # load pre-tained model
    checkpoint_fp = 'ckpts_basis156_0811/SynergyNet_checkpoint_epoch_100.pth.tar' 
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
            files = sorted(glob.glob(args.files+'*/*.JPG'))
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
            print("nme", np.mean(nmes))
            vertices_lst.append(vx_ge0)

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
    print("nme", np.mean(nmes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', default='', help='path to a single image or path to a folder containing multiple images')
    parser.add_argument("--png", action="store_true", help="if images are with .png extension")
    parser.add_argument('--img_size', default=120, type=int)
    parser.add_argument('-b', '--batch-size', default=1, type=int)

    args = parser.parse_args()
    main(args)
