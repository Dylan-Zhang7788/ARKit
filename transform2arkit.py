import numpy as np
import scipy
from math import sin, cos, atan2
from scipy.spatial.transform import Rotation
import pdb
import open3d  as o3d
from ply import save_ply
from infer4arkit_v2_hifi import load_3dmm_basis

def ralign(X,Y):
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
    Xc =  X - np.tile(mx, (n, 1)).T
    Yc =  Y - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(Xc*Xc, 0))

    Sxy = np.dot(Yc, Xc.T) / n

    U,D,V = np.linalg.svd(Sxy,full_matrices=True,compute_uv=True)
    V=V.T.copy()

    S = np.eye(m)

    R = np.dot( np.dot(U, S ), V.T)

    s = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - s * np.dot(R, mx)

    return R,s,t


if __name__ == '__main__':
    trans_file = open('id_dict_hifi3d2arkit_jawOpen.txt', "r")
    trans = {}
    for line in trans_file.readlines():
        line = line.strip().split()
        trans[int(line[0])]  = int(line[1])   
    print("load trans done")
    
    vertices_s = []
    # source_file = open('face.obj', 'r')    
    # for line in source_file.readlines():        
    #     if line.startswith('v'):
    #         line = line.strip().split()[1:]
    #         if len(line)<3 :
    #             continue
    #         vertices_s.append([float(line[0])*1e5, float(line[1])*1e5, float(line[2])*1e5])
    files_3dmm = '3dmm_data'
    mu_shape = load_3dmm_basis(files_3dmm + '/AI-NEXT-Shape.mat',
                               files_3dmm + '/AI-NEXT-Albedo-Global.mat',
                               is_whole_uv=True)["mu_shape"].T
    keypoints = []
    for idx in range(len(trans)):
        for i in range(3):
            keypoints.append(trans[idx]+i)
    key = np.load("array_d.npy")
    mu_shape = np.asarray(mu_shape[key]).T
    vertices_ss = np.reshape(mu_shape, [mu_shape.shape[1] // 3, 3]).tolist()
    # vertices_ss = []
    # for i in range(len(trans)):
    #     vertices_ss.append(vertices_s[trans[i]])
    # print("load vertices_s done")

    # D =[]
    # C = np.asarray(vertices_ss).reshape(3660, 1)
    # for x in C:
    #     indices = np.where(mu_shape == x)
    #     if indices[0].size > 0:
    #         D.append(indices[1][0])
    # file_name = "array_d.npy"
    # np.save(file_name, D)


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices_ss)
    o3d.io.write_point_cloud("trans2.ply", pcd)
    

    vertices_g = []
    target_file = open('Neutral_arkit.obj', "r")
    # vertices_ss = []
    idx = 0
    for line in target_file.readlines():
        if line.startswith('v'):
            line = line.strip().split()[1:]
            vertices_g.append([float(line[0]), float(line[1]), float(line[2])])
            # vertices_ss.append(vertices_s[trans[idx]])
            idx+=1
    print("load vertices_g done")

    vertices_g = np.array(vertices_g)
    vertices_ss = np.array(vertices_ss)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices_g)
    o3d.io.write_point_cloud("trans1.ply", pcd)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices_ss)
    o3d.io.write_point_cloud("trans2.ply", pcd)


    print(vertices_g.shape)
    print(vertices_ss.shape)
    R, ss, tt =  ralign(vertices_ss, vertices_g)
    print('R: ', R)
    print('ss: ', ss)
    print('tt: ', tt)

    vx_ge0 = ss * vertices_ss @ R.T + tt
    print("vx_ge0", vx_ge0.shape)
    
    save_ply(vx_ge0, vertices_g, "point_cloud_compare_hifi_vs_arkit.ply", [1, 0, 0], [0, 1, 0])
