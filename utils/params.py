import os.path as osp
import numpy as np
from .io import _load
import scipy.io as sio

def make_abs_path(d):
    return osp.join(osp.dirname(osp.realpath(__file__)), d)

class ParamsPack():
	"""Parameter package"""
	def __init__(self):
		try:
		
			d = "3dmm_data"
			meta = _load(osp.join(d, 'param_whitening.pkl'))
			self.param_mean = meta.get('param_mean')
			self.param_std = meta.get('param_std')


			m = sio.loadmat(osp.join(d,"BFM_model_front.mat"))
			self.u = m['meanshape'].reshape(-1, 1) * 1e5
			self.tex_mean = m['meantex'].reshape(1, -1, 3)[..., ::-1].copy()
			self.w_shp = m['idBase'] * 1e5
			self.w_exp = m['exBase'] * 1e5
			#self.keypoints = _load(osp.join(d, 'bfm2arkit_v2.npy')).reshape(-1)
			keypoints = (m['keypoints'] - 1).reshape(-1)
			self.keypoints = [int(key) * 3 + ii for key in keypoints for ii in range(3)]

			self.u_base = self.u[self.keypoints]
			self.w_shp_base = self.w_shp[self.keypoints]
			self.w_exp_base = self.w_exp[self.keypoints]

			self.w_tex = m['texBase']
			self.triangles = m['tri'] - 1
                        
			self.std_size = 120

			'''
			d = make_abs_path('../3dmm_data')
			self.keypoints = _load(osp.join(d, 'bfm2arkit.npy'))
			
			# PCA basis for shape, expression, texture
			self.w_shp = _load(osp.join(d, 'w_shp_sim.npy'))
			self.w_exp = _load(osp.join(d, 'w_exp_sim.npy'))
			# param_mean and param_std are used for re-whitening
			meta = _load(osp.join(d, 'param_whitening.pkl'))
			self.param_mean = meta.get('param_mean')
			self.param_std = meta.get('param_std')
			# mean values
			self.u_shp = _load(osp.join(d, 'u_shp.npy'))
			self.u_exp = _load(osp.join(d, 'u_exp.npy'))
			self.u = self.u_shp + self.u_exp
			self.w = np.concatenate((self.w_shp, self.w_exp), axis=1)
			# base vector for landmarks
			self.w_base = self.w[self.keypoints]
			self.w_norm = np.linalg.norm(self.w, axis=0)
			self.w_base_norm = np.linalg.norm(self.w_base, axis=0)
			self.u_base = self.u[self.keypoints].reshape(-1, 1)
			self.w_shp_base = self.w_shp[self.keypoints]
			self.w_exp_base = self.w_exp[self.keypoints]
			self.dim = self.w_shp.shape[0] // 3
            '''

		except:
			raise RuntimeError('Missing data')


