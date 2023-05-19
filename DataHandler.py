import os
from random import shuffle
import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import torch as t
from torchvision import datasets
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from Params import args

device = "cuda" if t.cuda.is_available() else "cpu"

class DataHandler():
	def __init__(self):
		if args.data == 'Yelp':
			predir = 'Data/Yelp/'
		elif args.data == 'CiaoDVD':
			predir = 'Data/CiaoDVD/'
		elif args.data == 'Epinions':
			predir = 'Data/Epinions/'
		self.predir = predir
		self.trnfile = predir + 'train.csv'
		self.tstfile = predir + 'test_Data.csv'
		self.uufile = predir + 'trust.csv'
	
	def normalizeAdj(self, mat):
		"""Create symmetrically normalized Laplacian matrix.

		Args:
			mat: Original matrix.

		Returns:
			Symmetrically normalized Laplacian matrix.
		"""
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()
		
	def makeTorchAdj(self, mat):
		"""Create tensor-based adjacency matrix for user-item graph.
			
		Args:
			mat: Adjacency matrix.
		
		Returns:
			Tensor-based adjacency matrix.
		"""
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat+ sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)
		
		# make cuda tensor
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		return t.sparse.FloatTensor(idxs, vals, shape).to(device)

	def makeTorchuAdj(self, mat):
		"""Create tensor-based adjacency matrix for user social graph.
			
		Args:
			mat: Adjacency matrix.
		
		Returns:
			Tensor-based adjacency matrix.
		"""
		mat = (mat != 0) * 1.0
		mat = (mat+ sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)
		
		# make cuda tensor
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		return t.sparse.FloatTensor(idxs, vals, shape).to(device)

	def LoadData(self):
		with open(self.trnfile, 'rb') as fs: # csr
			trnMat = pickle.load(fs)
		with open(self.tstfile, 'rb') as fs: # list
			testData = pickle.load(fs)
		with open(self.uufile, 'rb') as fs: # csr
			uuMat = pickle.load(fs)
		args.user, args.item = trnMat.shape
		args.edgeNum = len(trnMat.data)
		args.uuEdgeNum = len(uuMat.data)
		self.torchAdj = self.makeTorchAdj(trnMat)
		self.torchuAdj = self.makeTorchuAdj(uuMat)

		trnMat = trnMat.tocoo()
		uuMat = uuMat.tocoo()
		self.trnMat = trnMat
		self.uuMat = uuMat
		trainData = np.hstack([trnMat.row.reshape(-1, 1), trnMat.col.reshape(-1, 1)]).tolist() # (u, v) list
		uuData = np.hstack([uuMat.row.reshape(-1, 1), uuMat.col.reshape(-1, 1)]).tolist()

		trnData = BPRData(trainData, trnMat, uuData, uuMat, isTraining=True)
		tstData = BPRData(testData, trnMat, uuData, uuMat, isTraining=False)
		self.trnLoader = DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
		self.tstLoader = DataLoader(tstData, batch_size=args.test_batch*1000, shuffle=False, num_workers=0, pin_memory=True)

class BPRData(Dataset):
	def __init__(self, data, coomat, uuData, uumat, negNum=None, isTraining=None):
		super(BPRData, self).__init__()
		self.data = data
		self.uuData = uuData
		self.dokmat = coomat.todok()
		self.uuDokmat = uumat.todok()
		self.negNum = negNum
		self.isTraining = isTraining
		self.negs = np.zeros(len(self.data)).astype(np.int32)
		self.uuNegs = np.zeros(len(self.uuData)).astype(np.int32)
		
	def negSampling(self):
		assert self.isTraining, 'No need to sample when testing'
		for i in range(len(self.data)):
			u = self.data[i][0]
			while True:
				iNeg = np.random.randint(args.item)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg

		for i in range(len(self.uuData)):
			u = self.uuData[i][0]
			while True:
				uNeg = np.random.randint(args.user)
				if (u, uNeg) not in self.uuDokmat:
					break
			self.uuNegs[i] = uNeg
		
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		if self.isTraining: # # usr: pos: neg = 1: 1: 1
			return self.data[idx][0], self.data[idx][1], self.negs[idx]#, self.uuData[idx][0], self.uuData[idx][1], self.uuNegs[idx]
		else:
			return self.data[idx][0], self.data[idx][1]