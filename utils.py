import numpy as np
import pandas as pd


def cart2spherical(x, y, z):
	r = np.linalg.norm([x,y,z], axis=0)
	rho = np.linalg.norm([x,y], axis=0)
	phi = np.arctan2(y, x)
	theta = np.arctan2(rho, z)
	return r, phi, theta

class UnionFind:
	def __init__(self, n):
		self.parents = list(range(n))

	def find(self, a):
		if a == self.parents[a] : return a
		pa = self.find(self.parents[a])
		self.parents[a] = pa
		return pa

	def join(self, a, b):
		pa = self.find(a)
		pb = self.find(b)
		if pa != pb:
			self.parents[pa] = pb

	def connect(self, a, b):
		return self.find(a) == self.find(b)
