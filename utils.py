import numpy as np
import pandas as pd


def cart2spherical(x, y, z):
	r = np.linalg.norm([x,y,z], axis=0)
	rho = np.linalg.norm([x,y], axis=0)
	phi = np.arctan2(y, x)
	theta = np.arctan2(rho, z)
	return r, phi, theta
