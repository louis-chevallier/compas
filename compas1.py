import sympy
import torch
from torch import tensor as T1
from torch import optim, sqrt
import numpy as np
from utillc import *
import sympy.abc as X
from sympy import Point2D, symbols, solve, cos, sin, lambdify
import sympy
from sympy.utilities.lambdify import lambdastr
import os, sys
import random
from collections import namedtuple
import inspect
import matplotlib.pyplot as plt

print_everything()

EKO()

dev = "cuda" if torch.cuda.is_available() else "cpu"


V = lambda x : torch.tensor(x, requires_grad=True, device=dev)
T = lambda x : torch.stack(x)
Tsr = lambda x : torch.tensor(x, device=dev)
P = lambda x, y : T([Tsr(x),Tsr(y)])

def rotation_matrix(a) :
	s = torch.sin(a)
	c = torch.cos(a)
	rot = torch.stack([torch.stack([c, -s]),
					   torch.stack([s, c])])
	return rot

def rot(A, Center, a) :
		"""
		rotate a around Center by angle a
		"""
		CA = A - Center
		rotm = rotation_matrix(a).permute(2,1,0)
		R = torch.matmul(rotm, CA[:,:,None])
		X = R[:,:,0] + Center
		return X

def proj(A, B, l) :
		"""
		if dist(A,B) = d
		return X such that AX = AB / d * l
		"""
		d = (A-B).norm()
		X = (B-A)*l/d+A
		return X

def proj2(A, B, l) :
		"""
		if dist(A,B) = d
		return X such that AX = AB / d * l
		"""
		d = (A-B).norm(dim=1)
		f = (d+l)/d
		X1 = (B-A)*f[:,None]
		X = X1+A
		return X

u, s, px, x, py, y, qx, qy  = symbols("u s px x py y qx qy")
eq1 = ((px-x)**2 + (py-y)**2) - u ** 2
eq2 = ((qx-x)**2 + (qy-y)**2) - s ** 2
EKOT("solving..")
sol = solve([eq1, eq2], [x, y])

lf1 = lambdify([px, py, qx, qy, u, s], sol[0], "numpy")
lf2 = lambdify([px, py, qx, qy, u, s], sol[1], "numpy")

EKOX(sol[0])

lf1_code = "from torch import sqrt\n" + inspect.getsource(lf1)
with open("lf1_code.py", "w") as fd : fd.write(lf1_code.replace("_lambdifygenerated", "lf1"))


lf2_code = "from torch import sqrt\n" + inspect.getsource(lf2)
with open("lf2_code.py", "w") as fd : fd.write(lf2_code.replace("_lambdifygenerated", "lf2"))

from lf1_code import lf1
from lf2_code import lf2

				
def joint(a, b, u, s, i=0) :
		"""
		define a point P with dist(a,P) = u and dist(b, P) = s
		- equation solved by sympy
		"""
		X = (lf1 if i == 0 else lf2)(a[:, 0], a[:, 1],
									 b[:, 0], b[:, 1],
									 u, s)

		X = torch.vstack(X).T
		return X




def f1(a, A, B, C, J, u, s, v, o, b, q, n, w) :
		"""
				a en rd
		"""
		Cp= rot(C, A, a)
		K = joint(Cp, J, u, s)
		N = proj2(J, K, v)
		F = joint(Cp, B, o, b)
		H = proj2(Cp, F, q)
		O = joint(H, N, n, w, i=1)
		return Cp, F, N, H, O, K

pi = Tsr(np.pi)


def build(W=1) :
		A, B, C, J = P(-16., 10.),  P(-16., 6.),  P(-11., 9.), P(-14., 10.)
		u, s = map(V, [1.9, 2.])
		v = V(6.5)
		o, b = V(3.8), V(4.5)
		q = V(2.4)
		n, w = V(4.8), V(6.)
		ddd, u,   s,   v,   o,   b,   q,   n,   w = map(V, [
				0.,  1.9, 2.0, 6.5, 3.8, 4.5, 2.4, 4.8, 6.0])
		expand = lambda x : x.expand(W,-1)
		A, B, C, J = list(map(expand, (A,B,C,J)))
		
		return A, B, C, J,  u, s, v, o, b, q, n, w


def step(optimizer, ctxt, _n=0) :
				los, lhs = [], []
				optimizer.zero_grad()
				ard, A, B, C, J,  u, s, v, o, b, q, n, w = ctxt
				# version vectorisée
				lps = Cp, F, N, H, O, K = f1(ard, A, B, C, J,
											 u, s, v, o, b, q, n, w)
				lps = torch.stack(lps).permute(1,0,2)
				mask = ~torch.any(lps.isnan(),dim=(1,2))
				lps = lps[mask]
				lhs = lps[:, 3, :] # H
				los = lps[:, 4, :] # O
				hmx = torch.amax(lhs, dim=0)
				hmn = torch.amin(lhs, dim=0)
				omx = torch.amax(los, dim=0)
				omn = torch.amin(los, dim=0)
				
				loss = (hmx[0] - hmn[0]) # on veut excursion horizontale minimum
				loss += torch.abs(hmx[1] - hmn[1] - 300) / 300 # et de hauteur donnée
				
				loss += (los[0][1] - lhs[0][1]).norm() # on veut la hauteur de O et H au départ egale
				loss += (los[-1][0] - lhs[-1][0]).norm() # on veut l'abscisse de O et H a la fin egale
				EKON(_n, lps.shape, loss.item())
				loss.backward()
				optimizer.step()				
				return loss

def optimize() :
		"""
				H en haut : (258, 442), en bas : (297, 766)
				O en haut : (599, 440), en bas : (107, 1043)
		"""

		degree = 2 * torch.pi / 360
		a = Tsr(21 * degree)
		
		aa = torch.arange(-40., 0, 0.1, device=dev)
		"""
		aa = torch.arange(40., 0, -1., device=dev)
		aa = torch.arange(40., 0, -9., device=dev)
		aa = Tsr([40., 31., 22., 12., 1.])
		aa = Tsr([40., 31., 22., 1.])		
		aa = Tsr([40., 31., 22.])
		aa = Tsr([39.])
		"""
		W = aa.shape[0]
		ard = aa / 360 * 2 * pi
		A, B, C, J,  u, s, v, o, b, q, n, w = build(W)
		variables = [u, s, v, o, b, q, n, w]
		EKOX(variables)
		optimizer = optim.SGD(variables, lr=0.01, momentum=0.9)
		optimizer = optim.Adam(variables, lr=0.01)
		lps = Cp, F, N, H, O, K = f1(ard, A, B, C, J, u, s, v, o, b, q, n, w)
		lps = torch.stack(lps).permute(1,0,2)
		EKOX(lps.shape)
		mask = ~torch.any(lps.isnan(),dim=(1,2))
		lps = lps[mask]
		EKOX(lps.shape)
		ctxt = ard, A, B, C, J,  u, s, v, o, b, q, n, w
		losses = []
		for _n in range(300) :
				ll = step(optimizer, ctxt, _n=_n)
				losses.append(ll.item())
				
		plt.plot(losses); plt.show()
		return u, s, v, o, b, q, n, w
#optimize()
#EKON(u,   s,   v,   o,   b,   q,   n,   w)

