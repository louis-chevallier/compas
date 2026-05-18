import sympy
import torch
from torch import tensor as T, optim, sqrt
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


print_everything()

EKO()

V = lambda x : torch.tensor(x, requires_grad=True)

P = lambda x, y : T([x,y])

def rotation_matrix(a) :
				return T([[torch.cos(a), -torch.sin(a)],
						  [torch.sin(a), torch.cos(a)]])

def rot(A, Center, a) :
		"""
		rotate a around Center by angle a
		"""
		CA = A - Center
		R = rotation_matrix(a) @ CA
		X = R + Center
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
		d = (A-B).norm()
		X = (B-A)*(d+l)/d+A
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
		Ax_2, Ay_3 = a
		Bx_4, By_5 = b
		u_12, s_13 = u, s

		lf1(a[0], a[1], b[0], b[1], u, s)		
		
		X = (lf1 if i == 0 else lf2)(a[0], a[1],
									 b[0], b[1],
									 u, s)
		X = torch.hstack(X)
		return X

degree = 2 * torch.pi / 360
A, B, C, J = P(-16., 10.),  P(-16., 6.),  P(-11., 9.), P(-14., 10.)

a = T(21 * degree)

u, s = map(V, [1.9, 2.])
v = V(6.5)
o, b = V(3.8), V(4.5)
q = V(2.4)
n, w = V(4.8), V(6.)

ddd, u,   s,   v,   o,   b,   q,   n,   w = map(V, [
0.,  1.9, 2.0, 6.5, 3.8, 4.5, 2.4, 4.8, 6.0])
EKON(u,   s,   v,   o,   b,   q,   n,   w)
def f1(a) :
				"""
				a en rd
				"""
				Cp= rot(C, A, a)
				K = joint(Cp, J, u, s)
				N = proj2(J, K, v)
				F = joint(Cp, B, o, b)
				H = proj2(Cp, F, q)
				O = joint(H, N, n, w, i=1)
				return A, B, C, J, Cp, F, N, H, O, K
pi = T(np.pi)

def optimize() :
				"""
				H en haut : (258, 442), en bas : (297, 766)
				O en haut : (599, 440), en bas : (107, 1043)
				"""
				
				variables = [u, s, v, o, b, q, n, w]
				optimizer = optim.SGD(variables, lr=0.01, momentum=0.9)
				optimizer = optim.Adam(variables, lr=0.1)
				for _n in range(100) :
								los, lhs = [], []
								optimizer.zero_grad()
								for ia, a in enumerate(torch.arange(40., 0, -0.1)) :
												rd = a / 360 * 2 * pi
												lps = A, B, C, J, Cp, F, N, H, O, K = f1(rd)
												lps = torch.cat(lps)
												if lps.isnan().any() :
																EKOX(ia)
																break
												
												los.append(O)
												lhs.append(H)

								lhs = torch.stack(lhs)
								hmx = torch.amax(lhs, dim=0)
								hmn = torch.amin(lhs, dim=0)

								los = torch.stack(los)
								omx = torch.amax(los, dim=0)
								omn = torch.amin(los, dim=0)
								
								loss = (hmx[0] - hmn[0]) # on veut excursion horizontale minimum
								loss += torch.abs(hmx[1] - hmn[1] - 300) # et de hauteur donnée

								loss += (los[0][1] - lhs[0][1]).norm() # on veut la hauteur de O et H au départ egale
								loss += (los[-1][0] - lhs[-1][0]).norm() # on veut l'abscisse de O et H a la fin egale
								EKON(_n, loss.item())
								loss.backward()
								optimizer.step()				
												
optimize()
EKON(u,   s,   v,   o,   b,   q,   n,   w)

