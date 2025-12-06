import sympy
import torch
import numpy as np
from utillc import *
import sympy.abc as X
from sympy import Point2D, symbols, solve, cos, sin, lambdify
import sympy
import os, sys
import random
from collections import namedtuple
print_everything()

EKO()
experiment = False

"""
geogebra
connect google drive
compas-v2


"""


def dist2(A, B) :
	px, py = A.x, A.y
	x, y = B.x, B.y
	return (px-x)**2 + (py-y)**2


if experiment :
	"""
	a partir d'une equation, y = x*x 
	resoudre l'équation en fct de x : x = sqrt(y) 
	transformer la solution en code *torch* !
	permet de construire le graphe de calcul en torch pour faire la GD
	"""
	x, y, z, u, v, px, py, qx, qy  = symbols("x y z u v px py qx qy")

	"""
	on donne 2 equations et 2 variables u et v
	
	"""


	eq1 = sympy.sqrt((px-x)**2 + (py-y)**2) - u
	eq2 = sympy.sqrt((qx-x)**2 + (qy-y)**2) - v

	eq1 = ((px-x)**2 + (py-y)**2) - u**2
	eq2 = ((qx-x)**2 + (qy-y)**2) - v**2	

	s = solve([eq1, eq2], [x, y])
	
	EKOX(s)
	
	e1, e2 = y**2 + z, z - y
	eq = [ e1 - u, e2 - v]

	ss = [(y,2),(z,3)]
	EKOX((e1.subs(ss), e2.subs(ss)))
	

	s = solve(eq, [y, z])
	EKOX(s)
	f1 = lambdify([ u, v], s, "numpy")
	os.makedirs("gen", exist_ok = True)
	with open("gen/f.py", "w") as fd :
		ss = """
from torch import *
def f(u, v) :
	return %s """
		fd.write(ss % str(s))
		imp = __import__("f")
		EKOX(imp.f(torch.tensor(7.) ,torch.tensor(1.)))
		EKOX(f1(7., 1.))


	sys.exit(0)

deqs = {}
mobiles = []
scalars = []
bras = []
eee = []

l_symbols = list(range(10000))
a, r, ra, rb = symbols("a r ra rb")

def S(n="S") :
	s = symbols("%s_%d" % (n, l_symbols.pop(0)))
	scalars.append(s)
	return s

def Pf(n="s") :
	x = symbols("%s_%d" % (n, l_symbols.pop(0)))
	return x

def P2(pn) :
	p = Point2D(Pf(pn + "x"), Pf(pn + "y"))
	return p
EKOX(P2("P"))

A, B, C, J = fixed = [ P2("A"), P2("B"), P2("C"), P2("J") ]



def choose(l) :
	i = random.randint(0, len(l)-1)
	return l[i] 

def brasf(P1, P2, d) :
	"""
	rend l'eq correspondant a un bras de longueur d entre P1 et P2
	"""
	eq = [ P1.distance(P2) - d]
	return { "eq" : eq, "p" : [P1, P2] }

"""
mobiles : liste des points mobiles
fixed : liste des points fixes
scalars : liste des valeurs scalaires ( variables a optimiser )
bras : liste des bras 

les strats ajoutent des points fixes, des points mobiles, des bras 
et des scalaires
Xf un pt fixe
Xm un pt mobiles
B9 un bras


depart : Af - un point fixe => a, d, Pm, bras(Af, Pm, d) / d(Af, Pm) == d, 
initialisation : 
mobiles = [Pm0], fixes = [ Af0 ], bras = [ bras(Af0, Pm0, d) ], scalars = [d]
a = angle(Af0_x, Af0_Pm)


Strat :
	init() :
	   solve(eq, p)
	   f = torch(eq)
	exec() => return f(angle)

1/ => Af, nouveau point fixe

2/ P1, P2 => d1, d2, Qm, bras(Qm, P1, d1), bras(Qm, P2, d2) 

3/ bras(P, Q) => d, R, bras(P, R, d1), bras(P, Q), bras(Q, R) / R = P + v(P,Q) * u 


construit strats
optimizer = optim(scalars)

construction du graphe:
start = next_strat()

Ta_0, Tb_0 : position initiale des extremités du segment
Ta_90, Tb_90 : position finale des extremités du segment

loop :
	for s in strats :
	   l0.append(exec(s, angle = 0))
	   l90.append(exec(s, angle = 90))

	p0_1, p0_2 = l0[-2:-1]
	p90_1, p90_2 = l90[-2:-1]
	loss = dist(p0_1, Ta_0) + dist(p0_2, Tb_0) 
	loss += dist(p90_1, Ta_90) + dist(p00_2, Tb_90) 

	loss.back()
	step()

"""

def strat1() :
	"""
	called once
	use alpha = angle of b1
	pick 2 fix points pf1, pf2
	define 2 mobile points pm1, pm2
	define 3 distances d1, d2, d3
	b1 = bras(pf1, pm1, d1)
	b2 = bras(pf2, pm2, d2)
	b3 = bras(pm1, pm2, d3)
	yield :
		   pm1, pm2 : mobile
		   d1, d2, d3 : scalaires
		   b1, b2, b3 : bras
	"""
	
	Pf1, Pf2 = choose(fixed), choose(fixed)
	Pm2 = Pf("p2")
	d1, d2, d3 = S("d1"), S("d2"), S("d3")
	xp = Pf1.x + d1 * cos(a)
	yp = Pf1.y + d1 * sin(a)
	Pm1 = Point2D(xp, yp)
	
	b1 = brasf(Pf1, Pm1, d1)
	b2 = brasf(Pf2, Pm2, d2)
	b3 = brasf(Pm1, Pm2, d3)
	
	return { "mobiles" : [ Pm1],
			 "fixed" : [Pf1, Pf2],
			 "scalars" : [ d1, d2, d3],
			 "bras" : [ b1, b2, b3]}

def strat4() :
	"""
	pick fix point pf, mobile point pm1
	b1 = bras(pf, pm, d1)
	b2 = bras(pm1, pm, d2)
	yield :
		 pm, d1, d2, b1, b2
	"""
	Pf1 = choose(fixed)
	Pm1 = choose(mobiles)
	d1, d2 = S("d1"), S("d2")
	Pm = Pf("pm")
	b1 = brasf(Pf1, Pm, d1)
	b2 = brasf(Pm1, Pm, d2)
	
	return { "mobiles" : [ Pm ], "fixed" : [], "scalars" : [ d1, d2], "bras" : [ b1, b2]}
	

def add(strt) :
	"""
	ajoute des éléments décrits par str
	"""
	global fixed, mobiles, scalars, bras
	d = strt()
	fixed += d["fixed"]
	mobiles += d["mobiles"]
	scalars += d["scalars"]
	bras += d["bras"]
	
	eqs = [ ee for e in d["bras"] for ee in e["eq"] ]
	EKOX(eqs)
	EKOX(d["mobiles"])
	s = solve(eqs, [ c for p in d["mobiles"] for c in p.coordinates] )
	EKOX(s)
	eee.append(s)

if False :
	add(strat1)
	add(strat4)

	EKOX(bras)
	EKOX(mobiles)
	eqs = [ ee for e in bras for ee in e["eq"] ]
	EKOX(eqs)
	EKOX(mobiles)
	#s = solve(eqs, [ c for p in mobiles for c in p.coordinates] )
	#EKOX(s)

def strat2() :
	"""
	pick 2 fixed points
	define 2 params
	pf = u pf1 + v pf2
	yield :
		   pf1, pf2
	"""

def strat3() :
	"""
	pick a bras b = bras(A, B, p)
	define 2 params
	yield : 
		   pm = u A + v B
	"""

def rot(a, b, nm="rr") :
	s = S(nm)
	rr = a.rotate(s, b)
	P = Point2D(rr.x.simplify(), rr.y.simplify())
	return P, s

def joint(a, b, nm1="j", nm2="jj", i=0) :
	P = P2("xx")
	u, s = S(nm1), S(nm2)

	symbs = "x y z vu vv px py qx qy".split(" ")
	def create_symbol(n) :
			s = symbols("%s_%d" % (n, l_symbols.pop(0)))
			return s
	syms = map(create_symbol, symbs)
	x, y, z, vu, vv, px, py, qx, qy  = syms
	eq1 = ((px-x)**2 + (py-y)**2) - u ** 2
	eq2 = ((qx-x)**2 + (qy-y)**2) - s ** 2
	EKO()
	sol = solve([eq1, eq2], [x, y])
	EKO()
	deqs["joint"] = sol
	ls = [(px, a.x), (py, a.y), (qx, b.x), (qy, b.y)]
	rx = sol[i][0].subs(ls)
	EKOX(rx)
	rx = rx.simplify()
	EKO()
	ry = sol[i][1].subs(ls)
	EKO()
	ry = ry.simplify()
	EKO()
	P = Point2D(rx, ry)

	bras.append((a, P))
	bras.append((b, P))
	
	return P, u, s

def proj(a, b, nm="pp") :
	u = S(nm)
	ax, ay, bx, by  = symbols("ax ay bx by")
	d = sympy.sqrt(((ax-bx)**2 + (ay-by)**2))
	rx = ( bx + (bx-ax)/d * u)
	ry = ( by + (by-ay)/d * u)
	ls = [(ax, a.x), (ay, a.y), (bx, b.x), (by, b.y)]
	rx = rx.subs(ls).simplify()
	ry = ry.subs(ls).simplify()
	#P = a + (b-a) / d * u
	P = Point2D(rx, ry)
	bras.append((a, P))
	return P, u

t1 = torch.tensor(1.)

lf = [ n for p in fixed for n in [ p.x, p.y]]
lf1 = [ n for p in fixed for n in [ t1, t1]]
lfi = [ (-16., 12.),
		(-16., 6.),
		(-10., 12.),
		(-14., 8.)]
		
lf1 = [ n for p in fixed for n in [ t1, t1]]
lf1 = [ n for p in lfi for n in p]
EKOX(lf)
#EKOX(bras)

lsciv = [-25.  / 180 * sympy.pi,
		 2.4, # u
		 1.6, # s
		 5.1, # v
		 5.2, # o
		 5.6, # b
		 2.4, # q
		 4.8, # n
		 6]   #w

sciv = map(torch.tensor, lsciv)

flat = lambda l : [ e for p in l for e in p]
dumpx = lambda e : sympy.N(e.x.subs(list(zip(lf, flat(lfi)))).subs(list(zip(scalars, lsciv))))
dumpy = lambda e : sympy.N(e.y.subs(list(zip(lf, flat(lfi)))).subs(list(zip(scalars, lsciv))))

dump = lambda e : sympy.N(e.subs(list(zip(scalars, lsciv))).subs(list(zip(lf, flat(lfi)))))

EKO()
Cp, a = rot(C, A, "angle")
EKO()
K, u, s = joint(Cp, J, "u", "s")
EKO()
N, v = proj(J, K, "v")
EKO()
F, o, b = joint(Cp, B, "o", "b")
EKO()
H, q = proj(Cp, F, "q")
EKO()
O, n, w = joint(H, N, "n", "w")

EKOX(list(zip(scalars, lsciv)))
EKOX(list(zip(fixed, lfi)))

#EKOX(F)

EKOX(dump(Cp))
EKOX(dump(K))
EKOX(dump(N))
EKOX(dumpx(F))
EKOX(dumpy(F))
EKOX(dump(F))
EKOX(dump(H))

eee = O.subs(list(zip(scalars, lsciv)))
EKO()
eee = eee.subs(list(zip(lf, flat(lfi))))
#EKOX(eee)

EKOX(sympy.N(eee))

EKOX(dump(O))


#EKOX(H)
#EKOX(O)
#f1 = lambdify(scalars, H, "numpy")

EKOX(",".join(map(str, scalars)))

EKOX(fixed)


sc = ",".join(map(str, scalars))
sci = ",".join(map(str, [t1 for _ in scalars]))
sci = ",".join(map(str, sciv))
slf = ','.join(map(str, lf))
slf1 = ','.join(map(str, lf1))

with open("gen/f.py", "w") as fd :
	ss = """
import sympy
import torch
import numpy as no
from utillc import *
import os, sys
import random
from collections import namedtuple
from torch import *

def Point2D(x, y) :
	return x, y


%s = %s

def f(%s) :
	return %s, %s 

EKOX(f(%s))

"""
	fd.write(ss % (slf, slf1, sc, str(F), str(F), sci))
#	imp = __import__("f")
#	EKOX(imp.f(torch.tensor(7.) ,torch.tensor(1.)))
#	EKOX(f1(7., 1.))
