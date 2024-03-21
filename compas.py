import sympy
import torch
import numpy as no
from utillc import *
import sympy.abc as X
from sympy import Point2D, symbols, solve, cos, sin, lambdify
import os, sys
import random

EKO()
experiment = False

if experiment :
    """
    a partir d'une equation, y = x*x 
    resoudre l'équation en fct de x : x = sqrt(y) 
    transformer la solution en code *torch* !
    permet de construire le graphe de calcul en torch pour faire la GD
    """
    x, y, z, u, v  = symbols("x y z u v")

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



l_symbols = list(range(10000))
a, r, ra, rb = symbols("a r ra rb")

def S(n="") :
    s = symbols("%s_%d" % (n, l_symbols.pop(0)))
    return s

def Pf(n="") :
    x = symbols("x%s_%d" % (n, l_symbols.pop(0)))
    y = symbols("y%s_%d" % (n, l_symbols.pop(0)))
    return Point2D(x, y)


A = Pf("A")
B = Pf("B")
xp = A.x + ra * cos(a)
yp = A.y + ra * sin(a)
P, Q = Point2D(xp, yp), Pf("Q")

"""
étant donnés 2 points A et B
P tourne autour de A ( rayon ra)
l'angle Ax, AP = a
Q tourne autour de B ( rayon rb)
dist(P,Q) = r

trouver les positions de P et Q
"""

eq = [ P.distance(Q) - r, P.distance(A) - ra, Q.distance(B) - rb]
EKO()
sol = solve(eq, list(P.coordinates) + list(Q.coordinates))
EKOX(sol)

"""
listes :
bras, fix point, mobile point, scalars (distances) 

build system : calls to stratX()
distances are params
select 2 points mobiles, pm1, pm2
objective = pm1 pm2 location at t=0 and t=end
optimize distances wrt objectives

"""

alpha = S("alpha")

fixed = [ Pf("A"), Pf("B")]
mobiles = []
scalars = []
bras = []
eee = []

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
"""

"""
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
    
    return { "mobiles" : [ Pm1], "fixed" : [Pf1, Pf2], "scalars" : [ d1, d2, d3], "bras" : [ b1, b2, b3]}

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




