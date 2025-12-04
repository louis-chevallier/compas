
from torch import *
def f(u, v) :
    return [(-sqrt(4*u - 4*v + 1)/2 - 1/2, v - sqrt(4*u - 4*v + 1)/2 - 1/2), (sqrt(4*u - 4*v + 1)/2 - 1/2, v + sqrt(4*u - 4*v + 1)/2 - 1/2)] 