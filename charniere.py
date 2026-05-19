from tkinter import *
from tkinter.ttk import *
from funcy import compose
import numpy as np
import math as m
from utillc import *
import torch
from torch import tensor as T
from torch import tensor as T1
from torch import optim, sqrt

import compas1

v = lambda x : np.asarray(x)
to_np = lambda x : x.cpu().detach().numpy()


def rotate(coordinates, angle_degree):
	(x,y) = coordinates
	angler = angle_degree*m.pi/180
	newx = x*m.cos(angler) - y*m.sin(angler)
	newy = x*m.sin(angler) + y*m.cos(angler)
	return v((newx, newy))

ps2 = 180/2
ps4 = ps2/2
Tsr = lambda x : torch.tensor(x, device=compas1.dev)

class Shape:
	L=50
	def compute(self, xy=(10, 60),
				longueur = 50,
				epaisseur=10,
				rot=0) :
		d = v((longueur, 0))
		de = v((0, epaisseur))
		dr = rotate(d, rot)
		der = rotate(de, rot)
		xy1 = v(xy) + dr + der 
		x0, y0 = xy
		xy2 = xy + dr
		x1, y1 = xy1
		return x0, y0, x1, y1
		
		
	def block(self, xy=(10, 60),
			  longueur = 50, epaisseur=10, rot=0) :
		x0, y0, x1, y1 = self.compute(xy, longueur, epaisseur, rot)
		rect = self.canvas.create_rectangle(x0, y0, x1, y1,
									 outline="black", fill="blue",
									 width=2)
		return rect

		
	def __init__(self, master=None):
		self.master = master
		
		self.create()


		degree = 2 * torch.pi / 360
		a = Tsr(21 * degree)
		
		aa = torch.arange(-40., 0, 0.1, device=compas1.dev)
		"""
		aa = torch.arange(40., 0, -1., device=dev)
		aa = torch.arange(40., 0, -9., device=dev)
		aa = Tsr([40., 31., 22., 12., 1.])
		aa = Tsr([40., 31., 22., 1.])		
		aa = Tsr([40., 31., 22.])
		aa = Tsr([39.])
		"""
		W = aa.shape[0]
		ard = aa / 360 * 2 * np.pi
		A, B, C, J,  u, s, v, o, b, q, n, w = compas1.build(W)
		self.lvs = ard, A, B, C, J,  u, s, v, o, b, q, n, w
		
		variables = [u, s, v, o, b, q, n, w]
		self.var = tuple(variables) 

		self.var = compas1.optimize()

		
		EKOX(variables)
		optimizer = optim.SGD(variables, lr=0.01, momentum=0.9)
		self.optimizer = optim.Adam(variables, lr=0.01)
		self.count = 0
		self.vvv = 32
		#self.tick()
		#self.master.after(100, self.tick)
		
	def tick(self) :
		self.master.after(100, self.tick)
		compas1.step(self.optimizer, self.lvs, self.count)
		self.count += 1
		self.dessiner(self.vvv)

	def dessiner(self, vvv):
			self.vvv = vvv = float(vvv)
			
			r=5
			shift = lambda x : ((x + (30-14, 0)) * 70 + ( 20, 20)) * ( 1, -1) + (180, 900)
			def kkk(NN) :
					vvvs = [vvv] * NN
					#EKOX(vvvs)
					A, B, C, J,  u, s, v, o, b, q, n, w = compas1.build(NN)
					u, s, v, o, b, q, n, w = self.var

					lps = compas1.f1(T(vvvs), A, B, C, J, u, s, v, o, b, q, n, w)
					lps1 = torch.stack(lps).permute(1,0,2)
					#EKOX(lps.isnan().any())
					assert(not lps1.isnan().any())
					
					lll = map(to_np, lps)
					lll = list(map(shift, lll))
					Cp, F, N, H, O, K = tuple(lll)
					#EKON(Cp, F, N, H, O, K)
					#EKON(A, B, C, J)
					
					#EKOX(list(lll))
					squeeze = lambda x : x[0]
					Cp, F, N, H, O, K = map(squeeze, tuple(lll))
					A, B, C, J = map(compose(shift,  to_np, squeeze), (A, B, C, J))

					return A, B, C, J, Cp, F, N, H, O, K
			
			A, B, C, J, Cp, F, N, H, O, K = kkk(1)
			dcc = lambda p, pc : (self.canvas.coords(p[0],
													 pc[0]-r, pc[1]-r, pc[0]+r, pc[1]+r),
								  self.canvas.coords(p[1],
													 pc[0], pc[1]-r*3))
			dcl = lambda p, pa, pb : self.canvas.coords(p, pa[0], pa[1], pb[0], pb[1])
			
			dcl(self.sacp, A, Cp)
			dcl(self.sbf, B, F)
			dcl(self.scpf, Cp, F)
			dcl(self.sfh, F, H)
			dcl(self.sjn, J, N)
			dcl(self.sho, H, O)
			dcl(self.sno, N, O)
			dcl(self.scpk, Cp, K)

			dcc(self.ca, A)
			dcc(self.cb, B)
			dcc(self.ccp, Cp)
			dcc(self.cc, C)
			dcc(self.cj, J)
			dcc(self.cf, F)
			dcc(self.cn, N)
			dcc(self.ch, H)
			dcc(self.co, O)			
			dcc(self.ck, K)			


	def ff(self, x) :
			self.label.config(text="%.2f" % float(x))
			self.canvas.after(10, self.dessiner, float(x) / 360*np.pi*2)
		
	def create(self):		
			self.canvas = Canvas(self.master)
			slider = Scale(orient='horizontal',
						   #label = 'Amount',
						   length = 500,
						   command = self.ff,
						   from_= -40, to=0) #-np.pi, to = np.pi)
			self.label = Label(text="text")
			c, r = (30, 40), 5

			css_arc = lambda : self.canvas.create_arc(c[0]-r, c[0]-r,
													  c[1]+r, c[1]+r,
													  fill="green",
													  outline="",
													  start=0,
													  extent=359)
			css_box = lambda : self.canvas.create_rectangle(c[0]-r*2, c[0]-r*2,
															c[1]+r*2, c[1]+r*2,
															fill="yellow",
															outline="black")
							
			ca = lambda txt, squared=False : (css_box() if squared else css_arc() ,
											   self.canvas.create_text(100,10,
																	   fill="darkblue",
																	   font="Helvetica 20 italic bold",
																	   text=txt))

			cl = lambda : self.canvas.create_line(c[0], c[1],
												  c[0]+20, c[1]+20,
												  fill="red",
												  width=3)
			self.ca = ca("A", True)
			self.cb = ca("B", True)
			self.ccp = ca("Cp")
			self.cc = ca("C", True)
			self.cj = ca("J", True)			
			self.cf = ca("F")
			self.cn = ca("N")
			self.ch = ca("H")
			self.co = ca("O")
			self.ck = ca("K")			
			
			self.sacp = cl()
			self.sbf= cl()
			self.scpf = cl()
			self.sfh = cl()
			self.sjn = cl()
			self.sho = cl()
			self.sno = cl()
			self.scpk = cl()			
			
		
			#self.dessiner(0)
			#amount = slider.get()		
			slider.pack()
			self.label.pack()
			self.canvas.pack(fill=BOTH, expand=1)

			EKOX(23)
			degree = 23.
			rd = degree / 360 * 2 * np.pi
			A, B, C, J,  u, s, v, o, b, q, n, w = compas1.build()			
			lll = map(to_np, compas1.f1(T([rd]), A, B, C, J, u, s, v, o, b, q, n, w))


			

if __name__ == "__main__":
	
	master = Tk()
	shape = Shape(master)

	master.title("Shapes")
	
	master.geometry("1000x1000+300+100")

	mainloop()
