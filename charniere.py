from tkinter import *
from tkinter.ttk import *

import numpy as np
import math as m
from utillc import *
import torch
from torch import tensor as T
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

	def tick(self) :
		self.master.after(100, self.tick)
		
	def __init__(self, master=None):
		self.master = master
		self.master.after(100, self.tick)
		self.create()

	def dessiner(self, vvv):
			vvv = float(vvv)
			r=5
			shift = lambda x : ((x + (30-14, 0)) * 70 + ( 20, 20)) * ( 1, -1) + (20, 900)
			lll = map(to_np, compas1.f1(T(vvv)))
			lll = list(map(shift, lll))
			#EKOX(list(lll))
			A, B, C, J, Cp, F, N, H, O, K = tuple(lll)

			EKOX(H)
			EKOX(O)

			
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
						   from_= -20, to=180) #-np.pi, to = np.pi)
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


if __name__ == "__main__":
	
	master = Tk()
	shape = Shape(master)

	master.title("Shapes")
	
	master.geometry("830x820+300+100")

	mainloop()
