from tkinter import *
from tkinter.ttk import *

import numpy as np
import math as m
from utillc import *
import torch

v = lambda x : np.asarray(x)


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
			x0, y0, x1, y1 = self.compute(rot=vvv)
			self.canvas.coords(self.b1, x0, y0, x1, y1)
			
			A = (30, 40)
			B = (30, 20)
			Cp = (60, 40)
			F = (80, 30)		
			r=5
			dcc = lambda p, pc : self.canvas.coords(p, pc[0]-r, pc[1]-r, pc[0]+r, pc[1]+r)
			dcl = lambda p, pa, pb : self.canvas.coords(p, pa[0], pa[1], pb[0], pb[1])
			
			
			dcc(self.ca, A)
			dcc(self.cb, B)
			dcc(self.ccp, Cp)
			dcc(self.cf, F)
			
			dcl(self.sacp, A, Cp)
			dcl(self.sbf, B, F)
			dcl(self.scpf, Cp, F)
			

	def ff(self, x) :
			self.canvas.after(10, self.dessiner, x)			
		
	def create(self):		
			self.canvas = Canvas(self.master)
			slider = Scale(orient='horizontal',
						   #label = 'Amount',
						   length = 500,
						   command = self.ff,
						   from_= -np.pi, to = np.pi)
			self.b1 = self.block(longueur=100, rot=0)
			c, r = (30, 40), 5

			ca = lambda : self.canvas.create_arc(c[0]-r, c[0]-r,
												 c[1]+r, c[1]+r,
												 fill="green",
												 outline="",
												 start=0,
												 extent=359)
			
			cl = lambda : self.canvas.create_line(c[0], c[1],
												  c[0]+20, c[1]+20,
												  fill="pink",
												  width=3)
			self.ca = ca()
			self.cb = ca()
			self.ccp = ca()
			self.cf = ca()
			
			self.sacp = cl()
			self.sbf= cl()
			self.scpf = cl()
			
		
			#self.dessiner(0)
			#amount = slider.get()		
			slider.pack()
			self.canvas.pack(fill=BOTH, expand=1)


if __name__ == "__main__":
	
	master = Tk()
	shape = Shape(master)

	master.title("Shapes")
	
	master.geometry("830x820+300+100")

	mainloop()
