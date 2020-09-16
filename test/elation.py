import numpy as np
import cv2
from Appendix import *

img = np.zeros((500,500,3), np.uint8)

#v = np.array([0,0,1])
#a = np.array([1./250, 1./300,0])
v = np.array([1.,0.,1.])
a = np.array([0, 1./300,0])

mu=100
H_p = np.eye(3) +mu*np.outer(v,a)
tx = 50
ty = 35 
th = np.pi/6.0
H_e = np.array([[np.cos(th), -np.sin(th), tx], [np.sin(th), np.cos(th), ty],[0,0,1]])
H_e[:2,:2]*=3
H_a = np.eye(3)
H_a[:2,:2]=np.random.rand(2,2)*100
H_a[1,0] = 0

print(H_p)
H = H_e@H_a@H_p
print(H)

'''
D, Q = np.linalg.eig(H_p)
print(D)
print(Q)
H = np.random.rand(3,3)
H[2,2]=0
print(H@np.array([0,0,1]))
invHt =np.linalg.inv(H).T 
Ht = H.T
l = Ht@np.array([0,0,1]) # source image vanishing line
x_o = np.array([0,0,1]) # source image origin
print(l)

'''
