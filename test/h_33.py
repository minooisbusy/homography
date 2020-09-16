from Appendix import drawLine
from homography_SL import *
import numpy as np
import cv2
import sys

if __name__=='__main__':
    src = np.zeros((500,500,3), dtype=np.uint8)
    dst = np.zeros((500,500,3), dtype=np.uint8)

    S = 3.*255.0/(500.0**2*2)


    for x in range(0,500):
        for y in range(0, 500):
            dist = float(x)**2+float(y)**2
            dist*=S # 0~765
            r = dist%512 if dist>512 else dist
            dist = dist-512 if dist>=512 else dist # 0<=dist<512
            b = dist%(256) if dist>256 else dist 
            dist = dist-256 if dist>=256 else dist 
            g = dist
            src[x,y,0]=b
            src[x,y,1]=g
            src[x,y,2]=r
    

    pts = []
    res = []
    pts.append([125.,125.,1.])
    pts.append([125.,375.,1.])
    pts.append([375.,125.,1.])
    pts.append([375.,375.,1.])

    src_line=[]
    dst_line=[]
    src_line.append([-1.0, 23.0, 0.]) # Origin = (0,0,1)을 지나는 라인 y=x
    src_line.append([0.0, 1, -250.])
    src_line.append([1.0, 0, -125.])
    src_line.append([1.0, 0, -375.])

    dst_line.append([0., 0., 1.]) # line at infinity
    dst_line.append([0., 1., -250.])
    dst_line.append([1., 0., -125.])
    dst_line.append([1., 0., -375.])
    '''
    dst_line.append([1, 1, -375])
    dst_line.append([1, 0, -125])
    dst_line.append([1, -1, -375])
    '''
    src_line = np.array(src_line).T
    dst_line = np.array(dst_line).T

    src_line = src_line.astype(float)
    dst_line = dst_line.astype(float)

    ''' Homography est'''
    T1, fp, T2, tp = normalize(src_line, dst_line)
    H_l= homography(T1, fp, T2, tp)
    #print(H)
    H_p = np.linalg.inv(H_l.T)
    print(H_p)
    print('is line at infinity?=',H_l@src_line[:,0])
    print('is point at infinity?=',H_p@np.array([0,0,1]))


    sys.exit()


    pts = np.array(pts)
    A = (np.random.rand(3,3))*1700
    A[2,2]=0
    A[0,:] /= A[2,0]+A[2,1]+A[2,2]
    A[1,:] /= A[2,0]+A[2,1]+A[2,2]
    A[2,:] /= (A[2,0]+A[2,1]+A[2,2])**1.8

    L,U = np.linalg.eig(A)
    print(A)
    print(U)
    v1 = U[:,0]
    v2 = U[:,1]
    v3 = U[:,2]/U[2,2]
    l = np.cross(v1,v2)
    l /=l[2]
    distance = l@v3/np.linalg.norm(l[:2])
    print("distance between line and point=",distance)
    v1 = U[:,0]/U[2,0]
    v2 = U[:,1]/U[2,1]
    v3 = U[:,2]/U[2,2]
    v1=v1[:2].astype(int)
    v2=v2[:2].astype(int)
    v3=v3[:2].astype(int)

    c = (0,255,255)
    cv2.circle(src, tuple(v1),5, c,-1)
    cv2.circle(src, tuple(v2),5, c,-1)
    cv2.circle(src, tuple(v3),5, c,-1)
    print(L)
    
    cv2.rectangle(src,(125,125), (375,375), (0,0,255),3)

    for i in range(len(pts)):
        tmp = A@pts[i]
        tmp /= tmp[2]
        tmp[0]+=.5
        tmp[1]+=.5
        tmp=tmp.astype(int)
        res.append(tmp)

    
    color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
    color2 = [(0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0)]
    pts=pts.astype(int)
    dst=cv2.warpPerspective(src,A,(500,500))
    cv2.circle(dst, tuple(v1),5, c,-1)
    cv2.circle(dst, tuple(v2),5, c,-1)
    cv2.circle(dst, tuple(v3),5, c,-1)
    print(v1,v2,v3)
    for i, c in enumerate(color):
        print('{} -> {}'.format(pts[i][:2], res[i][:2]))
        cv2.circle(src, tuple(pts[i][:2]),5, c,-1)
        cv2.circle(dst, tuple(res[i][:2]),3, c,-1)

    cv2.imshow("src", src)
    cv2.imshow("dst",dst) 
    cv2.waitKey(0)
