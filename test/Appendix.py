import numpy as np
import cv2
import sys

def drawLine(l, image, color):
    a = l[0]
    b = l[1]
    c = l[2]
    H = image.shape[0]
    W = image.shape[1]
    T = (-c/a, 0)
    L = (0, -c/b)
    R = (W, -(a*W+c)/b)
    B = (-(b*H+c)/a, H)

    bT = T[0]>=0 and T[0]<W
    bL = L[1]>=0 and L[1]<H
    bR = R[1]>=0 and R[1]<H
    bB = B[0]>=0 and B[0]<W
    if bT and bL:
        pt1 = T
        pt2 = L
    elif bT and bB:
        pt1 = T
        pt2 = B
    elif bT and bR:
        pt1 = T
        pt2 = R
    elif bL and bB:
        pt1 = L
        pt2 = B
    elif bL and bR:
        pt1 = L
        pt2 = R
    elif bB and bR:
        pt1 = bB
        pt2 = bR
    else:
        print('error!!')
        print(T)
        print(L)
        print(R)
        print(B)
        sys.exit()

    pt1 = (int(round(pt1[0])), int(round(pt1[1])))
    pt2 = (int(round(pt2[0])), int(round(pt2[1])))
    cv2.line(image, pt1, pt2, color, 2)

        

if __name__=='__main__':
    th = np.pi/6

    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])

    D,Q = np.linalg.eig(R)

    #Q = np.random.randint(0,500,(3,3))
    Q  = np.array([[250, 100, 1], [45, 375, 1], [375, 400,1]])
    Q = Q.T
    Q = Q.astype(float)
    iQ = np.linalg.inv(Q)
    D = np.diag([5.0,1.0,1.0])

    A = Q@D@iQ

    D, Q_ = np.linalg.eig(A)
    A /=D[2]

    #print(Q)

    #print(Q_/Q_[2,2])
    #print(Q/Q[2,2])
    iQ_ = np.linalg.inv(Q_)
    e1 = Q[:,0].astype(int)
    e2 = Q[:,1].astype(int)
    e3 = Q[:,2].astype(int)
    _e1 = A@e1
    _e2 = A@e2
    _e3 = A@e3
    _e1 /= _e1[2]
    _e2 /= _e2[2]
    _e3 /= _e3[2]
    print('Ae1={}, {} <- vertex'.format(e1,_e1))
    print('Ae2={}, {} <- a fixed point construct axis'.format(e2,_e2))
    print('Ae3={}, {} <- the other fixed point construct axis'.format(e3,_e3))
    print('eigenvalues={}'.format(D))

    pts = []

    src = np.zeros((500,500,3), dtype=np.uint8)
    dst = np.zeros((500,500,3), dtype=np.uint8)

    pts.append([125,125,1])
    pts.append([125,375,1])
    pts.append([375,125,1])
    pts.append([375,375,1])

    pts = np.array(pts)
    res = []

    axis = np.cross(e2,e3)
    vertex = e1.astype(int)




    for i in range(len(pts)):
        tmp = A@pts[i]
        tmp /= tmp[2]
        tmp=tmp.astype(int)
        res.append(tmp)
    l = []
    #TODO 모든 점들이 직선 만들도록 만들기, 직선 그리는 함수 만들기.
    for i in range(len(pts)):
        tmp = np.cross(pts[i],res[i])
        tmp = tmp.astype(int)
        l.append(tmp)

    color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
    color2 = [(0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0)]

    # Draw Axis
    e2 = e2.astype(int)
    e3 = e3.astype(int)
    drawLine(axis,src, (0,128,128))
    cv2.circle(src, tuple(vertex[:2]),5,(0,255,255),4)
    cv2.circle(src, tuple(e2[:2]),5,(0,255,255),4)
    cv2.circle(src, tuple(e3[:2]),5,(0,255,255),4)


    for i, c in enumerate(color2):
        drawLine(l[i],src, c)

    for i, c in enumerate(color):
        print('{} -> {}'.format(pts[i][:2], res[i][:2]))
        cv2.circle(src, tuple(pts[i][:2]),5, c,-1)
        cv2.circle(src, tuple(res[i][:2]),5, c,-1)

    # pts1, pts2
    lpts1 = np.cross(pts[0], pts[1]) # line joining between x1 and x2
    lpts2 = np.cross(res[0], res[1])# line joining between x1' and x2'

    x_12 = np.cross(lpts1, lpts2)
    x_12 =x_12/x_12[2]
    x_12 = x_12[:2].astype(int)

    print(x_12)
    cv2.circle(src, tuple(x_12),5,(255,255,255),1)


    cv2.imshow("src", src)
    cv2.waitKey(0)
