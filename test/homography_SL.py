import getopt
import sys
import math
import random
import itertools
import cv2
import PIL
import numpy as np
from scipy.stats.distributions import chi2
from matplotlib import pyplot as plt

# https://engineering.purdue.edu/RVL/ECE661_2018/Homeworks/HW5/2BestSolutions/1.pdf


def cvtGRAY(img):
    img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # h, w = img.shape[:2]
    return gimg

def split_color(img, c):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    b, g, r = cv2.split(img)
    if c == "b":
        return b
    elif c == 'r':
        return r
    elif c == 'g':
        return g
    else:
        print("r, g, b 중에서 하나를 입력")

def homoCoord(inhomo):
    one = np.ones((1, inhomo.shape[0]))
    homo = np.concatenate((inhomo.T, one), axis=0)
    return homo

def inhomoCoord(homo):
    inhomo = homo[0:2, :].T
    return inhomo

def SIFT(img1, img2):
    # Feature Matching using SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # # correspondences
    src_pts = []
    dst_pts = []
    correspondenceList = []
    for i in range(len(good)):
        # print(int(kp1[good[i][0].queryIdx].pt[0]), int(kp1[good[i][0].queryIdx].pt[1]),
        # '->', int( kp2[good[i][0].trainIdx].pt[0]), int(kp2[good[i][0].trainIdx].pt[1]))
        (x1, y1) = kp1[good[i][0].queryIdx].pt
        (x2, y2) = kp2[good[i][0].trainIdx].pt
        src_pts.append([x1, y1])
        dst_pts.append([x2, y2])
        # correspondenceList.append([x1, y1, x2, y2])

    src = np.array(src_pts)
    dst = np.array(dst_pts)

    if src.shape != dst.shape:
        raise RuntimeError
    # corrs = np.array(correspondenceList)

    # MIN_MATCH_COUNT = 10
    # if len(good2) > MIN_MATCH_COUNT:
    #     src_pts2 = np.float32([kp1[m.queryIdx].pt for m in good2]).reshape(-1, 1, 2)
    #     dst_pts2 = np.float32([kp2[m.trainIdx].pt for m in good2]).reshape(-1, 1, 2)

    # Matching image
    matchImg = cv2.drawMatchesKnn(gray1, kp1, gray2, kp2, good, None, flags=2)
    # cv2.imshow("Feature Matching", matchImg)

    return src, dst

def diffImg(img1, img2):
    RImg = split_color(img1, "r")

    Bgray2 = split_color(img2, "b")
    Ggray2 = split_color(img2, "g")
    Rgray2 = split_color(img2, "r")

    diff = cv2.merge(([Bgray2, Ggray2, RImg]))
    # cv2.imshow("difference", diff)
    return diff

def test(src, dst):
    HH, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    resImg = cv2.warpPerspective(gray1, HH, (w, h))

    print("real H = ", HH)

    # diff = diffImg(resImg, gray2)
    #
    # cv2.imshow("Real Result", resImg)

    # RresImg = split_color(resImg, "r")
    #
    # Bgray2 = split_color(gray2, "b")
    # Ggray2 = split_color(gray2, "g")
    # Rgray2 = split_color(gray2, "r")
    # #
    # diff = cv2.merge(([Bgray2, Ggray2, RresImg]))
    diff = diffImg(resImg, gray2)
    # cv2.imshow("difference using Function", diff)

def arecolinear(points):
    x1 = points[0][0]
    y1 = points[0][1]
    x2 = points[1][0]
    y2 = points[1][1]
    x3 = points[2][0]
    y3 = points[2][1]

    # Calculation the area of triangle
    # 0.5 * [x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)]
    a = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    if (a == 0):
        return True
    else:
        return False

def RandomFour(corrs):
    n = len(corrs)
    idx = range(0, n)
    sampleList = random.sample(idx, 4)

    sampleCorrsList = []
    for _, idx in enumerate(sampleList):
        corr = corrs[idx]
        sampleCorrsList.append(corr)
    sampleCorrs = np.array(sampleCorrsList)
    return sampleCorrs

def generalFour(corrs):
    sampleCorrs = RandomFour(corrs)
    for points in itertools.combinations(sampleCorrs, 3):
        while arecolinear(points):
            print("collinear")
            sampleCorrs = RandomFour(corrs)
    return sampleCorrs

def matrixT(hpts):
    m = np.mean(hpts[:2], axis=1)
    stdX, stdY = np.std(hpts[:2], axis=1)

    T = np.diag([1/stdX, 1/stdY, 1])
    T[0][2] = -m[0] / stdX
    T[1][2] = -m[1] / stdY

    p = np.dot(T, hpts)
    p = inhomoCoord(p)
    return T, p

def matrixA(src, dst):
    np = len(src)
    A = []

    for i in range(0, np):
        x1, y1, w1 = src[i, 0], src[i, 1]
        x2, y2, w2 = dst[i, 0], dst[i, 1]

        A.append([0, 0, 0, -x1, -y1, -w1, y2 * x1, y2 * y1, y2*w1])
        A.append([x1, y1, w1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2*w2])
        # print("Stacked matrix A shape:", np.shape(A))
    return A

def computeSVD(A):
    A = np.array(A)
    U, S, VT = np.linalg.svd(A)
    return VT

def computeH(VT):
    h = VT[-1,:]
    h = h.reshape(3,3)
    return h

def denormH(T1, T2, H_tilde):
    H = np.dot(np.dot(np.linalg.pinv(T2), H_tilde), T1)
    H = H / H.item(8)
    return H

def normalize(src, dst):

    # normalize
    # fp, tp: inhomo
    T1, fp = matrixT(src)
    T2, tp = matrixT(dst)

    return T1, fp, T2, tp

def sampling(corrs):
    # 4 Sampling
    sample = generalFour(corrs)
    sfp = sample[:, 0:2]
    stp = sample[:, 2:4]
    return sfp, stp

def homography(T1, fp, T2, tp):
    A = matrixA(fp, tp)
    VT = computeSVD(A)
    H_tilde = computeH(VT)
    H = denormH(T1, T2, H_tilde)
    return H

def find_inliers(H, corr):
    x = np.transpose(np.array([corr[0], corr[1], 1]))
    p = np.transpose(np.array([corr[2], corr[3], 1]))

    Y = np.dot(H, x)

    error = Y - x
    sq_error = error ** 2
    d_sq = np.sum(sq_error, axis=0)
    print("d_sq", d_sq)
    ind = np.where(d_sq <= 10)
    return [corr[i] for i in ind[0]]

# symmetric transfer error
def error(corr, H):
    x = np.transpose(np.array([corr[0], corr[1], 1]))
    p = np.transpose(np.array([corr[2], corr[3], 1]))

    X = np.dot(np.linalg.pinv(H), p)
    P = np.dot(H, x)
    X = X / X[2]
    P = P / P[2]

    error1 = np.linalg.norm(x-X)
    error2 = np.linalg.norm(p-P)

    return error1 + error2


def sampleN(e = 0.5, p = 0.99, s = 4):
    w = 1 - e   # probability inliers
    # p = 1 - (1 - w**s)**N
    N = np.log(1 - p) / np.log(1 - (w**s) + 1e-7)
    # N = np.log(1 - p) / np.log(1 - pow(w, s))
    return int(math.ceil(N))

def consensus(e, n):
    T = (1 - e) * n
    return round(T)


def Ransac(src, dst):
    maxInliers = []
    finalH = None
    N = sampleN(e = 0.5, p = 0.99, s = 4) # 72

    corrs = np.hstack([src, dst])  # inhomo (x1 y1 x2 y2) stack

    for t in range(N):
        sfp, stp = sampling(corrs)
        T1, fp, T2, tp = normalize(sfp, stp)
        H = homography(T1, fp, T2, tp)

        #
        # inliers_tmp = find_inliers(H, corrs)
        # print("inliers")
        # print(len(inliers_tmp))

        inliers = []
        for i in range(len(corrs)):
            d = error(corrs[i], H)
            # print(d)

            if d < 5:
                inliers.append(corrs[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = H
        print("Corr size: ", len(corrs), "\tNumInliers: ", len(inliers), "\tMax inliers: ", len(maxInliers))

        #
        # w = len(inliers) / len(corrs)
        # e = 1 - w
        # n = len(corrs)
        # T = round((1 - e) * n)
        # N = sampleN(e, p, s = 4)

        # if len(maxInliers) > T:

        if len(maxInliers) > (len(corrs)*0.6):

            break
    return finalH, maxInliers


if __name__ == "__main__":
    img1 = cv2.imread('CAU1.jpg')
    img2 = cv2.imread('CAU2.jpg')

    # BGR to GRAY
    gray1 = cvtGRAY(img1)
    gray2 = cvtGRAY(img2)

    h, w = gray1.shape[:2]

    # Feature Matching using SIFT
    src, dst = SIFT(gray1, gray2)

    # using cv2.findHomography
    # test(src, dst)

    # RANSAC + Homography
    H, inliers = Ransac(src, dst)
    print(H)

    resImg = cv2.warpPerspective(gray1, H, (gray1.shape[1], gray1.shape[0]))
    # cv2.imshow("result image", resImg)

    # cv2.imshow("image2", gray2)
    diff = diffImg(resImg, gray2)
    cv2.imshow("diff image", diff)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
