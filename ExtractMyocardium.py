import os, math
import matplotlib.pyplot as plt
plt.set_cmap('gray')
import sys, cv2
import numpy as np
sys.path.append((R'/Users/varunkansal/Desktop/Project'))
import SupportFunctions as SF

LGEStr, maskStr = 'LGEImages', 'MyocardiumMask'

def findCentre(mim, im):
    #reg = mim[90:140,50:100]
    mim[mim>100] = 255
    mim[mim<=100] = 0
    ker = np.ones((6, 6), np.uint8)
    mim = cv2.morphologyEx(mim, cv2.MORPH_CLOSE, ker)

    conList, hierarchy = cv2.findContours(mim, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #_,
    m = cv2.moments(conList[0])
    cx, cy = int(m['m10'] // m['m00']), int(m['m01'] // m['m00'])
    #cx, cy = int(m['m10'] // m['m00'])+50, int(m['m01'] // m['m00'])+90
    rad = int(math.sqrt(cv2.contourArea(conList[0]) / math.pi))
    return (rad, (cx, cy))

#---------------------------------------------------------------------------------------------
""" It resamples the 3D myocardium by opening up the myocardium along long axis and create a 3D image in r-p-s dimension.
It does it by converting the donut shaped hollow myocaridum along the perimeter for r = rmin to rmax, where rmin and rmax
are the minimum and maximum distance of any point on endocardium and pericardium respectively. Thus, the image is converted 
from cartesian coordinates to polar coordinates from centre of myocardium. """

def getMyo(dirLoc):
    global LGEStr, maskStr

    imList = []
    radList = []
    flist = [file for file in os.listdir(dirLoc) if file.startswith(LGEStr)]
    mlist = [file for file in os.listdir(dirLoc) if file.startswith(maskStr)]
    for i in range(len(flist)): # from largest LV (basal) to smallest (apex)
        im = SF.readImage(dirLoc+'//'+flist[i])
        mim = SF.readImage(dirLoc+'//'+mlist[i])
        #SF.showImage(im, flist[i], 121), SF.showImage(mim, None, 122, show=True)
        rad, cen = findCentre(mim, im)
        strip, mean = SF.getStrip(im, (cen[0], cen[1]), max(0,rad-5), rad+10)
        strip = cv2.rotate(strip, cv2.ROTATE_90_COUNTERCLOCKWISE)   # so that y axis represents r and x axis p
        imList.append(strip)
        radList.append(rad), radList
    return imList, radList
#---------------------------------------------------------------------------------------------
def adjustRList(rlist):
    rmin, rmax = np.min(rlist), np.max(rlist)
    i = 0
    while i < len(rlist) and rlist[i + 1] > rlist[i]:
        rlist[i] = rlist[i + 1]
        i += 1

    while i < len(rlist):
        if rlist[i] > rlist[max(i - 1, 0)]: rlist[i] = rlist[max(0, i - 1)]
        i += 1
    return

#---------------------------------------------------------------------------------------------
def resize(im, rlist):  # It scales the image in vertical (s) direction so that it looks normal, otherwise
    # it looks very flattened as there are only 9 slices compared to image size

    adjustRList(rlist)
    s, r, p = im.shape
    rad = [24,24,31,31,31,30,26,20,16]
    rlist = rad
    rmin, rmax = np.min(rlist), np.max(rlist)
    for i in range(s):
        xSft = int(180*(1-rlist[i]/rmax))
        if (rlist[i] < 2):
            print('radius too small')
        xsize = int(p*rlist[i]/rmax)
        rim = cv2.resize(im[i],(xsize, r))   # rescale the x dimension (along perimeter p) so that size
        #of the myocarium (in r-p-s) is in proportion to the radius.
        ysize, xsize = rim.shape
        im[i] = rmax//2 # the area outside the rescaled myocardium is set to a gray value so that myocardium is easily visible.
        im[i,:,xSft:xSft+xsize] = rim[:,:]     # so that images for different slice position (i value) have the same size and are centred
    return(im, rlist)
#---------------------------------------------------------------------------------------------
def fitToMaxSize(rpsIm):
    sh = (0, 0)
    for im in rpsIm:
        if im.shape > sh:
            sh = im.shape
    sh = [len(rpsIm), sh[0], sh[1]]
    im3D = np.zeros(sh, np.uint8)
    for i in range(len(im3D)):
        sh = rpsIm[i].shape
        im3D[i][:sh[0], :sh[1]] = rpsIm[i]

    return im3D
#---------------------------------------------------------------------------------------------

def getRPS(dir):
    imList, rList = getMyo(dir) # a list of 2D images containing myocaridum

    rpsIm = fitToMaxSize(imList)  # creating 3D image of myocardium as a numpy array
    rpsIm, rList = resize(rpsIm, rList)
    return (rpsIm, rList)

