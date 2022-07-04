"""
Desc    :   Pressing right mouse button in top left segment (myocardium) and moving it up / down will navigate show p-s image
for increasing / decreasing r value (the r-p-s image is a 3D image).
Pressing the left mouse button moves SAX plane. It selects a slice location that is used to show time series images for
that slice position in bottom left (SAX) segment.
Clicking of mouse in the bottom left segment shows time series images (for slice position as selected by SAX line in the
top left segment) as a movie in bottom left segment.
Clicking the mouse in the bottom right segment shows the LAX time series images as a movie in bottom right segment.

V3.0    :   1. Prints fibrosis percentage value for all 16 segments
            2. Lines dividing myocardium into three sections also displayed

Input:      Reads extracted myocardium masks and corrresponding contrast images, from "MyocardiumExtracted" folder
            under the patient, to display the Bulls Eye View
Returns:    None
Bug     :   Intensity value of myocardium in RPS view to be checked and normalized
"""

import cv2
from pathlib import Path
import os
import pydicom
import numpy as np
import matplotlib
from pydicom import Dataset
from io import StringIO
import PIL
matplotlib.use('Qt5Agg')
#print (matplotlib.matplotlib_fname())
from moviepy.video.io.bindings import mplfig_to_npimage #new module
import io
from pydicom.pixel_data_handlers.numpy_handler import pack_bits
import matplotlib.pyplot as plt

plt.set_cmap('gray')
import ExtractMyocardium as EM

SIZE, NUM_SLICE = 256, 9
SECTIONS, SEGMENTS = [2, 5, NUM_SLICE-2, NUM_SLICE], [6, 6, 4, 1]
if len(SECTIONS) != len(SEGMENTS): exit(-1)

RPS_IDX, SAX_IDX = 0, 0
im_canvas = []
SAXPlane = 100

def showSeqPlt(seg, imList, delay=0.05):
    flag = True
    #plt.figure(1)

    while (flag):
        for im in imList:
            plt.subplot(224),plt.imshow(im), plt.xticks([]),plt.yticks([])
            plt.show(block=False)
            print("loop bottom right starts")
            plt.savefig('foo2.png')
            if (plt.waitforbuttonpress(timeout=0.05)):
                print("loop bottom right ends")
                flag = False
                break
            else: flag = True
#----------------------------------------------------------------------------------
def showSeqPlts(seg, imList, solay,solayr):
    flag = True
   # plt.figure(1)

    while (flag):
        for im in imList:

            plt.subplot(223),plt.imshow(im,alpha=0.7), plt.xticks([]),plt.yticks([])
            plt.subplot(223),plt.imshow(solay,alpha=0.25),plt.xticks([]),plt.yticks([])
            plt.subplot(223),plt.imshow(solayr,alpha=0.25),plt.xticks([]),plt.yticks([])

            plt.show(block=False)
            print("loop bottom left starts")
            plt.savefig('foo.png')
          #  print(plt.waitforbuttonpress())
            if (plt.waitforbuttonpress(timeout=0.05)):
                print("loop bottom left ends")
                flag = False
                break
            else: flag = True
#----------------------------------------------------------------------------------
def readImage(file):
    if file!='/Users/varunkansal/Desktop/Project/Dynamic Fibrosis Visualization/MB18043554/CINE_Segmented_SAX_b8/.DS_Store':
        if file[-4:] != '.png':
            global olay,olayr
            dimg = pydicom.dcmread(file,force=True)   # read the dicom image
            sLoc = dimg.data_element('SliceLocation').value    #get_item('SliceLocation') returns full string
            imgf = dimg.pixel_array     # extract the pixel data from the dicom image file

            olay = dimg.overlay_array(0x6000)
            olayr = dimg.overlay_array(0x6002)

            #change-------- https://pydicom.github.io/pydicom/dev/reference/generated/pydicom.dataelem.DataElement.html#pydicom.dataelem.DataElement
            #packed_bytes=pack_bits(olay.flatten())


            #elem=pydicom.dataelem.DataElement(0x60000000 + 0x3000, 'OW', packed_bytes)
            #dimg.add(elem)


            #packed_bytes2 = pack_bits(olayr.flatten())

            #dimg.add(pydicom.dataelem.DataElement(0x60020000 + 0x3000, 'OW', packed_bytes2))

            #img2=dimg.pixel_array
            #--------

            #plt.imshow(dimg.pixel_array, alpha=0.8, cmap=plt.cm.gray)
            #plt.imshow(olay, alpha=0.4, cmap=plt.cm.gray)
            #plt.imshow(olayr, alpha=0.4, cmap=plt.cm.gray)

            olay = np.uint8(olay * (255 / np.max(olay)))
            olayr = np.uint8(olayr * (255 / np.max(olayr)))

            img = np.uint8(imgf * (255 / np.max(imgf)))  # to create ndarray of bytes needed for displaying image
            return (imgf, sLoc, olay, olayr)
        else:
            olay=[]
            olayr=[]
            img = cv2.imread(file, 0)
            img = cv2.resize(img, (256, 256))
            return (img, 0, olay, olayr)

#--------------------------------------------------------------------------------
def readImageL(file):
    if file!='/Users/varunkansal/Desktop/Project/Dynamic Fibrosis Visualization/MB18043554/CINE_Segmented_SAX_b8/.DS_Store':
        if file[-4:] != '.png':
            global olay,olayr
            dimg = pydicom.dcmread(file,force=True)   # read the dicom image
            sLoc = dimg.data_element('SliceLocation').value    #get_item('SliceLocation') returns full string
            imgf = dimg.pixel_array     # extract the pixel data from the dicom image file






            img = np.uint8(imgf * (255 / np.max(imgf)))  # to create ndarray of bytes needed for displaying image
            return (img, sLoc)
        else:

            img = cv2.imread(file, 0)
            img = cv2.resize(img, (256, 256))
            return (img, 0)
#--------------------------------------------------------------------------------
def showSlicewiseFibroses(img):
    s, p, r = img.shape
    for i in range(s):
        im = cv2.rotate(np.array(img[i], np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE)
        #im = cv2.resize(im, (p, size))
        plt.imshow(im), plt.show(block=True)

#--------------------------------------------------------------------------------
import pickle as pkl
def readPerc(dir):
    file = r'D:\DSI\Dataset\Myocardium'+'\\'+'percent.pkl'
    with open(file, 'rb') as f:
        perc = pkl.load(f)
    p2 = np.array(perc, float)
    sz = len(perc)
    for i in range(sz):
        for j in range(len(perc[i])):
            if np.isnan(perc[i,j]): perc[i,j] = 0
            p2[sz - i - 1, j] = perc[i, j]
    return p2
#--------------------------------------------------------------------------------
def createOverlay(size, numSlices, rlist, fibPerc=None, segAngles=None):
    global SECTIONS, SEGMENTS
    h, w = size[0], size[1]
    rmax = np.max(rlist)
    ht, lIdx = h // numSlices, 0
    segBdrPts = [[[] for j in range(SEGMENTS[i]+1)] for i in range(len(SECTIONS))]
    secBdrPts, textPts = [], []

    for sl in range(numSlices):
        scale = rlist[sl] / rmax
        xSft = int(w*(1-scale)/2)
        segWidth = int((w // SEGMENTS[lIdx]) * scale)
        for seg in range(SEGMENTS[lIdx]+1):
            if segAngles[lIdx][seg] > 360: angle = segAngles[lIdx][seg]%360
            else: angle = segAngles[lIdx][seg]
            segBdr = xSft + (angle) / 360 * w * scale
            #segAngles contains segment border as angle in degree (it may be more than 360)
            x = min(int(segBdr), w-1)
            segBdrPts[lIdx][seg].append((x, sl * ht))
            segBdrPts[lIdx][seg].append((x, (sl+1) * ht))

            if seg < SEGMENTS[lIdx] and sl == SECTIONS[lIdx]-1:
                textPts.append((x + segWidth // 2 - 10, sl * ht + 15, '%3.1f'%(fibPerc[lIdx][seg])))
        if sl == SECTIONS[lIdx]:
            secBdrPts.append(((0, (sl+1) * ht), (w, (sl+1) * ht)))
            lIdx += 1
    return (secBdrPts, segBdrPts, textPts)
#--------------------------------------------------------------------------------
def composeRPSImgs(RPSdir, size):
    global NUM_SLICE

    img, rlist = EM.getRPS(RPSdir)
    s,r,p = img.shape
    avgImg = np.zeros((size, p), int)     # to store Average image
    MIP = np.zeros((size, p), np.uint8)      # to store MIP image
    imList = []
    for i in range(r):
        im = cv2.resize(img[:,i,:], (p, size))
        imList.append(im)
        avgImg += im
        MIP[im > MIP] = im[im > MIP]    # get the pixels that is greater of the two
    avgImg = np.array(avgImg // r, np.uint8)
    return ([avgImg]+[MIP]+imList, rlist)

#-------------------------------------------------------------------------------
def getMouseSegment(event):
    bot_left = (430, 90)
    spacing = (490, 60)
    winSize = (310, 335)

    x, y = [event.x - bot_left[0], event.y -  bot_left[1]]

    if (x > 0 and x < winSize[0] and y > 0 and y < winSize[1]):  rval =  (3, 0)   # If  button has been pressed in the bottom-left segment
    elif (x > winSize[0]+spacing[0] and x < spacing[0]+ 2*winSize[0]) and (y > 0 and y < winSize[1]): rval =  (4, 0)
    elif (y > winSize[1]+spacing[1] and y < spacing[1]+ 2*winSize[1]) and (x > 0 and x < winSize[0]):
        RPSHeight = int(SIZE *((2*winSize[1]+spacing[1])-y) / winSize[1])    # offset from top-left in number of pixels
        rval = (1, RPSHeight)   # If  button has been pressed in the top-left segment
    elif ((x > winSize[0]+spacing[0] and x < spacing[0]+ 2*winSize[0]) and
        (y > winSize[1] + spacing[1] and y < spacing[1] + 2 * winSize[1])):  rval =  (2, 0)
    else: rval =  (-1, 0)   # button has been pressed outside of any window
    print('Location is (x=%d, y=%d), segment is %d and offset (for top-left) is %d'%(x, y, rval[0], rval[1]))
    return rval

#------------------------------------------------------------------------
def plotsegBoundariesAndText(im, overlay):
    secBdrPts, segBdrPts, textPts = overlay[0], overlay[1], overlay[2]
    for sec in range(len(secBdrPts)):
        stPt, endPt = secBdrPts[sec][0], secBdrPts[sec][1]
        x, y = [stPt[0], endPt[0]], [stPt[1], endPt[1]]
        plt.plot(x, y, 'g')

    for sl in range(len(segBdrPts)):
        for seg in range(len(segBdrPts[sl])):
            x = [segBdrPts[sl][seg][i][0] for i in range(len(segBdrPts[sl][seg]))]
            y = [segBdrPts[sl][seg][i][1] for i in range(len(segBdrPts[sl][seg]))]
            plt.plot(x, y, 'r')

    for tpts in textPts:
        plt.text(tpts[0], tpts[1], tpts[2], fontsize=15, color="red")
#------------------------------------------------------------------------
from PIL import Image
def refreshIm(CLEAR = 0):
    global im_canvas, NUM_SLICE, RPS_IDX, SAX_IDX, SAXPlane
    fig, cid, im, overlay = im_canvas[0], im_canvas[1], im_canvas[2], im_canvas[3]
    h, w = im[0][0].shape[0:2]  #im[0] is a list of images consisting of average, MIP and (r,p,s) volume image
    SAXPlane = min(h, max(0, SAXPlane))
    SAX_IDX = NUM_SLICE - (SAXPlane * NUM_SLICE) // h - 1
    print('SAXPlane and SAX_IDX are: ', SAXPlane, SAX_IDX)
    # since slice positions are measured from apex to basal, whereas, SAXPlane is measured from origin of image (top-left)
    if CLEAR == 1: plt.clf()   # Clear earlier images in the canvas
    maxVal = np.max(im[0])
    plt.ion()   # chanege

    plt.subplot(221), plt.imshow(im[0][RPS_IDX], vmax=min(maxVal, 255)),plt.xticks([]),plt.yticks([])
    plotsegBoundariesAndText(im[0][RPS_IDX], overlay)
    plt.plot([SAXPlane] * w)
    plt.subplot(222), plt.imshow(im[1]),plt.xticks([]),plt.yticks([]),plt.xticks([]),plt.yticks([])
    #plt.subplot(223), plt.imshow(im[2][SAX_IDX][0])

    #CHANGE ------------
    #olay = dimg.overlay_array(0x6000)
    #olayr = dimg.overlay_array(0x6002)

    print("SAX_ID =",SOLAYR[SAX_IDX])
    #plt.subplot(223), plt.plot(SOLAYR[SAX_IDX])
    plt.subplot(223), plt.imshow(im[2][SAX_IDX][0]), plt.xticks([]), plt.yticks([])
    #plt.subplot(223), plt.imshow(SOLAYR[SAX_IDX], alpha=0.5,cmap='Reds'),plt.xticks([]),plt.yticks([])
    #plt.subplot(223), plt.imshow(SOLAY[SAX_IDX], alpha=0.5,cmap='Blues'),plt.xticks([]),plt.yticks([])

    #-----------------------------


    plt.subplot(224), plt.imshow(im[3][0][0]),plt.xticks([]),plt.yticks([])

    plt.show(block=False)
    #plt.waitforbuttonpress(1)
    plt.ioff()     #change

#------------------------------------------------------------------------
lastPos = 0
def onDraw(event):
    increment = lambda x, y: x+1 if x < y-1 else y-1
    decrement = lambda x: x-1 if x > 0 else 0
    global SAXPlane, im_canvas, RPS_IDX, NUM_SLICE, lastPos, SAX_IDX

    if event.button == None: return    # nothing to be done if only mouse has been moved without a button press
    print(event)
    seg, pos = getMouseSegment(event)
    #seg=4
    if seg == -1:  return  # button has been pressed outside of any window
    if seg == 3: showSeqPlts(223, im_canvas[2][2][SAX_IDX],SOLAYR[SAX_IDX],SOLAY[SAX_IDX])   #   Mouse button has been pressed in bottom left segment
    elif seg == 4: showSeqPlt(224, im_canvas[2][3][0])    # Mouse button has been pressed in bottom right segment
    elif seg == 2: print ('In segment 2 (volume image)')
    else:   # If left / right button has been pressed and moved over top-left segment (seg 3)
        print('position, lastPos and event is: ', pos, lastPos, event.x, event.y)
        if abs(pos - lastPos) < 5: return    # Update image and move line only when there is a significant mouse motion
        if event.button == 3:  # if right mouse button is pressed
            print("hello");
            if pos > lastPos:  RPS_IDX = increment(RPS_IDX, NUM_SLICE)
            else:  RPS_IDX = decrement(RPS_IDX)
            lastPos = pos
        else:  SAXPlane = pos
        refreshIm(1)    # refresh image in all segments and line in top left segment

    return

#------------------------------------------------------------------------
def LoadSeqImg(dir):
    cdir = Path(dir)
    imList = []
    olayList=[]
    olayrList=[]
    for file in os.listdir(dir):
        fname = os.path.join(cdir, file)
        im, sLoc,olay,olayr = readImage(fname)
        olay=cv2.resize(olay,(SIZE,SIZE))
        olay = cv2.cvtColor(olay, cv2.COLOR_GRAY2RGB)
        cv2.putText(olay, file, (20, 20), cv2.QT_FONT_NORMAL, 0.7, (0, 255, 0))

        olayr = cv2.resize(olayr, (SIZE, SIZE))
        olayr = cv2.cvtColor(olayr, cv2.COLOR_GRAY2RGB)
        cv2.putText(olayr, file, (20, 20), cv2.QT_FONT_NORMAL, 0.7, (0, 255, 0))

        #im = cv2.resize(im, (SIZE, SIZE))
        #im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        cv2.putText(im, file, (20,20),cv2.QT_FONT_NORMAL, 0.7, (0, 255, 0))

        imList.append(im)
        olayList.append(olay)
        olayrList.append(olayr)

    return imList, sLoc ,olayList, olayrList

# ----------------------------------------------------------------------------------
def LoadSeqImgL(dir):
    cdir = Path(dir)
    imList = []
    olayList=[]
    olayrList=[]
    for file in os.listdir(dir):
        fname = os.path.join(cdir, file)
        im, sLoc = readImageL(fname)


        im = cv2.resize(im, (SIZE, SIZE))
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        cv2.putText(im, file, (20,20),cv2.QT_FONT_NORMAL, 0.7, (0, 255, 0))

        imList.append(im)


    return imList, sLoc
# ----------------------------------------------------------------------------------
def slicePosition(sList, sLoc):
    if len(sList) == 0:
        sList.append(sLoc)
        return 0

    idx = 0
    while idx < len(sList) and sList[idx] < sLoc: idx += 1
    sList.insert(idx, sLoc)
    return idx
# ----------------------------------------------------------------------------------
""" It loads time series images for each slice position. It assumes series name to contain 'name' as part of it
    and the folder to contain time series images at that slice position. Different folders with 'name' prefix 
    contain time series images (stored as a list) for different slice position (this is also true for LAX time series images).
    The time series images are sorted based on slice position and returned as list of list of images."""
def LoadSeqSeries(dir, name):
    sList, imgSeries = [], []
    olaySeries,olayrSeries=[],[]
    cdir = Path(dir)
    if not cdir.exists():
        print('Dicom dir does not exist')
        return (-1)

    for file in os.listdir(dir):
        if file.find(name) == -1: continue
        fname = os.path.join(dir, file)
        imgList, sLoc, olayList,olayrList = LoadSeqImg(fname)
        idx = slicePosition(sList, int(sLoc))
        imgSeries.insert(idx, imgList)  # The series images for a given slice location are sorted in ascending order
        olaySeries.insert(idx,olay)
        olayrSeries.insert(idx,olayr)

    return imgSeries,olaySeries,olayrSeries
#----------------------------------------------------------------------------------
def LoadSeqSeriesL(dir, name):
    sList, imgSeries = [], []
    olaySeries,olayrSeries=[],[]
    cdir = Path(dir)
    if not cdir.exists():
        print('Dicom dir does not exist')
        return (-1)

    for file in os.listdir(dir):
        if file.find(name) == -1: continue
        fname = os.path.join(dir, file)
        imgList, sLoc= LoadSeqImgL(fname)
        idx = slicePosition(sList, int(sLoc))
        imgSeries.insert(idx, imgList)  # The series images for a given slice location are sorted in ascending order


    return imgSeries
# ----------------------------------------------------------------------------------

def createLayout(imgs ,overlay):
    global im_canvas, SAXPlane
    fig = plt.figure(1,figsize=(1920/200, 855/200))
    figManager = plt.get_current_fig_manager()
    #figManager.window.showMaximized()

    cid = fig.canvas.mpl_connect('button_press_event', onDraw) # Mouse callback fn for motion is associated with this figure
    cid = fig.canvas.mpl_connect('motion_notify_event', onDraw)

    im_canvas = [fig ,cid , imgs, overlay]
    refreshIm()
    plt.show()
    #plt.close()
# ----------------------------------------------------------------------------------
SAX_NAME = 'CINE_Segmented_SAX_'
LAX_NAME = r'FOUR_CH_LAX_1'
root = r'/Users/varunkansal/Desktop/Project/Dynamic Fibrosis Visualization'
VolImFile = root + '//3D_VolumeView-Image.png'

# ----------------------------------------------------------------------------------
def create3DDynView(root, pat, fibPercent = 0, segAngles = 0):
    global SAX_NAME, LAX_NAME, SIZE, VolImFile,SOLAY,SOLAYR,LOLAY,LOLAYR
    fPath = root + '//' + pat
    imList, rlist = composeRPSImgs(fPath, SIZE)
    #if fibPercent == 0: fibPercent = readPerc(fPath)
    size = imList[0].shape
    overlay = createOverlay((size[0], size[1]), NUM_SLICE, rlist, fibPercent, segAngles)
    # since first two images are average and MIP images, so number of slices are equal to NUM_SLICE-2

    VolIm, sLoc, olay, olayr = readImage(VolImFile)
    SAXImgs,SOLAY,SOLAYR = LoadSeqSeries(fPath, SAX_NAME)
    LAXImgs = LoadSeqSeriesL(fPath, LAX_NAME)
    imgs = [imList, VolIm, SAXImgs, LAXImgs]
    createLayout(imgs, overlay)

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

patList = ['MB18043554']
fibPerc = [[1,2,3,4,5,6],
           [2,2,3,3,4,4],
           [4,4,4,4,0,0],
           [6,6,6,6,6,6]]
segAngles = [[0,60,120,180,240,300,360,420],
             [0,60,120,180,240,300,360,420],
             [45,135,225,315,405],
             [0,360]]

create3DDynView(root, patList[0], fibPerc, segAngles)