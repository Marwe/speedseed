#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import cv2.cv as cv # for missing constants
import sys
import numpy
import argparse
import os
import csv
import itertools
import time

#display=False
#display=True
#keypress=False
#keypress=True
#keytimeout=500
#interactive=True

#TODO : -d and -g: set to store_true

parser = argparse.ArgumentParser(description='Process R/IR/DIFF images and do row + plant in the row detection',epilog='writes out the features to stdout and a csv')
parser.add_argument('-d', '--display', action="store_true", dest='display',
                   help='display the images in a viewer')
parser.add_argument('-g', '--gui', action="store_false", dest='gui',
                   help='open interactive gui')
parser.add_argument('-k', '--keypress', action="store_false", dest='keypress',
                   help='wait for key')
parser.add_argument('-K', '--keytimeout', type=int, default=500, dest='keytimeout',
                   help='timeout for key in ms')
parser.add_argument('-C', '--channel', type=int, default=0, dest='channel', choices=range(0,3),
                   help='channel, 0=blue, 1=green')

#parser.add_argument('-m', '--minh',default=20, type=int, choices=range(0,256), dest='minh',
                   #help='min H threshold')
#parser.add_argument('-M', '--maxh',default=170, type=int, choices=range(0,180), dest='maxh',
                   #help='max H threshold')
#parser.add_argument('-s', '--mins',default=35, type=int, choices=range(0,256), dest='mins',
                   #help='min S threshold')
#parser.add_argument('-S', '--maxs',default=255, type=int, choices=range(0,256), dest='maxs',
                   #help='max S threshold')
#parser.add_argument('-v', '--minv',default=50, type=int, choices=range(0,256), dest='minv',
                   #help='min V threshold')
#parser.add_argument('-V', '--maxv',default=180, type=int, choices=range(0,256), dest='maxv',
                   #help='max V threshold')

parser.add_argument('-b', '--minb',default=20, type=int, choices=range(0,255), dest='minb',
                   help='min blue threshold')
parser.add_argument('-B', '--maxb',default=255, type=int, choices=range(0,255), dest='maxb',
                   help='max blue threshold')
parser.add_argument('-x', '--xul',default=0, type=int, dest='xul',
                   help='x of UL corner of ROI rectangle')
parser.add_argument('-y', '--yul',default=0, type=int, dest='yul',
                   help='y of UL corner of ROI rectangle')
parser.add_argument('-X', '--Xsize',default=10000, type=int, dest='X',
                   help='x size of ROI rectangle')
parser.add_argument('-Y', '--Ysize',default=10000, type=int, dest='Y',
                   help='y size of ROI rectangle')
parser.add_argument('-c', '--colhistthreshperc',default=25, type=int, choices=range(0,100), dest='colhistthreshperc',
                   help='threshold for column histogram in percent')
parser.add_argument('-r', '--rowhistthreshperc',default=25, type=int, choices=range(0,100), dest='rowhistthreshperc',
                   help='threshold for row histogram in percent')
parser.add_argument('-t', '--thresholdblue', action="store_true", dest='thresholdblue', 
                   help='threshold image first with --minb --maxb thresholds  (default: directly use gray levels) ')
parser.add_argument('-D', '--distancewrap', type=int, default=500, dest='distancewrap', 
                   help='distance [pixel] to wrap the histogram (add up object pixels with this distance)')
parser.add_argument('-s', '--saveresults', action="store_true", dest='saveresults', 
                   help='save results (images)')


#parser.add_argument('-a', '--minarea',default=0, type=int, dest='minarea',
                   #help='min area threshold')
#parser.add_argument('-A', '--maxarea',default=-1, type=int, dest='maxarea',
                   #help='max area threshold')
#parser.add_argument('-o', '--opening',default=0, type=int, choices=range(-31,32,2)+[0], dest='opening',
                   #help='morphological opening, size of element. negative values: closing')
#parser.add_argument('-p', '--prefix', default='', type=str, dest='prefix',
                   #help='prefix for the filenames of the results')
#parser.add_argument('-w', '--writedb', default='', type=str, dest='writedb',
                   #help='csv DB name to store the data')
parser.add_argument(metavar='I', type=str, nargs='*', default=["/tmp/sample.png"], dest='imagefilenames',
                   help='RGB images to be processed')
args = parser.parse_args()
print args

def checkminmax(value,minval=0,maxval=255,name="value"):
    res=value
    if(value<minval):
        res=minval
        print "warning: invalid "+name+": "+str(value)+",setting to "+str(res)
    if(value>maxval):
        res=maxval
        print "warning: invalid "+name+": "+str(value)+",setting to "+str(res)
    return res

#print "prefix: ("+args.prefix+")"
#display=args.display
#keypress=args.keypress
#keytimeout=args.keytimeout
#numimages=checkminmax(args.numimages,1,len(args.imagefilenames),"numimages range (max=nr of images)")
#areamm=[checkminmax(args.minarea,0,65000*65000,"minarea"),checkminmax(args.maxarea,-1,65000*65000,"maxarea")]
## print args.imagefilenames, len(args.imagefilenames)
#print "areaminmax: "+str(areamm)
#optfilename=time.strftime("shaperecolass_%Y-%m-%d-%H-%M-%S",time.gmtime())+".log"
#with open(optfilename, 'wb') as csvfile:
        #print "optfile: "+optfilename
        #csvwriter = csv.writer(csvfile, delimiter=" ", quotechar='', quoting=csv.QUOTE_NONE)
        #csvwriter.writerow(sys.argv)

## morphology requested?
##print "morphology: ",args.opening
#morphology=False
#op=cv2.MORPH_OPEN
#if args.opening !=0:
    #morphology=True
    #se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(args.opening),abs(args.opening)))    
    #if args.opening<0:
        #op=cv2.MORPH_CLOSE


#VSH?
##HSVt=[ 35, 255, 50, 255, 20, 170]

##class lktracker(object):
##    graystack=None
##    lk_params = dict( winSize  = (10, 10),
##                  maxLevel = 3,
##                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
##    feature_params = dict( maxCorners = 2500,
##                       qualityLevel = 0.005,
##                       minDistance = 7,
##                       blockSize = 7 )
##    # from lk_track.py example App() class
##    self.track_len = 10
##    self.detect_interval = 5
##    self.tracks = []
##    #self.cam = video.create_capture(video_src)
##    self.frame_idx = 0
##    def __init__(self,graystack,**kwargs):
##        self.graystack=graystack
##    def trackstack(self):
##        if self.graystack.shape[3] < 2:
##            print "stack not of size > 1"
##            return None
##        self.prev_gray=self.graystack[0]
##        for (i in range(1,self.graystack.shape[3])):
##            frame_gray=self.graystack[i]
##            if len(self.tracks) > 0:
##                img0, img1 = self.prev_gray, frame_gray
##                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
##                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
##                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
##                d = abs(p0-p0r).reshape(-1, 2).max(-1)
##                good = d < 1
##                new_tracks = []
##                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
##                    if not good_flag:
##                        continue
##                    tr.append((x, y))
##                    if len(tr) > self.track_len:
##                        del tr[0]
##                    new_tracks.append(tr)
##                    #cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
##                self.tracks = new_tracks
##                #cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
##                #draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))
##
##            if self.frame_idx % self.detect_interval == 0:
##                mask = np.zeros_like(frame_gray)
##                mask[:] = 255
##                #for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
##                    #cv2.circle(mask, (x, y), 5, 0, -1)
##                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
##                if p is not None:
##                    for x, y in np.float32(p).reshape(-1, 2):
##                        self.tracks.append([(x, y)])
##            self.frame_idx += 1
##            self.prev_gray = frame_gray
##     
        

class ShowImg(object):
    def __init__(self, image, name="display", show=False, key=False, keytime=500):
        if args.display:
            cv2.namedWindow(name);
            cv2.imshow(name, image)
            if key:
                cv2.waitKey(0)
            else:
                cv2.waitKey(keytime)
            cv2.destroyWindow(name)        

class HSVthresh(object):
    """An HSV thresholder class"""
    w=None
    wname=None
    wc=None
    wcname=None
    #hsvminmax=None
    vshimage=None
    hsvtc=None
    threshimg=None
    threshimgmorph=None
    threshres=None
    sumimg=None
    display=False
    keypress=False
    keytimeout=0
    roi=[0,0,0,0]
    hsvth=[0, 255, 0, 255, 0, 180]
    _op=cv2.MORPH_OPEN
    _se=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    moments=None
    def __init__(self, bgrimage,roi=[0,0,0,0],display=False,keypress=False,keytimeout=0):
        self.display=display
        self.keypress=keypress
        self.keytimeout=keytimeout
        self.imgorig=bgrimage
        self.setroi(roi)
        # init hsv image
        self.hsvtrans(self.imgorig)
        # init result matrices
        self.hsvtc=numpy.zeros((self.vshimage.shape[0], self.vshimage.shape[1], 3), self.vshimage[0].dtype)
        self.sumimg=numpy.zeros((self.vshimage.shape[0], self.vshimage.shape[1]), self.vshimage[0].dtype)
        self.threshimg=numpy.zeros((self.vshimage.shape[0], self.vshimage.shape[1]), self.vshimage[0].dtype)
        self.threshimgmorph=numpy.zeros((self.vshimage.shape[0], self.vshimage.shape[1]), self.vshimage[0].dtype)
        self.threshres=numpy.zeros((self.vshimage.shape[0], self.vshimage.shape[1]), self.vshimage[0].dtype)
    def hsvtrans(self, bgrimage):
        print "hsvtrans"
        self.vshimage=cv2.cvtColor(bgrimage, cv2.COLOR_BGR2HSV)
        self.vshchannels=self.splitchannel(self.vshimage)
        return self.vshimage
    def setroi(self,roicoord):
        # TODO implement
        # set bgrimage background to 0
        roi=roicoord
        roiapply=False
        if roi[0]<0:
            roi[0]=0
        if roi[0]>self.imgorig.shape[0]:
            roiapply=True
            roi[0]=0
        if roi[1]<0:
            roi[1]=0
        if roi[1]>self.imgorig.shape[1]:
            roiapply=True
            roi[1]=0
        if roi[2]>0:
            roiapply=True
            #check fit of ROI to image size
            if roi[2]+roi[0]>self.imgorig.shape[0]:
                roi[2]=self.imgorig.shape[0]-roi[0]
        if roi[3]>0:
            roiapply=True
            if roi[3]+roi[1]>self.imgorig.shape[1]:
                roi[3]=self.imgorig.shape[1]-roi[1]
        self.roi=roi
        if roiapply:
            print "roiapply"
            print self.roi
            #newbgr=numpy.zeros((self.bgrimage.shape[0], self.bgrimage.shape[1], 3), self.bgrimage[0].dtype)
            mask = numpy.zeros(self.imgorig.shape, dtype=numpy.uint8)
            roi_corners = numpy.array([[(roi[0],roi[1]),
                                     (roi[0]+roi[2],roi[1]),
                                     (roi[0]+roi[2],roi[1]+roi[3]),
                                     (roi[0],roi[1]+roi[3])]], dtype=numpy.int32)
            white = (255, 255, 255)
            cv2.fillPoly(mask, roi_corners, white)
            # apply the mask
            masked_image = cv2.bitwise_and(self.imgorig, mask)
            #cv2.imwrite("/tmp/roimasked.png",masked_image)
            self.imgorig=masked_image
    def splitchannel(self,image):
        return cv2.split(image)
    def sethsvthresh(self,hmin,hmax,smin,smax,vmin,vmax):
        self.hsvth=[hmin,hmax,smin,smax,vmin,vmax]
    def updatedisp(self,dummmy=None):
        self.threshhsv()
        self.morphthreshimg()
        ityp=cv2.getTrackbarPos("Channel", self.wcname)
        if (0<=ityp and 2>=ityp):
            print type(self.hsvtc[:,:,ityp]), self.hsvtc[:,:,ityp]
            cv2.imshow(self.wname, self.hsvtc[:,:,ityp])
        if 3==ityp:
            cv2.imshow(self.wname, self.imgorig)
        if 4==ityp:
            cv2.imshow(self.wname, self.hsvtc)
        if 5==ityp:
            cv2.imshow(self.wname, self.sumimg)
        if 6==ityp:
            cv2.imshow(self.wname, self.threshimg)
        if 7==ityp:
            cv2.imshow(self.wname, self.vshimage)
        if 8==ityp:
            if self.threshimgmorph is not None:
                cv2.imshow(self.wname, self.threshimgmorph)
        self.printcmdlineparm()

    def threshhsv(self):
        #VSHt=[vmin,vmax,smin,smax,hmin,hmax]
        self.sumimg=numpy.zeros((self.vshimage.shape[0], self.vshimage.shape[1]), self.vshimage[0].dtype)
        self.hsvtc=numpy.zeros((self.vshimage.shape[0], self.vshimage.shape[1], 3), self.vshimage[0].dtype)
        for i in range(3):
            # threshold min values
            #ret, timgmin=cv2.threshold(self.vshchannels[i], HSVt[i*2], 255, cv2.THRESH_BINARY)
            # threshold max values
            #ret, timgmax=cv2.threshold(self.vshchannels[i], HSVt[i*2+1], 255, cv2.THRESH_BINARY)            
            #self.hsvtr[0:imgshape[0],0:imgshape[1],i]=timgmin-timgmax
            #print "thresholds fo channel %d: %d %d" % (i,self.hsvth[i*2],self.hsvth[i*2+1])
            tmpt=self.threshinterval(self.vshchannels[i],self.hsvth[i*2],self.hsvth[i*2+1])
            self.hsvtc[0:tmpt.shape[0],0:tmpt.shape[1],i]=tmpt
            self.sumimg+=self.hsvtc[:,:,i]/3
        ret, self.threshimg=cv2.threshold(self.sumimg, 244, 255, cv2.THRESH_BINARY)
        self.threshres=self.threshimg
        return self.hsvtc
    """ thresholds the self.vshimage with min and max values """
    def threshhsvinterval(self,hmin,hmax,smin,smax,vmin,vmax):
        self.sethsvthresh(hmin,hmax,smin,smax,vmin,vmax)
        self.threshhsv()
        return self.hsvtc
    def threshinterval(self,channel,min,max):
        ret, tmin = cv2.threshold(channel, min, 255, cv2.THRESH_BINARY)
        ret, tmax = cv2.threshold(channel, max, 255, cv2.THRESH_BINARY)
        return tmin-tmax
    def morphthreshimg(self,se=None,op=None):
        if se is not None:
            self._se=se
        if op is not None:
            self._op=op
        if self.threshimg!=None:
            self.threshimgmorph=cv2.morphologyEx(self.threshimg,self._op,self._se)
            self.threshres=self.threshimgmorph
        return self.threshimgmorph
    #def channelname(self,nchannel):
    #    if (value == 0):
    #        print "Image"
    #    elif (value == 1):
    #        print "First channel"
    #    elif (value == 2):
    #        print "Second channel"
    #    elif (value == 3):
    #        print "Third channel"
    def meanshiftf(srcimage,sp=3,sr=3,maxLevel=1):
        return cv2.pyrMeanShiftFiltering(srcimage,sp=sp,sr=sr,maxLevel=maxLevel)
    # cv2.pyrMeanShiftFiltering(src, sp, sr

    def createwin(self, wname="Display"):
        if self.wc is None:
            self.wcname="Controls"
            self.w=cv2.namedWindow(self.wcname)
            cv2.createTrackbar("Channel", self.wcname, 4, 8, self.updatedisp)
            cv2.createTrackbar("Hmin", self.wcname, 0, 179, self.dothreshhmin)
            cv2.setTrackbarPos("Hmin",self.wcname,self.hsvth[0])
            cv2.createTrackbar("Hmax", self.wcname, 0, 179, self.dothreshhmax)
            cv2.setTrackbarPos("Hmax",self.wcname,self.hsvth[1])
            cv2.createTrackbar("Smin", self.wcname, 0, 255, self.dothreshsmin)
            cv2.setTrackbarPos("Smin",self.wcname,self.hsvth[2])
            cv2.createTrackbar("Smax", self.wcname, 0, 255, self.dothreshsmax)
            cv2.setTrackbarPos("Smax",self.wcname,self.hsvth[3])
            cv2.createTrackbar("Vmin", self.wcname, 0, 255, self.dothreshvmin)
            cv2.setTrackbarPos("Vmin",self.wcname,self.hsvth[4])
            cv2.createTrackbar("Vmax", self.wcname, 0, 255, self.dothreshvmax)
            cv2.setTrackbarPos("Vmax",self.wcname,self.hsvth[5])
            #cv2.createTrackbar("Vmin", wname, 0, 3, self.doThresh)
            #cv2.createTrackbar("Vmax", wname, 0, 3, self.doThresh)
            self.updatedisp()
        if self.w is None:
            self.wname=wname
            self.w=cv2.namedWindow(self.wname)
            if self.display:
                ShowImg(self.imgorig,self.wname,show=self.display,key=self.keypress,keytime=self.keytimeout)
            #cv2.imshow(self.wname, self.imgorig)
    #def (self, wname="Display"):
    def dothreshhmin(self,value):
        if not (value>=0 and value<=179):
            print "hmin out of range: "+str(value)
            return False
        if not (value<=self.hsvth[1]):
            print "hmin > hmax, setting to hmax"
            value=self.hsvth[1]
            cv2.setTrackbarPos("Hmin",self.wcname,value)
        self.hsvth[0]=value
        self.updatedisp()
    def dothreshhmax(self,value):
        if not (value>=0 and value<=179):
            print "hmax out of range: "+str(value)
            return False
        if not (value>=self.hsvth[0]):
            print "hmax < hmin, setting to hmin"
            value=self.hsvth[0]
            cv2.setTrackbarPos("Hmax",self.wcname,value)
        self.hsvth[1]=value
        self.updatedisp()
    def dothreshsmin(self,value):
        if not (value>=0 and value<=255):
            print "smin out of range: "+str(value)
            return False
        if not (value<=self.hsvth[3]):
            print "smin > smax, setting to smax"
            value=self.hsvth[3]
            cv2.setTrackbarPos("Smin",self.wcname,value)
        self.hsvth[2]=value
        self.updatedisp()
    def dothreshsmax(self,value):
        if not (value>=0 and value<=255):
            print "smax out of range: "+str(value)
            return False
        if not (value>=self.hsvth[2]):
            print "smax < smin, setting to smin"
            value=self.hsvth[2]
            cv2.setTrackbarPos("Smax",self.wcname,value)
        self.hsvth[3]=value
        self.updatedisp()
    def dothreshvmin(self,value):
        if not (value>=0 and value<=255):
            print "vmin out of range: "+str(value)
            return False
        if not (value<=self.hsvth[5]):
            print "vmin > vmax, setting to vmax"
            value=self.hsvth[5]
            cv2.setTrackbarPos("Vmin",self.wcname,value)
        self.hsvth[4]=value
        self.updatedisp()
    def dothreshvmax(self,value):
        if not (value>=0 and value<=255):
            print "vmax out of range: "+str(value)
            return False
        if not (value>=self.hsvth[4]):
            print "vmax < vmin, setting to vmin"
            value=self.hsvth[4]
            cv2.setTrackbarPos("Vmax",self.wcname,value)
        self.hsvth[5]=value
        self.updatedisp()
    def printcmdlineparm(self):
        print "-m %d -M %d -s %d -S %d -v %d -V %d" % (self.hsvth[0],self.hsvth[1],self.hsvth[2],self.hsvth[3],self.hsvth[4],self.hsvth[5])
    def gui(self,image=None):
        if image is None:
            image=self.vshimage
        self.createwin()
    def shapefeatures(self,binaryimage=None):
        if binaryimage is None:
            binaryimage=self.threshimg
        #print type(binaryimage)
        #print binaryimage
        binaryimage2=binaryimage.copy()
        self.contours,hierarchy = cv2.findContours(binaryimage2,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        #centers = []
        #radii = []
        self.shapefeat=[]
        for h,cnt in enumerate(self.contours):
            sf=cv2.moments(cnt)
            hu=cv2.HuMoments(sf)
            # center of gravity
            if sf['m00']!=0:
                sf['cogx']=sf['m10']/sf['m00']
                sf['cogy']=sf['m01']/sf['m00']
            else:
                sf['cogx']=-1
                sf['cogy']=-1
                #print hu
            for h in enumerate(hu):
                sf["hu"+str(h[0]+1)]=h[1][0]
            (cx,cy),cr = cv2.minEnclosingCircle(cnt)
            sf['circx']=cx
            sf['circy']=cy
            sf['circr']=cr
            #TODO: other shape features
            sf['area']=cv2.contourArea(cnt)
            sf['arclength']=cv2.arcLength(cnt,True)
            self.shapefeat.append(sf)
            br=cv2.boundingRect(cnt)
            sf['bbulx']=br[0]
            sf['bbuly']=br[1]
            sf['bblrx']=br[2]
            sf['bblry']=br[3]
            #center = (int(x),int(y))
            #radius = int(radius)
            #centers.append(center)
            #radii.append(radius)
#            cv2.circle(self.imgorig,center,radius,(255,0,0))                
        #self.centersradii=zip(centers,radii)
        return self.shapefeat
    def optflow(self,previmg,currentimg):
        #cv2.calcOpticalFlowPyrLK(prevImg, nextImg, prevPts[, nextPts[, status[, err[, winSize[, maxLevel[, criteria[, flags[, minEigThreshold]]]]]]]]) → nextPts, status, err
        #features, status, track_error = cv2.calcOpticalFlowPyrLK(prev_gs, current_gs, good_features, None,**lk_params)
        # http://jayrambhia.wordpress.com/2012/08/09/lucas-kanade-tracker/
        # https://gist.github.com/jayrambhia/3295631
        lk_params = dict( winSize  = (10, 10), maxLevel = 5, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))   
        feature_params = dict( maxCorners = 3000, qualityLevel = 0.5, minDistance = 3, blockSize = 3)
        pt = cv2.goodFeaturesToTrack(previmg, **feature_params)
        #bb=[0,0,previmg]
        #for i in xrange(len(pt)):
        #    pt[i][0][0] = pt[i][0][0]+bb[0]
        #    pt[i][0][1] = pt[i][0][1]+bb[1]
        p0 = numpy.float32(pt).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(previmg, currentimg, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(currentimg, previmg, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_pts = []
        for pts, isgood in itertools.izip(p1, good):
            if isgood:
                new_pts.append([pts[0][0], pts[0][1]])
        return new_pts
        # TODO: go on here

##def channelName(value):
##        if (value == 0):
##                print "Showing H";
##        elif (value == 1):
##                print "Showing S";
##        elif (value == 2):
##                print "Showing V";


# init hsvthresh stack
#stack=list()
#previmg=None
roi=[args.xul, args.yul, args.X, args.Y]

for filecnt in range(len(args.imagefilenames)):
    print filecnt,args.imagefilenames[filecnt]
    image=cv2.imread(args.imagefilenames[filecnt])
    if None==image:
        print "error reading image"
        continue
    
    # check parameters
    imageshape=image.shape
    roix=checkminmax(args.xul,0,imageshape[0])
    roiy=checkminmax(args.yul,0,imageshape[1])
    roiX=checkminmax(args.xul+args.X,0,imageshape[0])
    roiY=checkminmax(args.yul+args.Y,0,imageshape[1])
    
    # threshold the blue channel
    # optionally work on grayscale, might be faster
    if args.thresholdblue:
        ret, binimg=cv2.threshold(image[roix:roiX,roiy:roiY,args.channel],args.minb,args.maxb, cv2.THRESH_BINARY)
    
        # cv.Reduce(src, dst, dim=-1, op=CV_REDUCE_SUM) → None¶
        colhist=cv2.reduce(src=binimg,dim=0,rtype=cv.CV_REDUCE_AVG)
    else:
        #TODO: needs to be scaled before thresholding, or adapt the threshold values?
        binimg=image[roix:roiX,roiy:roiY,args.channel]
        colhist=cv2.reduce(src=binimg,dim=0,rtype=cv.CV_REDUCE_AVG)
    
    # threshold histogram based on percentage of columns
    colselected=(colhist>255*args.colhistthreshperc/100)
    print (float(sum(colselected[0]))/binimg.shape[1])
    colselbinimg=numpy.compress(colselected[0],binimg,1)
    
    rowhist=cv2.reduce(src=colselbinimg,dim=1,rtype=cv.CV_REDUCE_AVG)
    rowselected=(rowhist>255*args.rowhistthreshperc/100)
    
    # TODO: evaluate the positions, 
    # wrap rowhist
    
    

    print (float(sum(rowselected))/binimg.shape[0])
    
    # show selection
    if (args.saveresults|args.display):
        chred=numpy.resize(numpy.array(numpy.repeat(rowhist,binimg.shape[1])),binimg.shape)
        chgreen=numpy.transpose(numpy.resize(numpy.array(numpy.repeat(colselected*255,binimg.shape[0]), dtype=numpy.uint8),[binimg.shape[1],binimg.shape[0]]))
        combimg=cv2.merge((binimg,chred,chgreen))
        outfilename=args.imagefilenames[filecnt]+".rowcol.png"
        cv2.imwrite(outfilename, combimg)
        
    if args.display:
        print ("display on")
        #ShowImg(image,"orig",show=True,key=args.keypress,keytime=args.keytimeout)
        # create an rgb image from binimg
        # cv2.cvtColor(binimg,cv.CV_GRAY2BGR)
        # red channel is row wise
        ShowImg(combimg,show=True,key=args.keypress,keytime=args.keytimeout)
        





















sys.exit(0)
if True: # jsut for the indent
    #print "tresholds:", args.minv, args.maxv, args.mins, args.maxs, args.minh, args.maxh
    #ht=HSVthresh(image[2],roi,display=args.display,keypress=args.keypress,keytimeout=args.keytimeout)
    #ht.threshhsvinterval(args.minh, args.maxh, args.mins, args.maxs, args.minv, args.maxv)
    #if morphology:
        #ht.morphthreshimg(se,op)
    #if args.gui:
        #ht.gui()
    #if filecnt<numimages:
        #stack.append(ht)
        #print "filling stack ",len(stack)
    #else:
        #stack[filecnt%numimages]=ht
        ##print "using stack ",filecnt%numimages
    #if previmg is None:
        #previmg=ht.threshres
    # iterate over stack, adding thresholds
    #st=numpy.zeros((ht.threshimg.shape[0], ht.threshimg.shape[1]),numpy.uint8)
    
    st=stack[0].threshimg/255
    for scnt in range(len(stack)):
        st+=stack[scnt].threshres/255
    ret, allt=cv2.threshold(st,len(stack),255, cv2.THRESH_BINARY)
    # show results?
    #ShowImg(ht.hsvtc,"hsvthresh",display,keypress,keytimeout)
    #ShowImg(ht.sumimg,"sumimg",display,keypress,keytimeout)
    #ShowImg(ht.threshimg,"thresh",display,keypress,keytimeout)
    #ShowImg(st*255/len(stack),"stackthresh",display,keypress,keytimeout)
    #ShowImg(allt,"allthresh",display,keypress,keytimeout)

    #optf=ht.optflow(previmg,ht.threshres)
    #print optf
    previmg=ht.threshres
    # write out results
    outfilenamebase=os.path.join(os.path.dirname(args.imagefilenames[filecnt]),args.prefix+os.path.basename(args.imagefilenames[filecnt]))
    outfilename=outfilenamebase+".vsh.png"
    #print "writing to %s" % (outfilename)
    #cv2.imwrite(outfilename, ht.vshimage)

    #ms=cv2.pyrMeanShiftFiltering(image,sp=5,sr=5,maxLevel=2)
    #ShowImg(ms,"meanshift")
    #outfilename=outfilenamebase+".ms.png"
    #print "writing to %s" % (outfilename)
    #cv2.imwrite(outfilename, ms)

    outfilename=outfilenamebase+".threshsum.png"
    #print "writing to %s" % (outfilename)
    #cv2.imwrite(outfilename, ht.sumimg)

    outfilename=outfilenamebase+".thresh.png"
    #print "writing to %s" % (outfilename)
    #cv2.imwrite(outfilename, ht.threshimg)

    outfilename=outfilenamebase+".threshcolor.png"
    #print "writing to %s" % (outfilename)
    #cv2.imwrite(outfilename, ht.hsvtc)

    outfilename=outfilenamebase+".stackthresh.png"
    print "writing to %s" % (outfilename)
    cv2.imwrite(outfilename, allt)

    # positions
    shapef=ht.shapefeatures(ht.threshres)
    outfilename=outfilenamebase+".postions.csv"
    with open(outfilename, 'wb') as csvfile:
        print "csvfile: "+outfilename
        csvwriter = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #csvwriter.writerow(['filename','cx','cy','radius'])
        rowcnt=0
        for sf in shapef:
            sf['filename']=os.path.basename(args.imagefilenames[filecnt])
            # sort them
            #sf=sorted(sf.iteritems())
            #print sf
            if 0==rowcnt:
                # header sorted
                rowdat=zip(*sorted(sf.items()))[0]
                #csvwriter.writerow(rowdat)
            rowcnt=rowcnt+1;
            areachk=True
            if (areamm[0] >= sf['area']):
                areachk=False
                if (areamm[1] < 0) or (areamm[1] >= sf['area']):
                    areachk=False
            if areachk:
                rowdat=zip(*sorted(sf.items()))[1]
                csvwriter.writerow(rowdat)

