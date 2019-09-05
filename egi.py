#!/usr/bin/env python
#!/usr/bin/python
# -*- coding: utf-8 -*-

# TODO: clipping (done in float32uint8) does not work properly?!? Output files are garbled (OpenCV version problem?)
# TODO: add more indices as option

import cv2
import sys
import os
import numpy
import argparse
#import pyexiv2
import tempfile



parser = argparse.ArgumentParser(description='Process RGB images to EGI index, optionally apply gray value scaling + thresholding, saving index/mask to files and/or alpha channel')
parser.add_argument(metavar='I', type=str, nargs='+', dest='imagefilenames',
                   help='RGB images to be processed')
parser.add_argument('-d', '--display', action="store_true", dest='display',
                   help='display the images in a viewer')
parser.add_argument('-T', '--autothresh', action="store_true", dest='autothresh',
                   help='autothreshold EGI image')
parser.add_argument('-s', '--scalemin', action="store_true", dest='scalemin',
                   help='scale min value')
parser.add_argument('-S', '--scalemax', action="store_true", dest='scalemax',
                   help='scale max value')
parser.add_argument('-t', '--threshold', type=int, default=-1, choices=list(range(0,256)), dest='threshold',
                   help='threshold for EGI image (overrides autothresh)')
parser.add_argument('-r', '--redfactor', type=float, default=-1, dest='rfactor',
                   help='factor for red channel')
parser.add_argument('-g', '--greenfactor', type=float, default=2, dest='gfactor',
                   help='factor for green channel')
parser.add_argument('-b', '--bluefactor', type=float, default=-1, dest='bfactor',
                   help='factor for blue channel')
parser.add_argument('-D', '--debug', action="store_true", dest='debug',
                   help='print debug information')
parser.add_argument('-e', '--egisuffix', default='.egi.png', type=str, dest='egisuffix',
                   help='suffix for the EGI results, appended to the input filenames')
parser.add_argument('-E', '--egithreshsuffix', default='.egi.thresh.png', type=str, dest='egithreshsuffix',
                   help='suffix for the thresholded EGI results, appended to the input filenames')
parser.add_argument('-a', '--alphaimage', action="store_true", dest='alphaimage',
                   help='create an image with alpha channel containing the index (original with additional transparency).')
parser.add_argument('-A', '--alphaimagesuffix', default='.egi.alpha.png', type=str, dest='alphaimagesuffix',
                   help='suffix for the image with alpha channel, appended to the input filenames')
parser.add_argument('-m', '--maskedimage', action="store_true", dest='maskedimage',
                   help='create a masked image (original with additional transparency mask). Use -a or -t, too')
parser.add_argument('-M', '--maskedimagesuffix', default='.egi.masked.png', type=str, dest='maskedimagesuffix',
                   help='suffix for the masked image, appended to the input filenames')

#parser.add_argument('-I', default='', type=str, dest='indexnames', 
#                   help='comma-separated list of index names')

def printstderr(*objs):
    sys.stderr.write("".join(map(str,objs)))
    sys.stderr.write(os.linesep)

class ImgProc(object):
    """An image processing class"""
    def __init__(self, image=None):
            self.setorig(image)
    def egi(self,rfactor=-1.0,gfactor=2.0,bfactor=-1.0):
        if self.imgorig is not None:
            # create the float32 type array filled with 0.0
            self.egiimg=numpy.zeros((self.imgorig.shape[0], self.imgorig.shape[1]), numpy.float32)
            # add the result of the egi computation
            # explicitly cast the imgorig to float32!
            self.egiimg+=(bfactor*self.imgorig[:,:,0].astype('float32')+
                       gfactor*self.imgorig[:,:,1].astype('float32')+
                       rfactor*self.imgorig[:,:,2].astype('float32'))
        else:
            self.egiimg = None
        return self.egiimg
    def setorig(self, image):
        self.imgorig = image
        if self.imgorig == None:
            self.setdummy()
            self.ok = False
        else:
                if 3==self.image.shape[3]:
                    self.ok = True
                else:
                    self.ok = False
    def setdummy(self):
            # set imgorig to black pixel
            self.imgorig=numpy.zeros((1, 1, 3), numpy.uint8)
    def imreadbgr(self,filename=None):
        if filename is None:
            self.ok=False
            self._metadata=None
        else:
            #img=cv2.imread(filename,cv2.CV_LOAD_IMAGE_COLOR)
            # read in unchanged, do not assume 3 channel images, also RGBA etc.
            # CV_LOAD_IMAGE_COLOR, CV_LOAD_IMAGE_GRAYSCALE, CV_LOAD_IMAGE_ANYCOLOR, CV_LOAD_IMAGE_ANYDEPTH, and CV_LOAD_IMAGE_UNCHANGED will be removed in future versions of OpenCV.
            img=cv2.imread(filename,cv2.IMREAD_UNCHANGED)
            if img is None:
                self.ok=False
                self._metadata=None
            else:
                #self.readmetadata(filename=filename)
                self.imgorig=img
                self.ok=True
        return self.ok
        
    def getorigrgba(self,alphachannel=None):
        if self.imgorig is None:
            return None
        if alphachannel is None:
            return numpy.append(self.imgorig,255*numpy.ones(self.imgorig.shape[0],self.imgorig.shape[1],1).astype('uint8'),axis=2)
        if self.imgorig.shape[0:2] != alphachannel.shape[0:2]:
            return None
        return numpy.append(self.imgorig,alphachannel.reshape(self.imgorig.shape[0],self.imgorig.shape[1],1),axis=2)
        
    def imreadstr(self,imstr):
        if type(str()) != type(imstr):
            # not a string
            self.ok=False
        else:
                tmpf = tempfile.NamedTemporaryFile(delete=False)
                tmpf.write(imstr)
                tmpf.close()
                self.imreadbgr(filename=tmpf.name)
                #self.readmetadata(filename=tmpf.name)
                os.unlink(tmpf.name)
        return self.ok
    #def readmetadata(self,filename):
    #    try:
            #self._metadata=pyexiv2.ImageMetadata(filename)
            #pyexiv2self._metadata.read()
    #    except IOError as e:
    #        self._metadata=None
    #def getmetadata(self):
    #    return self._metadata
    def scale(self,image,minval=0.0,maxval=255.0):
        if 0!=(maxval-minval):
            return (image-minval)*(255.0/(maxval-minval))
        else:
            return image
    # convert float to uint with clipping
    def float32uint8(self,fimage,clipmin=0.0,clipmax=255.0):
        #self.imguint8=numpy.zeros((fimage.shape[0], fimage.shape[1]), numpy.uint8)
        #self.imguint8=numpy.rint(numpy.clip(fimage,clipmin,clipmax)).astype('uint8')
        self.imguint8=numpy.clip(fimage,clipmin,clipmax).astype('uint8')
        return self.imguint8

def checkminmax(value,minval=0,maxval=255,name="value"):
    res=value
    if(value<minval):
        res=minval
        printstderr("warning: invalid "+name+": "+value+",setting to "+res)
    if(value>maxval):
        res=maxval
        printstderr("warning: invalid "+name+": "+value+",setting to "+res)
    return res

args = parser.parse_args()
if args.debug:
    printstderr(args)

##bfactor=-1.0
##gfactor=2.0
##rfactor=-1.0
##scalemax=True
###scalemax=False
##scalemin=True
##scalemin=False
dothresh=False
headerprinted=False
if args.autothresh:
    dothresh=True
if args.threshold>-1:
    dothresh=True

for filecnt in range(len(args.imagefilenames)):
##    print filecnt,args.imagefilenames[filecnt]
    imgfilename=args.imagefilenames[filecnt]
#    image=cv2.imread(imgfilename)
    ic=ImgProc(None)
    if args.debug:
        printstderr("reading file "+imgfilename)
    if not ic.imreadbgr(imgfilename):
        printstderr("ERROR: ","loading image: "+imgfilename+" failed")
        continue
    #image=ic.imgorig
    #if args.debug:
        #print "image content"
        #print ic.imgorig
        #print "factors rgb: %f %f %f" % (args.rfactor, args.gfactor, args.bfactor)
    egiimg=ic.egi(rfactor=args.rfactor,gfactor=args.gfactor,bfactor=args.bfactor)

    #image=cv2.imread(imgfilename)
    #bgrc=cv2.split(image)
#    egi=numpy.zeros((image.shape[0], image.shape[1]), numpy.float32)
#    egi+=(args.bfactor*image[:,:,0]+args.gfactor*image[:,:,1]+args.rfactor*image[:,:,2])

    if args.debug:
        printstderr("min/max egi: %g %g" % (numpy.min(egiimg),numpy.max(egiimg)))
    # set defaults (no scaling requested)
    egimin=0.0
    egimax=255.0
    # if scaling requested, get min and max to be scaled to [0,255]
    doscale=False
    if args.scalemax:
        egimax=numpy.max(egiimg)
        doscale=True
    if args.scalemin:
        egimin=numpy.min(egiimg)
        doscale=True
    # scale grayvalues, if requested
    if doscale:
        if args.debug:
            printstderr("scaling EGI")
            printstderr(str(numpy.min(egiimg))+" "+str(numpy.max(egiimg))+" "+str(egimin)+" "+str(egimax))
        egiimg=ic.scale(egiimg,egimin,egimax)
        #span=(egimax-egimin)
        #if span != 0:
        #    egiimg=(egiimg-egimin)*(255.0/(egimax-egimin))
        #else:
        #    pass
        #    #print "span is zero"
        #print str(numpy.min(egiimg))+" "+str(numpy.max(egiimg))
    #if args.debug:
        #printstderr("min/max (egi after scaling): %g %g" % (numpy.min(egiimg),numpy.max(egiimg)))
    if args.display:
        #printstderr(type(egiimg[0,0]))
        cv2.namedWindow("egi")
        cv2.imshow("egi",egiimg/255)
        cv2.waitKey(1000)
    # convert to 8 bit
    egi8=ic.float32uint8(egiimg)
    if args.display:
        #printstderr(type(egi8[0,0]))
        cv2.namedWindow("egi")
        cv2.imshow("egi",egi8)
        cv2.waitKey(1000)

    outfilename = imgfilename+args.egisuffix
    if args.debug:
        printstderr("debug: ","writing EGI image to %s " % outfilename)
    cv2.imwrite(outfilename,egi8)
    if args.alphaimage:
        outfilename = imgfilename+args.alphaimagesuffix
        if args.debug:
            printstderr("debug: ","writing alpha image to %s " % outfilename)
        cv2.imwrite(outfilename,ic.getorigrgba(egi8))

    if (dothresh):
        #TODO check thresholdval here?
        ttype=cv2.THRESH_BINARY
        # convert to 8bit with
        # cv.ConvertScaleAbs(src, dst, scale=1.0, shift=0.0) -> None
        if (args.autothresh):
            ttype=(ttype|cv2.THRESH_OTSU)
        ret, egit=cv2.threshold(egi8,args.threshold,255,ttype)
        if not headerprinted:
            print("imagefilename\tthreshold\tcovpix\tcovperc")
            headerprinted=True
        wpix=int(numpy.sum(egit)/255.0)
        print("%s\t%d\t%d\t%f" % (imgfilename, ret, wpix, 1.0*wpix/egit.shape[0]/egit.shape[1]))
        #TODO: flush here?
        outfilename = imgfilename+args.egithreshsuffix
        if args.debug:
            printstderr("debug: ","writing EGI thresh image to %s " % outfilename)
        cv2.imwrite(outfilename,egit)
        if args.maskedimage:
          outfilename = imgfilename+args.maskedimagesuffix
          if args.debug:
              printstderr("debug: ","writing masked image to %s " % outfilename)
          cv2.imwrite(outfilename,ic.getorigrgba(egit))
        
        sys.stdout.flush()

        if args.display:
            cv2.imshow("egi", egit)
            cv2.waitKey(1000)
