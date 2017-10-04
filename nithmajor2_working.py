import numpy as np
from firebase1 import firebase
import cv2
import math
import datetime
import matplotlib.pyplot as plt
cap=cv2.VideoCapture('D:\mazor\VID_20170320_174611.mp4')
firebase=firebase.FirebaseApplication('https://major-f1a75.firebaseio.com/',None)
#print cap.set(3,640)
#print cap.set(4,480)
cap.set(1,476)
centerPositions=[]
global blobs
blobs=[]
INENTRY=0
OUTENTRY=0
heavy=0
prev=0
prev1=0
#########################################################
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
######################################################################
ret,imgFrame1Copy=cap.read()
mn=imgFrame1Copy
mn=cv2.flip(mn,-1)
plt.imshow(mn, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()
ret,imgFrame2Copy=cap.read()
carCount=0
twovehiclecount=0
blnFirstFrame = True
fps=0
###################################################### crossing line reference
crossingLine=np.zeros((2,2),np.float32)
horizontalLinePosition=434
crossingLine[0][0]= 231
crossingLine[0][1]= horizontalLinePosition
crossingLine[1][0]= 1000
crossingLine[1][1] =horizontalLinePosition
#########################################################
def millis_interval(start,end):
    """start and end are datetime instances"""
    diff = end - start
    millis = diff.days * 24 * 60 * 60 * 1000
    millis += diff.seconds * 1000
    millis += diff.microseconds / 1000
    return millis

##################################################################################
#define blob filter for blob analysing and filtering of bad blobs
class blobz(object): 
    def __init__(self,contour):
        global currentContour 
        global currentBoundingRect 
        global centerPosition
        global centerPositions
        global cx
        global cy
        global dblCurrentDiagonalSize 
        global dblCurrentAspectRatio 
        global intCurrentRectArea
        global blnCurrentMatchFoundOrNewBlob 
        global blnStillBeingTracked 
        global intNumOfConsecutiveFramesWithoutAMatch 
        global predictedNextPosition
        global numPositions
        self.predictedNextPosition=[]
        self.centerPosition=[]
        currentBoundingRect=[]
        currentContour=[]
        self.centerPositions=[]
        self.currentContour=contour
        self.currentBoundingArea=cv2.contourArea(contour)
        x,y,w,h = cv2.boundingRect(contour)
        self.currentBoundingRect=[x,y,w,h]
        cx=(2*x+w)/2
        cy=(2*y+h)/2
        self.centerPosition=[cx,cy]
        self.dblCurrentDiagonalSize=math.sqrt(w*w+h*h)
        self.dblCurrentAspectRatio=(w/(h*1.0))
        self.intCurrentRectArea=w*h
        self.blnStillBeingTracked = True
        self.blnCurrentMatchFoundOrNewBlob = True
        self.intNumOfConsecutiveFramesWithoutAMatch = 0
        self.centerPositions.append(self.centerPosition)    
    def predictNextPosition(self):
     #### next position prediction algorithm based on last 5 weighing sum of tracked blob positions
        numPositions=len(self.centerPositions)
        if (numPositions == 1):
            self.predictedNextPosition=[self.centerPositions[-1][-2],self.centerPositions[-1][-1]]
        if(numPositions >= 2):
            deltaX = self.centerPositions[1][0]-self.centerPositions[0][0]
            deltaY =self.centerPositions[1][1] -self.centerPositions[0][1]
            self.predictedNextPosition =[self.centerPositions[-1][-2]+deltaX,self.centerPositions[-1][-1]+deltaY ]
       # if (numPositions == 3):
        #    sumOfXChanges= ((self.centerPositions[2][0] - self.centerPositions[1][0]) * 2) +((self.centerPositions[1][0] - self.centerPositions[0][0]) * 1)
         #   deltaX=(sumOfXChanges / 3)
          #  sumOfYChanges= ((self.centerPositions[2][1] - self.centerPositions[1][1]) * 2) +((self.centerPositions[1][1] - self.centerPositions[0][1]) * 1)
           # deltaY=(sumOfYChanges / 3)
            #self.predictedNextPosition=[self.centerPositions[-1][-2]+deltaX,self.centerPositions[-1][-1]+deltaY ]
        #if (numPositions == 4):
         #   sumOfXChanges= ((self.centerPositions[3][0] - self.centerPositions[2][0]) * 3) +((self.centerPositions[2][0] - self.centerPositions[1][0]) * 2) +((self.centerPositions[1][0] - self.centerPositions[0][0]) * 1)
          #  deltaX=(sumOfXChanges / 6)
           # sumOfYChanges= ((self.centerPositions[3][1] - self.centerPositions[2][1]) * 3) +((self.centerPositions[2][1] - self.centerPositions[1][1]) * 2) +((self.centerPositions[1][1] - self.centerPositions[0][1]) * 1)
            #deltaY= (sumOfYChanges / 6)
            #self.predictedNextPosition=[self.centerPositions[-1][-2]+deltaX,self.centerPositions[-1][-1]+deltaY ]
        #if (numPositions >= 5):
         #   sumOfXChanges= ((self.centerPositions[numPositions-1][0] - self.centerPositions[numPositions-2][0]) * 4) +((self.centerPositions[numPositions-2][0] - self.centerPositions[numPositions-3][0]) * 3) +((self.centerPositions[numPositions-3][0] - self.centerPositions[numPositions-4][0]) * 2) +((self.centerPositions[numPositions-4][0] - self.centerPositions[numPositions-5][0]) * 1)
          #  sumOfYChanges= ((self.centerPositions[numPositions-1][1] - self.centerPositions[numPositions-2][1]) * 4) +((self.centerPositions[numPositions-2][1] - self.centerPositions[numPositions-3][1]) * 3) +((self.centerPositions[numPositions-3][1] - self.centerPositions[numPositions-4][1]) * 2) +((self.centerPositions[numPositions-4][1] - self.centerPositions[numPositions-5][1]) * 1)
           # deltaX= (sumOfXChanges / 10)
            #deltaY=(sumOfYChanges / 10)
            self.predictedNextPosition=[self.centerPositions[-1][-2]+deltaX,self.centerPositions[-1][-1]+deltaY ]
##########################################################################################
def matchCurrentFrameBlobsToExistingBlobs(blobs,currentFrameBlobs):
    for existingBlob in blobs:
        existingBlob.blnCurrentMatchFoundOrNewBlob = False
        existingBlob.predictNextPosition()
    for currentFrameBlob in currentFrameBlobs:
        intIndexOfLeastDistance = 0
        dblLeastDistance = 1000000.0
        for i in range(len(blobs)):
            if (blobs[i].blnStillBeingTracked == True):
                dblDistance=distanceBetweenPoints(currentFrameBlob.centerPositions[-1],blobs[i].predictedNextPosition)
                if (dblDistance < dblLeastDistance):
                    dblLeastDistance = dblDistance
                    intIndexOfLeastDistance = i
        if (dblLeastDistance < (currentFrameBlob.dblCurrentDiagonalSize * 1.0)/1.2):
            blobs=addBlobToExistingBlobs(currentFrameBlob, blobs, intIndexOfLeastDistance)
        else:
            blobs,currentFrameBlob=addNewBlob(currentFrameBlob, blobs)
    for existingBlob in blobs:
        if (existingBlob.blnCurrentMatchFoundOrNewBlob == False):
            existingBlob.intNumOfConsecutiveFramesWithoutAMatch = existingBlob.intNumOfConsecutiveFramesWithoutAMatch + 1
        if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >=3):
            existingBlob.blnStillBeingTracked =False
    return blobs   
##########################################################################################################
def distanceBetweenPoints(pos1,pos2):
    if (pos2==[]):
        dblDistance=math.sqrt((pos1[0])**2+(pos1[1])**2)
    else:
        dblDistance=math.sqrt((pos2[0]-pos1[0])**2+(pos2[1]-pos1[1])**2)
    return dblDistance
def addBlobToExistingBlobs(currentFrameBlob, blobs, intIndex):
    blobs[intIndex].currentContour = currentFrameBlob.currentContour
    blobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect
    blobs[intIndex].centerPositions.append(currentFrameBlob.centerPositions[-1])
    blobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize
    #blobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio
    blobs[intIndex].blnStillBeingTracked = True
    blobs[intIndex].blnCurrentMatchFoundOrNewBlob = True
    return blobs
def addNewBlob(currentFrameBlob,Blobs):
    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = True
    blobs.append(currentFrameBlob)
    return blobs,currentFrameBlob
##################################################################################################################
def drawBlobInfoOnImage(blobs,m1):
    for i in range(len(blobs)):
        if (blobs[i].blnStillBeingTracked == True):
            x,y,w,h=blobs[i].currentBoundingRect
            cx=blobs[i].centerPositions[-1][-2]
            cy=blobs[i].centerPositions[-1][-1]
            print cx,cy
            cv2.rectangle(m1,(x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(m1,(int(cx),int(cy)),2,(0,0,0),-1)
            text =str(i)
            cv2.putText(m1,"ob{}".format(text),(blobs[i].centerPositions[-1][-2],blobs[i].centerPositions[-1][-1]),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255 ,0), 2) 
    return m1                             
######################################################################################
def drawCarCountOnImage(carCount,twovehiclecount,m1,fps):
    initText = "Car Counter: "
    text =initText+str(carCount) +  "   Two Wheelers Counter:"+str(     twovehiclecount)+"   FRAME RATE:" + str(int(fps))
    cv2.putText(m1, "Traffic Status: {}".format(text), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0 ,255), 2)
    cv2.putText(m1, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, m1.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    return m1
###############################################################################
def checkIfBlobsCrossedTheLine(blobs,horizontalLinePosition,carCount,twovehiclecount,INENTRY,OUTENTRY):
    atLeastOneBlobCrossedTheLine= False
    for blob in blobs:
        if (blob.blnStillBeingTracked == True and len(blob.centerPositions) >= 2):
            cx=blob.centerPositions[-1][-2]
            cy=blob.centerPositions[-1][-1]
            prevFrameIndex= len(blob.centerPositions) - 2
            currFrameIndex= len(blob.centerPositions) - 1
            if (blob.centerPositions[prevFrameIndex][-1] >= horizontalLinePosition and blob.centerPositions[currFrameIndex][-1] < horizontalLinePosition) and cx>231 and cx<1000:
                x,y,w,h=blob.currentBoundingRect
                if (w>100 and h>100):
                    heavy=heavy+1
                    INENTRY=INENTRY+1
                if (w>100 and h>30):
                    carCount = carCount + 1
                    INENTRY=INENTRY+1
                else:
                    twovehiclecount=twovehiclecount+1
                    INENTRY=INENTRY+1
                atLeastOneBlobCrossedTheLine = True
            if (blob.centerPositions[prevFrameIndex][-1] <= horizontalLinePosition and blob.centerPositions[currFrameIndex][-1] > horizontalLinePosition) and cx>231 and cx<1000:
                x,y,w,h=blob.currentBoundingRect
                if (w>100 and h>100):
                    heavy=heavy+1
                    INENTRY=INENTRY+1
                if (w>100 and h>30):
                    carCount = carCount + 1
                    OUTENTRY=OUTENTRY+1
                else:
                    twovehiclecount=twovehiclecount+1
                    OUTENTRY=OUTENTRY+1
                atLeastOneBlobCrossedTheLine = True
    return atLeastOneBlobCrossedTheLine,carCount,twovehiclecount,INENTRY,OUTENTRY
###################################################################################
def snipping(a):
    mask=np.zeros(a.shape,dtype=np.uint8)
    roi_corner=np.array([[(63,475),(443,488),(754,472),(1013,450),(843,417),(719,407),(605,399),(514,392),(412,393),(319,392),(265,428),(149,463)]],dtype=np.int32)
    channel_count=a.shape[2]
    ignore_mask_color=(255,)*channel_count
    cv2.fillPoly(mask,roi_corner,ignore_mask_color)
    masked_image=cv2.bitwise_and(a,mask)
    return masked_image
while  (True):
    startTime=datetime.datetime.now()
    m1=imgFrame1Copy
    n1=imgFrame2Copy
    m1=cv2.flip(m1,-1)
    n1=cv2.flip(n1,-1)
    masked_image=snipping(m1)
    s1=masked_image
    masked_image=snipping(n1)
    s2=masked_image
    a1 = cv2.cvtColor(s1,cv2.COLOR_BGR2GRAY)
    b1 = cv2.cvtColor(s2,cv2.COLOR_BGR2GRAY)
    a2 = cv2.GaussianBlur(a1,(5,5),0)
    b2 = cv2.GaussianBlur(b1,(5,5),0)
    imgDifference=cv2.absdiff(b2,a2)
    ret1,th1 = cv2.threshold(imgDifference,30,255,cv2.THRESH_BINARY)
    th1 = cv2.dilate(th1,kernel,iterations = 1)
    th1 = cv2.dilate(th1,kernel,iterations = 1)
    fgmask = cv2.erode(th1,kernel,iterations = 1)
    frameNo=cap.get(1)
    fgmask = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel2)
    fg2=np.zeros((fgmask.shape[0],fgmask.shape[1],3),np.uint8)
    fg3=np.zeros((fgmask.shape[0],fgmask.shape[1],3),np.uint8)
    contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fg2, contours, -1, (255,255,255), -1)
    hulls=[]
    for c in range(len(contours)):
        hull=cv2.convexHull(contours[c])
        hulls.append(hull)
    curFrameblobs=[]
    for c in range(len(hulls)):
        ec=blobz(hulls[c])
        if(ec.intCurrentRectArea>100 and ec.dblCurrentDiagonalSize>30 and ec.currentBoundingRect[2]>20 and ec.currentBoundingRect[3]>20 and (ec.currentBoundingArea*1.0/ec.intCurrentRectArea)>.4):
            curFrameblobs.append(ec)
    contor=[]
    for af in curFrameblobs:
        contor.append(af.currentContour)
    if (blnFirstFrame ==True):
        for f1 in curFrameblobs:
            blobs.append(f1)
    else: 
        blobs=matchCurrentFrameBlobsToExistingBlobs(blobs,curFrameblobs)                     
    m1=drawBlobInfoOnImage(blobs,m1)
    atLeastOneBlobCrossedTheLine,carCount,twovehiclecount,INENTRY,OUTENTRY=checkIfBlobsCrossedTheLine(blobs, horizontalLinePosition, carCount,twovehiclecount,INENTRY,OUTENTRY)
    if (atLeastOneBlobCrossedTheLine):
        cv2.line(m1,(crossingLine[0][0],crossingLine[0][1]),(crossingLine[1][0],crossingLine[1][1]),(0,255,0), 2)
    else:
        cv2.line(m1,(crossingLine[0][0],crossingLine[0][1]),(crossingLine[1][0],crossingLine[1][1]),(0,0,255), 2)
    m1=drawCarCountOnImage(carCount,twovehiclecount,m1,fps)
    endTime=datetime.datetime.now()
    millis=millis_interval(startTime,endTime)
    fps=(2.0*1000)/millis
    if INENTRY>prev:
        num1=str(26-INENTRY+OUTENTRY)
        OUT=firebase.put('/parkinzone','zone1',num1)
    if OUTENTRY>prev1:
        num2=str(26+OUTENTRY-INENTRY)
        OUT=firebase.put('/parkinzone','zone1',num2)   
    cv2.imshow('original',m1)
    #out.write(m1)
    cv2.drawContours(fg3, contours, -1, (255,255,255), -1)
    cv2.imshow('contour',fg3)
    cv2.drawContours(fg2, hulls, -1, (255,255,255), -1)
    cv2.imshow('convexhulls',fg2)
    prev=INENTRY
    prev1=OUTENTRY
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    imgFrame1Copy=imgFrame2Copy
    if cap.get(1)==1374:
        cap.set(1,1790)
    if cap.get(1)==2450:
        cap.set(1,3502)
    if cap.get(1)==6302:
        cap.set(1,6400) 
    ret,imgFrame2Copy=cap.read()
    ret,imgFrame2Copy=cap.read()
    ret,imgFrame2Copy=cap.read()
    ret,imgFrame2Copy=cap.read()
    if not ret:
        break
    blnFirstFrame = False
    if cap.get(1)==7000:
        break
    if cap.get(1)==7002:
        break
cap.release()
cv2.destroyAllWindows()
