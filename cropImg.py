import cv2
import os

origin_train = 'origin_train.jpg'
origin_gt = 'origin_gt.jpg'

trainOut = 'train.jpg'
gtOut = 'gt.jpg'

resize = True
width = 1024
height = 1024

if os.path.exists(trainOut):
  os.remove(trainOut)
train = cv2.imread(origin_train, cv2.IMREAD_COLOR)

left = 0
top = 0
bottom = int(train.shape[0])
right = int(train.shape[1])

cropTrain = train[top:bottom, left:right]
if resize:
  cropTrain = cv2.resize(cropTrain, (width, height))
cv2.imwrite(trainOut, cropTrain)

if os.path.exists(gtOut):
  os.remove(gtOut)
gt = cv2.imread(origin_gt, cv2.IMREAD_COLOR)
cropGt = gt[top:bottom, left:right]
if resize:
  cropGt = cv2.resize(cropGt, (width, height))
cv2.imwrite(gtOut, cropGt)
