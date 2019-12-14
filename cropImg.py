<<<<<<< HEAD
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
left = int(train.shape[1] * 0.2)
top = 0
bottom = int(train.shape[0])
cropTrain = train[top:bottom, left:]
if resize:
  cropTrain = cv2.resize(cropTrain, (width, height))
cv2.imwrite(trainOut, cropTrain)

if os.path.exists(gtOut):
  os.remove(gtOut)
gt = cv2.imread(origin_gt, cv2.IMREAD_COLOR)
left = int(gt.shape[1] * 0.2)
top = 0
bottom = int(gt.shape[0])
cropGt = gt[top:bottom, left:]
if resize:
  cropGt = cv2.resize(cropGt, (width, height))
cv2.imwrite(gtOut, cropGt)
=======
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
top = 200
bottom = int(train.shape[0])
right = 800 # int(train.shape[1])

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
>>>>>>> 8efd0c446aea9233b8149fc5a5144f8eed2310d7
