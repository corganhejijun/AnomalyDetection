# -*- coding: utf-8 -*- 
import os
import cv2

test_file = 'train'
test_path = 'test'
ext = '.png'
hist_size = 128
out_file = 'hist_result.csv'

outFile = open(out_file, 'w')
outFile.write('filename,Correlation,Chi-Square,Intersection,Bhattacharyya,HELLINGER')

fileList = os.listdir(test_path)
for index, file in enumerate(fileList):
  if not file.endswith(ext):
    continue
  print("processing " + file + " " + str(index) + ' of total ' + str(len(fileList)))
  nameValue = file.split('.')[0].split('_')
  step = int(nameValue[5])
  x = int(nameValue[3])
  y = int(nameValue[4])
  img = cv2.imread(os.path.join(test_path, file), 0)
  height = img.shape[1]
  origin = img[y+height:y+step+height, x:x+step]
  imgOut = img[y:y+step, x:x+step]
  originHist = cv2.calcHist([origin], [0], None, [hist_size], [0.0, 255.0])
  imgHist = cv2.calcHist([imgOut], [0], None, [hist_size], [0.0, 255.0])
  dist1 = cv2.compareHist(originHist, imgHist, cv2.HISTCMP_CORREL)
  dist2 = cv2.compareHist(originHist, imgHist, cv2.HISTCMP_CHISQR)
  dist3 = cv2.compareHist(originHist, imgHist, cv2.HISTCMP_INTERSECT)
  dist4 = cv2.compareHist(originHist, imgHist, cv2.HISTCMP_BHATTACHARYYA)
  dist5 = cv2.compareHist(originHist, imgHist, cv2.HISTCMP_HELLINGER)
  outFile.write(file + ',' + str(dist1) + ',' + str(dist2) + ',' + str(dist3) + ',' + str(dist4) + ',' + str(dist5))