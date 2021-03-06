# -*- coding: utf-8 -*- 
import os
import cv2
from shutil import copyfile
import numpy as np

class Evaluator:
  def __init__(self, myList, name, test_path, save_dir, fine_size, useHeadCrop = True):
    self.USE_AVERAGE = True
    self.HIST_COUNT_COUNT = 5
    self.fine_size = fine_size
    self.SAVE_FILE_COUNT = 30
    self.save_dir = save_dir
    self.myList = myList
    self.name = name
    self.test_path = test_path
    self.origin_file = 'train.jpg'
    self.gt_file = 'gt.jpg'
    self.useHeadCrop = useHeadCrop

  def sortList(self, item):
    return item[0]

  def saveList(self, llist, dest):
    for item in llist:
      filename = item[1]
      copyfile(os.path.join(self.test_path, filename), os.path.join(dest, filename))

  def saveFirstAndLast(self):
    print("saveing " + self.name)
    self.myList.sort(key=self.sortList)
    os.mkdir(os.path.join(self.save_dir, self.name + '_top'))
    self.saveList(self.myList[:self.SAVE_FILE_COUNT], os.path.join(self.save_dir, self.name + '_top'))
    os.mkdir(os.path.join(self.save_dir, self.name + '_last'))
    self.saveList(self.myList[-self.SAVE_FILE_COUNT:], os.path.join(self.save_dir, self.name + '_last'))

  def sortHistByArea(self):
    countList = []
    img = cv2.imread(self.origin_file, cv2.IMREAD_COLOR)
    for index, item in enumerate(self.myList):
      value = item[0]
      file = item[1]
      posList = file.split('_')
      pos = posList[1] + '_' + posList[2]
      if int(posList[1]) + self.fine_size > img.shape[1] or int(posList[2]) + self.fine_size > img.shape[0]:
        continue
      found = -1
      for j, c in enumerate(countList):
        if c[1] == pos:
          found = j
          break
      if found > -1:
        if self.USE_AVERAGE:
          countList[found][2] += index
          countList[found][0] = (value + countList[found][0]) / 2
        else:   # use index range
          countList[found][0] += index
          countList[found][2] = (value + countList[found][2]) / 2
      else:
        if self.USE_AVERAGE:
          countList.append([value, pos, index])
        else:
          countList.append([index, pos, value])
    countList.sort(key=self.sortList)
    return countList

  def saveHistCountImage(self, countList):
    img = cv2.imread(self.origin_file, cv2.IMREAD_COLOR)
    for i in range(self.HIST_COUNT_COUNT):
      if self.useHeadCrop:
        head = countList[i][1].split('_')
        x = int(head[0])
        y = int(head[1])
        cv2.rectangle(img, (x, y), (x+self.fine_size, y+self.fine_size), (0, 0, 255), 3)
        if self.USE_AVERAGE:
          cv2.putText(img, "{}:{:.5f}".format(i, countList[i][0]), (x, y+int(self.fine_size/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
          cv2.putText(img, "{}:{:.5f}".format(i, countList[i][2]), (x, y+int(self.fine_size/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
      else:
        tail = countList[-i-1][1].split('_')
        x = int(tail[0])
        y = int(tail[1])
        cv2.rectangle(img, (x, y), (x+self.fine_size, y+self.fine_size), (255, 0, 0), 3)
        if self.USE_AVERAGE:
          cv2.putText(img, "{}:{:.5f}".format(i, countList[-i-1][0]), (x, y+int(self.fine_size/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
          cv2.putText(img, "{}:{:.5f}".format(i, countList[-i-1][2]), (x, y+int(self.fine_size/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    if not os.path.isdir('png'):
      os.mkdir('png')
    cv2.imwrite('png/' + self.name + '.png', img)

  def saveHistCount(self):
    print("saveing " + self.name + " count list")
    countList = self.sortHistByArea()
    self.saveHistCountImage(countList)

  def getCurve(self, gtImg, rect_TP_FP):
    curve = []
    FN_ALL = np.count_nonzero(gtImg == 255)
    TN_ALL = np.count_nonzero(gtImg == 0)
    TP_SUM = 0
    FP_SUM = 0
    for item in rect_TP_FP:
      TP_SUM += item[0]
      FP_SUM += item[1]
      FN = FN_ALL - TP_SUM
      TN = TN_ALL - FP_SUM
      TPR = TP_SUM / (TP_SUM + FN)
      FPR = FP_SUM / (FP_SUM + TN)
      curve.append([FPR, TPR])
    lastPt = curve[-1]
    if lastPt[0] < 1 or lastPt[1] < 1:
      curve.append([1,1])
    return curve

  def getROC(self, countList, gtImg):
    # explain
    # https://www.cnblogs.com/dlml/p/4403482.html
    rect_TP_FP = []
    for i in range(len(countList)):
      rect = None
      if self.useHeadCrop:
        rect = countList[i][1].split('_')
      else:
        rect = countList[-i-1][1].split('_')
      x = int(rect[0])
      y = int(rect[1])
      TP = np.count_nonzero(gtImg[y:y+self.fine_size, x:x+self.fine_size] == 255)
      FP = np.count_nonzero(gtImg[y:y+self.fine_size, x:x+self.fine_size] == 0)
      rect_TP_FP.append([TP, FP])
    return self.getCurve(gtImg, rect_TP_FP)

  def writeCurveFile(self, curve, path):
    if not os.path.isdir(path):
      os.mkdir(path)
    out_file = path + '/' + self.name + '_ROC_curve.csv'
    outFile = open(out_file, 'w')
    outFile.write('x, y\n')
    for i in range(len(curve)):
      outFile.write(str(curve[i][0]) + ', ' + str(curve[i][1]) + '\n')
    outFile.close()

  def ROCCurve(self):
    print("calculating " + self.name + " ROC curve")
    countList = self.sortHistByArea()
    gtImg = cv2.imread(self.gt_file, cv2.IMREAD_GRAYSCALE)
    curve = self.getROC(countList, gtImg)
    self.writeCurveFile(curve, 'csv')

  def getPos(self, file):
      posList = file.split('_')
      x = int(posList[1])
      y = int(posList[2])
      x1 = int(posList[3])
      y1 = int(posList[4])
      width = int(posList[5].split('.')[0])
      x += x1
      y += y1
      return x, y, width

  def testRocCurve(self, reverse):
    print("calculating " + self.name + " test ROC curve")
    gtImg = cv2.imread(self.gt_file, cv2.IMREAD_GRAYSCALE)
    self.myList.sort(key=self.sortList, reverse=reverse)
    rect_TP_FP = []
    for item in self.myList:
      value = item[0]
      file = item[1]
      x, y, width = self.getPos(file)
      TP = np.count_nonzero(gtImg[y:y+width, x:x+width] == 255)
      FP = np.count_nonzero(gtImg[y:y+width, x:x+width] == 0)
      rect_TP_FP.append([TP, FP])

    curve = self.getCurve(gtImg, rect_TP_FP)
    self.writeCurveFile(curve, 'test_csv')

  def drawRange(self, percent, reverse):
    print("drawing " + self.name + " range image")
    img = cv2.imread(self.origin_file)
    self.myList.sort(key=self.sortList, reverse=reverse)
    for index in range(int(len(self.myList) * percent)):
      item = self.myList[index]
      x, y, width = self.getPos(item[1])
      cv2.rectangle(img, (x, y), (x+width, y+width), (0, 0, 255), 3)
    if not os.path.isdir('rangePng'):
      os.mkdir('rangePng')
    cv2.imwrite('rangePng/' + self.name + '_' + str(percent) + '.png', img)
