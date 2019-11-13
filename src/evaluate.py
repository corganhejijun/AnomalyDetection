# -*- coding: utf-8 -*- 
import os
import cv2
from shutil import copyfile

class Evaluator:
  def __init__(self, myList, name, test_path, save_dir):
    self.USE_AVERAGE = True
    self.HIST_COUNT_COUNT = 5
    self.fine_size = 128
    self.SAVE_FILE_COUNT = 30
    self.save_dir = save_dir
    self.myList = myList
    self.name = name
    self.test_path = test_path
    self.origin_file = 'train.jpg'
    self.gt_file = 'gt.jpg'

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
    for index, item in enumerate(self.myList):
      value = item[0]
      file = item[1]
      posList = file.split('_')
      pos = posList[1] + '_' + posList[2]
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
      head = countList[i][1].split('_')
      tail = countList[-i-1][1].split('_')
      x = int(head[0])
      y = int(head[1])
      cv2.rectangle(img, (x, y), (x+self.fine_size, y+self.fine_size), (0, 0, 255), 3)
      if self.USE_AVERAGE:
        cv2.putText(img, "{}:{:.5f}".format(i, countList[i][0]), (x, y+int(self.fine_size/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
      else:
        cv2.putText(img, "{}:{:.5f}".format(i, countList[i][2]), (x, y+int(self.fine_size/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
      x = int(tail[0])
      y = int(tail[1])
      cv2.rectangle(img, (x, y), (x+self.fine_size, y+self.fine_size), (255, 0, 0), 3)
      if self.USE_AVERAGE:
        cv2.putText(img, "{}:{:.5f}".format(i, countList[-i-1][0]), (x, y+int(self.fine_size/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
      else:
        cv2.putText(img, "{}:{:.5f}".format(i, countList[-i-1][2]), (x, y+int(self.fine_size/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imwrite(self.name + '.png', img)

  def saveHistCount(self):
    print("saveing " + self.name + " count list")
    countList = self.sortHistByArea()
    self.saveHistCountImage(countList)

  def insideRect(self, pos, rect):
    x = pos[0]
    y = pos[1]
    top = rect[0]
    bottom = rect[1]
    left = rect[2]
    right = rect[3]
    if x >= left and x <= right and y >= top and y <= bottom:
      return True
    return False

  def insideList(self, gtX, gtY, count, head, countList):
    for i in range(count):
      x = y = None
      if head:
        head = countList[i][1].split('_')
        x = int(head[0])
        y = int(head[1])
      else:
        tail = countList[-i-1][1].split('_')
        x = int(tail[0])
        y = int(tail[1])
      if self.insideRect([gtX, gtY], [y, y+ self.fine_size, x, x + self.fine_size]):
        return True
    return False

  def getROC(self, countList, count, head, gtImg):
    # explain
    # https://www.cnblogs.com/dlml/p/4403482.html
    TP = FN = FP = TN = 0
    for i in range(gtImg.shape[0]):
      for j in range(gtImg.shape[1]):
        if gtImg[i, j] == 255:
          if self.insideList(j, i, count, head, countList):
            TP += 1
          else:
            FN += 1
        if gtImg[i, j] == 0:
          if self.insideList(j, i, count, head, countList):
            FP += 1
          else:
            TN += 1
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return [FPR, TPR]

  def writeCurveFile(self, curve):
    out_file = self.name + '_ROC_curve.csv'
    outFile = open(out_file, 'w')
    outFile.write('x, y\n')
    for i in range(len(curve)):
      outFile.write(curve[i][0] + ', ' + curve[i][1])
    outFile.close()

  def ROCCurve(self):
    print("calculating " + self.name + " ROC curve")
    countList = self.sortHistByArea()
    gtImg = cv2.imread(self.gt_file, cv2.IMREAD_GRAYSCALE)
    curve = []
    for i in range(len(countList)):
      point = self.getROC(countList, i, True, gtImg)
      curve.append(point)
    self.writeCurveFile(curve)