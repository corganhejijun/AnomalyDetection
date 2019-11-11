# -*- coding: utf-8 -*- 
import os
import cv2
from shutil import copyfile

class Evaluator:
  def __init__(self, myList, name, test_path):
    self.USE_AVERAGE = True
    self.HIST_COUNT_COUNT = 5
    self.fine_size = 128
    self.SAVE_FILE_COUNT = 30
    self.save_dir = 'save_folder'
    self.myList = myList
    self.name = name
    self.test_path = test_path
    os.mkdir(save_dir)
    self.origin_file = 'train.jpg'

  def sortList(self, item):
    return item[0]

  def saveList(self, dest):
    for item in self.myList:
      filename = item[1]
      copyfile(os.path.join(self.test_path, filename), os.path.join(dest, filename))

  def saveFirstAndLast(self):
    print("saveing " + self.name)
    self.myList.sort(key=self.sortList)
    os.mkdir(os.path.join(self.save_dir, self.name + '_top'))
    self.saveList(self.myList[:self.SAVE_FILE_COUNT], os.path.join(self.save_dir, self.name + '_top'), self.test_path)
    os.mkdir(os.path.join(self.save_dir, self.name + '_last'))
    self.saveList(self.myList[-self.SAVE_FILE_COUNT:], os.path.join(self.save_dir, self.name + '_last'), self.test_path)

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
    for i in range(HIST_COUNT_COUNT):
      head = countList[i][1].split('_')
      tail = countList[-i-1][1].split('_')
      x = int(head[0])
      y = int(head[1])
      cv2.rectangle(img, (x, y), (x+self.fine_size, y+self.fine_size), (0, 0, 255), 3)
      if USE_AVERAGE:
        cv2.putText(img, "{}:{:.5f}".format(i, countList[i][0]), (x, y+int(self.fine_size/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
      else:
        cv2.putText(img, "{}:{:.5f}".format(i, countList[i][2]), (x, y+int(self.fine_size/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
      x = int(tail[0])
      y = int(tail[1])
      cv2.rectangle(img, (x, y), (x+self.fine_size, y+self.fine_size), (255, 0, 0), 3)
      if USE_AVERAGE:
        cv2.putText(img, "{}:{:.5f}".format(i, countList[-i-1][0]), (x, y+int(self.fine_size/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
      else:
        cv2.putText(img, "{}:{:.5f}".format(i, countList[-i-1][2]), (x, y+int(self.fine_size/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imwrite(self.name + '.png', img)

  def saveHistCount():
    print("saveing " + self.name + " count list")
    countList = sortHistByArea(self.myList)
    saveHistCountImage(countList)