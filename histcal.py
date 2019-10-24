# -*- coding: utf-8 -*- 
import os
import cv2
import skimage
from shutil import copyfile

USE_STEP = True
test_file = 'train'
test_path = 'test'
ext = '.png'
hist_size = 128
out_file = 'hist_result.csv'
SAVE_FILE_COUNT = 30
save_dir = 'save_folder'
os.mkdir(save_dir)
HIST_COUNT_COUNT = 5
origin_file = 'train.jpg'
fine_size = 128

outFile = open(out_file, 'w')
outFile.write('filename,Correlation,Chi-Square,Intersection,Bhattacharyya,HELLINGER,SSIM,PSNR\n')

hist1List = []
hist2List = []
hist3List = []
hist4List = []
hist5List = []
psnrList = []
ssimList = []

def sortList(item):
  return item[0]

def saveList(l, dest):
  for item in l:
    filename = item[1]
    copyfile(os.path.join(test_path, filename), os.path.join(dest, filename))

def saveFirstAndLast(l, name):
  l.sort(key=sortList)
  os.mkdir(os.path.join(save_dir, name + '_top'))
  saveList(l[:SAVE_FILE_COUNT], os.path.join(save_dir, name + '_top'))
  os.mkdir(os.path.join(save_dir, name + '_last'))
  saveList(l[-SAVE_FILE_COUNT:], os.path.join(save_dir, name + '_last'))

def saveHistCount(l, name):
  countList = []
  for index, item in enumerate(l):
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
      countList[found][0] += index
    else:
      countList.append([index, pos])
  countList.sort(key=sortList)
  img = cv2.imread(origin_file, cv2.IMREAD_COLOR)
  for i in range(HIST_COUNT_COUNT):
    head = countList[i][1].split('_')
    tail = countList[-i-1][1].split('_')
    x = int(head[0])
    y = int(head[1])
    cv2.rectangle(img, (x, y), (x+fine_size, y+fine_size), (0, 0, 255), 3)
    cv2.putText(img, str(i), (x, y+int(fine_size/2)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 4)
    x = int(tail[0])
    y = int(tail[1])
    cv2.rectangle(img, (x, y), (x+fine_size, y+fine_size), (255, 0, 0), 3)
    cv2.putText(img, str(i), (x, y+int(fine_size/2)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 4)
  cv2.imwrite(name+'.png', img)

fileList = os.listdir(test_path)
for index, file in enumerate(fileList):
  if not file.endswith(ext):
    continue
  print("processing " + file + " " + str(index) + ' of total ' + str(len(fileList)))
  nameValue = file.split('.')[0].split('_')
  x = int(nameValue[3])
  y = int(nameValue[4])
  img = cv2.imread(os.path.join(test_path, file), cv2.IMREAD_GRAYSCALE)
  height = img.shape[1]
  if USE_STEP:
    step = int(nameValue[5])
    origin = img[y+height:y+step+height, x:x+step]
    imgOut = img[y:y+step, x:x+step]
  else:
    origin = img[height:, :]
    imgOut = img[:height,:]
  originHist = cv2.calcHist([origin], [0], None, [hist_size], [0.0, 255.0])
  imgHist = cv2.calcHist([imgOut], [0], None, [hist_size], [0.0, 255.0])
  dist1 = cv2.compareHist(originHist, imgHist, cv2.HISTCMP_CORREL)
  dist2 = cv2.compareHist(originHist, imgHist, cv2.HISTCMP_CHISQR)
  dist3 = cv2.compareHist(originHist, imgHist, cv2.HISTCMP_INTERSECT)
  dist4 = cv2.compareHist(originHist, imgHist, cv2.HISTCMP_BHATTACHARYYA)
  dist5 = cv2.compareHist(originHist, imgHist, cv2.HISTCMP_HELLINGER)
  psnr = skimage.measure.compare_psnr(origin, imgOut)
  ssim = skimage.measure.compare_ssim(origin, imgOut)
  outFile.write(file + ',' + str(dist1) + ',' + str(dist2) + ',' + str(dist3) + ',' + str(dist4) + ',' + str(dist5) 
                  + ',' + str(ssim) + ',' + str(psnr) + '\n')
  
  hist1List.append([dist1, file])
  hist2List.append([dist2, file])
  hist3List.append([dist3, file])
  hist4List.append([dist4, file])
  hist5List.append([dist5, file])
  psnrList.append([psnr, file])
  ssimList.append([ssim, file])

print("saveing corr_hist")
saveFirstAndLast(hist1List, 'corr_hist')
print("saveing chi_hist")
saveFirstAndLast(hist2List, 'chi_hist')
print("saveing inter_hist")
saveFirstAndLast(hist3List, 'inter_hist')
print("saveing bhatt_hist")
saveFirstAndLast(hist4List, 'bhatt_hist')
print("saveing hell_hist")
saveFirstAndLast(hist5List, 'hell_hist')
print("saveing psnr_hist")
saveFirstAndLast(psnrList, 'psnr_hist')
print("saveing ssim_hist")
saveFirstAndLast(ssimList, 'ssim_hist')

print("saveing corr_hist count list")
saveHistCount(hist1List, 'corr_hist')
print("saveing chi_hist count list")
saveHistCount(hist2List, 'chi_hist')
print("saveing inter_hist count list")
saveHistCount(hist3List, 'inter_hist')
print("saveing bhatt_hist count list")
saveHistCount(hist4List, 'bhatt_hist')
print("saveing hell_hist count list")
saveHistCount(hist5List, 'hell_hist')
print("saveing psnr_hist count list")
saveHistCount(psnrList, 'psnr_hist')
print("saveing ssim_hist count list")
saveHistCount(ssimList, 'ssim_hist')
