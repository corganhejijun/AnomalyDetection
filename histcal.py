# -*- coding: utf-8 -*- 
import os
import cv2
import skimage
import tensorflow as tf
import numpy as np
import facenet
import shutil
from sklearn import metrics as mr

from src.evaluate import Evaluator

USE_STEP = True
test_path = 'test'
ext = '.png'
hist_size = 128
model_path = "./models/20180402-114759"
fine_size = 128

save_dir = 'save_folder'
if os.path.isdir(save_dir):
  print(save_dir + " exists delete it.")
  shutil.rmtree(save_dir)
os.mkdir(save_dir)

models = ["corr_hist", "chi_hist", "inter_hist", "bhatt_hist", 
          "hell_hist", "psnr", "ssim", "fid", "mututal"]

modelList = []
out_file = 'hist_result.csv'
outFile = open(out_file, 'w')
title = ""
for item in models:
  title += ", " + item
  modelList.append([])
outFile.write('filename' + title + '\n')

def calFID(img1, img2, sess):
  img1 = cv2.resize(img1, (160,160))
  img2 = cv2.resize(img2, (160,160))
  imga = np.zeros((img1.shape[0], img1.shape[1], 3))
  imga[:, : ,0] = imga[:, :, 1] = imga[:, :, 2] = img1
  imgb = np.zeros((img1.shape[0], img1.shape[1], 3))
  imgb[:, : ,0] = imgb[:, :, 1] = imgb[:, :, 2] = img2
  images = np.stack([imga, imgb])

  images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
  embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
  phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

  feed_dict = { images_placeholder: images, phase_train_placeholder:False }
  emb = sess.run(embeddings, feed_dict=feed_dict)
  dist = np.sqrt(np.sum(np.square(np.subtract(emb[0,:], emb[1,:]))))
  return dist

fileList = os.listdir(test_path)

with tf.Graph().as_default():
  with tf.Session() as sess:
    facenet.load_model(model_path)
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
      fid = calFID(origin, imgOut, sess)
      mutual = mr.mutual_info_score(np.reshape(origin, -1), np.reshape(imgOut, -1))
      outFile.write(file + ',' + str(dist1) + ',' + str(dist2) + ',' + str(dist3) + ',' + str(dist4) + ',' + str(dist5) 
                      + ',' + str(ssim) + ',' + str(psnr) + ',' + str(fid) + ',' + str(mutual) + '\n')
      
      modelList[0].append([dist1, file])
      modelList[1].append([dist2, file])
      modelList[2].append([dist3, file])
      modelList[3].append([dist4, file])
      modelList[4].append([dist5, file])
      modelList[5].append([psnr, file])
      modelList[6].append([ssim, file])
      modelList[7].append([fid, file])
      modelList[8].append([mutual, file])

outFile.close()

evaluatorList = []
for index in range(len(models)):
  if models[index] == 'fid':
    evaluatorList.append(Evaluator(modelList[index], models[index], test_path, save_dir, fine_size, False))
  else:
    evaluatorList.append(Evaluator(modelList[index], models[index], test_path, save_dir, fine_size))

for item in evaluatorList:
  item.saveFirstAndLast()
  item.saveHistCount()
  item.ROCCurve()
  item.testRocCurve()
