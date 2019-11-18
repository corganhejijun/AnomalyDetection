# -*- coding: utf-8 -*- 
import os
import cv2
import skimage
import tensorflow as tf
import numpy as np
import facenet
import shutil

from src.evaluate import Evaluator

USE_STEP = True
test_path = 'test'
ext = '.png'
hist_size = 128
model_path = "./models/20180402-114759"

save_dir = 'save_folder'
if os.path.isdir(save_dir):
  print(save_dir + " exists delete it.")
  shutil.rmtree(save_dir)
os.mkdir(save_dir)

out_file = 'hist_result.csv'
outFile = open(out_file, 'w')
outFile.write('filename,Correlation,Chi-Square,Intersection,Bhattacharyya,HELLINGER,SSIM,PSNR,fid\n')

hist1List = []
hist2List = []
hist3List = []
hist4List = []
hist5List = []
psnrList = []
ssimList = []
fidList = []

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
      outFile.write(file + ',' + str(dist1) + ',' + str(dist2) + ',' + str(dist3) + ',' + str(dist4) + ',' + str(dist5) 
                      + ',' + str(ssim) + ',' + str(psnr) + ',' + str(fid) + '\n')
      
      hist1List.append([dist1, file])
      hist2List.append([dist2, file])
      hist3List.append([dist3, file])
      hist4List.append([dist4, file])
      hist5List.append([dist5, file])
      psnrList.append([psnr, file])
      ssimList.append([ssim, file])
      fidList.append([fid, file])

outFile.close()

corrEva = Evaluator(hist1List, 'corr_hist', test_path, save_dir)
chiEva = Evaluator(hist1List, 'chi_hist', test_path, save_dir)
interEva = Evaluator(hist3List, 'inter_hist', test_path, save_dir)
bhattEva = Evaluator(hist4List, 'bhatt_hist', test_path, save_dir)
hellEva = Evaluator(hist5List, 'hell_hist', test_path, save_dir)
psnrEva = Evaluator(psnrList, 'psnr', test_path, save_dir)
ssimEva = Evaluator(ssimList, 'ssim', test_path, save_dir)
fidEva = Evaluator(fidList, 'fid', test_path, save_dir)

corrEva.saveFirstAndLast()
chiEva.saveFirstAndLast()
interEva.saveFirstAndLast()
bhattEva.saveFirstAndLast()
hellEva.saveFirstAndLast()
psnrEva.saveFirstAndLast()
ssimEva.saveFirstAndLast()
fidEva.saveFirstAndLast()

corrEva.saveHistCount()
chiEva.saveHistCount()
interEva.saveHistCount()
bhattEva.saveHistCount()
hellEva.saveHistCount()
psnrEva.saveHistCount()
ssimEva.saveHistCount()
fidEva.saveHistCount()

corrEva.ROCCurve()
chiEva.ROCCurve()
interEva.ROCCurve()
bhattEva.ROCCurve()
hellEva.ROCCurve()
psnrEva.ROCCurve()
ssimEva.ROCCurve()
fidEva.ROCCurve()