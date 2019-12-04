# AnomalyDetection
anomaly detection

## 1. training
set the normal sample filename into "train.jpg"

python train.py --phase train

## 2. generate sample for anormaly detection
set the sample to be detected into "train.jpg"

python cropImg.py

## 3. generate test sample for detection
python train.py --phase test

## 4. detection procedure
python train.py --phase cal

## 5. get the comparition fid psnr and ssim results
python histcal.py

## 6. draw ROC curve and calculate AUC value
matlab showCsv.m
