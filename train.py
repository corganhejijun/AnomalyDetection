# -*- coding: utf-8 -*- 
import argparse
import os

from src.model import ScaleGan
import tensorflow as tf

parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', default='train', help='phase')
parser.add_argument('--dataname', dest='dataname', default='train.jpg', help='name of the data')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--train_size', dest='train_size', type=int, default=500, help='# images used to train')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')

args = parser.parse_args()

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
        
    with tf.Session() as sess:
        model = ScaleGan(sess, dataname=args.dataname)

        if args.phase == 'train':
            model.train(args)
        else:
            model.test(args)

if __name__ == '__main__':
    tf.app.run()