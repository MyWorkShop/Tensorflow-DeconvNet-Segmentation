import numpy as np
import os
import os.path
import tensorflow as tf
from tqdm import tqdm
from natsort import natsorted
import random
import time
import re
from pprint import pprint
import argparse

import matplotlib.pyplot as plt

global fig
fig = plt.figure()


def imshow(img, block=True, rgb=False):
    global fig
    plt.figure(fig.number)
    if not rgb:
        plt.imshow(img, interpolation='nearest')
    else:
        plt.imshow(img)
    plt.show(block=block)


def read_filelist(img_path, seg_path):
    print('Reading from: {}, \n\t{}'.format(img_path, seg_path))
    imgList = [
        os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(img_path) for f in files
        # if f.endswith('.png')
    ]
    imgList = natsorted(imgList)
    print("No of files: %i" % len(imgList))
    imgFiles = tf.train.string_input_producer(imgList, shuffle=False)

    segList = [
        os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(seg_path) for f in files
        # if f.endswith('.png')
    ]
    segList = natsorted(segList)
    print("No of files: %i" % len(segList))
    segFiles = tf.train.string_input_producer(segList, shuffle=False)

    return imgFiles, len(imgList), segFiles, len(segList)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def replace(v, org_val, new_val):
    comparison = tf.equal(v, org_val)
    return tf.where(comparison, tf.ones_like(v) * new_val, v)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--imgpath', help="path to images", default="data/images/")
    parser.add_argument(
        '--segpath',
        help="path to segmentations",
        default="data/segmentations/")
    parser.add_argument(
        '--outpath', help="path to write tfrecord", default="tfrecords/")
    parser.add_argument(
        '--rec', help="tfrecord file name to write", default="pascalvoc2012")
    parser.add_argument(
        '--crop', help="should we crop/pad images to fixed size?", default="y")
    args = parser.parse_args()

    # read file list from directory into queues
    iQ, imgLen, mQ, segLen = read_filelist(args.imgpath, args.segpath)

    reader = tf.WholeFileReader()
    key, ivalue = reader.read(iQ)
    key, mvalue = reader.read(mQ)
    myImg = tf.image.decode_png(ivalue)  #reads images into tensor
    mySeg = tf.image.decode_png(mvalue)  #reads images into tensor

    if args.crop == "y":
        print('Resizing images to 224x224')
        myImg = tf.image.resize_images(
            myImg, [96, 96], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        all_val = [
            0, 14, 19, 33, 37, 38, 52, 57, 72, 75, 89, 94, 108, 112, 113, 128,
            132, 147, 150, 220
        ]
        start = 0
        for i in all_val:
            mySeg=replace(mySeg, i, start)
            start += 1
        mySeg = tf.image.resize_images(
        mySeg, [96, 96], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    init_global = tf.global_variables_initializer()  # v0.12

    with tf.Session() as sess:

        # filename = os.path.join(args.outpath + args.rec + '.tfrecords')
        filename = os.path.join(args.outpath)
        print('Writing', filename)
        sess.run(init_global)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        writer = tf.python_io.TFRecordWriter(filename)
        myImg.eval()
        uniq = []
        for index in tqdm(range(imgLen)):  #(11648)
            image = myImg.eval()
            # imshow(image.reshape([96, 96, 3]))
            mask = mySeg.eval()
            # uniq += list(np.unique(mask))
            # uniq = list(np.unique(uniq))
            # print(uniq)
            # imshow(mask.reshape([96, 96]))
            imageRaw = image.tostring()
            maskRaw = mask.tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image_raw': _bytes_feature(imageRaw),
                        'mask_raw': _bytes_feature(maskRaw)
                    }))
            writer.write(example.SerializeToString())

        coord.request_stop()
        coord.join(threads)
