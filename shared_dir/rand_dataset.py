#!/usr/bin/python
"""
複数のディレクトリから
ランダムに抽出して
ロバストに対応した
1つの
データセットを
作成する
"""
from pathlib import Path
import argparse
import sys
import natsort
import random
import cv2
import os
import glob

def rand_ints_nodup(min, max, length):
  ns = []
  while len(ns) < length:
    n = random.randint(min, max)
    if not n in ns:
      ns.append(n)
  return ns

def main():
    black_list = []
    path = '/home/pytorch/PetImages/Dog'
    os.makedirs(str(Path(path + '/train')), exist_ok=True)
    os.makedirs(str(Path(path + '/val')), exist_ok=True)

    img_list = natsort.natsorted(Path(path).glob('*.jpg'))
    num = len(img_list)
    for i, img in enumerate(img_list):
        print(str(img))
        rgb = cv2.imread(str(img), 2)
        if i <= 10000:
            save_path = '/home/pytorch/PetImages/Dog/train/'
        else:
            save_path = '/home/pytorch/PetImages/Dog/val/'
        name = img.stem
        if i == 1308:
            print(rgb)
        if rgb is None:
            black_list.append(str(img))
        else:
            cv2.imwrite(save_path + name + '.jpg', rgb)
    print(black_list)
    print(len(black_list))

if __name__=='__main__':
    # args=parser()
    main()