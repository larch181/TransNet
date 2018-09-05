#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# by ruihui li

import os
from functools import reduce
import shutil

def file_filter(file_dir, extension):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == extension:
                L.append(os.path.join(root, file))
    return L

def batch_rename(filedir):
    idx = 0
    for f in sorted([x for x in os.listdir(filedir)]):
        os.rename(os.path.join(filedir,f),os.path.join(filedir,'%06d.'%idx+f.rsplit('.')[1]))
        idx+=1

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(root)  # 当前目录路径
        print(dirs)  # 当前路径下所有子目录
        print(files)  # 当前路径下所有非目录子文件



def str2int(s):
    def fn(x, y):
        return x * 10 + y

    def char2num(s):
        return {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}[s]

    return reduce(fn, map(char2num, s))


def get_train_data(fileDir):
    L = file_filter(fileDir, '.xyz')
    lable_num = -1
    type_num = -1
    for file in L:
        lable_num = max(lable_num, str2int(file.split('_')[-2]))
        type_num = max(type_num, str2int(os.path.splitext(file)[0].split('_')[-1]))

    type_num = type_num + 1
    lable_num = lable_num + 1
    fileName = L[0].split('_')[0]
    return type_num,lable_num,fileName


def CleanDir( Dir ):
    if os.path.isdir( Dir ):
        paths = os.listdir( Dir )
        for path in paths:
            filePath = os.path.join( Dir, path )
            if os.path.isfile( filePath ):
                    os.remove( filePath )
            elif os.path.isdir( filePath ):
                if filePath[-4:].lower() == ".svn".lower():
                    continue
                shutil.rmtree(filePath,True)
    return True


if __name__ == '__main__':
    dir = 'E:/DeepLearning/PointCloud/Dataset/Data/segment/'
    type_num, lable_num, fileName = get_train_data(dir)
    file = (fileName+'_%d_%d.xyz')%(0,0)
    print(type_num, lable_num, file)
