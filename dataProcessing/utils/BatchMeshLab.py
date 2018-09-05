#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# by ruihui li

import os


ROOT_FOLDER = 'E:/DeepLearning/PointCloud/Dataset/Data/'
MeshLab_BIN = 'E:/DailySoftWare/MeshLab/meshlabserver.exe'

MLX_File = 'script/'


def get_xyzNormal_command(input, output,script):
    cmd = MeshLab_BIN + ' -i ' + input
    cmd += ' -o ' + output
    cmd += ' -m vc vn '
    cmd += ' -s  ' + MLX_File+script
    return cmd


if __name__ == '__main__':

    input = ROOT_FOLDER+'einstein_noise.xyz'
    output = ROOT_FOLDER+'einstein@normal.xyz'
    script = 'calNormal.mlx'
    cmd = get_xyzNormal_command(input,output,script)
    os.system(cmd)
    print()
