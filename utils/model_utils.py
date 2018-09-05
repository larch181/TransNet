#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# by ruihui li
import tensorflow as tf
import os

def pre_load_checkpoint(checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # print(" [*] Reading checkpoint from {}".format(ckpt.model_checkpoint_path))
        print(ckpt.model_checkpoint_path)

        print(os.path.basename(ckpt.model_checkpoint_path))
        epoch_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        return epoch_step, ckpt.model_checkpoint_path
    else:
        return 0, None