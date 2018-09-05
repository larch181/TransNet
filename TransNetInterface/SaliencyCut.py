import tensorflow as tf
import numpy as np
import os
from scipy import misc
import argparse
from pydensecrf import densecrf as dcrf
import sys
import cv2
import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)

g_mean = np.array(([126.88, 120.24, 112.19])).reshape([1, 1, 3])

class SaliencyCut:
    def __init__(self):
        self.graph_path = 'models/salience_model/my-model.meta'
        self.model_path = 'models/salience_model'
        self.init_model()

    def init_model(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options = gpu_options))
        saver = tf.train.import_meta_graph( self.graph_path)
        saver.restore(self.sess, tf.train.latest_checkpoint(self.model_path))
        self.image_batch = tf.get_collection('image_batch')[0]
        self.pred_mattes = tf.get_collection('mask')[0]

    def segment(self,img,bbox):
        height, width, _ = img.shape
        factor = 0.51
        x1 = max(int(bbox[0] - factor * bbox[2] / 2), 0)
        y1 = max(int(bbox[1] - factor * bbox[3] / 2), 0)
        x2 = min(int(bbox[0] + bbox[2] + factor * bbox[2] / 2), height)
        y2 = min(int(bbox[1] + bbox[3] + factor * bbox[3] / 2), width)

        rgb = img[y1:y2, x1:x2, :].copy()
        if rgb.shape[2] == 4:
            rgb = self.rgba2rgb(rgb)
        origin_shape = rgb.shape[:2]
        rgb = np.expand_dims(
            misc.imresize(rgb.astype(np.uint8), [320, 320, 3], interp="nearest").astype(np.float32) - g_mean, 0)

        feed_dict = {self.image_batch: rgb}
        pred_alpha = self.sess.run(self.pred_mattes, feed_dict=feed_dict)
        final_alpha = misc.imresize(np.squeeze(pred_alpha), origin_shape)

        #final_alpha = self.crf_refine(img[y1:y2, x1:x2, :].copy(),final_alpha)
        mask = np.zeros(img.shape[:2], np.uint8)

        mask2 = np.where((final_alpha>50), 1, 0).astype('uint8')
        mask[y1:y2, x1:x2] = mask2
        final_alpha = mask*255
        return final_alpha,(x1,y1,x2,y2)
        #misc.imsave(os.path.join(output_folder, 'alpha.png'), final_alpha)

    # codes of this function are borrowed from https://github.com/Andrew-Qibin/dss_crf
    def crf_refine(self,img, annos):
        def _sigmoid(x):
            return 1 / (1 + np.exp(-x))

        assert img.dtype == np.uint8
        assert annos.dtype == np.uint8
        assert img.shape[:2] == annos.shape

        # img and annos should be np array with data type uint8

        EPSILON = 1e-8

        M = 2  # salient or not
        tau = 1.05
        # Setup the CRF model
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

        anno_norm = annos / 255.

        n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
        p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

        U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
        U[0, :] = n_energy.flatten()
        U[1, :] = p_energy.flatten()

        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

        # Do the inference
        infer = np.array(d.inference(1)).astype('float32')
        res = infer[1, :]

        res = res * 255
        res = res.reshape(img.shape[:2])
        return res.astype('uint8')

    def rgba2rgb(self, img):
        return img[:, :, :3] * np.expand_dims(img[:, :, 3], 2)

    def parse_arguments(argv):
        parser = argparse.ArgumentParser()

        parser.add_argument('--rgb', type=str,
                            help='input rgb', default='./test/animal2.jpg')
        parser.add_argument('--rgb_folder', type=str,
                            help='input rgb', default=None)
        parser.add_argument('--gpu_fraction', type=float,
                            help='how much gpu is needed, usually 4G is enough', default=0.5)
        return parser.parse_args(argv)


#if __name__ == '__main__':
   # main(parse_arguments(sys.argv[1:]))
