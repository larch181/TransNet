import collections
import os
import sys
import time

import numpy as np
import pygraph.algorithms
import pygraph.algorithms.minmax
import pygraph.classes.graph
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy import spatial
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'dataProcessing'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
import tf_sampling
import ProcessUtils
import FileOperator
from glob import glob


def load_off(path):
    # Reset this mesh:
    verts = []
    faces = []
    # Open the file for reading:
    with open(path, 'r') as f:
        lines = f.readlines()

    # Read the number of verts and faces
    if lines[0] != 'OFF\n' and lines[0] != 'OFF\r\n':
        params = lines[0][3:].split()
        nVerts = int(params[0])
        nFaces = int(params[1])
        # split the remaining lines into vert and face arrays
        vertLines = lines[1:1 + nVerts]
        faceLines = lines[1 + nVerts:1 + nVerts + nFaces]
    else:
        params = lines[1].split()
        nVerts = int(params[0])
        nFaces = int(params[1])
        # split the remaining lines into vert and face arrays
        vertLines = lines[2:2 + nVerts]
        faceLines = lines[2 + nVerts:2 + nVerts + nFaces]

    # Create the verts array
    for vertLine in vertLines:
        XYZ = vertLine.split()
        verts.append([float(XYZ[0]), float(XYZ[1]), float(XYZ[2])])

    # Create the faces array
    for faceLine in faceLines:
        XYZ = faceLine.split()
        faces.append(verts[int(XYZ[1])] + verts[int(XYZ[2])] + verts[int(XYZ[3])])
    return np.asarray(faces)


def sampling_from_edge(edge):
    itervel = edge[:, 3:6] - edge[:, 0:3]
    point = np.expand_dims(edge[:, 0:3], axis=-1) + np.linspace(0, 1, 100) * np.expand_dims(itervel, axis=-1)
    point = np.transpose(point, (0, 2, 1))
    point = np.reshape(point, [-1, 3])
    return point


face_grid = []
for i in range(0, 11):
    for j in range(0, 11 - i):
        face_grid.append([i, j])
face_grid = np.asarray(face_grid) / 10.0


def sampling_from_face(face):
    points = []
    for item in face:
        pp = np.reshape(item, [3, 3])
        point = (1.0 - face_grid[:, 0:1] - face_grid[:, 1:2]) * pp[0, :] + face_grid[:, 0:1] * pp[1, :] + face_grid[:,
                                                                                                          1:2] * pp[2,
                                                                                                                 :]
        points.append(point)
    points = np.concatenate(points, axis=0)
    return points

def shuffle_data(data):
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    return data[idx, :],idx

class GKNN():
    def __init__(self, point_path, edge_path=None, mesh_path=None, patch_size=1024, patch_num=30, sampleRatio=[],
                 clean_point_path=None,
                 normalization=False, add_noise=False, usenormal=False):
        print(point_path, edge_path, mesh_path)
        self.name = point_path.split('/')[-1][:-4]
        self.wholedata,idx = shuffle_data(np.loadtxt(point_path))

        print(self.wholedata.shape)

        self.data = self.wholedata[:, 0:3]
        self.dataNormal = None
        if usenormal:
            self.dataNormal = self.wholedata[:, 3:6]
        self.sampleRatio = sampleRatio

        if  clean_point_path is not None:
            print("Use clean data to construct the graph", clean_point_path)
            self.clean_data = np.loadtxt(clean_point_path)[:, 0:3]
            self.clean_data = self.clean_data[idx,:]
            assert len(self.clean_data) == len(self.data)
        else:
            self.clean_data = self.data

        # self.data = self.data[np.random.permutation(len(self.data))[:100000]]
        self.centroid = np.mean(self.data, axis=0, keepdims=True)
        self.furthest_distance = np.amax(np.sqrt(np.sum((self.data - self.centroid) ** 2, axis=-1)), keepdims=True)

        if normalization:
            print("Normalize the point data")
            self.data = (self.data - self.centroid) / self.furthest_distance
            if clean_point_path is not None:
                self.clean_data = (self.clean_data - self.centroid) / self.furthest_distance
        if add_noise:
            print("Add gaussian noise into the point")
            self.data = ProcessUtils.jitter_perturbation_point_cloud(np.expand_dims(self.data, axis=0),
                                                                     sigma=self.furthest_distance * 0.004,
                                                                     clip=self.furthest_distance * 0.01)
            self.data = self.data[0]

        print("Total %d points" % len(self.data))

        if edge_path is not None:
            self.edge = np.loadtxt(edge_path)
            print("Total %d edges" % len(self.edge))
        else:
            self.edge = None
        if mesh_path is not None:
            self.face = load_off(mesh_path)
            print("Total %d faces" % len(self.face))
        else:
            self.face = None
        self.patch_size = patch_size
        self.patch_num = patch_num

        start = time.time()
        self.nbrs = spatial.cKDTree(self.clean_data)
        # dists,idxs = self.nbrs.query(self.clean_data,k=6,distance_upper_bound=0.2)
        dists, idxs = self.nbrs.query(self.clean_data, k=16)
        self.graph = []
        for item, dist in zip(idxs, dists):
            item = item[dist < 0.07]  # use 0.03 for chair7 model; otherwise use 0.05
            self.graph.append(set(item))
        print("Build the graph cost %f second" % (time.time() - start))

        self.graph2 = pygraph.classes.graph.graph()
        self.graph2.add_nodes(range(len(self.clean_data)))
        sid = 0
        for idx, dist in zip(idxs, dists):
            for eid, d in zip(idx, dist):
                if not self.graph2.has_edge((sid, eid)) and eid < len(self.clean_data):
                    self.graph2.add_edge((sid, eid), d)
            sid = sid + 1
        print("Build the graph cost %f second" % (time.time() - start))

        return

    def bfs_knn(self, seed=0, patch_size=1024):
        q = collections.deque()
        visited = set()
        result = []
        q.append(seed)
        while len(visited) < patch_size and q:
            vertex = q.popleft()
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                if len(q) < patch_size * 5:
                    q.extend(self.graph[vertex] - visited)
        return result

    def geodesic_knn(self, seed=0, patch_size=1024):
        _, dist = pygraph.algorithms.minmax.shortest_path(self.graph2, seed)
        dist_list = np.asarray([dist[item] if dist.has_key(item) else 10000 for item in xrange(len(self.data))])
        idx = np.argsort(dist_list)
        return idx[:patch_size]

    def estimate_single_density(self, id=0, patch_size=128):
        query_point = self.data[id]
        try:
            point = self.bfs_knn(id, patch_size=patch_size)
        except:
            return np.asarray([0])
        dist = np.sum((query_point - point) ** 2, axis=-1)
        avg_dist = np.sum(dist) / patch_size
        return avg_dist

    def estimate_density(self):
        self.density = []
        for id in tqdm(range(len(self.data))):
            dist = self.estimate_single_density(id)
            self.density.append(dist)
        self.density = np.asarray(self.density)
        plt.hist(self.density)
        plt.show()

    def get_seed_fromdensity(self, seed_num, idx=None):
        if idx is None:
            candidata_num = min(len(self.data), seed_num * 50)
            print("Total %d candidata random points" % candidata_num)
            idx = np.random.permutation(len(self.data))[:candidata_num]
        density = []
        for item in tqdm(idx):
            dist = self.estimate_single_density(item)
            density.append(dist)
        density = np.asarray(density)
        density = density * density
        density = density / np.sum(density)
        idx = np.random.choice(idx, size=seed_num, replace=False, p=density)
        return idx

    def get_idx(self, num, random_ratio=0.7):
        if self.edge != None and self.edge.shape[0] != 0 and random_ratio != 0.0:
            # select seed from the edge
            select_points = []
            prob = np.sqrt(np.sum(np.power(self.edge[:, 3:6] - self.edge[:, 0:3], 2), axis=-1))
            prob = prob / np.sum(prob)
            idx1 = np.random.choice(len(self.edge), size=int(1.5 * num * random_ratio), p=prob)
            for item in idx1:
                edge = self.edge[item]
                point = edge[0:3] + np.random.random() * (edge[3:6] - edge[0:3])
                select_points.append(point)
            select_points = np.asarray(select_points)
            dist, idx = self.nbrs.query(select_points, k=50)
            idx1 = idx[dist[:, -1] < 0.05, 0][:int(num * random_ratio)]
        else:
            idx1 = np.asarray([])

        # randomly select seed
        idx2 = np.random.randint(0, len(self.clean_data) - 20, [num - len(idx1), 1])
        idx2 = idx2 + np.arange(0, 20).reshape((1, 20))
        point = self.clean_data[idx2]
        point = np.mean(point, axis=1)
        _, idx2 = self.nbrs.query(point, k=10)

        return np.concatenate([idx1, idx2[:, 0]])

    def get_subedge(self, dist):
        threshold = 0.05
        # sort the  edge according to the distance to point
        dist = np.sort(dist, axis=0)
        second_dist = dist[5, :]
        idx = np.argsort(second_dist)
        second_dist.sort()
        subedge = self.edge[idx[second_dist < threshold]]
        # subedge = self.edge[np.sum(dist<threshold,axis=0)>=2]
        if len(subedge) == 0:
            print
            "No subedge"
            subedge = np.asarray([[10, 10, 10, 20, 20, 20]])
        return subedge

    def get_subface(self, dist):
        threshold = 0.03
        # sort the  edge according to the distance to point
        dist = np.sort(dist, axis=0)
        second_dist = dist[5, :]
        idx = np.argsort(second_dist)
        second_dist.sort()
        subface = self.face[idx[second_dist < threshold]]
        # subface = self.face[np.sum(dist<threshold,axis=0)>=2]
        return subface

    def crop_patch(self, save_root_path, use_dijkstra=True, loadseed=False):
        if save_root_path[-1] == '/':
            save_root_path = save_root_path[:-1]
        if not os.path.exists(save_root_path):
            os.makedirs(save_root_path)

        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = str(0)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        sample_data = np.expand_dims(self.data, axis=0)
        sample_seed = tf_sampling.farthest_point_sample(self.patch_num,sample_data)

        # with tf.Session(config=config) as sess:
        seeds = self.sess.run(sample_seed)
        # seeds = self.get_idx(self.patch_num, random_ratio=random_ratio)

        if loadseed:
            seeds = np.loadtxt(OUT_Seed_file)

        seeds = np.squeeze(seeds)
        np.savetxt(OUT_Seed_file,seeds,fmt='%d',delimiter="\n")
        id = -1
        for scale_ratio in self.sampleRatio:
            assert scale_ratio >= 1.0
            i = -1
            id = id + 1
            for seed in tqdm(seeds):
                t0 = time.time()
                i = i + 1
                # patch_size = self.patch_size*np.random.randint(1,scale_ratio+1)
                patch_size = int(self.patch_size * scale_ratio)
                try:
                    if use_dijkstra:
                        idx = self.geodesic_knn(seed, patch_size)
                    else:
                        idx = self.bfs_knn(seed, patch_size)
                except:
                    print("has exception")
                    continue


                point = self.data[idx,...]

                if self.dataNormal is not None:
                    outNormal = self.dataNormal[idx,...]
                    point = np.concatenate((point,outNormal),axis=1)

                idx = np.arange(point.shape[0])
                np.random.shuffle(idx)
                point = point[idx,...]
                #assert point.shape[0] == self.patch_size
                t2 = time.time()
                # print t1-t0, t2-t1
                print("patch:%d  point:%d " % (i, patch_size))
                np.savetxt('%s/%s_%d_%d.xyz' % (save_root_path, self.name, i,id), point, fmt='%0.6f')
        self.sess.close()


def postProcessing(dataFolder,outFolder,patch_size=1024,nomalization=True,usenormal=False):
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(0)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    samples = glob(dataFolder + "/*.xyz")
    print('File Num: ',len(samples))
    for i in range(len(samples)):
        filename = samples[i].split('/')[-1]
        print(filename)

        data = np.loadtxt(samples[i])
        point = data[:,:3]
        if usenormal:
            dataNormal = data[:,3:6]
        pointNum = point.shape[0]
        idx = np.arange(pointNum)
        if pointNum < patch_size:

            offset = pointNum - patch_size
            idx = np.concatenate([np.arange(pointNum), np.random.randint(pointNum, size=offset)], axis=0)
            np.random.shuffle(idx)

        elif pointNum > patch_size:

            sample_data = np.expand_dims(point, axis=0)
            sample_seed = tf_sampling.farthest_point_sample(patch_size, sample_data)

            # with tf.Session(config=config) as sess:
            idx = sess.run(sample_seed)

            idx = np.squeeze(idx)

        point = point[idx, ...]

        if nomalization:
            centroid = np.mean(point, axis=0, keepdims=True)
            furthest_distance = np.amax(np.sqrt(np.sum((point - centroid) ** 2, axis=-1)),
                                        keepdims=True)
            point = (point - centroid) / furthest_distance

        if usenormal:
            dataNormal = dataNormal[idx, ...]
            point = np.concatenate((point, dataNormal), axis=1)
        np.savetxt(os.path.join(outFolder,filename), point, fmt='%0.6f')

Noise_file = '/data/lirh/pointcloud/Dataset/Data/einstein@normal.xyz'
Clean_file = '/data/lirh/pointcloud/Dataset/Data/einstein_no_noise.xyz'
Raw_mesh = '/data/lirh/pointcloud/Dataset/Data/einstein_normalized.off'


PATCH_SIZE=1024
PATCH_NUM = 1024
SCALE_RATIO = [1.0,1.8]
HAS_NORMAL  = True

OUT_DIR = '/data/lirh/pointcloud/Dataset/data/segmentS2/'
OUT_Seed_file = os.path.join(OUT_DIR,'seeds.txt')
OUT_Processing_Folder = os.path.join(OUT_DIR,'processing')
OUT_PostProcessing_Folder = os.path.join(OUT_DIR,'postprocessing')

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
if not os.path.exists(OUT_Processing_Folder):
    os.makedirs(OUT_Processing_Folder)
if not os.path.exists(OUT_PostProcessing_Folder):
    os.makedirs(OUT_PostProcessing_Folder)



if __name__ == '__main__':
    gm = GKNN(Noise_file, None,Raw_mesh,patch_size=PATCH_SIZE, patch_num=PATCH_NUM, sampleRatio=SCALE_RATIO,usenormal=HAS_NORMAL,
              clean_point_path=Clean_file)
    t1 = time.time()
    gm.crop_patch(OUT_Processing_Folder, use_dijkstra=False,loadseed=False)
    t3 = time.time()
    print  "use is %f" % (t3 - t1)
    print('start to postprocessing')
    postProcessing(OUT_Processing_Folder,OUT_PostProcessing_Folder,patch_size=PATCH_SIZE,nomalization=True,usenormal=HAS_NORMAL)



