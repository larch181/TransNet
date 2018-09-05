#import numpy as np
cimport numpy as np
cimport cython


def fetch_data(np.ndarray[int, ndim=1] point_removed, np.ndarray[tuple, ndim=1] vertices,np.ndarray[float, ndim=2] data):
    cdef int i
    cdef int count = 0
    for i in range(vertices.shape[0]):
        if point_removed[i]==1:
            # print(vertices[i][0],vertices[i][1],vertices[i][2])
            data[count,0] =  vertices[i][0]
            data[count,1] =  vertices[i][1]
            data[count,2] =  vertices[i][2]
            count = count + 1
    return count

def fetch_data2(np.ndarray[tuple, ndim=1] vertices,np.ndarray[float, ndim=2] data):
    cdef int i
    for i in range(vertices.shape[0]):
        # print(vertices[i][0],vertices[i][1],vertices[i][2])
        data[i,0]  =  -vertices[i][0]
        data[i,1]  =  vertices[i][1]
        data[i,2]  =  vertices[i][2]
    return vertices.shape[0]

def fetch_data_vert_frag(np.ndarray[tuple, ndim=1] vertices,np.ndarray[tuple, ndim=1] tex_coords,np.ndarray[float, ndim=2] _vertices,np.ndarray[float, ndim=2] _tex_coords):
    cdef int i
    for i in range(vertices.shape[0]):
        # print(vertices[i][0],vertices[i][1],vertices[i][2])
        _vertices[i,0]  =  vertices[i][0]
        _vertices[i,1]  =  vertices[i][1]
        _vertices[i,2]  =  vertices[i][2]
        _tex_coords[i,0] = tex_coords[i][0]
        _tex_coords[i,1] = tex_coords[i][1]


    return vertices.shape[0]



def fetch_data3(tuple vertices):

    return [vertices[0],vertices[1],vertices[2]]