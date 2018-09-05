// DataOperator.cpp : Defines the entry point for the console application.
//

#include "DataOperator.h"
#include <stdio.h>
#include <math.h>
#define SIZE 100000
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
static double distance[SIZE];
static int visit[SIZE];
double euclidean(double x1, double y1, double z1,double x2, double y2, double z2) {
	int i;
	double dis=0.0;
	dis += pow(x1 - x2,2);
	dis += pow(y1 - y2,2);
	dis += pow(z1 - z2,2);
	return dis;
}

int dist_point2list(int pt_id, double* points, int size) {
	int i = 0, index = 0;
	double dis_max = -INFINITY;
	for (;i<size;i++)
	{
	    if(visit[i]==0){
            distance[i] = MIN(distance[i], euclidean(points[pt_id*3+0],points[pt_id*3+1],points[pt_id*3+2], points[i*3+0],points[i*3+1],points[i*3+2]));
            if (distance[i] > dis_max) {
                dis_max = distance[i];
                index = i;
            }
		}
	}
	return index;
}

void init_distance() {
	int i;
	for (i=0;i<SIZE;i++)
	{
		distance[i] = INFINITY;
		visit[i] = 0;
	}
}


void farthest_point_sampling(double * points, int * out_index, int size, int K) {

	int i,index=0;
    double dis=0.0;
    init_distance();
    out_index[0]=0;
    distance[0] = 0.0;
    visit[0] = 1;
	for (i = 1; i < K; i++) {
		index = dist_point2list(index, points, size);
		out_index[i] = index;
		visit[index] = 1;
		distance[index] = 0.0;
	}
}

void proj_point2pixel(double* vertices, double* intrinsic, int* out_pixels,int size){

    double x=0.0;
    double y=0.0;
    int i=0;

    for(i=0;i<size;i++){

        x = intrinsic[0]*(-vertices[4*i+0]/vertices[4*i+2]) + intrinsic[2];
        y = intrinsic[1]*(vertices[4*i+1]/vertices[4*i+2]) + intrinsic[3];
        out_pixels[i*2] = (int)x;
        out_pixels[i*2+1] = (int)y;
    }

}


