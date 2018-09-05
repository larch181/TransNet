#ifndef _DATA_OPERATOR_H
#define _DATA_OPERATOR_H
double euclidean(double x1, double y1, double z1,double x2, double y2, double z2);
int dist_point2list(int index, double* points, int size);
void farthest_point_sampling(double * points, int * out_index, int size, int K);
#endif