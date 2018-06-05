#ifndef __CUDA_RENDERER_H__
#define __CUDA_RENDERER_H__

#ifndef uint
#define uint unsigned int
#endif

#include <tuple>
#include <list>
#include <iostream>


typedef struct edge_tuple{
    int u;
    int v;
}edge_tuple_t;


class CudaTriangleCounter {

public:

    int numNodes;
    int node_list_size;
    int numEdges;

    int *node_list;
    int *list_len;
    int *start_addr;

    //edge_tuple_t *edge_list;

    int *cudaDeviceNodeList;
    int *cudaDeviceListLen;
    int *cudaDeviceStartAddr;
    //edge_tuple_t *cudaDeviceEdgeList;

    CudaTriangleCounter(char *);
    virtual ~CudaTriangleCounter();

    void setup();
    void countTriangles();
};


#endif
