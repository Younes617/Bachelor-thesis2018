/*
 * Triangle counter with workload balancing
 *
 * @author: Manish Jain
 * @author: Vashishtha Adtani
 */

#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <thrust/scan.h>                                                        
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "cudaTriangleCounter.h"

#define BLOCK_SIZE 32

struct GlobalConstants {

    int *NodeList;
    int *ListLen;
    int numNodes;
    int numEdges;
};

__constant__ GlobalConstants cuConstCounterParams;

void
CudaTriangleCounter::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CountingTriangles\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);

    }
    // printf("---------------------------------------------------------\n");

    // By this time the graph should be loaded.  Copying graph to 
    // data structures into device memory so that it is accessible to
    // CUDA kernels
    //

    cudaMalloc(&cudaDeviceListLen, sizeof(int ) * numNodes);
    cudaMemcpy(cudaDeviceListLen, list_len, sizeof(int) * numNodes, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&cudaDeviceNodeList, node_list_size * sizeof(int));
    cudaMemcpy(cudaDeviceNodeList, node_list, sizeof(int) * node_list_size, cudaMemcpyHostToDevice);

    GlobalConstants params;
    params.ListLen = cudaDeviceListLen;
    params.NodeList = cudaDeviceNodeList;
    params.numNodes = numNodes;
    params.numEdges = numEdges;
    cudaMemcpyToSymbol(cuConstCounterParams, &params, sizeof(GlobalConstants));
}

CudaTriangleCounter::CudaTriangleCounter(char *fileName) {
    clock_t start, diff, malloc_diff;
    int node, edge_id, temp = 0;
    int total_nodes = 0;
    int total_edges = 0;
    int msec;

    std::string line;
    std::ifstream myfile;
    myfile.open(fileName);

    std::string token;                                                             
    if (strstr(fileName,"new_orkut") != NULL) {                                    
        printf("This is the NEW_ORKUT FILE **\n");                             
        total_nodes = 3072600;                                                     
        total_edges = 117185083 + 1;                                               
    } else {                                                                       
        std::getline(myfile,line);                                                 
        std::stringstream lineStream(line);                                        
        while (lineStream >> token) {                                              
            if (temp == 0) {                                                       
                total_nodes = std::stoi(token, NULL, 10) + 1;                      
            } else if (temp == 1) {                                                
                total_edges = std::stoi(token, NULL, 10) + 1;                      
            } else {                                                               
                printf("!!!!!!!!!!!! TEMP IS %d\n ", temp);                        
                break;                                                             
            }                                                                      
            temp++;                                                                
        }                                                                          
    }

    start = clock();

    numNodes = total_nodes;
    node_list_size = total_edges * 2;
    numEdges = total_edges;

    printf("total_nodes %d\n", total_nodes);
    printf("node_list_size %d\n", node_list_size);
    printf("numEdges %d\n", numEdges);

    list_len = (int *)calloc(total_nodes, sizeof(int));
    start_addr = (int *)calloc(total_nodes, sizeof(int));
    node_list = (int *)calloc(node_list_size, sizeof(int));

    malloc_diff = clock() - start;
    msec = malloc_diff * 1000 / CLOCKS_PER_SEC;

    printf("memory allocated ......\n");
    node = 1;
    temp = 1;
    int neighbors;
    while(std::getline(myfile, line)) {
        neighbors = 0;
        std::stringstream lineStream(line);
        std::string token;
        while(lineStream >> token)
        {
            edge_id = std::stoi(token, NULL, 10);
            if (edge_id > node) {
                node_list[temp++] = edge_id;
                neighbors++;
            }
        }

        list_len[node] = neighbors;
        node++;
    }

    printf("graph created......\n");
    diff = clock() - start;
    msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);

    myfile.close();
}

CudaTriangleCounter::~CudaTriangleCounter() {

    free(node_list);
    free(list_len);
}


/*
 * Kernel to count number of triangles formed by a single edge. And store the count
 * in an array on which we will run reduction later to find total number of triangles
 * in the given graph.
 */
__global__ void countTriangleKernel(int *countArray, edge_tuple_t *compressed_list, int *start_addr, int num) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= num) {
        return;
    }

    if (i == 0) {
        countArray[i] = 0;
        return;
    }

    int j = 0, k = 0, count=0;
    int *node_list = cuConstCounterParams.NodeList;
    int *list_len = cuConstCounterParams.ListLen;
    edge_tuple_t *edgeList = compressed_list;

    int u = edgeList[i].u;
    int v = edgeList[i].v;

    /* Fetching neigbour vertices from the node list */
    int *list1 = node_list + start_addr[u-1] + 1;
    int len1 = list_len[u];

    int *list2 = node_list + start_addr[v-1] + 1;
    int len2 = list_len[v];

    /* 
     * Traversing both lists to find the common nodes. Each common node
     * will be counted as a triangle
     */
    while ( j < len1 && k < len2) {

        if (list1[j] == list2[k]) {
            count++;
            j++;
            k++;
        } else if (list1[j] < list2[k]) {
            j++;
        } else {
            k++;
        }
    }

    countArray[i] = count;
}


/*
 * Creating data structure which stores all the edges
 */
__global__ void createEdgeList(edge_tuple_t *edge_list, int *start_addr) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= cuConstCounterParams.numNodes) {
        return;
    }

    if (i == 0) {
        return;
    }

    int *node_list = cuConstCounterParams.NodeList;
    int *list_len = cuConstCounterParams.ListLen;
    int start_index = start_addr[i-1] + 1;
    int *list = node_list + start_addr[i-1] + 1;
    int len = list_len[i];

    for (int j=0; j<len; j++) {
        edge_list[start_index].u = i;
        edge_list[start_index].v = list[j];
        start_index++;
    }
}

#define THRESHOLD 50000

__global__ void segregateList(edge_tuple_t *edge_list, int *small_edge, int *large_edge) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int *list_len = cuConstCounterParams.ListLen;

    if ( i >= cuConstCounterParams.numEdges) {
        return;
    }

    if (i == 0) {
        large_edge[i] = 0;
        small_edge[i] = 0;
        return;
    }

    int u = edge_list[i].u;
    int v = edge_list[i].v;

    if ((list_len[u] > THRESHOLD) || (list_len[v] > THRESHOLD)) {
        large_edge[i] = 1;
        small_edge[i] = 0;
    } else {
        large_edge[i] = 0;
        small_edge[i] = 1;
    }
}

__global__ void createSmallList(edge_tuple_t *edge_list, edge_tuple_t *small_edge_list, int *small_edge) { 
 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 

    if ( i >= (cuConstCounterParams.numEdges)) {
        return;
    }

    if (small_edge[i] != small_edge[i+1]) {
        int index = small_edge[i];
        small_edge_list[index] = edge_list[i];
    }
}

__global__ void createLargeList(edge_tuple_t *edge_list, edge_tuple_t *large_edge_list, int *large_edge) { 
 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 

    if ( i >= (cuConstCounterParams.numEdges-1)) {
        return;
    }

    if (large_edge[i] != large_edge[i+1]) {
        int index = large_edge[i];
        large_edge_list[index] = edge_list[i];
    }
}


/*
 * Counts the number of triangles in the given graph. We first find out the
 * starting address of each list where list stores the neighbours of particular
 * node. We then create the list of all edges from the given nodes and their
 * neighbours.
 */
void
CudaTriangleCounter::countTriangles() {

    dim3 blockdim  = BLOCK_SIZE;
    dim3 griddim = (numEdges + BLOCK_SIZE)/BLOCK_SIZE;
    dim3 griddim1 = (numNodes + BLOCK_SIZE)/BLOCK_SIZE;
    int count;
    edge_tuple_t *edge_list, *small_edge_list, *large_edge_list;
    int *small_edge, *large_edge;
    int num_small_edges, num_large_edges;
    int *temp;

    /* Calculating start address of each neighbour list */
    cudaMalloc(&cudaDeviceStartAddr, sizeof(int ) * numNodes);
    thrust::device_ptr<int> dev_ptr1(cudaDeviceListLen);
    thrust::device_ptr<int> output_ptr(cudaDeviceStartAddr);
    thrust::inclusive_scan(dev_ptr1, dev_ptr1 + numNodes, output_ptr);

    /* Create a list of all edges present in the graph */
    cudaMalloc((void **)&edge_list, numEdges * sizeof(edge_tuple_t));
    createEdgeList<<<griddim1, blockdim>>>(edge_list, cudaDeviceStartAddr);
    cudaDeviceSynchronize();

    cudaMalloc(&small_edge, sizeof(int ) * numEdges);
    cudaMalloc(&large_edge, sizeof(int ) * numEdges);

    segregateList<<<griddim, blockdim>>>(edge_list, small_edge, large_edge);
    cudaDeviceSynchronize();

    thrust::device_ptr<int> small_ptr(small_edge);
    thrust::inclusive_scan(small_ptr, small_ptr + numEdges, small_ptr);

    thrust::device_ptr<int> large_ptr(large_edge);
    thrust::inclusive_scan(large_ptr, large_ptr + numEdges, large_ptr);

    temp = (int *) malloc (numEdges * sizeof(int));
    cudaMemcpy(temp, small_edge, sizeof(int) * numEdges, cudaMemcpyDeviceToHost);

    cudaMemcpy(&num_small_edges, &small_edge[numEdges-1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&num_large_edges, &large_edge[numEdges-1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMalloc((void **)&small_edge_list, ( 1 +num_small_edges) * sizeof(edge_tuple_t));
    cudaMalloc((void **)&large_edge_list, ( 1 + num_large_edges) * sizeof(edge_tuple_t));

    createSmallList<<<griddim, blockdim>>>(edge_list, small_edge_list, small_edge);
    cudaDeviceSynchronize();

    createLargeList<<<griddim, blockdim>>>(edge_list, large_edge_list, large_edge);
    cudaDeviceSynchronize();

    int *countArraySmall, *countArrayLarge;
    
    cudaMalloc((void **)&countArraySmall, (2 + num_small_edges) * sizeof(int));
    cudaMalloc((void **)&countArrayLarge, (2 + num_large_edges) * sizeof(int));

    dim3 griddim2 = (num_small_edges + 1 + BLOCK_SIZE)/BLOCK_SIZE;

    /* Applying intersection rule on all small edges to find number of triangles */
    countTriangleKernel<<<griddim2, blockdim>>>(countArraySmall, small_edge_list, cudaDeviceStartAddr, num_small_edges+1);
    cudaDeviceSynchronize();

    thrust::device_ptr<int> dev_ptr2(countArraySmall);
    thrust::inclusive_scan(dev_ptr2, dev_ptr2 + num_small_edges+1, dev_ptr2);

    int count1, count2;
    cudaMemcpy(&count1, &countArraySmall[num_small_edges], sizeof(int), cudaMemcpyDeviceToHost);


    dim3 griddim3 = (num_large_edges + 1 + BLOCK_SIZE)/BLOCK_SIZE;

    /* Applying intersection rule on all large edges to find number of triangles */
    countTriangleKernel<<<griddim3, blockdim>>>(countArrayLarge, large_edge_list, cudaDeviceStartAddr, num_large_edges+1);
    cudaDeviceSynchronize();

    thrust::device_ptr<int> dev_ptr3(countArrayLarge);
    thrust::inclusive_scan(dev_ptr3, dev_ptr3 + num_large_edges + 1, dev_ptr3);
    cudaMemcpy(&count2, &countArrayLarge[num_large_edges], sizeof(int), cudaMemcpyDeviceToHost);

    count = count1 + count2;
    printf("count %d\n", count);
}

