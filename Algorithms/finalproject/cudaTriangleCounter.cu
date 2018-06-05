/*
 * Triangle counter without workload balancing
 *
 * @author: Manish Jain
 * @author: Vashishtha Adtani
 *
 * edited by: Younes Ouazref 
 *
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

#define BLOCK_SIZE 112

using namespace std;


float firstAllocation = 0;


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

    // printf("---------------------------------------------------------\n");
    // printf("Initializing CUDA for CountingTriangles\n");
    // printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        // printf("Device %d: %s\n", i, deviceProps.name);
        // printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        // printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        // printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);

    }
    // printf("---------------------------------------------------------\n");

    // By this time the graph should be loaded.  Copying graph to 
    // data structures into device memory so that it is accessible to
    // CUDA kernels
    //


    // Create cuda events to keep track of the GPU time
    cudaEvent_t Malloc1Start, Malloc1Stop;
    cudaEventCreate(&Malloc1Start);
    cudaEventCreate(&Malloc1Stop);
    // ----------------------------
    cudaEvent_t Memcpy1Start, Memcpy1Stop;
    cudaEventCreate(&Memcpy1Start);
    cudaEventCreate(&Memcpy1Stop);
    // ----------------------------
    cudaEvent_t Malloc2Start, Malloc2Stop;
    cudaEventCreate(&Malloc2Start);
    cudaEventCreate(&Malloc2Stop);
    // ----------------------------
    cudaEvent_t Memcpy2Start, Memcpy2Stop;
    cudaEventCreate(&Memcpy2Start);
    cudaEventCreate(&Memcpy2Stop);
    // ----------------------------
    cudaEvent_t MemcpyToS1Start, MemcpyToS1Stop;
    cudaEventCreate(&MemcpyToS1Start);
    cudaEventCreate(&MemcpyToS1Stop);
    // ----------------------------

    cudaEventRecord(Malloc1Start, 0);
    cudaMalloc(&cudaDeviceListLen, sizeof(int ) * numNodes);
    cudaEventRecord(Malloc1Stop, 0);

    cudaEventRecord(Memcpy1Start, 0);
    cudaMemcpy(cudaDeviceListLen, list_len, sizeof(int) * numNodes, cudaMemcpyHostToDevice);
    cudaEventRecord(Memcpy1Stop, 0);

    cudaEventRecord(Malloc2Start, 0);
    cudaMalloc((void **)&cudaDeviceNodeList, node_list_size * sizeof(int));
    cudaEventRecord(Malloc2Stop, 0);

    cudaEventRecord(Memcpy2Start, 0);
    cudaMemcpy(cudaDeviceNodeList, node_list, sizeof(int) * node_list_size, cudaMemcpyHostToDevice);
    cudaEventRecord(Memcpy2Stop, 0);


    GlobalConstants params;
    params.ListLen = cudaDeviceListLen;
    params.NodeList = cudaDeviceNodeList;
    params.numNodes = numNodes;
    params.numEdges = numEdges;


    cout << "Num vertices: " << numNodes << endl;
    cout << "Num edges: " << numEdges << endl;


    cudaEventRecord(MemcpyToS1Start, 0);
    cudaMemcpyToSymbol(cuConstCounterParams, &params, sizeof(GlobalConstants));
    cudaEventRecord(MemcpyToS1Stop, 0);


    cudaEventSynchronize(Malloc1Stop);
    float Malloc1ElapsedTime;
    cudaEventElapsedTime(&Malloc1ElapsedTime, Malloc1Start, Malloc1Stop);
    cout << "Malloc1 copying took: " << Malloc1ElapsedTime << " milliseconds" << endl;
    cudaEventDestroy(Malloc1Start);
    cudaEventDestroy(Malloc1Stop);

    cudaEventSynchronize(Memcpy1Stop);
    float Memcpy1ElapsedTime;
    cudaEventElapsedTime(&Memcpy1ElapsedTime, Memcpy1Start, Memcpy1Stop);
    cout << "Memcpy1 copying took: " << Memcpy1ElapsedTime << " milliseconds" << endl;
    cudaEventDestroy(Memcpy1Start);
    cudaEventDestroy(Memcpy1Stop);


    cudaEventSynchronize(Malloc2Stop);
    float Malloc2ElapsedTime;
    cudaEventElapsedTime(&Malloc2ElapsedTime, Malloc2Start, Malloc2Stop);
    cout << "Malloc2 copying took: " << Malloc2ElapsedTime << " milliseconds" << endl;
    cudaEventDestroy(Malloc2Start);
    cudaEventDestroy(Malloc2Stop);


    cudaEventSynchronize(Memcpy2Stop);
    float Memcpy2ElapsedTime;
    cudaEventElapsedTime(&Memcpy2ElapsedTime, Memcpy2Start, Memcpy2Stop);
    cout << "Memcpy2 copying took: " << Memcpy2ElapsedTime << " milliseconds" << endl;
    cudaEventDestroy(Memcpy2Start);
    cudaEventDestroy(Memcpy2Stop);


    cudaEventSynchronize(MemcpyToS1Stop);
    float MemcpyToS1ElapsedTime;
    cudaEventElapsedTime(&MemcpyToS1ElapsedTime, MemcpyToS1Start, MemcpyToS1Stop);
    cout << "MemcpyToS1 copying took: " << MemcpyToS1ElapsedTime << " milliseconds" << endl;
    cudaEventDestroy(MemcpyToS1Start);
    cudaEventDestroy(MemcpyToS1Stop);


    // Save the time to use at the bottom of the file
    firstAllocation = Malloc1ElapsedTime + Memcpy1ElapsedTime + Malloc2ElapsedTime + Memcpy2ElapsedTime + MemcpyToS1ElapsedTime;



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

    // printf("total_nodes %d\n", total_nodes);
    // printf("node_list_size %d\n", node_list_size);
    // printf("numEdges %d\n", numEdges);




    list_len = (int *)calloc(total_nodes, sizeof(int));
    start_addr = (int *)calloc(total_nodes, sizeof(int));
    node_list = (int *)calloc(node_list_size, sizeof(int));

    malloc_diff = clock() - start;
    msec = malloc_diff * 1000 / CLOCKS_PER_SEC;

    // printf("memory allocated ......\n");
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

    // printf("graph created......\n");
    diff = clock() - start;
    msec = diff * 1000 / CLOCKS_PER_SEC;
    // printf("time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);

    myfile.close();
}

CudaTriangleCounter::~CudaTriangleCounter() {

    free(node_list);
    free(list_len);
}

/************************* Sequential implementation *************************/

#if 0
void CudaTriangleCounter::countTriangles() {
    int i, j, k, m, count=0;

    for (i=1; i<numNodes; i++) {

        int *list = node_list + start_addr[i-1] + 1;

        int len = list_len[i];

        if (len < 2) {
            continue;
        }

        for (j=0; j<len-1; j++) {
            for (k=j+1; k<len; k++) {

                int idx1;
                int idx2;
                idx1 = list[j];
                idx2 = list[k];
                int *list1 = node_list + start_addr[idx1-1] + 1;
                int len1 = list_len[idx1];

                for (m=0; m<len1; m++) {

                    if (list1[m] == idx2) {
                        count++;
                    }
                }
            }

        }

    }
        printf("count for %d -> %d\n", i, count);

}
#endif

/***************** First implementation using vertices to count triangles **********************/
//Performs poorly that's why why we shifted to our final approach mentioned below
#if 0
__global__ void countTriangleKernel(int *countArray) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= cuConstCounterParams.numNodes) {
        return;
    }

    int j, k, m, count=0;
    int *node_list = cuConstCounterParams.NodeList;
    int *list_len = cuConstCounterParams.ListLen;
    int *start_addr = cuConstCounterParams.StartAddr;

    int *list = node_list + start_addr[i-1] + 1;
    int len = list_len[i];

    if (len < 2) {
        countArray[i] = 0;
        return;
    }

    for (j=0; j<len-1; j++) {
        for (k=j+1; k<len; k++) {

            int idx1;
            int idx2;
            idx1 = list[j];
            idx2 = list[k];
            int *list1 = node_list + start_addr[idx1-1] + 1;
            int len1 = list_len[idx1];

            for (m=0; m<len1; m++) {
                if (list1[m] == idx2) {
                    count++;
                }
            }
        }
    }

    countArray[i] = count;

   //printf("%d count %d\n", i, count);
}

void
CudaTriangleCounter::countTriangles() {

    dim3 blockdim  = 1024;
    dim3 griddim = (numNodes + 1024)/1024;
    int *countArray;
    int count;

    cudaMalloc((void **)&countArray, numNodes * sizeof(int));

    printf("countTriangleKernel\n");
    countTriangleKernel<<<griddim, blockdim>>>(countArray);
    cudaDeviceSynchronize();
    printf("countTriangleKernel done\n");

    thrust::device_ptr<int> dev_ptr(countArray);
    thrust::inclusive_scan(dev_ptr, dev_ptr + numNodes, dev_ptr);

    cudaMemcpy(&count, &countArray[numNodes-1], sizeof(int), cudaMemcpyDeviceToHost);

    printf("count %d\n", count);
}

#endif

/************** Final approach but without work load balancing *************/

/*
 * Kernel to count number of triangles formed by a single edge. And store the count
 * in an array on which we will run reduction later to find total number of triangles
 * in the given graph.
 */
__global__ void countTriangleKernel(int *countArray, edge_tuple_t *compressed_list, int *start_addr) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= cuConstCounterParams.numEdges) {
        return;
    }

    if (i == 0) {
        countArray[i] = 0;
        return;
    }

    int j = 0, k = 0;
    uint64_t count=0;
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
    int *countArray;
    int count;
    edge_tuple_t *edge_list;


    // Create CUDA events to keep track of the GPU time
    cudaEvent_t Malloc3Start, Malloc3Stop;
    cudaEventCreate(&Malloc3Start);
    cudaEventCreate(&Malloc3Stop);
    // ----------------------------
    cudaEvent_t GPUthrust1Start, GPUthrust1Stop;
    cudaEventCreate(&GPUthrust1Start);
    cudaEventCreate(&GPUthrust1Stop);
    // ----------------------------
    cudaEvent_t GPUthrust2Start, GPUthrust2Stop;
    cudaEventCreate(&GPUthrust2Start);
    cudaEventCreate(&GPUthrust2Stop);
    // ----------------------------
    cudaEvent_t GPUthrust3Start, GPUthrust3Stop;
    cudaEventCreate(&GPUthrust3Start);
    cudaEventCreate(&GPUthrust3Stop);
    // ----------------------------
    cudaEvent_t Malloc4Start, Malloc4Stop;
    cudaEventCreate(&Malloc4Start);
    cudaEventCreate(&Malloc4Stop);
    // ----------------------------
    cudaEvent_t CreateEdgesStart, CreateEdgesStop;
    cudaEventCreate(&CreateEdgesStart);
    cudaEventCreate(&CreateEdgesStop);
    // ----------------------------
    cudaEvent_t Malloc5Start, Malloc5Stop;
    cudaEventCreate(&Malloc5Start);
    cudaEventCreate(&Malloc5Stop);
    // ----------------------------
    cudaEvent_t TCountStart, TCountStop;
    cudaEventCreate(&TCountStart);
    cudaEventCreate(&TCountStop);
    // ----------------------------
    cudaEvent_t GPUthrust4Start, GPUthrust4Stop;
    cudaEventCreate(&GPUthrust4Start);
    cudaEventCreate(&GPUthrust4Stop);
    // ----------------------------
    cudaEvent_t GPUthrust5Start, GPUthrust5Stop;
    cudaEventCreate(&GPUthrust5Start);
    cudaEventCreate(&GPUthrust5Stop);
    // ----------------------------
    cudaEvent_t Memcpy3Start, Memcpy3Stop;
    cudaEventCreate(&Memcpy3Start);
    cudaEventCreate(&Memcpy3Stop);
    // ----------------------------

    /* Calculating start address of each neighbour list */

    cudaEventRecord(Malloc3Start, 0);
    cudaMalloc(&cudaDeviceStartAddr, sizeof(int ) * numNodes);
    cudaEventRecord(Malloc3Stop, 0);

    cudaEventRecord(GPUthrust1Start, 0);
    thrust::device_ptr<int> dev_ptr1(cudaDeviceListLen);
    cudaEventRecord(GPUthrust1Stop, 0);

    cudaEventRecord(GPUthrust2Start, 0);
    thrust::device_ptr<int> output_ptr(cudaDeviceStartAddr);
    cudaEventRecord(GPUthrust2Stop, 0);

    cudaEventRecord(GPUthrust3Start, 0);
    thrust::inclusive_scan(dev_ptr1, dev_ptr1 + numNodes, output_ptr);
    cudaEventRecord(GPUthrust3Stop, 0);


    /* Create a list of all edges present in the graph */
    cudaEventRecord(Malloc4Start, 0);
    cudaMalloc((void **)&edge_list, numEdges * sizeof(edge_tuple_t));
    cudaEventRecord(Malloc4Stop, 0);

    cudaEventRecord(CreateEdgesStart, 0);
    createEdgeList<<<griddim1, blockdim>>>(edge_list, cudaDeviceStartAddr);
    cudaEventRecord(CreateEdgesStop, 0);


    cudaDeviceSynchronize();


    cudaEventRecord(Malloc5Start, 0);
    cudaMalloc((void **)&countArray, numEdges * sizeof(int));
    cudaEventRecord(Malloc5Stop, 0);


    /* Applyinf intersection rule on all edges to find number of triangles */
    cudaEventRecord(TCountStart, 0);
    countTriangleKernel<<<griddim, blockdim>>>(countArray, edge_list, cudaDeviceStartAddr);
    cudaEventRecord(TCountStop, 0);


    cudaDeviceSynchronize();

    cudaEventRecord(GPUthrust4Start, 0);
    thrust::device_ptr<int> dev_ptr(countArray);
    cudaEventRecord(GPUthrust4Stop, 0);

    cudaEventRecord(GPUthrust5Start, 0);
    thrust::inclusive_scan(dev_ptr, dev_ptr + numEdges, dev_ptr);
    cudaEventRecord(GPUthrust5Stop, 0);

    cudaEventRecord(Memcpy3Start, 0);
    cudaMemcpy(&count, &countArray[numEdges-1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(Memcpy3Stop, 0);


    cudaEventSynchronize(Malloc3Stop);
    float Malloc3ElapsedTime;
    cudaEventElapsedTime(&Malloc3ElapsedTime, Malloc3Start, Malloc3Stop);
    cout << "Malloc3 copying took: " << Malloc3ElapsedTime << " milliseconds" << endl;
    cudaEventDestroy(Malloc3Start);
    cudaEventDestroy(Malloc3Stop);

    cudaEventSynchronize(GPUthrust1Stop);
    float GPUthrust1ElapsedTime;
    cudaEventElapsedTime(&GPUthrust1ElapsedTime, GPUthrust1Start, GPUthrust1Stop);
    cout << "GPUthrust1 copying took: " << GPUthrust1ElapsedTime << " milliseconds" << endl;
    cudaEventDestroy(GPUthrust1Start);
    cudaEventDestroy(GPUthrust1Stop);

    cudaEventSynchronize(GPUthrust2Stop);
    float GPUthrust2ElapsedTime;
    cudaEventElapsedTime(&GPUthrust2ElapsedTime, GPUthrust2Start, GPUthrust2Stop);
    cout << "GPUthrust2 copying took: " << GPUthrust2ElapsedTime << " milliseconds" << endl;
    cudaEventDestroy(GPUthrust2Start);
    cudaEventDestroy(GPUthrust2Stop);

    cudaEventSynchronize(GPUthrust3Stop);
    float GPUthrust3ElapsedTime;
    cudaEventElapsedTime(&GPUthrust3ElapsedTime, GPUthrust3Start, GPUthrust3Stop);
    cout << "GPUthrust3 copying took: " << GPUthrust3ElapsedTime << " milliseconds" << endl;
    cudaEventDestroy(GPUthrust3Start);
    cudaEventDestroy(GPUthrust3Stop);

    cudaEventSynchronize(Malloc4Stop);
    float Malloc4ElapsedTime;
    cudaEventElapsedTime(&Malloc4ElapsedTime, Malloc4Start, Malloc4Stop);
    cout << "Malloc4 copying took: " << Malloc4ElapsedTime << " milliseconds" << endl;
    cudaEventDestroy(Malloc4Start);
    cudaEventDestroy(Malloc4Stop);

    cudaEventSynchronize(CreateEdgesStop);
    float CreateEdgesElapsedTime;
    cudaEventElapsedTime(&CreateEdgesElapsedTime, CreateEdgesStart, CreateEdgesStop);
    cout << "CreateEdges copying took: " << CreateEdgesElapsedTime << " milliseconds" << endl;
    cudaEventDestroy(CreateEdgesStart);
    cudaEventDestroy(CreateEdgesStop);

    cudaEventSynchronize(Malloc5Stop);
    float Malloc5ElapsedTime;
    cudaEventElapsedTime(&Malloc5ElapsedTime, Malloc5Start, Malloc5Stop);
    cout << "Malloc5 copying took: " << Malloc5ElapsedTime << " milliseconds" << endl;
    cudaEventDestroy(Malloc5Start);
    cudaEventDestroy(Malloc5Stop);

    cudaEventSynchronize(TCountStop);
    float TCountElapsedTime;
    cudaEventElapsedTime(&TCountElapsedTime, TCountStart, TCountStop);
    cout << "TCount  took: " << TCountElapsedTime << " milliseconds" << endl;
    cudaEventDestroy(TCountStart);
    cudaEventDestroy(TCountStop);

    cudaEventSynchronize(GPUthrust4Stop);
    float GPUthrust4ElapsedTime;
    cudaEventElapsedTime(&GPUthrust4ElapsedTime, GPUthrust4Start, GPUthrust4Stop);
    cout << "GPUthrust4 copying took: " << GPUthrust4ElapsedTime << " milliseconds" << endl;
    cudaEventDestroy(GPUthrust4Start);
    cudaEventDestroy(GPUthrust4Stop);

    cudaEventSynchronize(GPUthrust5Stop);
    float GPUthrust5ElapsedTime;
    cudaEventElapsedTime(&GPUthrust5ElapsedTime, GPUthrust5Start, GPUthrust5Stop);
    cout << "GPUthrust5 copying took: " << GPUthrust5ElapsedTime << " milliseconds" << endl;
    cudaEventDestroy(GPUthrust5Start);
    cudaEventDestroy(GPUthrust5Stop);

    cudaEventSynchronize(Memcpy3Stop);
    float Memcpy3ElapsedTime;
    cudaEventElapsedTime(&Memcpy3ElapsedTime, Memcpy3Start, Memcpy3Stop);
    cout << "Memcpy3 copying took: " << Memcpy3ElapsedTime << " milliseconds" << endl;
    cudaEventDestroy(Memcpy3Start);
    cudaEventDestroy(Memcpy3Stop);

    // Output the total GPU time to save in a csv file.
    cout << "Total GPU time: " << firstAllocation + Memcpy3ElapsedTime + GPUthrust5ElapsedTime + GPUthrust4ElapsedTime + TCountElapsedTime + Malloc5ElapsedTime +
            CreateEdgesElapsedTime + Malloc4ElapsedTime + GPUthrust3ElapsedTime + GPUthrust2ElapsedTime + GPUthrust1ElapsedTime +
            Malloc3ElapsedTime << " milliseconds" << endl;

    // printf("count %d\n", count);
}

