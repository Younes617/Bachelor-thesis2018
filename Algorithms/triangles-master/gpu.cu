// Original author: Adam Polak
// 
// edited by: Younes Ouazref
// 

#include "gpu.h"

#include "gpu-thrust.h"
#include "timer.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
using namespace std;

#define NUM_THREADS 64
#define NUM_BLOCKS_GENERIC 112
#define NUM_BLOCKS_PER_MP 8

template<bool ZIPPED>
__global__ void CalculateNodePointers(int n, int m, int* edges, int* nodes) {
  int from = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;
  for (int i = from; i <= m; i += step) {
    int prev = i > 0 ? edges[ZIPPED ? (2 * (i - 1) + 1) : (m + i - 1)] : -1;
    int next = i < m ? edges[ZIPPED ? (2 * i + 1) : (m + i)] : n;
    for (int j = prev + 1; j <= next; ++j)
      nodes[j] = i;
  }
}

__global__ void CalculateFlags(int m, int* edges, int* nodes, bool* flags) {
  int from = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;
  for (int i = from; i < m; i += step) {
    int a = edges[2 * i];
    int b = edges[2 * i + 1];
    int deg_a = nodes[a + 1] - nodes[a];
    int deg_b = nodes[b + 1] - nodes[b];
    flags[i] = (deg_a < deg_b) || (deg_a == deg_b && a < b);
  }
}

__global__ void UnzipEdges(int m, int* edges, int* unzipped_edges) {
  int from = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;
  for (int i = from; i < m; i += step) {
    unzipped_edges[i] = edges[2 * i];
    unzipped_edges[m + i] = edges[2 * i + 1];
  }
}

__global__ void CalculateTriangles(
    int m, const int* __restrict__ edges, const int* __restrict__ nodes,
    uint64_t* results, int deviceCount = 1, int deviceIdx = 0) {
  int from =
    gridDim.x * blockDim.x * deviceIdx +
    blockDim.x * blockIdx.x +
    threadIdx.x;
  int step = deviceCount * gridDim.x * blockDim.x;
  uint64_t count = 0;

  for (int i = from; i < m; i += step) {
    int u = edges[i], v = edges[m + i];

    int u_it = nodes[u], u_end = nodes[u + 1];
    int v_it = nodes[v], v_end = nodes[v + 1];

    int a = edges[u_it], b = edges[v_it];
    while (u_it < u_end && v_it < v_end) {
      int d = a - b;
      if (d <= 0)
        a = edges[++u_it];
      if (d >= 0)
        b = edges[++v_it];
      if (d == 0)
        ++count;
    }
  }

  results[blockDim.x * blockIdx.x + threadIdx.x] = count;
}

void CudaAssert(cudaError_t status, const char* code, const char* file, int l) {
  if (status == cudaSuccess) return;
  cerr << "Cuda error: " << code << ", file " << file << ", line " << l << endl;
  exit(1);
}

#define CUCHECK(x) CudaAssert(x, #x, __FILE__, __LINE__)

int NumberOfMPs() {
  int dev, val;
  CUCHECK(cudaGetDevice(&dev));
  CUCHECK(cudaDeviceGetAttribute(&val, cudaDevAttrMultiProcessorCount, dev));
  return val;
}

size_t GlobalMemory() {
  int dev;
  cudaDeviceProp prop;
  CUCHECK(cudaGetDevice(&dev));
  CUCHECK(cudaGetDeviceProperties(&prop, dev));
  return prop.totalGlobalMem;
}

Edges RemoveBackwardEdgesCPU(const Edges& unordered_edges) {
  int n = NumVertices(unordered_edges);
  int m = unordered_edges.size();

  vector<int> deg(n);
  for (int i = 0; i < m; ++i)
    ++deg[unordered_edges[i].first];

  vector< pair<int, int> > edges;
  edges.reserve(m / 2);
  for (int i = 0; i < m; ++i) {
    int s = unordered_edges[i].first, t = unordered_edges[i].second;
    if (deg[s] > deg[t] || (deg[s] == deg[t] && s > t))
      edges.push_back(make_pair(s, t));
  }

  return edges;
}

uint64_t MultiGPUCalculateTriangles(
    int n, int m, int* dev_edges, int* dev_nodes, int device_count) {
  vector<int*> multi_dev_edges(device_count);
  vector<int*> multi_dev_nodes(device_count);

  multi_dev_edges[0] = dev_edges;
  multi_dev_nodes[0] = dev_nodes;

  for (int i = 1; i < device_count; ++i) {
    CUCHECK(cudaSetDevice(i));
    CUCHECK(cudaMalloc(&multi_dev_edges[i], m * 2 * sizeof(int)));
    CUCHECK(cudaMalloc(&multi_dev_nodes[i], (n + 1) * sizeof(int)));
    int dst = i, src = (i + 1) >> 2;
    CUCHECK(cudaMemcpyPeer(
          multi_dev_edges[dst], dst, multi_dev_edges[src], src,
          m * 2 * sizeof(int)));
    CUCHECK(cudaMemcpyPeer(
          multi_dev_nodes[dst], dst, multi_dev_nodes[src], src,
          (n + 1) * sizeof(int)));
  }

  vector<int> NUM_BLOCKS(device_count);
  vector<uint64_t*> multi_dev_results(device_count);

  for (int i = 0; i < device_count; ++i) {
    CUCHECK(cudaSetDevice(i));
    NUM_BLOCKS[i] = NUM_BLOCKS_PER_MP * NumberOfMPs();
    CUCHECK(cudaMalloc(
          &multi_dev_results[i],
          NUM_BLOCKS[i] * NUM_THREADS * sizeof(uint64_t)));
  }

  for (int i = 0; i < device_count; ++i) {
    CUCHECK(cudaSetDevice(i));
    CUCHECK(cudaFuncSetCacheConfig(CalculateTriangles, cudaFuncCachePreferL1));
    CalculateTriangles<<<NUM_BLOCKS[i], NUM_THREADS>>>(
        m, multi_dev_edges[i], multi_dev_nodes[i], multi_dev_results[i],
        device_count, i);
  }

  uint64_t result = 0;

  for (int i = 0; i < device_count; ++i) {
    CUCHECK(cudaSetDevice(i));
    CUCHECK(cudaDeviceSynchronize());
    result += SumResults(NUM_BLOCKS[i] * NUM_THREADS, multi_dev_results[i]);
  }

  for (int i = 1; i < device_count; ++i) {
    CUCHECK(cudaSetDevice(i));
    CUCHECK(cudaFree(multi_dev_edges[i]));
    CUCHECK(cudaFree(multi_dev_nodes[i]));
  }

  for (int i = 0; i < device_count; ++i) {
    CUCHECK(cudaSetDevice(i));
    CUCHECK(cudaFree(multi_dev_results[i]));
  }

  cudaSetDevice(0);
  return result;
}

uint64_t GpuForward(const Edges& edges) {
  return MultiGpuForward(edges, 1);
}

uint64_t MultiGpuForward(const Edges& edges, int device_count) {
  Timer* timer = Timer::NewTimer();

  CUCHECK(cudaSetDevice(0));
  const int NUM_BLOCKS = NUM_BLOCKS_PER_MP * NumberOfMPs();

  int m = edges.size(), n;

  int* dev_edges;
  int* dev_nodes;

  // Creation of the cuda events which will count the GPU execution time.
  // -----------------------------
  cudaEvent_t Malloc1Start, Malloc1Stop;
  cudaEventCreate(&Malloc1Start);
  cudaEventCreate(&Malloc1Stop);
  // -----------------------------
  // ------------------------------
  cudaEvent_t Memcpy1Start, Memcpy1Stop;
  cudaEventCreate(&Memcpy1Start);
  cudaEventCreate(&Memcpy1Stop);
  // -------------------------------
  // -----------------------------------------
  cudaEvent_t GPUthrustVerticesStart, GPUthrustVerticesStop;
  cudaEventCreate(&GPUthrustVerticesStart);
  cudaEventCreate(&GPUthrustVerticesStop);
  // ------------------------------------------
  // --------------------------------------
  cudaEvent_t GPUthrustSortStart, GPUthrustSortStop;
  cudaEventCreate(&GPUthrustSortStart);
  cudaEventCreate(&GPUthrustSortStop);
  // --------------------------------------
  // ------------------------------
  cudaEvent_t Malloc2Start, Malloc2Stop;
  cudaEventCreate(&Malloc2Start);
  cudaEventCreate(&Malloc2Stop);
  // -------------------------------
  // --------------------------------
  cudaEvent_t CalcNodeP_TStart, CalcNodeP_TStop;
  cudaEventCreate(&CalcNodeP_TStart);
  cudaEventCreate(&CalcNodeP_TStop);
  // ---------------------------------
  // -----------------------------
  cudaEvent_t Malloc3Start, Malloc3Stop;
  cudaEventCreate(&Malloc3Start);
  cudaEventCreate(&Malloc3Stop);
  // -----------------------------
  // ---------------------------------
  cudaEvent_t CalcFlagsStart, CalcFlagsStop;
  cudaEventCreate(&CalcFlagsStart);
  cudaEventCreate(&CalcFlagsStop);
  // ----------------------------------
  // ---------------------------------------
  cudaEvent_t GPUthrustRemoveStart, GPUthrustRemoveStop;
  cudaEventCreate(&GPUthrustRemoveStart);
  cudaEventCreate(&GPUthrustRemoveStop);
  // ----------------------------------------
  // --------------------------
  cudaEvent_t UnzipStart, UnzipStop;
  cudaEventCreate(&UnzipStart);
  cudaEventCreate(&UnzipStop);
  // --------------------------
  // -------------------------------
  cudaEvent_t Memcpy2Start, Memcpy2Stop;
  cudaEventCreate(&Memcpy2Start);
  cudaEventCreate(&Memcpy2Stop);
  // --------------------------------
  // ------------------------------
  cudaEvent_t CalcNodeP_FStart, CalcNodeP_FStop;
  cudaEventCreate(&CalcNodeP_FStart);
  cudaEventCreate(&CalcNodeP_FStop);
  // -------------------------------
  // ----------------------------
  cudaEvent_t Malloc4Start, Malloc4Stop;
  cudaEventCreate(&Malloc4Start);
  cudaEventCreate(&Malloc4Stop);
  // ---------------------------
  // ------------------------
  cudaEvent_t TCountStart, TCountStop;
  cudaEventCreate(&TCountStart);
  cudaEventCreate(&TCountStop);
  // -------------------------
  // -------------------------------
  cudaEvent_t GPUthrustSumStart, GPUthrustSumStop;
  cudaEventCreate(&GPUthrustSumStart);
  cudaEventCreate(&GPUthrustSumStop);
  // -------------------------------


  if ((uint64_t)m * 4 * sizeof(int) < GlobalMemory()) {  // just approximation


    cudaEventRecord(Malloc1Start, 0);
    CUCHECK(cudaMalloc(&dev_edges, m * 2 * sizeof(int)));
    cudaEventRecord(Malloc1Stop, 0);

    

    cudaEventRecord(Memcpy1Start, 0);
    CUCHECK(cudaMemcpyAsync(
          dev_edges, edges.data(), m * 2 * sizeof(int),
          cudaMemcpyHostToDevice));
    cudaEventRecord(Memcpy1Stop, 0);

    
    CUCHECK(cudaDeviceSynchronize());
    // timer->Done("Memcpy edges from host do device");



    cudaEventRecord(GPUthrustVerticesStart, 0);
    n = NumVerticesGPU(m, dev_edges);
    cudaEventRecord(GPUthrustVerticesStop, 0);

    cout << "Num vertices: " << n << endl;
    // Undirected graph.
    cout << "Num edges: " << (m/2) << endl;


    // timer->Done("Calculate number of vertices");
    cudaEventRecord(GPUthrustSortStart, 0);
    SortEdges(m, dev_edges);
    cudaEventRecord(GPUthrustSortStop, 0);

    CUCHECK(cudaDeviceSynchronize());
    // timer->Done("Sort edges");



    cudaEventRecord(Malloc2Start, 0);
    CUCHECK(cudaMalloc(&dev_nodes, (n + 1) * sizeof(int)));
    cudaEventRecord(Malloc2Start, 0);



    cudaEventRecord(CalcNodeP_TStart, 0);
    CalculateNodePointers<true><<<NUM_BLOCKS, NUM_THREADS>>>(
        n, m, dev_edges, dev_nodes);
    cudaEventRecord(CalcNodeP_TStop, 0);


    CUCHECK(cudaDeviceSynchronize());
    // timer->Done("Calculate nodes array for two-way zipped edges");

    bool* dev_flags;



    cudaEventRecord(Malloc3Start, 0);
    CUCHECK(cudaMalloc(&dev_flags, m * sizeof(bool)));
    cudaEventRecord(Malloc3Stop, 0);



    cudaEventRecord(CalcFlagsStart, 0);
    CalculateFlags<<<NUM_BLOCKS, NUM_THREADS>>>(
        m, dev_edges, dev_nodes, dev_flags);
    cudaEventRecord(CalcFlagsStop, 0);



    cudaEventRecord(GPUthrustRemoveStart, 0);
    RemoveMarkedEdges(m, dev_edges, dev_flags);
    cudaEventRecord(GPUthrustRemoveStop, 0);


    CUCHECK(cudaFree(dev_flags));
    CUCHECK(cudaDeviceSynchronize());
    m /= 2;
    // timer->Done("Remove backward edges");


    cudaEventRecord(UnzipStart, 0);
    UnzipEdges<<<NUM_BLOCKS, NUM_THREADS>>>(m, dev_edges, dev_edges + 2 * m);
    cudaEventRecord(UnzipStop, 0);


    cudaEventRecord(Memcpy2Start, 0);
    CUCHECK(cudaMemcpyAsync(
          dev_edges, dev_edges + 2 * m, 2 * m * sizeof(int),
          cudaMemcpyDeviceToDevice));
    cudaEventRecord(Memcpy2Stop, 0);


    CUCHECK(cudaDeviceSynchronize());
    // timer->Done("Unzip edges");


  } else {
    Edges fwd_edges = RemoveBackwardEdgesCPU(edges);
    m /= 2;
    timer->Done("Remove backward edges on CPU");

    int* dev_temp;
    CUCHECK(cudaMalloc(&dev_temp, m * 2 * sizeof(int)));
    CUCHECK(cudaMemcpyAsync(
          dev_temp, fwd_edges.data(), m * 2 * sizeof(int), cudaMemcpyHostToDevice));
    CUCHECK(cudaDeviceSynchronize());
    timer->Done("Memcpy edges from host do device");

    SortEdges(m, dev_temp);
    CUCHECK(cudaDeviceSynchronize());
    timer->Done("Sort edges");

    CUCHECK(cudaMalloc(&dev_edges, m * 2 * sizeof(int)));
    UnzipEdges<<<NUM_BLOCKS, NUM_THREADS>>>(m, dev_temp, dev_edges);
    CUCHECK(cudaFree(dev_temp));
    CUCHECK(cudaDeviceSynchronize());
    timer->Done("Unzip edges");

    n = NumVerticesGPU(m, dev_edges);
    CUCHECK(cudaMalloc(&dev_nodes, (n + 1) * sizeof(int)));
    timer->Done("Calculate number of vertices");
  }


  cudaEventRecord(CalcNodeP_FStart, 0);
  CalculateNodePointers<false><<<NUM_BLOCKS, NUM_THREADS>>>(
      n, m, dev_edges, dev_nodes);
  cudaEventRecord(CalcNodeP_FStop, 0);



  CUCHECK(cudaDeviceSynchronize());
  // timer->Done("Calculate nodes array for one-way unzipped edges");

  uint64_t result = 0;




  if (device_count == 1) {
    uint64_t* dev_results;


    cudaEventRecord(Malloc4Start, 0);
    CUCHECK(cudaMalloc(&dev_results,
          NUM_BLOCKS * NUM_THREADS * sizeof(uint64_t)));
    cudaEventRecord(Malloc4Stop, 0);


    cudaFuncSetCacheConfig(CalculateTriangles, cudaFuncCachePreferL1);


    cudaProfilerStart();

    cudaEventRecord(TCountStart, 0);
    CalculateTriangles<<<NUM_BLOCKS, NUM_THREADS>>>(
        m, dev_edges, dev_nodes, dev_results);
    cudaEventRecord(TCountStop, 0);

    CUCHECK(cudaDeviceSynchronize());
    
    cudaProfilerStop();
    // timer->Done("Calculate triangles");


    cudaEventRecord(GPUthrustSumStart, 0);
    result = SumResults(NUM_BLOCKS * NUM_THREADS, dev_results);
    cudaEventRecord(GPUthrustSumStop, 0);


    CUCHECK(cudaFree(dev_results));



    // timer->Done("Reduce");
  } else {
    result = MultiGPUCalculateTriangles(
        n, m, dev_edges, dev_nodes, device_count);
    timer->Done("Calculate triangles on multi GPU");
  }

  CUCHECK(cudaFree(dev_edges));
  CUCHECK(cudaFree(dev_nodes));

  delete timer;






  CUCHECK(cudaEventSynchronize(Malloc1Stop));
  // print the time the kernel invocation took, without the copies!
  float Malloc1ElapsedTime;
  cudaEventElapsedTime(&Malloc1ElapsedTime, Malloc1Start, Malloc1Stop);
  cout << "Malloc1 copying took: " << Malloc1ElapsedTime << " milliseconds" << endl;
  cudaEventDestroy(Malloc1Start);
  cudaEventDestroy(Malloc1Stop);


  CUCHECK(cudaEventSynchronize(Memcpy1Stop));
  // print the time the kernel invocation took, without the copies!
  float Memcpy1ElapsedTime;
  cudaEventElapsedTime(&Memcpy1ElapsedTime, Memcpy1Start, Memcpy1Stop);
  cout << "Memcpy1 took: " << Memcpy1ElapsedTime << " milliseconds" << endl;
  cudaEventDestroy(Memcpy1Start);
  cudaEventDestroy(Memcpy1Stop);

  
  CUCHECK(cudaEventSynchronize(GPUthrustVerticesStop));
  // print the time the kernel invocation took, without the copies!
  float GPUthrustVerticesElapsedTime;
  cudaEventElapsedTime(&GPUthrustVerticesElapsedTime, GPUthrustVerticesStart, GPUthrustVerticesStop);
  cout << "GPU thrust num of vertices took: " << GPUthrustVerticesElapsedTime << " milliseconds" << endl;
  cudaEventDestroy(GPUthrustVerticesStart);
  cudaEventDestroy(GPUthrustVerticesStop);


  CUCHECK(cudaEventSynchronize(GPUthrustSortStop));
  // print the time the kernel invocation took, without the copies!
  float GPUthrustSortElapsedTime;
  cudaEventElapsedTime(&GPUthrustSortElapsedTime, GPUthrustSortStart, GPUthrustSortStop);
  cout << "GPU thrust sort took: " << GPUthrustSortElapsedTime << " milliseconds" << endl;
  cudaEventDestroy(GPUthrustSortStart);
  cudaEventDestroy(GPUthrustSortStop);


  CUCHECK(cudaEventSynchronize(Malloc2Stop));
  // print the time the kernel invocation took, without the copies!
  float Malloc2ElapsedTime;
  cudaEventElapsedTime(&Malloc2ElapsedTime, Malloc2Start, Malloc2Stop);
  cout << "Malloc2 copying took: " << Malloc2ElapsedTime << " milliseconds" << endl;
  cudaEventDestroy(Malloc2Start);
  cudaEventDestroy(Malloc2Stop);


  CUCHECK(cudaEventSynchronize(CalcNodeP_TStop));
  // print the time the kernel invocation took, without the copies!
  float CalcNodeP_TElapsedTime;
  cudaEventElapsedTime(&CalcNodeP_TElapsedTime, CalcNodeP_TStart, CalcNodeP_TStop);
  cout << "Calc node pointers T took: " << CalcNodeP_TElapsedTime << " milliseconds" << endl;
  cudaEventDestroy(CalcNodeP_TStart);
  cudaEventDestroy(CalcNodeP_TStop);


  CUCHECK(cudaEventSynchronize(Malloc3Stop));
  // print the time the kernel invocation took, without the copies!
  float Malloc3ElapsedTime;
  cudaEventElapsedTime(&Malloc3ElapsedTime, Malloc3Start, Malloc3Stop);
  cout << "Malloc3 took: " << Malloc3ElapsedTime << " milliseconds" << endl;
  cudaEventDestroy(Malloc3Start);
  cudaEventDestroy(Malloc3Stop);


  CUCHECK(cudaEventSynchronize(CalcFlagsStop));
  // print the time the kernel invocation took, without the copies!
  float CalcFlagsElapsedTime;
  cudaEventElapsedTime(&CalcFlagsElapsedTime, CalcFlagsStart, CalcFlagsStop);
  cout << "Calc flags took: " << CalcFlagsElapsedTime << " milliseconds" << endl;
  cudaEventDestroy(CalcFlagsStart);
  cudaEventDestroy(CalcFlagsStop);

  CUCHECK(cudaEventSynchronize(GPUthrustRemoveStop));
  // print the time the kernel invocation took, without the copies!
  float GPUthrustRemoveElapsedTime;
  cudaEventElapsedTime(&GPUthrustRemoveElapsedTime, GPUthrustRemoveStart, GPUthrustRemoveStop);
  cout << "GPU thrust remove took: " << GPUthrustRemoveElapsedTime << " milliseconds" << endl;
  cudaEventDestroy(GPUthrustRemoveStart);
  cudaEventDestroy(GPUthrustRemoveStop);


  CUCHECK(cudaEventSynchronize(UnzipStop));
  // print the time the kernel invocation took, without the copies!
  float UnzipElapsedTime;
  cudaEventElapsedTime(&UnzipElapsedTime, UnzipStart, UnzipStop);
  cout << "Unzipping took: " << UnzipElapsedTime << " milliseconds" << endl;
  cudaEventDestroy(UnzipStart);
  cudaEventDestroy(UnzipStop);

  CUCHECK(cudaEventSynchronize(Memcpy2Stop));
  // print the time the kernel invocation took, without the copies!
  float Memcpy2ElapsedTime;
  cudaEventElapsedTime(&Memcpy2ElapsedTime, Memcpy2Start, Memcpy2Stop);
  cout << "Memcpy2 took: " << Memcpy2ElapsedTime << " milliseconds" << endl;
  cudaEventDestroy(Memcpy2Start);
  cudaEventDestroy(Memcpy2Stop);


  CUCHECK(cudaEventSynchronize(CalcNodeP_FStop));
  // print the time the kernel invocation took, without the copies!
  float CalcNodeP_FElapsedTime;
  cudaEventElapsedTime(&CalcNodeP_FElapsedTime, CalcNodeP_FStart, CalcNodeP_FStop);
  cout << "Calc node pointers F took: " << CalcNodeP_TElapsedTime << " milliseconds" << endl;
  cudaEventDestroy(CalcNodeP_FStart);
  cudaEventDestroy(CalcNodeP_FStop);


  CUCHECK(cudaEventSynchronize(Malloc4Stop));
  // print the time the kernel invocation took, without the copies!
  float Malloc4ElapsedTime;
  cudaEventElapsedTime(&Malloc4ElapsedTime, Malloc4Start, Malloc4Stop);   
  cout << "Malloc4 took: " << Malloc4ElapsedTime << " milliseconds" << endl;
  cudaEventDestroy(Malloc4Start);
  cudaEventDestroy(Malloc4Stop);


  CUCHECK(cudaEventSynchronize(TCountStop));
  // print the time the kernel invocation took, without the copies!
  float TCountElapsedTime;
  cudaEventElapsedTime(&TCountElapsedTime, TCountStart, TCountStop);   
  cout << "Triangle count took: " << TCountElapsedTime << " milliseconds" << endl;
  cudaEventDestroy(TCountStart);
  cudaEventDestroy(TCountStop);


  CUCHECK(cudaEventSynchronize(GPUthrustSumStop));
  // print the time the kernel invocation took, without the copies!
  float GPUthrustSumElapsedTime;
  cudaEventElapsedTime(&GPUthrustSumElapsedTime, GPUthrustSumStart, GPUthrustSumStop);   
  cout << "Sum of triangles (of threads) took: " << GPUthrustSumElapsedTime << " milliseconds" << endl;
  cudaEventDestroy(GPUthrustSumStart);
  cudaEventDestroy(GPUthrustSumStop);

  // The output of the cuda events are being saved by the script that runs them.
  // the ':' char is used to recognize the data.
  cout << "Total GPU time: " << GPUthrustSumElapsedTime + TCountElapsedTime + Malloc4ElapsedTime + 
      CalcNodeP_TElapsedTime + Memcpy2ElapsedTime + UnzipElapsedTime + GPUthrustRemoveElapsedTime + CalcFlagsElapsedTime
      + Malloc3ElapsedTime + CalcNodeP_TElapsedTime + Malloc2ElapsedTime + GPUthrustSortElapsedTime
      + GPUthrustVerticesElapsedTime + Memcpy1ElapsedTime + Malloc1ElapsedTime << " milliseconds" << endl;


  return result;
}

void PreInitGpuContext(int device) {
  CUCHECK(cudaSetDevice(device));
  CUCHECK(cudaFree(NULL));
}
