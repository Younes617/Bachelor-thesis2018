#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "cudaTriangleCounter.h"

#define PRINT_TIME 1

void count_triangle(int total_nodes, int **node_list, int *list_len)
{
    int i, j, k, m, count=0;

    for (i=1; i<total_nodes; i++) {

        int *list = node_list[i];
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
                int *list1 = node_list[idx1];
                int len1 = list_len[idx1];

                for (m=0; m<len1; m++) {
                    if (list1[m] == idx2) {
                        count++;
                    }
                }

                /*
                if (graph[idx1][idx2] == 1) {
                    count++;
                }*/
            }

        }
    }

    printf("count %d\n", count);
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        // printf("usage: ./a.out <input_file>\n");
        exit(-1);
    }

    int msec;
    clock_t start, diff;

    CudaTriangleCounter *tCounter = new CudaTriangleCounter(argv[1]);
   
    tCounter->setup();
    start = clock();
    tCounter->countTriangles();
    diff = clock() - start;
    msec = diff * 1000 / CLOCKS_PER_SEC;
    // printf("counting taken %d seconds %d milliseconds\n", msec/1000, msec%1000);

    return 0;
}
