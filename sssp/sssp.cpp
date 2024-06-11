#include <iostream>
#include <vector>
#include <algorithm>
#include "defs.h"
#include "mpi.h"
#include <utility>
#include <set>
#include <map>

using namespace std;

extern "C" void init_sssp(graph_t *G)
{
}

extern "C" void finalize_sssp()
{
}

void sssp(vertex_id_t root, graph_t *G, weight_t *local_dist, double delta)
{
    map<int, set<vertex_id_t>> buckets;
    buckets[0] = set<vertex_id_t>();
    int *numToSend = (int *)malloc(G->nproc * sizeof(int)), *heavyNumToSend = (int *)malloc(G->nproc * sizeof(int)), *numToRecv = (int *)malloc(G->nproc * sizeof(int));
    memset(numToSend, 0, G->nproc * sizeof(int));
    memset(heavyNumToSend, 0, G->nproc * sizeof(int));
    uint64_t nedges = 0;
    if (root / G->local_n == G->rank) {
        buckets[0].insert(root % G->local_n);
        local_dist[root % G->local_n] = 0;
    }
    int continueIterating = 1;
    for (int numBucket = 0; continueIterating; numBucket++) {
        if (buckets.find(numBucket) == buckets.end()) {
            buckets[numBucket] = set<vertex_id_t>();
        }
        set<vertex_id_t> A = buckets[numBucket];
        map<vertex_id_t, weight_t> heavy;
        while (continueIterating) {
            map<vertex_id_t, weight_t> light;
            for (auto it : A) {
                for (edge_id_t j = G->rowsIndices[it]; j < G->rowsIndices[it + 1]; j++) {
                    if (G->weights[j] < delta) {
                        if (light.find(G->endV[j]) == light.end()) {
                            light[G->endV[j]] = local_dist[it] + G->weights[j];
                            numToSend[G->endV[j] / G->local_n]++;
                        }else if (light[G->endV[j]] > local_dist[it] + G->weights[j]) {
                            light[G->endV[j]] = local_dist[it] + G->weights[j];
                        }
                    } else {
                        if (heavy.find(G->endV[j]) == heavy.end()) {
                            heavy[G->endV[j]] = local_dist[it] + G->weights[j];
                            heavyNumToSend[G->endV[j] / G->local_n]++;
                        } else if (heavy[G->endV[j]] > local_dist[it] + G->weights[j]) {
                            heavy[G->endV[j]] = local_dist[it] + G->weights[j];
                        }
                    }
                    nedges++;
                }
            }
            MPI_Alltoall(numToSend, 1, MPI_INT, numToRecv, 1, MPI_INT, MPI_COMM_WORLD);
            int lenToSend = 0, lenToRecv = 0, *sendDispl = (int *)malloc(G->nproc * sizeof(int)), *recvDispl = (int *)malloc(G->nproc * sizeof(int));
            for (int i = 0; i < G->nproc; i++) {
                lenToRecv += numToRecv[i];
                lenToSend += numToSend[i];
            }
            sendDispl[0] = 0;
            recvDispl[0] = 0;
            for (int i = 0; i < G->nproc; i++) {
                sendDispl[i] = sendDispl[i-1] + numToSend[i-1];
                recvDispl[i] = recvDispl[i-1] + numToRecv[i-1];
            }
            vertex_id_t *verticesToSend = (vertex_id_t *)malloc(lenToSend * sizeof(vertex_id_t));
            vertex_id_t *verticesToRecv = (vertex_id_t *)malloc(lenToRecv * sizeof(vertex_id_t));
            weight_t *weightsToSend = (weight_t *)malloc(lenToSend * sizeof(weight_t));
            weight_t *weightsToRecv = (weight_t *)malloc(lenToRecv * sizeof(weight_t));
            int ind = 0;
            for (auto i : light) {
                verticesToSend[ind] = i.first;
                weightsToSend[ind] = i.second;
                ind++;
            }
            MPI_Alltoallv(verticesToSend, numToSend, sendDispl, MPI_INT, verticesToRecv, numToRecv, recvDispl, MPI_INT, MPI_COMM_WORLD);
            MPI_Alltoallv(weightsToSend, numToSend, sendDispl, MPI_DOUBLE, weightsToRecv, numToRecv, recvDispl, MPI_DOUBLE, MPI_COMM_WORLD);
            A.clear();
            for (int i = 0; i < lenToRecv; i++) {
                if (weightsToRecv[i] < local_dist[verticesToRecv[i] % G->local_n] || local_dist[verticesToRecv[i] % G->local_n] == -1) {
                    local_dist[verticesToRecv[i] % G->local_n] = weightsToRecv[i];
                    if (weightsToRecv[i] / delta < numBucket + 1) {
                        A.insert(verticesToRecv[i] % G->local_n);
                    } else {
                        int toBucket = floor(weightsToRecv[i] / delta);
                        if (buckets.find(toBucket) == buckets.end()) {
                            buckets[toBucket] = set<vertex_id_t>();
                        }
                        buckets[toBucket].insert(verticesToRecv[i] % G->local_n);
                    }
                }
            }
            memset(numToSend, 0, G->nproc * sizeof(int));
            free(verticesToRecv);
            free(verticesToSend);
            free(weightsToRecv);
            free(weightsToSend);
            free(recvDispl);
            free(sendDispl);
            continueIterating = !A.empty();
            MPI_Allreduce(MPI_IN_PLACE, &continueIterating, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        }

        MPI_Alltoall(heavyNumToSend, 1, MPI_INT, numToRecv, 1, MPI_INT, MPI_COMM_WORLD);
        int lenToSend = 0, lenToRecv = 0, *sendDispl = (int *)malloc(G->nproc * sizeof(int)), *recvDispl = (int *)malloc(G->nproc * sizeof(int));
        for (int i = 0; i < G->nproc; i++) {
            lenToRecv += numToRecv[i];
            lenToSend += heavyNumToSend[i];
        }
        sendDispl[0] = 0;
        recvDispl[0] = 0;
        for (int i = 0; i < G->nproc; i++) {
            sendDispl[i] = sendDispl[i-1] + heavyNumToSend[i-1];
            recvDispl[i] = recvDispl[i-1] + numToRecv[i-1];
        }
        vertex_id_t *verticesToSend = (vertex_id_t *)malloc(lenToSend * sizeof(vertex_id_t));
        vertex_id_t *verticesToRecv = (vertex_id_t *)malloc(lenToRecv * sizeof(vertex_id_t));
        weight_t *weightsToSend = (weight_t *)malloc(lenToSend * sizeof(weight_t));
        weight_t *weightsToRecv = (weight_t *)malloc(lenToRecv * sizeof(weight_t));
        int ind = 0;
        for (auto i : heavy) {
            verticesToSend[ind] = i.first;
            weightsToSend[ind] = i.second;
            ind++;
        }
        MPI_Alltoallv(verticesToSend, heavyNumToSend, sendDispl, MPI_INT, verticesToRecv, numToRecv, recvDispl, MPI_INT, MPI_COMM_WORLD);
        MPI_Alltoallv(weightsToSend, heavyNumToSend, sendDispl, MPI_DOUBLE, weightsToRecv, numToRecv, recvDispl, MPI_DOUBLE, MPI_COMM_WORLD);
        for (int i = 0; i < lenToRecv; i++) {
            if (local_dist[verticesToRecv[i] % G->local_n] == -1 || weightsToRecv[i] < local_dist[verticesToRecv[i] % G->local_n]) {
                local_dist[verticesToRecv[i] % G->local_n] = weightsToRecv[i];
                int toBucket = floor(weightsToRecv[i] / delta);
                if (buckets.find(toBucket) == buckets.end()) {
                    buckets[toBucket] = set<vertex_id_t>();
                }
                buckets[toBucket].insert(verticesToRecv[i] % G->local_n);
            }
        }
        free(verticesToRecv);
        free(verticesToSend);
        free(weightsToRecv);
        free(weightsToSend);
        free(recvDispl);
        free(sendDispl);
        memset(heavyNumToSend, 0, G->nproc * sizeof(int));

        continueIterating = buckets.size() - numBucket - 1; 
        MPI_Allreduce(MPI_IN_PLACE, &continueIterating, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
    G->numTraversedEdges[root] = nedges;
    free(numToSend);
    free(numToRecv);
    free(heavyNumToSend);
}

