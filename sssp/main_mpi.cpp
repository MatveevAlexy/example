#include <stdio.h>
#include <stdlib.h>
#include <error.h>
#include <assert.h>
#include <string.h>
#include "mpi.h"
#include <math.h>
#include "defs.h"
#include <iostream>
#include <unistd.h>
#include "sssp.cpp"

using namespace std;

int lgsize;
char* inFilename;
char* outFilename;
uint32_t rootNumberToValidate;

void usage(int argc, char **argv)
{
    printf("Usage:\n");
    printf("    %s -in <input> [options]\n", argv[0]);
    printf("Options:\n");
    printf("    -in <input> -- input graph filename\n");
    printf("    -out <output> -- output filename (distances from root vertex). By default output is '<input>.v.<process id>'\n");
    printf("    -root <root> -- root number for validation\n");
    exit(1);
}

void init (int argc, char** argv, graph_t* G)
{
    int i;
    inFilename = outFilename = NULL;
    rootNumberToValidate = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &G->nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &G->rank);
    

    for ( i = 1; i < argc; ++i) {
   		if (!strcmp(argv[i], "-in")) {
            inFilename = argv[++i];
        }
   		if (!strcmp(argv[i], "-out")) {
            outFilename = argv[++i];
        }
		if (!strcmp(argv[i], "-root")) {
			rootNumberToValidate = (int) atoi(argv[++i]);
        }
    }
    if (!inFilename) usage(argc, argv);
    if (!outFilename) {
        outFilename = (char *)malloc((strlen(inFilename) + 3) * sizeof(char));
        sprintf(outFilename, "%s.v", inFilename);
    }
}


void readGraph(graph_t *G, char *filename)
{
    edge_id_t arity;
    uint8_t align;
    FILE *F = fopen(filename, "rb");
    if (!F) error(EXIT_FAILURE, 0, "Error in opening file %s", filename);

    size_t objects_read = 0;

	objects_read = fread(&G->n, sizeof(vertex_id_t), 1, F);
    assert(objects_read == 1);
    objects_read = fread(&arity, sizeof(edge_id_t), 1, F);
    assert(objects_read == 1);
    G->m = G->n * arity;
    G->local_n = G->n / G->nproc;
    lgsize = log(G->nproc)/log(2);
    assert (lgsize < G->nproc);
    objects_read = fread(&G->directed, sizeof(bool), 1, F);
    assert(objects_read == 1);
    objects_read = fread(&align, sizeof(uint8_t), 1, F);
    assert(objects_read == 1);

	G->rowsIndices = (edge_id_t *)malloc((G->local_n+1) * sizeof(edge_id_t));
    assert(G->rowsIndices);
    fseek(F, G->rank * G->local_n * sizeof(edge_id_t), SEEK_CUR);
	objects_read = fread(G->rowsIndices, sizeof(edge_id_t), G->local_n+1, F);
    assert(objects_read == (G->local_n+1));
    fseek(F, (G->nproc - G->rank - 1) * G->local_n * sizeof(edge_id_t), SEEK_CUR);
    G->local_m = G->rowsIndices[G->local_n] - G->rowsIndices[0];

    G->endV = (vertex_id_t *)malloc((G->rowsIndices[G->local_n] - G->rowsIndices[0]) * sizeof(vertex_id_t));
    assert(G->endV);

    fseek(F, G->rowsIndices[0] * sizeof(vertex_id_t), SEEK_CUR);
    objects_read = fread(G->endV, sizeof(vertex_id_t), G->rowsIndices[G->local_n] - G->rowsIndices[0], F);
    assert(objects_read == G->rowsIndices[G->local_n] - G->rowsIndices[0]);
    fseek(F, (G->m - G->rowsIndices[G->local_n]) * sizeof(vertex_id_t), SEEK_CUR);

    objects_read = fread(&G->nRoots, sizeof(uint32_t), 1, F);
    assert(objects_read == 1);
    
    G->roots = (vertex_id_t *)malloc(G->nRoots * sizeof(vertex_id_t));
    assert(G->roots);
    G->numTraversedEdges = (edge_id_t *)malloc(G->nRoots * sizeof(edge_id_t));
    assert(G->numTraversedEdges);

    objects_read =  fread(G->roots, sizeof(vertex_id_t), G->nRoots, F);
    assert(objects_read == G->nRoots);
    objects_read =  fread(G->numTraversedEdges, sizeof(edge_id_t), G->nRoots, F);
    assert(objects_read == G->nRoots);

    fseek(F, G->rowsIndices[0] * sizeof(weight_t), SEEK_CUR);
    G->weights = (weight_t *)malloc(G->local_m * sizeof(weight_t));
    assert(G->weights);

    objects_read = fread(G->weights, sizeof(weight_t), G->local_m, F);
    assert(objects_read == G->local_m);

    
    for (int i = G->local_n; i >= 0; i -= 1) {
        G->rowsIndices[i] -= G->rowsIndices[0];
    }
    fclose(F);
}

void calc_traversed_edges(graph_t *G, weight_t* dist, uint64_t* traversed_edges)
{
    edge_id_t i;
    edge_id_t nedges_local = 0;
    edge_id_t nedges_global = 0;
    for ( i = 0; i < G->local_n; i++) {
        if ( dist[i] != -1 ) {
            nedges_local +=  G->rowsIndices[i+1] - G->rowsIndices[i];
        }
    }
    MPI_Allreduce(&nedges_local, &nedges_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    *traversed_edges = nedges_global;
}

/* write distances from root vertex to each others to output file. -1 = infinity */
void writeDistance(char* filename, weight_t *dist, vertex_id_t n)
{
    FILE *F = fopen(filename, "wb");
    assert(fwrite(dist, sizeof(weight_t), n, F) == n);
    fclose(F);
}

void write_validate(graph_t *G, weight_t* local_dist)
{
    int i, j;
    weight_t* dist_recv;
    weight_t* dist;

    if (G->rank == 0) dist_recv = (weight_t *)malloc(G->n * sizeof(weight_t));
    
    MPI_Gather(local_dist, G->local_n, MPI_DOUBLE, dist_recv, G->local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (G->rank == 0) {
        writeDistance(outFilename, dist_recv, G->n);
        free(dist_recv);
    }
}

void freeGraph(graph_t *G)
{
    free(G->rowsIndices);
    free(G->endV);
    free(G->roots);
    free(G->weights);
}


int main(int argc, char **argv) 
{
    weight_t* local_dist;
    graph_t g;
    uint64_t traversed_edges;
    uint64_t sssp_edges;
    uint32_t i;
    vertex_id_t j;
    double start, finish;
   
    /* initializing and reading the graph */
    MPI_Init (&argc, &argv);
    init(argc, argv, &g); 
    readGraph(&g, inFilename);
    double *perf = (double *)malloc(g.nRoots * sizeof(double)), min_perf, max_perf, avg_perf = 0;
    double *timing = (double *)malloc(g.nRoots * sizeof(double)), min_time, max_time, avg_time = 0;

    local_dist = (weight_t *)malloc(g.local_n * sizeof(weight_t));

    /* doing SSSP */
    for ( i = 0; i < g.nRoots; ++i) {
        /* initializing for validation, -1 = infinity */
        for ( j = 0; j < g.local_n; j++) {
            local_dist[j] = -1;
        }
        
        //FIXME: timings
        if (!g.rank) {
            start = MPI_Wtime();
        }
        sssp(i, &g, local_dist);
        edge_id_t nedges = 0;
        MPI_Reduce(&g.numTraversedEdges[i], &nedges, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        if (!g.rank) {
            finish = MPI_Wtime();
            timing[i] = finish - start;
            avg_time += timing[i];
            if (timing[i] < min_time || i == 0) min_time = timing[i];
            if (timing[i] > max_time || i == 0) max_time = timing[i];
            perf[i] = 0.000001 * nedges / (finish - start);
            avg_perf += perf[i];
            if (perf[i] < min_perf || i == 0) min_perf = perf[i];
            if (perf[i] > max_perf || i == 0) max_perf = perf[i];
        }

        // calc_traversed_edges(&g, local_dist);
    
        if (rootNumberToValidate == i) {
            /* writing for validation */
            write_validate(&g, local_dist);
        } 
    }
    if (g.rank == 0) {
        cout << "Time:" << endl;
        cout << "Average: " << avg_time / g.nRoots << " Min: " << min_time << " Max: " << max_time << endl;
        cout << "MTEPS:" << endl;
        cout << "Average: " << avg_perf / g.nRoots << " Min: " << min_perf << " Max: " << max_perf << endl;
    }
    free(local_dist);
    free(perf);
    free(timing);
    freeGraph(&g);
    MPI_Finalize();
    return 0;
}
