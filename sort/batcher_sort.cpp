#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <utility>
#include <float.h>
#include <cstring>
#include <algorithm>
#include <iterator>
#include <mpi.h>


void arr_gen(int n, int nproc, int myrank, std::vector<double>& arr, int arr_size)
{
    srand(nproc + myrank);
    int len = arr_size;
    if (myrank == nproc - 1) {
        if (n % arr_size) {
            len = n % arr_size;
        }
    }
    for (int i = 0; i < len; i++) {
        arr.push_back(((double) rand() / RAND_MAX) * 2000 - 1000);
    }
    if (len < arr_size) {
        for (int i = len; i < arr_size; i++) {
            arr.push_back(DBL_MAX);
        }
    }
}



void S (int first1, int first2, int step, int count1, int count2, std::vector<std::pair<int, int> >& comps, int myrank) {
    int count11, count21, i;
    if (count1 * count2 < 1) {
        return;
    }
    if ( count1 * count2 == 1) {
        if (myrank == first1 || myrank == first2) {
            comps.push_back(std::pair<int, int>(first1, first2));
        }
        return;
    }
    
    count11 = count1 - count1 / 2;
    count21 = count2 - count2 / 2;
    S(first1, first2, 2 * step, count11, count21, comps, myrank);
    S(first1 + step, first2 + step, 2 * step, count1 - count11, count2 - count21, comps, myrank);

    for (i = 1; i < count1 - 1; i += 2) {
        if (myrank == first1 + step * i || myrank == first1 + step * (i + 1)) {
            comps.push_back(std::pair<int, int>(first1 + step * i, first1 + step * (i + 1)));
        }
    }
    if (count1 % 2 == 0) {
        if (myrank == first1 + step * (count1 - 1) || myrank == first2) {
            comps.push_back(std::pair<int, int>(first1 + step * (count1 - 1), first2));
        }
        i = 1;
    } else {
        i = 0;
    }
    for (; i < count2 - 1; i += 2) {
        if (myrank == first2 + step * i || myrank == first2 + step * (i + 1)) {
            comps.push_back(std::pair<int, int>(first2 + step * i, first2 + step * (i + 1)));
        }
    }
}

void B (int first, int step, int count, std::vector<std::pair<int, int> >& comps, int myrank) {
    if (count < 2) {
        return;
    }
    if (count == 2) {
        if (myrank == first || myrank == first + step) {
            comps.push_back(std::pair<int, int>(first, first + step));
        }
        return;
    }
    int count1 = count / 2;
    B(first, step, count1, comps, myrank);
    B(first + step * count1, step, count - count1, comps, myrank);
    S(first, first + step * count1, step, count1, count - count1, comps, myrank);
}

void Batcher(int n, std::vector<std::pair<int, int> >& comps, int myrank)
{
    B(0, 1, n, comps, myrank);
}



int main(int argc, char **argv) {
    int n = atoi(argv[1]), nproc, myrank;
    double start = 0, end = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    std::vector<std::pair<int, int>> comps;
    int arr_size = (n + nproc - 1) / nproc;
    std::vector<double> arr;
    arr_gen(n, nproc, myrank, arr, arr_size);
    MPI_Barrier(MPI_COMM_WORLD);
    if (!myrank) {
        start = MPI_Wtime();
    }
    std::sort(arr.begin(), arr.end());
    Batcher(nproc, comps, myrank);
    double *tmp = new double[arr_size];
    double *res = new double[arr_size];

    for (auto i : comps) {
        int partner = i.first + i.second - myrank;
        MPI_Sendrecv(&arr[0], arr_size, MPI_DOUBLE, partner, 0, tmp, arr_size, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (partner > myrank) {
            for(int ia = 0, ib = 0, k = 0; k < arr_size; k++)
            {
                if(arr[ia] < tmp[ib]) {
                    res[k] = arr[ia];
                    ia++;
                } else {
                    res[k] = tmp[ib];
                    ib++;
                }
            }
        } else {
            for(int ia = arr_size - 1, ib = arr_size - 1, k = arr_size - 1; k >= 0; k--)
            {
                if(arr[ia] > tmp[ib]) {
                    res[k] = arr[ia];
                    ia--;
                } else {
                    res[k] = tmp[ib];
                    ib--;
                }
            }
        }
        for (int i = 0; i < arr_size; i++) {
            arr[i] = res[i];
        }
    }
    delete [] tmp;
    delete [] res;
    MPI_Barrier(MPI_COMM_WORLD);
    if (!myrank) {
        end = MPI_Wtime();
    }
    int flag = 0;
    double last = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    if (!myrank) {
        for (int j = 0; j < arr_size - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                flag = 1;
                std::cout << "Wrong sort on proc " << myrank << std::endl;
            }
        }
        last = arr[arr_size - 1];
        if (nproc != 1) {
            MPI_Send(&last, 1, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
            MPI_Send(&flag, 1, MPI_INT, 1, 2, MPI_COMM_WORLD);
        }
    }
    for (int i = 1; i < nproc; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (myrank == i) {
            MPI_Recv(&last, 1, MPI_DOUBLE, i - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&flag, 1, MPI_INT, i - 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (last > arr[0]) {
                flag = 1;
                std::cout << "Error between procs " << myrank - 1 << " and " << myrank << std::endl;
            }
            for (int j = 0; j < arr_size - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    flag = 1;
                    std::cout << "Wrong sort on proc " << myrank << std::endl;
                }
            }
            last = arr[arr_size - 1];
            if (i != nproc - 1) {
                MPI_Send(&last, 1, MPI_DOUBLE, i + 1, 1, MPI_COMM_WORLD);
            }
            MPI_Send(&flag, 1, MPI_INT, (i + 1) % nproc, 2, MPI_COMM_WORLD);
        }
    }
    if (!myrank) {
        if (nproc != 1) {
            MPI_Recv(&flag, 1, MPI_INT, nproc - 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (!flag) {
            std::cout << "Sorted correctly" << std::endl;
        }
        std::cout << "Time: " << end - start << std::endl;
    }
    MPI_Finalize();
    return 0;
}
