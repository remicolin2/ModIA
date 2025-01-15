#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cblas.h>
#include "utils.h"
#include "dsmat.h"
#include "gemms.h"

void p2p_transmit_A(int p, int q, Matrix *A, int i, int l) {
    int j;
    int me, my_row, my_col;
    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    node_coordinates_2i(p, q, me, &my_row, &my_col);

    Block *Ail;
    int node, tag, b;
    tag = 0;

    Ail = &A->blocks[i][l];
    b = A->b;

    /* TODO : transmit A[i,l] using MPI_Ssend & MPI_Recv */
    if (me == Ail -> owner) {  /* I own A[i,l]*/
        for (j = 0; j < q; j++) {
              node = get_node(p, q, my_row, j);
              if (node != me) {
                /* MPI_Ssend A[i,l] to my row */
                MPI_Ssend(Ail->c, b * b, MPI_FLOAT, node, tag, MPI_COMM_WORLD);
              }
        }
    } else if (my_row == Ail->row) {  /* A[i,l] is stored on my row */
        Ail->c = malloc(b * b * sizeof(float));
        node = Ail->owner;
        /* MPI_Recv A[i,l] */
        MPI_Recv(Ail->c, b * b, MPI_FLOAT, node, tag, MPI_COMM_WORLD, &status);
    }
    /* end TODO */
}


void p2p_transmit_B(int p, int q, Matrix *B, int l, int j) {
    int i;
    int me, my_row, my_col;
    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    node_coordinates_2i(p, q, me, &my_row, &my_col);

    int node, tag = 1, b;
    Block *Blj;

    Blj = &B->blocks[l][j];
    b = B->b;

    /* TODO : transmit B[l,j] using MPI_Ssend & MPI_Recv */
    if (me == Blj -> owner) {  /* I owned B[l,j]*/
        for (i = 0; i < p; i++) {
          node = get_node(p, q, i, my_col);
          if (node != me) {
              /* MPI_Ssend B[l,j] to my column */
              MPI_Ssend(Blj->c, b * b, MPI_FLOAT, node, tag, MPI_COMM_WORLD);
            }
        }
    } else if (my_col == Blj->col) {  /* B[l,j] is stored on my column */
        Blj->c = malloc(b * b * sizeof(float));
        node = Blj->owner;
        /* MPI_Recv B[l,j] */
        MPI_Recv(Blj->c, b * b, MPI_FLOAT, node, tag, MPI_COMM_WORLD, &status);
    }
    /* end TODO */
}

