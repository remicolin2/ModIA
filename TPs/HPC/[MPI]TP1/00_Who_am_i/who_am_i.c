#include <stdio.h>
#include <mpi.h>

int main( int argc, char *argv[] ) {

  int rank, size;
  int l;
  char name[MPI_MAX_PROCESSOR_NAME];
  int namelen ;
  char procname[MPI_MAX_PROCESSOR_NAME];
  // ...
  // Initialise MPI
  int MPI_Init(int* argc, char ***argv);

  // Get my rank
  int MPI_Comm_rank(MPI_Comm comm, int *rank);

  // Get tne number of processor in the communicator MPI_COMM_WORLD
  int MPI_Comm_size(MPI_Comm comm, int *size);

  // Get the name of the processor
  MPI_Get_processor_name(procname, &namelen);

  printf("Rank %d is on machine %s\n", rank, procname);
  printf("Hello world from process %d of %d on processor named %s\n", rank, size, procname);

  MPI_Finalize();
  // ...
  
  return 0;
}
