import mpi4py.MPI as MPI

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    print("Rank ID %s" % comm_rank)
