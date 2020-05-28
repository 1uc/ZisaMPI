#ifndef ZISA_MPI_DECL_HPP_KOIIO
#define ZISA_MPI_DECL_HPP_KOIIO

#include <zisa/memory/array_view.hpp>

namespace zisa {
namespace mpi {

class Request {
public:
  explicit Request(std::unique_ptr<MPI_Request> request);

  /// Wait until the data transfer is complete.
  void wait() const;

private:
  std::unique_ptr<MPI_Request> request = nullptr;
};

struct Status {
  int source;
  int tag;
  int error;

  Status(int source, int tag, int error)
      : source(source), tag(tag), error(error) {}

  explicit Status(const MPI_Status &status)
      : source(status.MPI_SOURCE),
        tag(status.MPI_TAG),
        error(status.MPI_ERROR) {}
};

void wait_all(const std::vector<Request> &requests);

int size(const MPI_Comm &mpi_comm = MPI_COMM_WORLD);
int rank(const MPI_Comm &mpi_comm = MPI_COMM_WORLD);
bool test_intra(const MPI_Comm &comm);
bool test_inter(const MPI_Comm &comm);

template <class T, int n_dims>
void send(const array_const_view<T, n_dims, row_major> &arr,
          int receiver,
          int tag,
          MPI_Comm comm);

template <class T, int n_dims>
Request isend(const array_const_view<T, n_dims, row_major> &arr,
              int receiver,
              int tag,
              MPI_Comm comm);

template <class POD>
Request isend_pod(const POD &pod, int receiver, int tag, MPI_Comm comm);

template <class T, int n_dims>
Status recv(const array_view<T, n_dims, row_major> &arr,
            int sender,
            int tag,
            MPI_Comm comm);

template <class T, int n_dims>
Request irecv(const array_view<T, n_dims, row_major> &arr,
              int sender,
              int tag,
              MPI_Comm comm);

template <class POD>
std::pair<POD, Status> recv_pod(int src, int tag, MPI_Comm comm);

template <class T>
void gather(const array_view<T, 1, row_major> &iobuff,
            int root,
            const MPI_Comm &comm);

template <class T>
void allgather(const array_view<T, 1, row_major> &arr, const MPI_Comm &comm);

template <class T, int n_dims>
void bcast(const array_view<T, n_dims, row_major> &view,
           int root,
           const MPI_Comm &comm);

MPI_Comm comm_split(MPI_Comm old_comm, int color, int rank);

void barrier(const MPI_Comm &comm);

}
}

#endif // ZISA_MPI_DECL_HPP
