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

void wait_all(const std::vector<Request> &requests);

int size(const MPI_Comm &mpi_comm = MPI_COMM_WORLD);
int rank(const MPI_Comm &mpi_comm = MPI_COMM_WORLD);

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

template <class T, int n_dims>
Request irecv(const array_view<T, n_dims, row_major> &arr,
              int sender,
              int tag,
              MPI_Comm comm);

}

template <class T, int n_dims>
void bcast(const array_view<T, n_dims, row_major> &view,
           int root,
           const MPI_Comm &comm);

}

#endif // ZISA_MPI_DECL_HPP