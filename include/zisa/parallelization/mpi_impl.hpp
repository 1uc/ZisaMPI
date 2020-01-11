#ifndef ZISA_MPI_IMPL_HPP_CUIUQ
#define ZISA_MPI_IMPL_HPP_CUIUQ

#include "mpi_decl.hpp"

namespace zisa {
namespace mpi {

template <class T, int n_dims>
void send(const array_const_view<T, n_dims, row_major> &arr,
          int receiver,
          int tag,
          MPI_Comm comm) {

  auto ptr = (void *)arr.raw();
  auto size = arr.size() * sizeof(T);

  auto code = MPI_Send(ptr, size, MPI_BYTE, receiver, tag, comm);
  LOG_ERR_IF(code != MPI_SUCCESS, string_format("MPI_Send failed. [%d]", code));
}

template <class T, int n_dims>
Request irecv(array_view<T, n_dims, row_major> &arr,
              int sender,
              int tag,
              MPI_Comm comm) {

  auto ptr = (void *)arr.raw();
  auto size = arr.size() * sizeof(T);
  auto request = std::make_unique<MPI_Request>();

  auto code = MPI_Irecv(ptr, size, MPI_BYTE, sender, tag, comm, request.get());
  LOG_ERR_IF(code != MPI_SUCCESS,
             string_format("MPI_Irecv failed. [%d]", code));

  return Request(std::move(request));
}

template <class T, int n_dims>
Request isend(const array_const_view<T, n_dims, row_major> &arr,
              int receiver,
              int tag,
              MPI_Comm comm) {

  auto ptr = (void *)arr.raw();
  auto size = arr.size() * sizeof(T);
  auto request = std::make_unique<MPI_Request>();

  auto code
      = MPI_Isend(ptr, size, MPI_BYTE, receiver, tag, comm, request.get());
  LOG_ERR_IF(code != MPI_SUCCESS,
             string_format("MPI_Isend failed. [%d]", code));

  return Request(std::move(request));
}

}
}
#endif // ZISA_MPI_IMPL_HPP
