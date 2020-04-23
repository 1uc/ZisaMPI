#ifndef ZISA_MPI_IMPL_HPP_CUIUQ
#define ZISA_MPI_IMPL_HPP_CUIUQ

#include "zisa/mpi/mpi_decl.hpp"

#if (ZISA_HAS_MPI != 1)
#error "Using MPI with MPI support."
#endif

namespace zisa {
namespace mpi {

template <class T, int n_dims>
void send(const array_const_view<T, n_dims, row_major> &arr,
          int receiver,
          int tag,
          MPI_Comm comm) {

  auto ptr = (void *)arr.raw();
  auto n_bytes = integer_cast<int>(arr.size() * sizeof(T));

  auto code = MPI_Send(ptr, n_bytes, MPI_BYTE, receiver, tag, comm);
  LOG_ERR_IF(code != MPI_SUCCESS, string_format("MPI_Send failed. [%d]", code));
}

template <class T, int n_dims>
Request isend(const array_const_view<T, n_dims, row_major> &arr,
              int receiver,
              int tag,
              MPI_Comm comm) {

  auto ptr = (void *)arr.raw();
  auto n_bytes = integer_cast<int>(arr.size() * sizeof(T));
  auto request = std::make_unique<MPI_Request>();
  *request = MPI_Request{};

  auto code
      = MPI_Isend(ptr, n_bytes, MPI_BYTE, receiver, tag, comm, request.get());
  LOG_ERR_IF(code != MPI_SUCCESS,
             string_format("MPI_Isend failed. [%d]", code));

  return Request(std::move(request));
}

template <class T, int n_dims>
Request irecv(const array_view<T, n_dims, row_major> &arr,
              int sender,
              int tag,
              MPI_Comm comm) {

  auto ptr = (void *)arr.raw();
  auto n_bytes = integer_cast<int>(arr.size() * sizeof(T));
  auto request = std::make_unique<MPI_Request>();
  *request = MPI_Request{};

  auto code
      = MPI_Irecv(ptr, n_bytes, MPI_BYTE, sender, tag, comm, request.get());
  LOG_ERR_IF(code != MPI_SUCCESS,
             string_format("MPI_Irecv failed. [%d]", code));

  return Request(std::move(request));
}

template <class T, int n_dims>
void bcast(const array_view<T, n_dims, row_major> &view,
           int root,
           const MPI_Comm &comm) {

  auto ptr = (void *)view.raw();
  auto n_bytes = integer_cast<int>(view.size() * sizeof(T));

  auto code = MPI_Bcast(ptr, n_bytes, MPI_BYTE, root, comm);
  LOG_ERR_IF(code != MPI_SUCCESS, string_format("MPI_Bcast failed. [%d]", code))
}

// Thanks to:
// https://stackoverflow.com/a/40808411

#ifndef MPI_SIZE_T
#if SIZE_MAX == UCHAR_MAX
#define MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
#define MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
#define MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
#error "`sizeof(size_t)` does not match anything we'd expect."
#endif
#else
#error "Someone already defined `MPI_SIZE_T`."
#endif

}
}
#endif // ZISA_MPI_IMPL_HPP
