#ifndef ZISA_MPI_IMPL_HPP_CUIUQ
#define ZISA_MPI_IMPL_HPP_CUIUQ

#include "zisa/mpi/mpi_decl.hpp"

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
  *request = MPI_Request{}; // Looks dodgy...

  auto code
      = MPI_Isend(ptr, n_bytes, MPI_BYTE, receiver, tag, comm, request.get());
  LOG_ERR_IF(code != MPI_SUCCESS,
             string_format("MPI_Isend failed. [%d]", code));

  return Request(std::move(request));
}

template <class POD>
Request isend_pod(const POD &pod, int receiver, int tag, MPI_Comm comm) {
  auto ptr = (void *)(&pod);
  auto n_bytes = integer_cast<int>(sizeof(POD));

  auto request = std::make_unique<MPI_Request>();
  *request = MPI_Request{};

  auto code
      = MPI_Isend(ptr, n_bytes, MPI_BYTE, receiver, tag, comm, request.get());

  LOG_ERR_IF(code != MPI_SUCCESS,
             string_format("MPI_Isend failed. [%d]", code));

  return Request(std::move(request));
}

template <class POD>
std::pair<POD, Status> recv_pod(int src, int tag, MPI_Comm comm) {
  auto pod = POD{};
  auto ptr = (void *)(&pod);
  auto n_bytes = integer_cast<int>(sizeof(pod));
  auto status = MPI_Status();

  auto code = MPI_Recv(ptr, n_bytes, MPI_BYTE, src, tag, comm, &status);
  LOG_ERR_IF(code != MPI_SUCCESS, "Failed `recv_pod`.");

  return {pod, Status(status)};
}

template <class T, int n_dims>
Status recv(const array_view<T, n_dims, row_major> &arr,
            int sender,
            int tag,
            MPI_Comm comm) {

  auto ptr = (void *)arr.raw();
  auto n_bytes = integer_cast<int>(arr.size() * sizeof(T));
  auto status = MPI_Status{};

  auto code = MPI_Recv(ptr, n_bytes, MPI_BYTE, sender, tag, comm, &status);
  LOG_ERR_IF(code != MPI_SUCCESS,
             string_format("MPI_Irecv failed. [%d]", code));

  return Status(status);
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

template <class T>
void gather(const array_view<T, 1, row_major> &iobuff,
            int root,
            const MPI_Comm &comm) {
  assert(zisa::mpi::test_intra(comm));
  auto rank = zisa::mpi::rank(comm);
  auto size = zisa::mpi::size(comm);

  auto ptr = (void *)raw_ptr(iobuff);
  auto n_elements
      = (rank == root ? iobuff.size() / int_t(size) : iobuff.size());
  auto n_bytes = integer_cast<int>(n_elements * sizeof(iobuff[0]));

  if (rank == root) {
    auto code = MPI_Gather(
        MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, ptr, n_bytes, MPI_BYTE, root, comm);
    LOG_ERR_IF(code != MPI_SUCCESS, "Root failed MPI_Allgather.");
  } else {
    auto code = MPI_Gather(
        ptr, n_bytes, MPI_BYTE, nullptr, 0, MPI_DATATYPE_NULL, root, comm);
    LOG_ERR_IF(code != MPI_SUCCESS, "Worker failed MPI_Allgather.");
  }
}

template <class T>
void allgather(const array_view<T, 1, row_major> &view, const MPI_Comm &comm) {
  // If it's an intracomm we can do inplace.
  assert(zisa::mpi::test_intra(comm));

  auto ptr = (void *)raw_ptr(view);
  auto n_bytes = integer_cast<int>(sizeof(view[0]));

  auto code = MPI_Allgather(
      MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, ptr, n_bytes, MPI_BYTE, comm);

  LOG_ERR_IF(code != MPI_SUCCESS, "Failed MPI_Allgather.");
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
