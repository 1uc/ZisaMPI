#include <zisa/mpi/mpi.hpp>

namespace zisa {
namespace mpi {

Request::Request(std::unique_ptr<MPI_Request> request)
    : request(std::move(request)) {}

void Request::wait() const {
  if (request != nullptr) {
    MPI_Wait(request.get(), nullptr);
  }
}

void wait_all(const std::vector<Request> &requests) {
  for(const auto &r : requests) {
    r.wait();
  }
}

int size(const MPI_Comm &mpi_comm) {
  int mpi_ranks = -1;
  MPI_Comm_size(mpi_comm, &mpi_ranks);
  return mpi_ranks;
}

int rank(const MPI_Comm &mpi_comm) {
  int rank = -1;
  MPI_Comm_rank(mpi_comm, &rank);
  return rank;
}

}

}