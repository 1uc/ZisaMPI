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

Request::~Request() {
  if (request != nullptr) {
    int is_complete;
    MPI_Test(request.get(), &is_complete, MPI_STATUS_IGNORE);

    // This would be an outright error, if it weren't for destructors being
    // implicitly `noexcept`.
    // This is probably never a harmless warning.
    LOG_WARN_IF(is_complete == 0,
                " !! Unfinished non-blocking communication !!!!!");
  }
}

void wait_all(const std::vector<Request> &requests) {
  for (const auto &r : requests) {
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

bool test_intra(MPI_Comm const &comm) { return !test_inter(comm); }
bool test_inter(MPI_Comm const &comm) {
  int flag = -1;
  auto code = MPI_Comm_test_inter(comm, &flag);
  LOG_ERR_IF(code != MPI_SUCCESS, "Failed MPI_Comm_test_inter.");

  return bool(flag);
}

void barrier(MPI_Comm const &comm) { MPI_Barrier(comm); }

MPI_Comm comm_split(MPI_Comm old_comm, int color, int rank) {
  MPI_Comm new_comm;
  auto status = MPI_Comm_split(old_comm, color, rank, &new_comm);
  LOG_ERR_IF(status != 0, string_format("Failed MPI_Comm_split. [%d]", status));

  return new_comm;
}

Status::Status(int source, int tag, int error)
    : source(source), tag(tag), error(error) {}

Status::Status(const MPI_Status &status)
    : source(status.MPI_SOURCE), tag(status.MPI_TAG), error(status.MPI_ERROR) {}
}

}