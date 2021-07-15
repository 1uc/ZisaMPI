// SPDX-License-Identifier: MIT
// Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

#include <mpi.h>
#include <zisa/mpi/mpi.hpp>

namespace zisa {
namespace mpi {

std::string error_message(int error_code) {
  int strlen = -1;
  char cstr[MPI_MAX_ERROR_STRING];
  auto code = MPI_Error_string(error_code, cstr, &strlen);
  LOG_ERR_IF(code != MPI_SUCCESS, "Failed `MPI_Error_string`.");

  return std::string(cstr);
}

void wait(const Request &request) { request.wait(); }

void wait(MPI_Request *const request_ptr) {
  auto code = MPI_Wait(request_ptr, MPI_STATUS_IGNORE);

  LOG_ERR_IF(code != MPI_SUCCESS,
             string_format("Failed `MPI_Wait`. [%x][%s]",
                           request_ptr,
                           zisa::mpi::error_message(code).c_str()));
}

Request::Request(std::unique_ptr<MPI_Request> request_) {
  this->request = std::move(request_);
}

void Request::wait() const {
  if (request != nullptr) {
    zisa::mpi::wait(request_ptr());
  }
}

Request::~Request() {
  if (request != nullptr) {
    int is_complete;
    auto status = MPI_Test(request_ptr(), &is_complete, MPI_STATUS_IGNORE);

    LOG_WARN_IF(status != MPI_SUCCESS,
                string_format("Failed `MPI_Test`. [%d]", status));

    // This would be an outright error, if it weren't for destructors being
    // implicitly `noexcept`.
    // This is probably never a harmless warning.
    LOG_WARN_IF(is_complete == 0,
                " !! Unfinished non-blocking communication !!!!!");
  }
}

MPI_Request *Request::request_ptr() const {
  if (request != nullptr) {
    return request.get();
  }
  LOG_ERR("Calling `request_ptr` on null request.");
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

void barrier() { zisa::mpi::barrier(MPI_COMM_WORLD); }
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

std::string comm_get_name(const MPI_Comm &comm) {
  auto comm_name = std::string(MPI_MAX_OBJECT_NAME, '\0');
  int comm_name_len = -1;
  auto code = MPI_Comm_get_name(comm, comm_name.data(), &comm_name_len);

  LOG_ERR_IF(
      code != MPI_SUCCESS,
      string_format("Failed `MPI_Comm_get_name`. [%s]", error_message(code)));

  return comm_name;
}
}
}
