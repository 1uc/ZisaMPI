cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(ZisaMPI VERSION 0.1)

include("cmake/has_build_type.cmake")
include("cmake/warning_flags.cmake")
include("cmake/fast_debug.cmake")
include("cmake/sanitizers.cmake")
include("cmake/gperftools.cmake")
include("cmake/clang_tidy.cmake")

# -- Dependencies --------------------------------------------------------------
add_library(mpi_dependencies INTERFACE)

# experimental/filesystem
target_link_libraries(mpi_dependencies INTERFACE stdc++fs)

# MPI
if(ZISA_HAS_MPI)
  find_package(MPI REQUIRED)
  target_compile_definitions(mpi_dependencies INTERFACE ZISA_HAS_MPI=1)
  target_link_libraries(mpi_dependencies INTERFACE MPI::MPI_CXX)
endif()

# Logging
if(ZISA_LOG_LEVEL)
  target_compile_definitions(mpi_dependencies INTERFACE ZISA_LOG_LEVEL=${ZISA_LOG_LEVEL})
endif()

# -- Internal dependencies -----------------------------------------------------
find_package(ZisaCore REQUIRED)
target_link_libraries(mpi_dependencies INTERFACE Zisa::core)

find_package(ZisaMemory REQUIRED)
target_link_libraries(mpi_dependencies INTERFACE Zisa::memory)


# -- Zisa ----------------------------------------------------------------------
# -- mpi -----------------------------------------------------------------------
add_library(mpi STATIC "")
set_target_properties(mpi PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(mpi PROPERTIES OUTPUT_NAME zisa_mpi)
set_target_properties(mpi PROPERTIES CXX_STANDARD 17)
target_include_directories(mpi PUBLIC
  $<BUILD_INTERFACE:${ZisaMPI_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(mpi PUBLIC mpi_dependencies)
target_link_libraries(mpi PRIVATE mpi_warning_flags)

# -- clang tidy ----------------------------------------------------------------
if(CLANG_TIDY_EXE)
  set_target_properties(mpi PROPERTIES CXX_CLANG_TIDY "${DO_CLANG_TIDY}")
endif()

# -- install -------------------------------------------------------------------
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/ZisaMPIConfigVersion.cmake"
  VERSION ${ZisaMPI_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(
  FILES
  "cmake/ZisaMPIConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/ZisaMPIConfigVersion.cmake"
  DESTINATION lib/cmake/zisa
)

add_library(Zisa::mpi ALIAS mpi)
install(
  TARGETS mpi mpi_dependencies mpi_warning_flags
  EXPORT ZisaMPITargets
)
include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT ZisaMPITargets
  FILE ZisaMPITargets.cmake
  NAMESPACE Zisa::
  DESTINATION lib/cmake/zisa
)


# -- Subdirectories ------------------------------------------------------------
add_subdirectory(src)
