# ZisaMPI
[![Build Status](https://github.com/1uc/ZisaMPI/actions/workflows/basic_integrity_checks.yml/badge.svg?branch=main)](https://github.com/1uc/ZisaMPI/actions)
[![Docs Status](https://github.com/1uc/ZisaMPI/actions/workflows/publish_docs.yml/badge.svg?branch=main)](https://1uc.github.io/ZisaMPI)

ZisaMPI provides a thin and incomplete wrapper around MPI, enabling sending
arrays from one MPI task to another with less code.

## Quickstart
Start by cloning the repository

    $ git clone https://github.com/1uc/ZisaMPI.git

and change into the newly created directory. Then proceed to install the
dependencies:

    $ bin/install_dir.sh COMPILER DIRECTORY DEPENDENCY_FLAGS

they will be placed into a subdirectory of `DIRECTORY` and print
part of the CMake command needed to include the dependencies. `COMPILER` must
be replaced with the compiler you want to use. The available `DEPENDENCY_FLAGS`
are

  * `--zisa_has_cuda={0,1}` to request CUDA.
  * `--zisa_has_hdf5={0,1}` to request HDF5 support for arrays.
  * `--zisa_has_netcdf={0,1}` to request NetCDF support for arrays.
  * `--zisa_has_mpi={0,1}` which defaults to `1`.

If this worked continue by running the `cmake` command and compiling the
library. Take a look at the [project specific flags] for CMake if you want to
modify something. If this didn't work, it's not going to be a quick start.
Please read [Dependencies] and then [Building].

[project specific flags]: https://1uc.github.io/ZisaMPI/md_building.html#cmake_flags
[Dependencies]: https://1uc.github.io/ZisaMPI/md_dependencies.html
[Building]: https://1uc.github.io/ZisaMPI/md_building.html
