include_directories(.)
set(FileIO_sources
    FileIO.cc
    FlatFileIO.cc
    SidreFileIO.cc
    SiloFileIO.cc
    PyFileIO.cc
    vectorstringUtilities.cc
    )

set(FileIO_headers
    DbFileIO.hh
    FileIO.hh
    FileIOInline.hh
    FlatFileIO.hh
    HDF5IO.hh
    HDF5Traits.hh
    HDF5Types.hh
    PyFileIO.hh
    SidreFileIO.hh
    SiloFileIO.hh
    vectorstringUtilities.hh
    )

spheral_install_python_files(
  GzipFileIO.py
  PlyFileIO.py
  )

spheral_add_obj_library(FileIO)

