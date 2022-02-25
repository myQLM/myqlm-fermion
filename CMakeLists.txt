project(qat-dqs)
cmake_minimum_required(VERSION 3.10.2)

include($ENV{QATDIR}/cmake-common/qlm.cmake)

#
# Python library
#
add_subdirectory(python)

#
# Unit tests
#
add_subdirectory(tests)
