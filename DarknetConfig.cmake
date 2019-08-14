# Config file for the Darknet package

get_filename_component(Darknet_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
list(APPEND CMAKE_MODULE_PATH "${Darknet_CMAKE_DIR}")

include(CMakeFindDependencyMacro)

if(1)
  find_dependency(OpenCV)
endif()

find_dependency(Stb)

if(TRUE)
  if(TRUE)
    find_dependency(CUDNN)
  endif()
endif()

set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_dependency(Threads)

if(1)
  find_dependency(PThreads_windows)
endif()

if(TRUE)
  find_dependency(OpenMP)
endif()

# Our library dependencies (contains definitions for IMPORTED targets)
include("${Darknet_CMAKE_DIR}/DarknetTargets.cmake")
