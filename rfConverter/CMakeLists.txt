cmake_minimum_required(VERSION 3.29)
project(rfConverter)

set(CMAKE_CXX_STANDARD 26)

add_executable(rfConverter main.cpp
        matplotlibcpp.h
)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(/usr/bin/python3 ${PYTHON_LIBRARIES})