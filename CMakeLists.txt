cmake_minimum_required(VERSION 3.10)
project(ciscy)

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/source/*.c")

add_executable(ciscy ${SOURCES})