cmake_minimum_required(VERSION 3.1)
project(openmp)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Wall -O2")

set(SOURCE_FILES
    lib/stb_image.h
    lib/stb_image_write.h
    src/main.c)

add_executable(openmp ${SOURCE_FILES})
target_link_libraries(openmp m)