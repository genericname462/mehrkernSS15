cmake_minimum_required(VERSION 3.1)
project(openmp)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/build)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -Wall -O3")

find_package(OpenMP)
if (OPENMP_FOUND)
    message("Found OpenMP! ^_^")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

set(HEADER_FILES lib/stb_image.h lib/stb_image_write.h src/async.h src/imagefilter.h)
set(SOURCE_FILES src/main.c src/async.c src/imagefilter.c)

add_executable(openmp ${SOURCE_FILES})
target_link_libraries(openmp m rt)


#For non-CMake user:
#gcc -fopenmp -std=gnu99 -O3 -o out.exe main.c async.c imagefilter.c -static -lgomp