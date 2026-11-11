cmake_minimum_required(VERSION 3.25)
project(TileStreamClient LANGUAGES CXX)

find_package(TileStream CONFIG REQUIRED)

add_executable(tilestream_client main.cpp)
target_link_libraries(tilestream_client PRIVATE TileStream::tilestream)
target_compile_features(tilestream_client PRIVATE cxx_std_17)
