cmake_minimum_required(VERSION 3.25)

project(TileStream VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(TIFF 4.5.1 REQUIRED)

option(TILESTREAM_ENABLE_CUDA "Build CUDA support" OFF)
option(TILESTREAM_ENABLE_NVCOMP "Build the optional nvCOMP adapter" OFF)
option(TILESTREAM_BUILD_TESTS "Build tests" ON)
option(TILESTREAM_BUILD_BENCHMARKS "Build benchmarks" ON)
option(TILESTREAM_BUILD_PYTHON "Build Python bindings" ON)
set(TILESTREAM_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures used by TileStream")

if(TILESTREAM_ENABLE_NVCOMP AND NOT TILESTREAM_ENABLE_CUDA)
    message(FATAL_ERROR "TILESTREAM_ENABLE_NVCOMP requires TILESTREAM_ENABLE_CUDA")
endif()

if(TILESTREAM_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit 13.0 REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    set(CMAKE_CUDA_ARCHITECTURES "${TILESTREAM_CUDA_ARCHITECTURES}")
endif()

if(TILESTREAM_ENABLE_NVCOMP)
    find_package(nvcomp CONFIG REQUIRED)
endif()

function(tilestream_enable_warnings target)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wsign-conversion>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wconversion,-Wsign-conversion>
    )
endfunction()

set(TILESTREAM_WITH_CUDA 0)
set(TILESTREAM_CUDA_VERSION "not-built")
if(TILESTREAM_ENABLE_CUDA)
    set(TILESTREAM_WITH_CUDA 1)
    set(TILESTREAM_CUDA_VERSION "${CUDAToolkit_VERSION}")
endif()

set(TILESTREAM_WITH_NVCOMP 0)
if(TILESTREAM_ENABLE_NVCOMP)
    set(TILESTREAM_WITH_NVCOMP 1)
endif()

configure_file(
    cmake/build_config.h.in
    "${CMAKE_CURRENT_BINARY_DIR}/generated/tilestream/build_config.h"
    @ONLY
)

set(TILESTREAM_SOURCES
    src/info.cpp
    src/sizing.cpp
    src/cpu/parser.cpp
    src/cpu/encoder.cpp
    src/cpu/decoder.cpp
    src/io/tiff.cpp
)
if(TILESTREAM_ENABLE_CUDA)
    list(APPEND TILESTREAM_SOURCES src/cuda/device_info.cu)
endif()

add_library(tilestream_library ${TILESTREAM_SOURCES})
add_library(TileStream::tilestream ALIAS tilestream_library)
set_target_properties(tilestream_library PROPERTIES
    EXPORT_NAME tilestream
    OUTPUT_NAME tilestream
    POSITION_INDEPENDENT_CODE ON
)
target_compile_features(tilestream_library PUBLIC cxx_std_17)
target_include_directories(tilestream_library
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
tilestream_enable_warnings(tilestream_library)
target_link_libraries(tilestream_library PRIVATE TIFF::TIFF)

if(TILESTREAM_ENABLE_CUDA)
    target_link_libraries(tilestream_library PUBLIC CUDA::cudart)
endif()

add_executable(tilestream_cli tools/tilestream.cpp)
set_target_properties(tilestream_cli PROPERTIES OUTPUT_NAME tilestream)
target_link_libraries(tilestream_cli PRIVATE TileStream::tilestream)
tilestream_enable_warnings(tilestream_cli)

if(TILESTREAM_BUILD_TESTS)
    include(CTest)
    add_executable(tilestream_tests tests/smoke_test.cpp)
    target_link_libraries(tilestream_tests PRIVATE TileStream::tilestream)
    tilestream_enable_warnings(tilestream_tests)
    add_test(NAME tilestream.cpu.smoke COMMAND tilestream_tests)
    set_tests_properties(tilestream.cpu.smoke PROPERTIES LABELS "cpu")

    add_executable(tilestream_sizing_tests tests/sizing_test.cpp)
    target_link_libraries(tilestream_sizing_tests PRIVATE TileStream::tilestream)
    target_include_directories(tilestream_sizing_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    tilestream_enable_warnings(tilestream_sizing_tests)
    add_test(NAME tilestream.cpu.sizing COMMAND tilestream_sizing_tests)
    set_tests_properties(tilestream.cpu.sizing PROPERTIES LABELS "cpu")

    find_package(GTest 1.14.0 EXACT CONFIG REQUIRED)
    add_executable(tilestream_cpu_codec_tests tests/cpu_codec_test.cpp)
    target_link_libraries(tilestream_cpu_codec_tests PRIVATE TileStream::tilestream GTest::gtest_main)
    target_compile_definitions(tilestream_cpu_codec_tests PRIVATE
        TILESTREAM_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/golden"
    )
    tilestream_enable_warnings(tilestream_cpu_codec_tests)
    add_test(NAME tilestream.cpu.codec COMMAND tilestream_cpu_codec_tests)
    set_tests_properties(tilestream.cpu.codec PROPERTIES LABELS "cpu")

    add_executable(tilestream_fp32_codec_tests tests/fp32_codec_test.cpp)
    target_link_libraries(tilestream_fp32_codec_tests PRIVATE TileStream::tilestream GTest::gtest_main)
    target_compile_definitions(tilestream_fp32_codec_tests PRIVATE
        TILESTREAM_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/golden"
    )
    tilestream_enable_warnings(tilestream_fp32_codec_tests)
    add_test(NAME tilestream.cpu.fp32 COMMAND tilestream_fp32_codec_tests)
    set_tests_properties(tilestream.cpu.fp32 PROPERTIES LABELS "cpu")

    add_executable(tilestream_tiff_tests tests/tiff_test.cpp)
    target_link_libraries(tilestream_tiff_tests PRIVATE TileStream::tilestream TIFF::TIFF GTest::gtest_main)
    tilestream_enable_warnings(tilestream_tiff_tests)
    add_test(NAME tilestream.io.tiff COMMAND tilestream_tiff_tests)
    set_tests_properties(tilestream.io.tiff PROPERTIES LABELS "cpu;tiff")

    add_executable(tilestream_tiff_interop tests/tiff_interop.cpp)
    target_link_libraries(tilestream_tiff_interop PRIVATE TileStream::tilestream)
    tilestream_enable_warnings(tilestream_tiff_interop)

    set_tests_properties(tilestream.cpu.codec tilestream.cpu.fp32 PROPERTIES LABELS "cpu;codec")

    if(TILESTREAM_ENABLE_CUDA)
        add_executable(tilestream_cuda_header_tests tests/cuda_headers_test.cu)
        target_link_libraries(tilestream_cuda_header_tests PRIVATE TileStream::tilestream)
        tilestream_enable_warnings(tilestream_cuda_header_tests)
        add_test(NAME tilestream.cuda.headers COMMAND tilestream_cuda_header_tests)
        set_tests_properties(tilestream.cuda.headers PROPERTIES LABELS "headers")
        add_executable(tilestream_cuda_tests tests/cuda_device_test.cu)
        target_link_libraries(tilestream_cuda_tests PRIVATE CUDA::cudart)
        tilestream_enable_warnings(tilestream_cuda_tests)
        add_test(NAME tilestream.cuda.device COMMAND tilestream_cuda_tests)
        set_tests_properties(tilestream.cuda.device PROPERTIES LABELS "cuda" SKIP_RETURN_CODE 77)

    endif()
endif()

if(TILESTREAM_BUILD_BENCHMARKS)
    add_executable(tilestream_benchmark benchmarks/smoke_benchmark.cpp)
    target_link_libraries(tilestream_benchmark PRIVATE TileStream::tilestream)
    tilestream_enable_warnings(tilestream_benchmark)
endif()

if(TILESTREAM_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 2.11.1 EXACT CONFIG REQUIRED)
    pybind11_add_module(tilestream_python MODULE bindings/python_module.cpp)
    set_target_properties(tilestream_python PROPERTIES OUTPUT_NAME _tilestream)
    target_link_libraries(tilestream_python PRIVATE TileStream::tilestream)
    tilestream_enable_warnings(tilestream_python)
endif()

install(TARGETS tilestream_library tilestream_cli
    EXPORT TileStreamTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/tilestream/build_config.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/tilestream"
)

set(TILESTREAM_CONFIG_NEEDS_CUDA "${TILESTREAM_ENABLE_CUDA}")
configure_package_config_file(
    cmake/TileStreamConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/TileStreamConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/TileStream"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/TileStreamConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)
install(EXPORT TileStreamTargets
    FILE TileStreamTargets.cmake
    NAMESPACE TileStream::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/TileStream"
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/TileStreamConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/TileStreamConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/TileStream"
)

if(TILESTREAM_BUILD_PYTHON)
    install(TARGETS tilestream_python LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/tilestream/python)
endif()
