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
option(TILESTREAM_BUILD_FUZZERS "Build libFuzzer targets (Clang)" OFF)
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
    # The library exports CUDA::cudart, so no target may add a second, static runtime.
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
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
    list(APPEND TILESTREAM_SOURCES
        src/cuda/device_info.cu src/cuda/plan.cpp src/cuda/buffer.cpp
        src/cuda/encoder.cu src/cuda/decoder.cu src/cuda/calibrate.cu)
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

    # The calibration error suite records the flags it was measured with. Host code never
    # contracts into an FMA, so the host reference rounds exactly like the device.
    string(TOUPPER "${CMAKE_BUILD_TYPE}" build_type)
    string(STRIP "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${build_type}} -ffp-contract=off"
        numerics_cxx_flags)
    string(STRIP "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${build_type}}" numerics_cuda_flags)
    function(tilestream_numerics target)
        target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
        target_compile_definitions(${target} PRIVATE
            TILESTREAM_NUMERICS_CXX_FLAGS="${numerics_cxx_flags}"
            TILESTREAM_NUMERICS_CUDA_FLAGS="${numerics_cuda_flags}"
        )
    endfunction()

    add_executable(tilestream_numerics_tests tests/numerics_test.cpp)
    target_link_libraries(tilestream_numerics_tests PRIVATE GTest::gtest_main)
    tilestream_numerics(tilestream_numerics_tests)
    tilestream_enable_warnings(tilestream_numerics_tests)
    add_test(NAME tilestream.cpu.numerics COMMAND tilestream_numerics_tests)
    set_tests_properties(tilestream.cpu.numerics PROPERTIES LABELS "cpu;numerics")
    set_tests_properties(tilestream.cpu.codec tilestream.cpu.fp32 PROPERTIES LABELS "cpu;codec")

    add_executable(tilestream_robustness_tests tests/robustness_test.cpp)
    target_link_libraries(tilestream_robustness_tests PRIVATE TileStream::tilestream GTest::gtest_main)
    target_include_directories(tilestream_robustness_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/fuzz)
    tilestream_enable_warnings(tilestream_robustness_tests)
    add_test(NAME tilestream.cpu.robustness COMMAND tilestream_robustness_tests)
    set_tests_properties(tilestream.cpu.robustness PROPERTIES LABELS "cpu;robustness")

    # The fuzz corpora and minimal reproducers of fixed bugs run as ordinary tests in every build.
    foreach(target parse mutation)
        set(inputs ${CMAKE_CURRENT_SOURCE_DIR}/fuzz/corpus/${target})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/fuzz/regressions/${target})
            list(APPEND inputs ${CMAKE_CURRENT_SOURCE_DIR}/fuzz/regressions/${target})
        endif()
        add_executable(tilestream_${target}_replay fuzz/${target}_fuzzer.cpp fuzz/replay.cpp)
        target_link_libraries(tilestream_${target}_replay PRIVATE TileStream::tilestream)
        target_include_directories(tilestream_${target}_replay PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/fuzz)
        tilestream_enable_warnings(tilestream_${target}_replay)
        add_test(NAME tilestream.cpu.fuzz_${target} COMMAND tilestream_${target}_replay ${inputs})
        set_tests_properties(tilestream.cpu.fuzz_${target} PROPERTIES LABELS "cpu;robustness")
        if(TILESTREAM_BUILD_FUZZERS)
            add_executable(tilestream_${target}_fuzzer fuzz/${target}_fuzzer.cpp)
            target_link_libraries(tilestream_${target}_fuzzer PRIVATE TileStream::tilestream)
            target_include_directories(tilestream_${target}_fuzzer PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/fuzz)
            target_link_options(tilestream_${target}_fuzzer PRIVATE -fsanitize=fuzzer)
            tilestream_enable_warnings(tilestream_${target}_fuzzer)
        endif()
    endforeach()

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

        foreach(suite encoder decoder api loading calibration numerics robustness)
            add_executable(tilestream_cuda_${suite}_tests
                tests/cuda_${suite}_test.cpp tests/cuda_test_main.cpp)
            target_link_libraries(tilestream_cuda_${suite}_tests PRIVATE
                TileStream::tilestream GTest::gtest)
            target_compile_definitions(tilestream_cuda_${suite}_tests PRIVATE
                TILESTREAM_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/golden"
            )
            tilestream_enable_warnings(tilestream_cuda_${suite}_tests)
            add_test(NAME tilestream.cuda.${suite} COMMAND tilestream_cuda_${suite}_tests)
            set_tests_properties(tilestream.cuda.${suite} PROPERTIES
                LABELS "cuda" SKIP_RETURN_CODE 77)
        endforeach()
        tilestream_numerics(tilestream_cuda_numerics_tests)
        target_include_directories(tilestream_cuda_robustness_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        set_tests_properties(tilestream.cuda.encoder tilestream.cuda.decoder PROPERTIES
            LABELS "cuda;codec")
        set_tests_properties(tilestream.cuda.calibration PROPERTIES LABELS "cuda;calibration")
        set_tests_properties(tilestream.cuda.numerics PROPERTIES LABELS "cuda;numerics")
        set_tests_properties(tilestream.cuda.robustness PROPERTIES LABELS "cuda;robustness")

        # Calibration error analysis; only fast_math.cu is built with --use_fast_math.
        add_executable(tilestream_numerics_report tests/numerics/report.cpp
            tests/numerics/variants.cu tests/numerics/fast_math.cu)
        target_link_libraries(tilestream_numerics_report PRIVATE TileStream::tilestream)
        set_source_files_properties(tests/numerics/fast_math.cu PROPERTIES
            COMPILE_OPTIONS "--use_fast_math")
        tilestream_numerics(tilestream_numerics_report)
        target_compile_definitions(tilestream_numerics_report PRIVATE
            TILESTREAM_NUMERICS_CUDA_COMPILER="${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}"
            TILESTREAM_NUMERICS_FAST_FLAGS="--use_fast_math"
            TILESTREAM_DATASETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/datasets/fixtures"
        )
        tilestream_enable_warnings(tilestream_numerics_report)
    endif()
endif()

if(TILESTREAM_BUILD_BENCHMARKS)
    add_executable(tilestream_benchmark benchmarks/smoke_benchmark.cpp)
    target_link_libraries(tilestream_benchmark PRIVATE TileStream::tilestream)
    tilestream_enable_warnings(tilestream_benchmark)
    if(TILESTREAM_ENABLE_CUDA)
        add_executable(tilestream_calibration_benchmark benchmarks/calibration_benchmark.cpp)
        target_link_libraries(tilestream_calibration_benchmark PRIVATE TileStream::tilestream)
        tilestream_enable_warnings(tilestream_calibration_benchmark)
    endif()
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
