include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# Common sources needed for tests
file(GLOB LMDB_SOURCES ${CMAKE_SOURCE_DIR}/third_party/mdbx/*.c)
set(ROARING_SOURCE ${CMAKE_SOURCE_DIR}/third_party/roaring_bitmap/roaring.c)

# Create the test executable
add_executable(ndd_filter_test filter_test.cpp ${LMDB_SOURCES} ${ROARING_SOURCE})

# Link against GTest
target_link_libraries(ndd_filter_test GTest::gtest_main)

# Set MDBX compile flags (same as main project)
set_source_files_properties(${LMDB_SOURCES} PROPERTIES
    COMPILE_FLAGS "-DMDBX_BUILD_SHARED_LIBRARY=0 -DMDBX_BUILD_FLAGS=\\\"NDD_EMBEDDED\\\""
)

# Allow tests to include private headers from src/
target_include_directories(ndd_filter_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/filter
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/json
)
# Add other necessary definitions
target_compile_definitions(ndd_filter_test PRIVATE MDB_MAXKEYSIZE=512)

include(GoogleTest)
gtest_discover_tests(ndd_filter_test)

# -------------------------------------------------------
# backup_race_test
# Tests that deleteIndex throws when a backup is active.
# Requires libarchive (backup_store.hpp) and SIMD (ndd.hpp).
# -------------------------------------------------------
add_executable(ndd_backup_race_test
    backup_race_test.cpp
    ${LMDB_SOURCES}
    ${ROARING_SOURCE}
)

target_include_directories(ndd_backup_race_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/storage
    ${CMAKE_SOURCE_DIR}/src/server
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/src/filter
    ${CMAKE_SOURCE_DIR}/src/sparse
    ${CMAKE_SOURCE_DIR}/src/quant
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/json
    ${LIBARCHIVE_INCLUDE_DIR}
    ${ASIO_INCLUDE_DIR}
    ${MSGPACK_INCLUDE_DIR}
    ${LMDB_INCLUDE_DIR}
    ${CURL_INCLUDE_DIRS}
    ${OPENSSL_INCLUDE_DIR}
)

target_compile_definitions(ndd_backup_race_test PRIVATE
    MDB_MAXKEYSIZE=512
    ASIO_STANDALONE
    ASIO_HAS_STD_CHRONO
    ASIO_HAS_STD_STRING_VIEW
)

# Match the SIMD flags used by the main binary
if(USE_AVX512)
    target_compile_options(ndd_backup_race_test PRIVATE -mavx512f -mavx512bw -mavx512vnni -mavx512fp16 -mavx512vpopcntdq)
    target_compile_definitions(ndd_backup_race_test PRIVATE USE_AVX512)
elseif(USE_AVX2)
    target_compile_options(ndd_backup_race_test PRIVATE -mavx2 -mfma -mf16c)
    target_compile_definitions(ndd_backup_race_test PRIVATE USE_AVX2)
elseif(USE_SVE2)
    target_compile_options(ndd_backup_race_test PRIVATE -march=armv8.6-a+sve2+fp16+dotprod)
    target_compile_definitions(ndd_backup_race_test PRIVATE USE_SVE2)
elseif(USE_NEON)
    if(APPLE AND (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64"))
        target_compile_options(ndd_backup_race_test PRIVATE -mcpu=native)
    else()
        target_compile_options(ndd_backup_race_test PRIVATE -march=armv8.2-a+fp16+fp16fml+dotprod)
    endif()
    target_compile_definitions(ndd_backup_race_test PRIVATE USE_NEON)
endif()

target_link_libraries(ndd_backup_race_test
    GTest::gtest_main
    archive_static
    Threads::Threads
    OpenSSL::SSL
    OpenSSL::Crypto
    CURL::libcurl
)

gtest_discover_tests(ndd_backup_race_test)
