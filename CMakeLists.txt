cmake_minimum_required(VERSION 3.15)
project(lidar_range_CNN VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Find dependencies
# Eigen3 - try find_package first, fallback to manual path
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    # Try to find Eigen3 manually
    find_path(EIGEN3_INCLUDE_DIR 
        NAMES Eigen/Core
        PATHS
            D:/lib/eigen3
            ${CMAKE_SOURCE_DIR}/third_party/eigen3
            ${CMAKE_SOURCE_DIR}/../eigen3
            ${CMAKE_SOURCE_DIR}/../../eigen3
            D:/eigen3
            D:/local/eigen3
            "C:/Program Files/eigen3"
            "C:/Program Files (x86)/eigen3"
            ENV EIGEN3_ROOT
            ENV EIGEN3_INCLUDE_DIR
        PATH_SUFFIXES include
    )
    
    if(EIGEN3_INCLUDE_DIR)
        message(STATUS "Eigen3 found manually at: ${EIGEN3_INCLUDE_DIR}")
        set(EIGEN3_FOUND TRUE)
    else()
        message(FATAL_ERROR 
            "Eigen3 not found!\n"
            "Please install Eigen3 using one of these methods:\n"
            "  1. vcpkg: vcpkg install eigen3:x64-windows\n"
            "  2. Download from http://eigen.tuxfamily.org/ and extract\n"
            "  3. Set EIGEN3_INCLUDE_DIR to the Eigen include directory\n"
            "     Example: cmake .. -DEIGEN3_INCLUDE_DIR=D:/lib/eigen3/include"
        )
    endif()
endif()

# PCL - try find_package first, fallback to manual path
find_package(PCL QUIET)
if(NOT PCL_FOUND)
    # Try to find PCL manually
    find_path(PCL_INCLUDE_DIRS
        NAMES pcl/point_cloud.h
        PATHS
            D:/lib/pcl
            ${CMAKE_SOURCE_DIR}/third_party/pcl
            C:/PCL
            C:/local/pcl
            "C:/Program Files/PCL"
            "C:/Program Files (x86)/PCL"
            ENV PCL_ROOT
            ENV PCL_DIR
        PATH_SUFFIXES include
    )
    
    find_library(PCL_LIBRARIES
        NAMES pcl_common
        PATHS
            D:/lib/pcl/lib
            C:/PCL/lib
            C:/local/pcl/lib
            "C:/Program Files/PCL/lib"
            ENV PCL_ROOT
            ENV PCL_DIR
        PATH_SUFFIXES lib
    )
    
    if(PCL_INCLUDE_DIRS AND PCL_LIBRARIES)
        message(STATUS "PCL found manually at: ${PCL_INCLUDE_DIRS}")
        set(PCL_FOUND TRUE)
        # PCL typically needs these components
        set(PCL_LIBRARIES 
            ${PCL_LIBRARIES}
            pcl_common pcl_io pcl_filters pcl_registration
        )
    else()
        message(WARNING 
            "PCL not found!\n"
            "The project uses PCL for ICP (Iterative Closest Point) algorithm.\n"
            "Installation options:\n"
            "  1. vcpkg: vcpkg install pcl:x64-windows\n"
            "  2. Download from: https://github.com/PointCloudLibrary/pcl/releases\n"
            "  3. Set PCL_DIR to PCL installation directory\n"
            "     Example: cmake .. -DPCL_DIR=D:/lib/pcl\n"
            "Note: PCL installation can be complex. Consider using vcpkg."
        )
        # For now, make PCL optional to see other dependencies
        set(PCL_FOUND FALSE)
    endif()
endif()

# Torch - required for neural network inference
# Workaround for nvToolsExt issue: create dummy target if missing
# (libtorch CUDA builds expect this, but we're using CPU-only)
if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
    message(STATUS "Created dummy CUDA::nvToolsExt target (CPU-only build)")
endif()
find_package(Torch QUIET)
if(NOT Torch_FOUND)
    find_path(TORCH_INCLUDE_DIRS
        NAMES torch/script.h
        PATHS
            ${CMAKE_SOURCE_DIR}/third_party/libtorch
            D:/lib/libtorch
            C:/libtorch
            C:/local/libtorch
            ENV TORCH_ROOT
            ENV CMAKE_PREFIX_PATH
        PATH_SUFFIXES include
    )
    
    if(TORCH_INCLUDE_DIRS)
        get_filename_component(TORCH_ROOT "${TORCH_INCLUDE_DIRS}/.." ABSOLUTE)
        find_library(TORCH_LIBRARIES
            NAMES torch torch_library
            PATHS
                ${TORCH_ROOT}/lib
            NO_DEFAULT_PATH
        )
        
        if(TORCH_LIBRARIES)
            message(STATUS "Torch found manually at: ${TORCH_ROOT}")
            set(TORCH_FOUND TRUE)
            set(TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
        endif()
    endif()
    
    if(NOT TORCH_FOUND)
        message(FATAL_ERROR 
            "libtorch (PyTorch C++ API) not found!\n"
            "Download libtorch from: https://pytorch.org/get-started/locally/\n"
            "Select: Windows, LibTorch, C++, CPU (or CUDA if you have GPU)\n"
            "Extract and set CMAKE_PREFIX_PATH:\n"
            "  cmake .. -DCMAKE_PREFIX_PATH=D:/lib/libtorch\n"
            "Or set TORCH_ROOT environment variable."
        )
    endif()
endif()

# PCL requires Boost (usually comes with PCL, but ensure it's found)
if(PCL_FOUND)
    if(PCL_BOOST_FOUND)
        message(STATUS "PCL Boost components found")
    endif()
endif()

# Print found packages
message(STATUS "Eigen3 found: ${EIGEN3_FOUND}")
if(EIGEN3_FOUND)
    message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")
endif()
message(STATUS "PCL found: ${PCL_FOUND}")
message(STATUS "Torch found: ${TORCH_FOUND}")

# Source files
set(SOURCES
    src/ErrorHandler.cpp
    src/FrontEnd.cpp
    src/KittiLoader.cpp
    src/LoopClosureDetector.cpp
    src/LoopClosureNet.cpp
    src/MapEvaluator.cpp
    src/MapManager.cpp
    src/PoseEvaluator.cpp
    src/PoseGraph.cpp
    src/RangeImageProjector.cpp
)

# Header files
set(HEADERS
    include/ErrorHandler.hpp
    include/FrontEnd.hpp
    include/KittiLoader.hpp
    include/LoopClosureDetector.hpp
    include/LoopClosureNet.hpp
    include/MapEvaluator.hpp
    include/MapManager.hpp
    include/PoseEvaluator.hpp
    include/PoseGraph.hpp
    include/RangeImageProjector.hpp
    include/Types.hpp
)

# Executable
add_executable(${PROJECT_NAME} src/main.cpp ${SOURCES} ${HEADERS})

# Test executable for DLL loading
add_executable(test_dll_loading test_dll_loading.cpp)
target_link_libraries(test_dll_loading ${TORCH_LIBRARIES})
target_include_directories(test_dll_loading PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)
if(PCL_FOUND)
    target_link_libraries(test_dll_loading ${PCL_LIBRARIES})
    target_include_directories(test_dll_loading PRIVATE ${PCL_INCLUDE_DIRS})
endif()

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${TORCH_LIBRARIES}
)

# Link PCL if found
if(PCL_FOUND)
    target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})
    target_compile_definitions(${PROJECT_NAME} PRIVATE ${PCL_DEFINITIONS} PCL_FOUND)
else()
    message(WARNING "Building without PCL - ICP functionality will be disabled")
    target_compile_definitions(${PROJECT_NAME} PRIVATE DISABLE_PCL)
endif()

# Include directories for target
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
)

# Add PCL includes if found
if(PCL_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${PCL_INCLUDE_DIRS})
    target_compile_definitions(${PROJECT_NAME} PRIVATE ${PCL_DEFINITIONS})
endif()

# Compiler flags
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Set properties for Torch
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

# Copy model file to build directory (if it exists)
if(EXISTS "${CMAKE_SOURCE_DIR}/src/pair_range_cnn_kitti_00_10.pt")
    configure_file(
        "${CMAKE_SOURCE_DIR}/src/pair_range_cnn_kitti_00_10.pt"
        "${CMAKE_BINARY_DIR}/bin/pair_range_cnn_kitti_00_10.pt"
        COPYONLY
    )
    message(STATUS "Model file will be copied to build directory")
endif()

