#=============================================================================
# 桥墩检测  地面剔除  边界提取
# 从桥梁扫描PCD中移除弧形船底, 保留桥墩并提取XY投影边界
#=============================================================================
cmake_minimum_required(VERSION 3.16)
project(pier-detect VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PCL 1.13 REQUIRED COMPONENTS common io filters features segmentation search)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc highgui imgcodecs)

add_executable(pier_detect
    src/main.cpp
    src/ground_remover.cpp
    src/boundary_extractor.cpp
    src/line_fitter.cpp
    src/coverage_planner.cpp
)

target_include_directories(pier_detect PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(pier_detect PRIVATE ${PCL_LIBRARIES} ${OpenCV_LIBS})

if(MSVC)
    target_compile_options(pier_detect PRIVATE /W4 /utf-8)
    target_compile_definitions(pier_detect PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
