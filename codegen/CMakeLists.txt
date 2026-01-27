cmake_minimum_required(VERSION 3.16)
project(solver_codegen C)

# 设置C标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 优化选项
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -ffast-math")
set(CMAKE_BUILD_TYPE Release)

# Conda环境路径
set(CONDA_ENV_PATH "$ENV{CONDA_PREFIX}")
if(NOT CONDA_ENV_PATH)
    message(FATAL_ERROR "请先激活conda环境: conda activate WBC")
endif()

message(STATUS "Conda环境路径: ${CONDA_ENV_PATH}")

# 头文件路径
include_directories(
    ${CONDA_ENV_PATH}/include
    ${CONDA_ENV_PATH}/lib/python3.12/site-packages/casadi/include
)

# 库文件路径
link_directories(
    ${CONDA_ENV_PATH}/lib
)

# 编译solver_function共享库
add_library(solver_function SHARED
    ${CMAKE_SOURCE_DIR}/../solver_function.c
)

# 链接Fatrop和Blasfeo库
target_link_libraries(solver_function
    fatrop
    blasfeo
)

# 设置输出目录
set_target_properties(solver_function PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib
    PREFIX ""  # 不添加lib前缀
)

# 如果存在retract_solution.c，也编译它
if(EXISTS "${CMAKE_SOURCE_DIR}/../retract_solution.c")
    add_library(retract_solution SHARED
        ${CMAKE_SOURCE_DIR}/../retract_solution.c
    )
    target_link_libraries(retract_solution
        fatrop
        blasfeo
    )
    set_target_properties(retract_solution PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib
        PREFIX ""
    )
endif()

message(STATUS "编译完成后，库文件将输出到: ${CMAKE_SOURCE_DIR}/lib")
