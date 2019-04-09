cmake_minimum_required(VERSION 2.8.3)
project(rhoban_regression_forests)

# Build options
option(RHOBAN_RF_USES_GP "Is support for gaussian processes activated" OFF)
option(RHOBAN_RF_BUILD_VIEWER "Build regression forests Viewer (require SFML)" OFF)
option(RHOBAN_RF_BUILD_EXAMPLES "Build examples/tools" OFF)


set (CATKIN_DEPS
  rhoban_utils
  rhoban_random
)

if (RHOBAN_RF_USES_GP)
  set (CATKIN_DEPS ${CATKIN_DEPS} rhoban_gp)
  add_definitions(-DRHOBAN_RF_USES_GP)
endif(RHOBAN_RF_USES_GP)
if (RHOBAN_RF_BUILD_VIEWER)
  set (CATKIN_DEPS ${CATKIN_DEPS} rhoban_viewer)
endif(RHOBAN_RF_BUILD_VIEWER)


find_package(catkin REQUIRED COMPONENTS ${CATKIN_DEPS})

set(DELEG_INCLUDE_DIRS include ${catkin_INCLUDE_DIRS})
set(DELEG_LIBRARIES rhoban_regression_forests ${catkin_LIBRARIES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS ${DELEG_INCLUDE_DIRS}
  LIBRARIES ${DELEG_LIBRARIES}
  CATKIN_DEPENDS ${CATKIN_DEPS}
)

###########
## Build ##
###########

include_directories(
  ${DELEG_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  )

# Directories containing sources files
set(DIRECTORIES
  src/rhoban_regression_forests/algorithms
  src/rhoban_regression_forests/approximations
  src/rhoban_regression_forests/core
  src/rhoban_regression_forests/tools
  )

# Build ALL_SOURCES
set (SOURCES)
set (ALL_SOURCES)
set (PREFIXED_SOURCES)
foreach (DIRECTORY ${DIRECTORIES})
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/${DIRECTORY}")
    include (${DIRECTORY}/sources.cmake)

    foreach (SOURCE ${SOURCES})
        set (PREFIXED_SOURCES ${PREFIXED_SOURCES} ${DIRECTORY}/${SOURCE})
    endforeach (SOURCE)

    set (ALL_SOURCES ${ALL_SOURCES} ${PREFIXED_SOURCES})
endforeach (DIRECTORY)

# if rhoban_viewer is found, then add the support for viewers
if (BUILD_VIEWER)
  set (ALL_SOURCES
    ${ALL_SOURCES}
    src/rhoban_regression_forests/tools/viewer.cpp
    )
endif(BUILD_VIEWER)

# Declare a C++ library
add_library(rhoban_regression_forests
  ${ALL_SOURCES}
)

target_link_libraries(rhoban_regression_forests
  ${catkin_LIBRARIES}
  ${EXTRA_LIBS}
  )


# Testing input output functions
if (RHOBAN_RF_BUILD_EXAMPLES)
  add_executable(test_rf_io src/test_io.cpp)
  target_link_libraries(test_rf_io ${catkin_LIBRARIES} ${DELEG_LIBRARIES} )
endif (RHOBAN_RF_BUILD_EXAMPLES)

# if rhoban_viewer is found, then add the viewer executable as target
if (RHOBAN_RF_BUILD_VIEWER)
  add_executable(viewer src/viewer.cpp)

  target_link_libraries(viewer
    ${catkin_LIBRARIES}
    ${DELEG_LIBRARIES}
    )
endif(RHOBAN_RF_BUILD_VIEWER)
