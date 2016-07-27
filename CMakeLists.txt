cmake_minimum_required(VERSION 2.8.3)
project(rosban_regression_forests)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED
  cmake_modules
  rosban_gp
  rosban_random
  rosban_utils
)

find_package(Eigen REQUIRED)

find_package(rosban_viewer)

set(DELEG_INCLUDE_DIRS include ${catkin_INCLUDE_DIRS})
set(DELEG_LIBRARIES rosban_regression_forests ${catkin_LIBRARIES})

# Warning: Need to clean build in order to create the viewer if it was not
#          available during the first build
if (rosban_viewer_FOUND)
  message("rosban_viewer found, adding support for regression_forests viewer")
  set (EXTRA_LIBS ${rosban_viewer_LIBRARIES})
  list(APPEND DELEG_INCLUDE_DIRS ${rosban_viewer_INCLUDE_DIRS})
  list(APPEND DELEG_LIBRARIES ${rosban_viewer_LIBRARIES})
else()
  message("rosban_viewer was not found, no support for regression_forests viewer")
endif(rosban_viewer_FOUND)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

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
  CATKIN_DEPENDS rosban_random rosban_utils
  DEPENDS Eigen
)

###########
## Build ##
###########

include_directories(
  ${DELEG_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIR}
  )

# Directories containing sources files
set(DIRECTORIES
  src/rosban_regression_forests/algorithms
  src/rosban_regression_forests/approximations
  src/rosban_regression_forests/core
  src/rosban_regression_forests/tools
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

# if rosban_viewer is found, then add the support for viewers
if (rosban_viewer_FOUND)
  set (ALL_SOURCES
    ${ALL_SOURCES}
    src/rosban_regression_forests/tools/viewer.cpp
    )
endif(rosban_viewer_FOUND)

# Declare a C++ library
add_library(rosban_regression_forests
  ${ALL_SOURCES}
)

target_link_libraries(rosban_regression_forests
  ${catkin_LIBRARIES}
  ${EXTRA_LIBS}
  )


# Testing input output functions
add_executable(test_io src/test_io.cpp)
target_link_libraries(test_io ${catkin_LIBRARIES} ${DELEG_LIBRARIES} )

# if rosban_viewer is found, then add the viewer executable as target
if (rosban_viewer_FOUND)
  add_executable(viewer src/viewer.cpp)

  target_link_libraries(viewer
    ${catkin_LIBRARIES}
    ${DELEG_LIBRARIES}
    )
endif(rosban_viewer_FOUND)