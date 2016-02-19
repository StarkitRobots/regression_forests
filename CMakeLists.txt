cmake_minimum_required(VERSION 2.8.3)
project(rosban_regression_forests)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED
  cmake_modules
  rosban_utils
)

find_package(Eigen REQUIRED)

find_package(rosban_viewer)

set(DELEG_INCLUDE_DIRS
  include
)
set(DELEG_LIBRARIES
  rosban_regression_forests
)

if (rosban_viewer_FOUND)
  message("rosban_viewer found, adding support for regression_forests viewer")
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
  CATKIN_DEPENDS rosban_utils
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

# if rosban_viewer is found, then add the support
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
  )

# if rosban_viewer is found, then add the support
if (rosban_viewer_FOUND)
endif(rosban_viewer_FOUND)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS rosban_regression_forests rosban_regression_forests_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_rosban_regression_forests.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
