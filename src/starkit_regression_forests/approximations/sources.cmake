set(SOURCES
  approximation.cpp
  approximation_factory.cpp
  composite_approximation.cpp
  pwc_approximation.cpp
  pwl_approximation.cpp
)

if(STARKIT_RF_USES_GP)
  set(SOURCES ${SOURCES} gp_approximation.cpp)
endif(STARKIT_RF_USES_GP)