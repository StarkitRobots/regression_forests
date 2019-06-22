#include "starkit_regression_forests/approximations/approximation.h"

namespace regression_forests
{
Approximation::ID Approximation::loadID(const std::string& str)
{
  if (str == "PWC")
    return Approximation::ID::PWC;
  if (str == "PWL")
    return Approximation::ID::PWL;
#ifdef STARKIT_RF_USES_GP
  if (str == "GP")
    return Approximation::ID::GP;
#endif
  throw std::runtime_error("Unknown string for loadID" + str);
}

std::string Approximation::idToString(ID id)
{
  switch (id)
  {
    case PWC:
      return "PWC";
    case PWL:
      return "PWL";
#ifdef STARKIT_RF_USES_GP
    case GP:
      return "GP";
#endif
  }
  std::ostringstream oss;
  oss << "Unknown id for idToString" << id;
  throw std::runtime_error(oss.str());
}

}  // namespace regression_forests
