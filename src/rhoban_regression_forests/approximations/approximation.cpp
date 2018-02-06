#include "rhoban_regression_forests/approximations/approximation.h"

namespace regression_forests
{

Approximation::ID Approximation::loadID(const std::string & str)
{
  if (str == "PWC") return Approximation::ID::PWC;
  if (str == "PWL") return Approximation::ID::PWL;
  if (str == "GP" ) return Approximation::ID::GP ;
  throw std::runtime_error("Unknown string for loadID" + str);
}

std::string Approximation::idToString(ID id)
{
  switch(id)
  {
    case PWC: return "PWC";
    case PWL: return "PWL";
    case GP : return "GP";
  }
  std::ostringstream oss;
  oss << "Unknown id for idToString" << id;
  throw std::runtime_error(oss.str());
}

}
