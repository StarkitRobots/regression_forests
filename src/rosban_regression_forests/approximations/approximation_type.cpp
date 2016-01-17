#include "rosban_regression_forests/approximations/approximation_type.h"

#include <stdexcept>

using regression_forests::ApproximationType;

namespace regression_forests
{
ApproximationType loadApproximationType(const std::string &s)
{
  if (s == "PWC")
  {
    return ApproximationType::PWC;
  }
  if (s == "PWL")
  {
    return ApproximationType::PWL;
  }
  throw std::runtime_error("Unknown approximation description '" + s + "'");
}

std::string to_string(regression_forests::ApproximationType at)
{
  switch (at)
  {
    case ApproximationType::PWC:
      return "PWC";
    case ApproximationType::PWL:
      return "PWL";
  }
  throw std::runtime_error("Unkown ApproximationType");
}
}
