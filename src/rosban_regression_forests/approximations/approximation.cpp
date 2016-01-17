#include "rosban_regression_forests/approximations/approximation.h"

std::ostream &operator<<(std::ostream &out, const regression_forests::Approximation &a)
{
  a.print(out);
  return out;
}
