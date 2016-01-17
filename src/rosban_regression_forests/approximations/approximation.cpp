#include "rosban_regression_forests/approximations/approximation.h"

std::ostream &operator<<(std::ostream &out, const Math::RegressionTree::Approximation &a)
{
  a.print(out);
  return out;
}
