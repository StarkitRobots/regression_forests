#include "Approximation.hpp"


std::ostream& operator<<(std::ostream& out,
                         const Math::RegressionTree::Approximation& a)
{
  a.print(out);
  return out;
}
