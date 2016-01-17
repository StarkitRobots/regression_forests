#pragma once

#include <Eigen/Core>

namespace Math
{
namespace RegressionTree
{
class OrthogonalSplit
{
public:
  int dim;
  double val;

  OrthogonalSplit();
  OrthogonalSplit(int dim, double val);

  /* Return true if input belongs to the lower-space */
  bool isLower(const Eigen::VectorXd &input) const;
};
}
}

std::ostream &operator<<(std::ostream &out, const Math::RegressionTree::OrthogonalSplit &split);
