#include "rosban_regression_forests/core/orthogonal_split.h"

#include <vector>

namespace regression_forests
{
OrthogonalSplit::OrthogonalSplit() : dim(-1), val(0)
{
}

OrthogonalSplit::OrthogonalSplit(int dim_, double val_) : dim(dim_), val(val_)
{
}

OrthogonalSplit::OrthogonalSplit(const OrthogonalSplit & other)
  : dim(other.dim), val(other.val)
{
}

bool OrthogonalSplit::isLower(const Eigen::VectorXd &input) const
{
  return input(dim) <= val;
}


void OrthogonalSplit::splitEntries(const Eigen::MatrixXd & input,
                                   Eigen::MatrixXd * lower_output,
                                   Eigen::MatrixXd * upper_output)
{
  /// Building separation
  std::vector<int> lower_indices;
  std::vector<int> upper_indices;
  for (int i = 0; i < input.cols(); i++)
  {
    if (isLower(input.col(i)))
      lower_indices.push_back(i);
    else
      upper_indices.push_back(i);
  }
  /// Filling lower output
  if (lower_output != nullptr)
  {
    *lower_output = Eigen::MatrixXd::Zero(input.rows(), lower_indices.size());
    for (int i = 0; i < lower_indices.size(); i++)
      lower_output->col(i) = input.col(lower_indices[i]);
  }
  /// Filling upper output
  if (upper_output != nullptr)
  {
    *upper_output = Eigen::MatrixXd::Zero(input.rows(), upper_indices.size());
    for (int i = 0; i < upper_indices.size(); i++)
      upper_output->col(i) = input.col(upper_indices[i]);
  }
}

void OrthogonalSplit::splitSpace(const Eigen::MatrixXd & space,
                                 Eigen::MatrixXd * lower_space,
                                 Eigen::MatrixXd * upper_space)
{
  *lower_space = space;
  *upper_space = space;
  (*lower_space)(dim,1) = val;
  (*upper_space)(dim,0) = val;
}

}

std::ostream &operator<<(std::ostream &out, const regression_forests::OrthogonalSplit &split)
{
  return out << "sd" << split.dim << "$v" << split.val << "$$";
}
