#pragma once

#include <Eigen/Core>

namespace regression_forests
{
class OrthogonalSplit
{
public:
  int dim;
  double val;

  OrthogonalSplit();
  OrthogonalSplit(int dim, double val);
  OrthogonalSplit(const OrthogonalSplit & other);

  /* Return true if input belongs to the lower-space */
  bool isLower(const Eigen::VectorXd &input) const;

  /// Separate all elements of input (one column is one input)
  void splitEntries(const Eigen::MatrixXd & input,
                    Eigen::MatrixXd * lower_output,
                    Eigen::MatrixXd * upper_output);

  /// Generate the lower and upper space
  void splitSpace(const Eigen::MatrixXd & space,
                  Eigen::MatrixXd * lower_space,
                  Eigen::MatrixXd * upper_space);
};
}

std::ostream &operator<<(std::ostream &out, const regression_forests::OrthogonalSplit &split);
