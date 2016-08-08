#pragma once

#include "rosban_regression_forests/approximations/approximation.h"

#include <vector>

namespace regression_forests
{
class CompositeApproximation : public Approximation
{
public:
  /// Merge the two Approximation with the given weight and return the result.
  static std::unique_ptr<Approximation>
  weightedMerge(std::shared_ptr<const Approximation> a1, double weight1,
                std::shared_ptr<const Approximation> a2, double weight2);

  /// Return the average absolute difference between the two approximations
  /// inside the provided limits
  static double avgDifference(std::shared_ptr<const Approximation> a1,
                              std::shared_ptr<const Approximation> a2,
                              const Eigen::MatrixXd & limits);
};
}
