#include "starkit_regression_forests/approximations/composite_approximation.h"

#include "starkit_regression_forests/approximations/pwc_approximation.h"
#include "starkit_regression_forests/approximations/pwl_approximation.h"

namespace regression_forests
{
std::unique_ptr<Approximation> CompositeApproximation::weightedMerge(std::shared_ptr<const Approximation> a1,
                                                                     double weight1,
                                                                     std::shared_ptr<const Approximation> a2,
                                                                     double weight2)
{
  // Catching type
  std::shared_ptr<const PWCApproximation> pwc1, pwc2;
  std::shared_ptr<const PWLApproximation> pwl1, pwl2;
  pwc1 = std::dynamic_pointer_cast<const PWCApproximation>(a1);
  pwc2 = std::dynamic_pointer_cast<const PWCApproximation>(a2);
  pwl1 = std::dynamic_pointer_cast<const PWLApproximation>(a1);
  pwl2 = std::dynamic_pointer_cast<const PWLApproximation>(a2);
  // Solving weighted merge
  std::unique_ptr<Approximation> result;
  // Two PWC case
  if (pwc1 && pwc2)
  {
    double meanValue = (pwc1->getValue() * weight1 + pwc2->getValue() * weight2) / (weight1 + weight2);
    result = std::unique_ptr<Approximation>(new PWCApproximation(meanValue));
  }
  // Two PWL case
  else if (pwl1 && pwl2)
  {
    Eigen::VectorXd factors1 = pwl1->getFactors();
    Eigen::VectorXd factors2 = pwl2->getFactors();
    Eigen::VectorXd mean_factors = (factors1 * weight1 + factors2 * weight2) / (weight1 + weight2);
    result = std::unique_ptr<Approximation>(new PWLApproximation(mean_factors));
  }
  // Unknown case
  else
  {
    throw std::runtime_error("WeightedMerge does not implement merge for the given types yet");
  }
  return result;
}

double CompositeApproximation::avgDifference(std::shared_ptr<const Approximation> a1,
                                             std::shared_ptr<const Approximation> a2, const Eigen::MatrixXd& limits)
{
  std::shared_ptr<const PWCApproximation> pwc1, pwc2;
  pwc1 = std::dynamic_pointer_cast<const PWCApproximation>(a1);
  pwc2 = std::dynamic_pointer_cast<const PWCApproximation>(a2);
  if (pwc1 && pwc2)
  {
    return std::fabs(pwc1->getValue() - pwc2->getValue());
  }
  std::shared_ptr<const PWLApproximation> pwl1, pwl2;
  pwl1 = std::dynamic_pointer_cast<const PWLApproximation>(a1);
  pwl2 = std::dynamic_pointer_cast<const PWLApproximation>(a2);
  if (pwl1 && pwl2)
  {
    Eigen::VectorXd center = (limits.col(0) + limits.col(1)) / 2;
    // Difference in average
    double avg_diff = std::fabs(pwl1->eval(center) - pwl2->eval(center));
    Eigen::VectorXd gradient = (pwl1->getGrad(center) - pwl2->getGrad(center)).cwiseAbs();
    // Computing average difference of gradient
    double grad_diff = gradient.dot(limits.col(1) - limits.col(0));
    for (int dim = 0; dim < limits.rows(); dim++)
    {
      grad_diff /= (limits(dim, 1) - limits(dim, 0));
    }
    return avg_diff + grad_diff;
  }
  throw std::runtime_error("difference is only implemented for PWCApproximations or PWLApproximations yet");
}
}  // namespace regression_forests
