#include "rosban_regression_forests/approximations/composite_approximation.h"

#include "rosban_regression_forests/approximations/pwc_approximation.h"
#include "rosban_regression_forests/approximations/pwl_approximation.h"

namespace regression_forests
{
CompositeApproximation::CompositeApproximation()
{
}

CompositeApproximation::~CompositeApproximation()
{
  for (Approximation *a : approximations)
  {
    delete (a);
  }
}

double CompositeApproximation::eval(const Eigen::VectorXd &state) const
{
  double sum = 0.0;
  for (Approximation *a : approximations)
  {
    sum += a->eval(state);
  }
  return sum / approximations.size();
}

void CompositeApproximation::updateMaxPair(const Eigen::MatrixXd &limits,
                                           std::pair<double, Eigen::VectorXd> &best) const
{
  double sum = 0.0;
  for (Approximation *a : approximations)
  {
    PWCApproximation *pwcA = dynamic_cast<PWCApproximation *>(a);
    if (pwcA == NULL)
    {
      throw std::runtime_error("CompositeApproximation: getMax of node is only "
                               "implemented for PWCApproximation yet");
    }
    sum += pwcA->eval(Eigen::VectorXd());
  }
  double value = sum / approximations.size();
  if (best.first < value)
  {
    best.first = value;
    for (int row = 0; row < limits.rows(); row++)
    {
      best.second(row) = (limits(row, 0) + limits(row, 1)) / 2;
    }
  }
}

void CompositeApproximation::push(Approximation *a)
{
  approximations.push_back(a);
}

Approximation *CompositeApproximation::clone() const
{
  CompositeApproximation *copy = new CompositeApproximation();
  for (Approximation *a : approximations)
  {
    copy->push(a->clone());
  }
  return copy;
}

void CompositeApproximation::print(std::ostream &out) const
{
  out << "ac";
  for (Approximation *a : approximations)
  {
    out << *a;
  }
  out << "$";
}

Approximation *CompositeApproximation::weightedMerge(Approximation *a1, double weight1, Approximation *a2,
                                                     double weight2)
{
  // Catching types
  PWCApproximation *pwc1 = dynamic_cast<PWCApproximation *>(a1);
  PWCApproximation *pwc2 = dynamic_cast<PWCApproximation *>(a2);
  PWLApproximation *pwl1 = dynamic_cast<PWLApproximation *>(a1);
  PWLApproximation *pwl2 = dynamic_cast<PWLApproximation *>(a2);
  // Solving weighted merge
  Approximation *result;
  // Two PWC case
  if (pwc1 != NULL && pwc2 != NULL)
  {
    double meanValue = (pwc1->getValue() * weight1 + pwc2->getValue() * weight2) / (weight1 + weight2);
    result = new PWCApproximation(meanValue);
    delete (pwc1);
    delete (pwc2);
  }
  // Two PWL case
  else if (pwl1 != NULL && pwl2 != NULL)
  {
    Eigen::VectorXd factors1 = pwl1->getFactors();
    Eigen::VectorXd factors2 = pwl2->getFactors();
    result = new PWLApproximation((factors1 * weight1 + factors2 * weight2) / (weight1 + weight2));
    delete (pwl1);
    delete (pwl2);
  }
  // Unknown case
  else
  {
    delete (a1);
    delete (a2);
    throw std::runtime_error("WeightedMerge does not implement merge for the given types yet");
  }
  return result;
}

double CompositeApproximation::difference(const Approximation *a1, const Approximation *a2)
{
  const PWCApproximation *pwc1 = dynamic_cast<const PWCApproximation *>(a1);
  const PWCApproximation *pwc2 = dynamic_cast<const PWCApproximation *>(a2);
  // Unsupported operation
  if (pwc1 == NULL || pwc2 == NULL)
  {
    throw std::runtime_error("difference is only implemented for PWCApproximations yet");
  }
  return pwc1->getValue() - pwc2->getValue();
}
}
