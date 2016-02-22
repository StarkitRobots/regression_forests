#include "rosban_regression_forests/approximations/pwc_approximation.h"

namespace regression_forests
{
PWCApproximation::PWCApproximation(double value_) : value(value_)
{
}

PWCApproximation::~PWCApproximation()
{
}

double PWCApproximation::getValue() const
{
  return value;
}

double PWCApproximation::eval(const Eigen::VectorXd &state) const
{
  // Just ignore state since the reward is piecewise constant
  (void)state;
  return value;
}

void PWCApproximation::updateMaxPair(const Eigen::MatrixXd &limits, std::pair<double, Eigen::VectorXd> &best) const
{
  if (best.first < value)
  {
    best.first = value;
    for (int row = 0; row < limits.rows(); row++)
    {
      best.second(row) = (limits(row, 0) + limits(row, 1)) / 2;
    }
  }
}

void PWCApproximation::updateMinPair(const Eigen::MatrixXd &limits,
                                     std::pair<double, Eigen::VectorXd> &best) const
{
  if (best.first > value)
  {
    best.first = value;
    for (int row = 0; row < limits.rows(); row++)
    {
      best.second(row) = (limits(row, 0) + limits(row, 1)) / 2;
    }
  }
}

Approximation *PWCApproximation::clone() const
{
  return new PWCApproximation(value);
}

void PWCApproximation::print(std::ostream &out) const
{
  out << "apwc" << value << "$";
}
}
