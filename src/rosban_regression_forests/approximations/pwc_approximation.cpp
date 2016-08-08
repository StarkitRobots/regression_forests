#include "rosban_regression_forests/approximations/pwc_approximation.h"

#include "rosban_utils/io_tools.h"

namespace regression_forests
{

PWCApproximation::PWCApproximation() : value(0) {}

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

Eigen::VectorXd PWCApproximation::getGrad(const Eigen::VectorXd &input) const
{
  return Eigen::VectorXd::Zero(input.rows());
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

std::unique_ptr<Approximation> PWCApproximation::clone() const
{
  return std::unique_ptr<Approximation>(new PWCApproximation(value));
}

int PWCApproximation::getClassID() const
{
  return Approximation::PWC;
}

int PWCApproximation::writeInternal(std::ostream & out) const
{
  return rosban_utils::write<double>(out, value);
}

int PWCApproximation::read(std::istream & in)
{
  return rosban_utils::read<double>(in, &value);
}

}
