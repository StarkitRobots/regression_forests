#pragma once

#include <ostream>

#include <Eigen/Core>

namespace regression_forests
{
class Approximation
{
public:
  virtual ~Approximation()
  {
  }

  virtual double eval(const Eigen::VectorXd &state) const = 0;

  virtual void updateMinPair(const Eigen::MatrixXd &limits, std::pair<double, Eigen::VectorXd> &best) const = 0;
  virtual void updateMaxPair(const Eigen::MatrixXd &limits, std::pair<double, Eigen::VectorXd> &best) const = 0;

  virtual Approximation *clone() const = 0;

  virtual void print(std::ostream &out) const = 0;
};
}

std::ostream &operator<<(std::ostream &out, const regression_forests::Approximation &a);
