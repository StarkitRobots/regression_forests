#pragma once

#include "rosban_regression_forests/approximations/approximation.h"

namespace regression_forests
{
class PWCApproximation : public Approximation
{
private:
  double value;

public:
  PWCApproximation(double value);
  virtual ~PWCApproximation();

  double getValue() const;

  virtual double eval(const Eigen::VectorXd &state) const override;
  virtual void updateMaxPair(const Eigen::MatrixXd &limits, std::pair<double, Eigen::VectorXd> &best) const override;

  virtual Approximation *clone() const override;

  virtual void print(std::ostream &out) const override;
};
}
}
