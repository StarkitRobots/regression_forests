#pragma once

#include "rosban_regression_forests/approximations/approximation.h"

namespace regression_forests
{
class PWLApproximation : public Approximation
{
private:
  // There is n+1 factors, if n is the dimension of states
  // val = f1*x1 + f2*x2 + ... + fn*xn + fn+1
  Eigen::VectorXd factors;

public:
  PWLApproximation(const Eigen::VectorXd &factors);
  PWLApproximation(const std::vector<Eigen::VectorXd> &inputs, const std::vector<double> &outputs);
  virtual ~PWLApproximation();

  const Eigen::VectorXd &getFactors() const;

  virtual double eval(const Eigen::VectorXd &state) const override;
  virtual void updateMaxPair(const Eigen::MatrixXd &limits, std::pair<double, Eigen::VectorXd> &best) const override;

  virtual Approximation *clone() const override;

  virtual void print(std::ostream &out) const override;
};
}
