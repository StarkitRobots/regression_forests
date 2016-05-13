#pragma once

#include "rosban_regression_forests/approximations/approximation.h"

#include "rosban_gp/core/gaussian_process.h"

namespace regression_forests
{

class GPApproximation : public Approximation
{
public:
  GPApproximation(const std::vector<Eigen::VectorXd> & inputs,
                  const std::vector<double> & outputs);
  virtual ~GPApproximation();

  virtual double eval(const Eigen::VectorXd & state) const override;

  virtual Approximation * clone() const override;

private:
  rosban_gp::GaussianProcess gp;
};

}
