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
  GPApproximation(const GPApproximation & other);
  virtual ~GPApproximation();

  /// Throws an error if gp has not run its precomputations
  virtual double eval(const Eigen::VectorXd & state) const override;

  virtual Eigen::VectorXd getGrad(const Eigen::VectorXd &input) const override;

  virtual Approximation * clone() const override;

  virtual void updateMinPair(const Eigen::MatrixXd &limits,
                             std::pair<double, Eigen::VectorXd> &best) const override;
  virtual void updateMaxPair(const Eigen::MatrixXd &limits,
                             std::pair<double, Eigen::VectorXd> &best) const override;

  virtual void print(std::ostream &out) const override;

  rosban_gp::GaussianProcess gp;

  static rosban_gp::RandomizedRProp::Config approximation_config;
};

}
