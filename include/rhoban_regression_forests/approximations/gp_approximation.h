#pragma once

#include "rhoban_regression_forests/approximations/approximation.h"

#include "rhoban_gp/core/gaussian_process.h"

namespace regression_forests
{
class GPApproximation : public Approximation
{
public:
  /// Internal approximation is set to default
  GPApproximation();
  /// Gaussian Process hyperParameters are tuned automatically with default conf
  GPApproximation(const std::vector<Eigen::VectorXd>& inputs, const std::vector<double>& outputs);
  /// Gaussian Process hyperParameters are tuned automatically with chosen conf
  GPApproximation(const std::vector<Eigen::VectorXd>& inputs, const std::vector<double>& outputs,
                  const rhoban_gp::RandomizedRProp::Config& conf);
  GPApproximation(const GPApproximation& other);
  virtual ~GPApproximation();

  /// Throws an error if gp has not run its precomputations
  virtual double eval(const Eigen::VectorXd& state) const override;

  virtual Eigen::VectorXd getGrad(const Eigen::VectorXd& input) const override;

  virtual std::unique_ptr<Approximation> clone() const override;

  virtual void updateMinPair(const Eigen::MatrixXd& limits, std::pair<double, Eigen::VectorXd>& best) const override;
  virtual void updateMaxPair(const Eigen::MatrixXd& limits, std::pair<double, Eigen::VectorXd>& best) const override;

  virtual int getClassID() const override;
  virtual int writeInternal(std::ostream& out) const override;
  virtual int read(std::istream& in) override;

  rhoban_gp::GaussianProcess gp;
};

}  // namespace regression_forests
