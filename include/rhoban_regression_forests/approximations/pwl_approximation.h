#pragma once

#include "rhoban_regression_forests/approximations/approximation.h"

#include <vector>

namespace regression_forests
{
class PWLApproximation : public Approximation
{
private:
  // There is n+1 factors, if n is the dimension of states
  // val = f1*x1 + f2*x2 + ... + fn*xn + fn+1
  Eigen::VectorXd factors;

public:
  PWLApproximation();
  PWLApproximation(const Eigen::VectorXd &factors);
  PWLApproximation(const std::vector<Eigen::VectorXd> &inputs,
                   const std::vector<double> &outputs);
  virtual ~PWLApproximation();

  const Eigen::VectorXd &getFactors() const;

  virtual double eval(const Eigen::VectorXd &state) const override;

  virtual Eigen::VectorXd getGrad(const Eigen::VectorXd &input) const override;

  virtual void updateMinPair(const Eigen::MatrixXd &limits, std::pair<double, Eigen::VectorXd> &best) const override;
  virtual void updateMaxPair(const Eigen::MatrixXd &limits, std::pair<double, Eigen::VectorXd> &best) const override;

  std::pair<double, Eigen::VectorXd> getMinPair(const Eigen::MatrixXd &limits) const;
  std::pair<double, Eigen::VectorXd> getMaxPair(const Eigen::MatrixXd &limits) const;

  virtual std::unique_ptr<Approximation> clone() const override;

  virtual int getClassID() const override;
  virtual int writeInternal(std::ostream & out) const override;
  virtual int read(std::istream & in) override;
};
}
