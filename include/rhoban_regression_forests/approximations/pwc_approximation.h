#pragma once

#include "rhoban_regression_forests/approximations/approximation.h"

namespace regression_forests
{
class PWCApproximation : public Approximation
{
private:
  double value;

public:
  PWCApproximation();
  PWCApproximation(double value);
  virtual ~PWCApproximation();

  double getValue() const;

  virtual double eval(const Eigen::VectorXd &state) const override;

  virtual Eigen::VectorXd getGrad(const Eigen::VectorXd &input) const override;

  virtual void updateMinPair(const Eigen::MatrixXd &limits, std::pair<double, Eigen::VectorXd> &best) const override;
  virtual void updateMaxPair(const Eigen::MatrixXd &limits, std::pair<double, Eigen::VectorXd> &best) const override;

  virtual std::unique_ptr<Approximation> clone() const override;

  virtual int getClassID() const override;
  virtual int writeInternal(std::ostream & out) const override;
  virtual int read(std::istream & in) override;
};
}
