#pragma once

#include <Eigen/Core>

namespace regression_forests
{
class Sample
{
private:
  Eigen::VectorXd input;
  double output;

public:
  Sample();
  Sample(const Eigen::VectorXd &input, double output);

  const Eigen::VectorXd &getInput() const;
  double getInput(size_t dim) const;
  double getOutput() const;
};
}
