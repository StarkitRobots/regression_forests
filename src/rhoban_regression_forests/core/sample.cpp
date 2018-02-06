#include "rhoban_regression_forests/core/sample.h"

namespace regression_forests
{
Sample::Sample() : input(), output(0)
{
}

Sample::Sample(const Eigen::VectorXd &input_, double output_) : input(input_), output(output_)
{
}

const Eigen::VectorXd &Sample::getInput() const
{
  return input;
}

double Sample::getInput(size_t dim) const
{
  return input(dim);
}

double Sample::getOutput() const
{
  return output;
}
}
