#include "rosban_regression_forests/approximations/pwl_approximation.h"

#include <Eigen/SVD>

#include <iostream>

namespace regression_forests
{
PWLApproximation::PWLApproximation(const Eigen::VectorXd &factors_) : factors(factors_)
{
}

PWLApproximation::PWLApproximation(const std::vector<Eigen::VectorXd> &inputs, const std::vector<double> &outputs)
{
  // Checking various stuff
  if (inputs.size() == 0)
  {
    throw std::runtime_error("PWLApproximation: inputs.size() == 0");
  }
  if (inputs.size() != outputs.size())
  {
    throw std::runtime_error("PWLApproximation: inputs.size() != outputs.size()");
  }
  int inputDim = inputs[0].rows();
  if (inputDim < 1)
  {
    throw std::runtime_error("PWLApproximation: inputDim < 1");
  }
  if (inputDim >= (int)inputs.size())
  {
    throw std::runtime_error("PWLApproximation: inputDim >= inputs.size() leastSquare impossible");
  }
  // Solving ax = b to find the hyperplan
  Eigen::MatrixXd a(inputs.size(), inputDim + 1);
  Eigen::VectorXd b(inputs.size());
  // Filling a and b
  for (size_t row = 0; row < inputs.size(); row++)
  {
    a.block(row, 0, 1, inputDim) = inputs[row].transpose();
    a(row, inputDim) = 1;  // Always 1 in front of the hyperplan offset
    b(row) = outputs[row];
  }
  // Solving (cf http://eigen.tuxfamily.org/dox-devel/group__LeastSquares.html)
  // factors = a.fullPivHouseholderQr().solve(b);// Weird results when testing
  // this one
  factors = a.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
  // std::cout << "################################" << std::endl;
  // std::cout << "Inputs"  << std::endl << a       << std::endl;
  // std::cout << "Outputs" << std::endl << b       << std::endl;
  // std::cout << "factors" << std::endl << factors << std::endl;
}

PWLApproximation::~PWLApproximation()
{
}

const Eigen::VectorXd &PWLApproximation::getFactors() const
{
  return factors;
}

double PWLApproximation::eval(const Eigen::VectorXd &state) const
{
  if (state.rows() != factors.rows() - 1)
  {
    std::ostringstream oss;
    oss << "PWLApproximation::eval -> received a state of dim : " << state.rows()
        << " expecting state of dim : " << (factors.rows() - 1);
    throw std::runtime_error(oss.str());
  }
  double value = factors(state.rows());  // offset of the hyperPlane
  for (int dim = 0; dim < state.rows(); dim++)
  {
    value += state(dim) * factors(dim);
  }
  return value;
}

void PWLApproximation::updateMaxPair(const Eigen::MatrixXd &limits, std::pair<double, Eigen::VectorXd> &best) const
{
  Eigen::VectorXd bestState(factors.rows() - 1);
  for (int dim = 0; dim < bestState.rows(); dim++)
  {
    if (factors(dim) > 0)
    {
      bestState(dim) = limits(dim, 1);
    }
    else if (factors(dim) < 0)
    {
      bestState(dim) = limits(dim, 0);
    }
    else
    {
      bestState(dim) = (limits(dim, 0) + limits(dim, 1)) / 2;
    }
  }
  double value = eval(bestState);
  if (best.first < value)
  {
    best.first = value;
    best.second = bestState;
  }
}

Approximation *PWLApproximation::clone() const
{
  return new PWLApproximation(factors);
}

void PWLApproximation::print(std::ostream &out) const
{
  out << "apwl" << factors.rows() << "|";
  for (int dim = 0; dim < factors.rows(); dim++)
  {
    out << factors(dim) << "|";
  }
  out << "$";
}
}
}
