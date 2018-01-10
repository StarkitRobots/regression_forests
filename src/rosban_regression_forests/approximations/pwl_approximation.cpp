#include "rosban_regression_forests/approximations/pwl_approximation.h"

#include "rhoban_utils/io_tools.h"

#include <Eigen/SVD>

#include <iostream>

namespace regression_forests
{

PWLApproximation::PWLApproximation() {}

PWLApproximation::PWLApproximation(const Eigen::VectorXd &factors_) : factors(factors_)
{
}

PWLApproximation::PWLApproximation(const std::vector<Eigen::VectorXd> &inputs,
                                   const std::vector<double> &outputs)
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
    std::ostringstream oss;
    oss << "PWLApproximation: inputsDim (" << inputDim
        << ") >= inputs.size() (" << inputs.size() << ") leastSquare impossible";
    throw std::runtime_error(oss.str());
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

Eigen::VectorXd PWLApproximation::getGrad(const Eigen::VectorXd &input) const
{
  return factors.segment(0,factors.rows() - 1);
}

std::pair<double,Eigen::VectorXd> PWLApproximation::getMinPair(const Eigen::MatrixXd &limits) const
{
  Eigen::VectorXd worst_state(factors.rows() - 1);
  for (int dim = 0; dim < worst_state.rows(); dim++)
  {
    if (factors(dim) > 0)
    {
      worst_state(dim) = limits(dim, 1);
    }
    else if (factors(dim) < 0)
    {
      worst_state(dim) = limits(dim, 0);
    }
    else
    {
      worst_state(dim) = (limits(dim, 0) + limits(dim, 1)) / 2;
    }
  }
  double value = eval(worst_state);
  return std::pair<double,Eigen::VectorXd>(value, worst_state);
}
std::pair<double,Eigen::VectorXd> PWLApproximation::getMaxPair(const Eigen::MatrixXd &limits) const
{
  Eigen::VectorXd best_state(factors.rows() - 1);
  for (int dim = 0; dim < best_state.rows(); dim++)
  {
    if (factors(dim) > 0)
    {
      best_state(dim) = limits(dim, 1);
    }
    else if (factors(dim) < 0)
    {
      best_state(dim) = limits(dim, 0);
    }
    else
    {
      best_state(dim) = (limits(dim, 0) + limits(dim, 1)) / 2;
    }
  }
  double value = eval(best_state);
  return std::pair<double,Eigen::VectorXd>(value, best_state);
}

void PWLApproximation::updateMinPair(const Eigen::MatrixXd &limits,
                                     std::pair<double, Eigen::VectorXd> &best) const
{
  auto new_pair = getMinPair(limits);
  if (best.first > new_pair.first)
  {
    best.first = new_pair.first;
    best.second = new_pair.second;
  }
}

void PWLApproximation::updateMaxPair(const Eigen::MatrixXd &limits,
                                     std::pair<double, Eigen::VectorXd> &best) const
{
  auto new_pair = getMaxPair(limits);
  if (best.first < new_pair.first)
  {
    best.first = new_pair.first;
    best.second = new_pair.second;
  }
}

std::unique_ptr<Approximation>PWLApproximation::clone() const
{
  return std::unique_ptr<Approximation>(new PWLApproximation(factors));
}

int PWLApproximation::getClassID() const
{
  return Approximation::PWL;
}

int PWLApproximation::writeInternal(std::ostream & out) const
{
  int bytes_written = 0;
  bytes_written += rhoban_utils::write<int>(out, factors.rows());
  bytes_written += rhoban_utils::writeArray<double>(out, factors.rows(), factors.data());
  return bytes_written;
}

int PWLApproximation::read(std::istream & in)
{
  int bytes_read = 0;
  int nb_factors;
  bytes_read += rhoban_utils::read<int>(in, &nb_factors);
  factors = Eigen::VectorXd(nb_factors);
  bytes_read += rhoban_utils::readArray<double>(in, nb_factors, factors.data());
  return bytes_read;
}

}
