#include "rosban_regression_forests/approximations/gp_approximation.h"

#include "rosban_gp/auto_tuning.h"
#include "rosban_gp/core/squared_exponential.h"

#include <stdexcept>

#include <iostream>

using rosban_gp::CovarianceFunction;
using rosban_gp::GaussianProcess;
using rosban_gp::SquaredExponential;

namespace regression_forests
{

GPApproximation::GPApproximation() {}

GPApproximation::GPApproximation(const std::vector<Eigen::VectorXd> & inputs,
                                 const std::vector<double> & outputs)
  : GPApproximation(inputs, outputs, rosban_gp::RandomizedRProp::Config())
{}

GPApproximation::GPApproximation(const std::vector<Eigen::VectorXd> & inputs,
                                 const std::vector<double> & outputs,
                                 const rosban_gp::RandomizedRProp::Config & conf)
{
  /// Checking conditions
  if (inputs.size() == 0) throw std::runtime_error("GPApproximation: inputs.size() == 0");
  if (inputs.size() != outputs.size())
  {
    std::ostringstream oss;
    oss << "GPApproximation: inputs.size() != outputs.size()" << std::endl
        << "\tinput : " << inputs.size() << std::endl
        << "\toutput: " << outputs.size();
    throw std::runtime_error(oss.str());
  }
  /// Converting data
  Eigen::MatrixXd input_mat (inputs[0].rows(), inputs.size());
  Eigen::VectorXd observations(outputs.size());
  for (int i = 0; i < inputs.size(); i++) {
    input_mat.col(i) = inputs[i];
    observations(i) = outputs[i];
  }
  /// Creating GP
  std::unique_ptr<CovarianceFunction> cov_func(new SquaredExponential(input_mat.rows()));
  gp = GaussianProcess(input_mat, observations, std::move(cov_func));
  gp.autoTune(conf);
}

GPApproximation::GPApproximation(const GPApproximation & other)
  : gp(other.gp)
{
}

GPApproximation::~GPApproximation() {}

double GPApproximation::eval(const Eigen::VectorXd & state) const
{
  return gp.getPrediction(state);
}

Eigen::VectorXd GPApproximation::getGrad(const Eigen::VectorXd & state) const
{
  return gp.getGradient(state);
}

Approximation * GPApproximation::clone() const
{
  return new GPApproximation(*this);
}


void GPApproximation::updateMinPair(const Eigen::MatrixXd &limits,
                                    std::pair<double, Eigen::VectorXd> &best) const
{
  throw std::logic_error("GPApproximation::updateMinPair: not implemented");
}
void GPApproximation::updateMaxPair(const Eigen::MatrixXd &limits,
                                    std::pair<double, Eigen::VectorXd> &best) const
{
  throw std::logic_error("GPApproximation::updateMaxPair: not implemented");
}

void GPApproximation::print(std::ostream &out) const
{
  out << "agp";
  gp.write(out);
  out << "$";
}

int GPApproximation::getClassID() const
{
  return Approximation::GP;
}

int GPApproximation::writeInternal(std::ostream & out) const
{
  return gp.write(out);
}

int GPApproximation::read(std::istream & in)
{
  return gp.read(in);
}

}
