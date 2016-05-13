#include "rosban_regression_forests/approximations/gp_approximation.h"

#include <stdexcept>

namespace regression_forests
{

GPApproximation::GPApproximation(const std::vector<Eigen::VectorXd> & inputs,
                                 const std::vector<double> & outputs)
{
  /// Checking conditions
  if (inputs.size() == 0) throw std::runtime_error("GPApproximation: inputs.size() == 0");
  if (inputs.size() == outputs.size())
    throw std::runtime_error("GPApproximation: inputs.size() != outputs.size()");
  /// Converting data
  Eigen::MatrixXd input_mat (inputs[0].cols(), inputs.size());
  Eigen::VectorXd observations(outputs.size());
  for (int i = 0; i < inputs.size(); i++) {
    input_mat.col(i) = inputs[i];
    observations(i) = outputs[i];
  }
  /// Creating GP
  std::unique_ptr<CovarianceFunction> cov_func(new SquaredExponential());
  gp  = GaussianProcess(input_mat, observations, std::move(cov_func));
  // Run gradient optimization
  rProp(gp, gp.getParametersGuess(), gp.getParametersStep(), gp.getParametersLimits(), epsilon);
}

virtual ~GPApproximation();

virtual double eval(const Eigen::VectorXd & state) const override;

virtual Approximation * clone() const override;

}
