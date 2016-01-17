#pragma once

#include <Eigen/Core>

namespace Math {
  namespace RegressionTree {

    class Sample {
    private:
      Eigen::VectorXd input;
      double output;

    public:
      Sample();
      Sample(const Eigen::VectorXd& input, double output);

      const Eigen::VectorXd& getInput() const;
      double getInput(size_t dim) const;
      double getOutput() const;
    };
  }
}
