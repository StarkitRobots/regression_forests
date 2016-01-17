#pragma once

#include <ostream>

#include <Eigen/Core>

namespace Math {
  namespace RegressionTree {

    class Approximation {
    public:
      virtual ~Approximation() {}

      virtual double eval(const Eigen::VectorXd& state) const = 0;

      virtual void updateMaxPair(const Eigen::MatrixXd& limits,
                                 std::pair<double, Eigen::VectorXd>& best) const = 0;

      virtual Approximation * clone() const = 0;

      virtual void print(std::ostream& out) const = 0;
    };
  }
}

std::ostream& operator<<(std::ostream& out,
                         const Math::RegressionTree::Approximation& a);
