#pragma once

#include "rhoban_utils/serialization/stream_serializable.h"

#include <Eigen/Core>

#include <ostream>
#include <memory>

namespace regression_forests
{

class Approximation: public rhoban_utils::StreamSerializable
{
public:
  virtual ~Approximation()
  {
  }

  virtual double eval(const Eigen::VectorXd &state) const = 0;

  virtual Eigen::VectorXd getGrad(const Eigen::VectorXd &input) const = 0;

  virtual void updateMinPair(const Eigen::MatrixXd &limits, std::pair<double, Eigen::VectorXd> &best) const = 0;
  virtual void updateMaxPair(const Eigen::MatrixXd &limits, std::pair<double, Eigen::VectorXd> &best) const = 0;

  virtual std::unique_ptr<Approximation> clone() const = 0;

  enum ID : int
  {
    PWC = 1,
      PWL = 2,
      GP = 3
      };

  static ID loadID(const std::string & str);
  static std::string idToString(ID id);
};
}
