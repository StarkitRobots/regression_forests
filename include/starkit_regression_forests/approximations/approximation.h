#pragma once

#include "starkit_utils/serialization/stream_serializable.h"

#include <Eigen/Core>

#include <ostream>
#include <memory>

namespace regression_forests
{
class Approximation : public starkit_utils::StreamSerializable
{
public:
  virtual ~Approximation()
  {
  }

  virtual double eval(const Eigen::VectorXd& state) const = 0;

  virtual Eigen::VectorXd getGrad(const Eigen::VectorXd& input) const = 0;

  virtual void updateMinPair(const Eigen::MatrixXd& limits, std::pair<double, Eigen::VectorXd>& best) const = 0;
  virtual void updateMaxPair(const Eigen::MatrixXd& limits, std::pair<double, Eigen::VectorXd>& best) const = 0;

  virtual std::unique_ptr<Approximation> clone() const = 0;

  enum ID : int
  {
    PWC = 1,
    PWL = 2,
#ifdef STARKIT_RF_USES_GP
    GP = 3
#endif
  };

  static ID loadID(const std::string& str);
  static std::string idToString(ID id);
};
}  // namespace regression_forests
