#pragma once

#include "rosban_regression_forests/approximations/approximation.h"

#include <vector>

namespace regression_forests
{
class CompositeApproximation : public Approximation
{
private:
  std::vector<Approximation *> approximations;

public:
  CompositeApproximation();
  CompositeApproximation(const CompositeApproximation &other) = delete;
  virtual ~CompositeApproximation();

  virtual double eval(const Eigen::VectorXd &state) const override;
  virtual void updateMaxPair(const Eigen::MatrixXd &limits, std::pair<double, Eigen::VectorXd> &best) const override;

  /**
   * The responsability of deleting the approximation is left to the
   * CompositeApproximation
   */
  void push(Approximation *approximation);

  virtual Approximation *clone() const override;

  virtual void print(std::ostream &out) const override;

  /**
   * Merge the two Approximation and return the result.
   * merge might delete or copy a1 and a2, but it will ensure that if
   * the result is deleted, then there will be no memory leak
   */
  // static Approximation * merge(Approximation * a1, Approximation * a2);

  /**
   * Merge the two Approximation with the given weight and return the result.
   * merge might delete or copy a1 and a2, but it will ensure that if
   * the result is deleted, then there will be no memory leak
   */
  static Approximation *weightedMerge(Approximation *a1, double weight1, Approximation *a2, double weight2);

  static double difference(const Approximation *a1, const Approximation *a2);
};
}
}
