#pragma once

#include "rosban_regression_forests/core/regression_node.h"

#include <memory>

namespace Math
{
namespace RegressionTree
{
class RegressionTree
{
protected:
  void fillProjection(std::vector<std::vector<Eigen::VectorXd>> &out, RegressionNode *currentNode,
                      const std::vector<int> &freeDimensions, Eigen::MatrixXd &limits);

  void addSubTree(RegressionNode *node, Eigen::MatrixXd &limits, const RegressionTree &other, double otherWeight);

public:
  RegressionNode *root;

  RegressionTree();
  RegressionTree(const RegressionTree &other) = delete;
  virtual ~RegressionTree();

  size_t maxSplitDim() const;
  size_t nbLeafs() const;

  double getValue(const Eigen::VectorXd &input) const;

  // Return max over all leafs
  double getMax(const Eigen::MatrixXd &limits) const;
  Eigen::VectorXd getArgMax(const Eigen::MatrixXd &limits) const;

  std::pair<double, Eigen::VectorXd> getMaxPair(const Eigen::MatrixXd &limits) const;

  /**
   * Return a vector containing all the approximations given by nodes for
   * the provided limits.
   * The size of the global vector is the number of nodes whose intersection
   * with the provided limits is not empty.
   * The size of the mid-level vector is always 2^d (the number of 'corners'
   * of the space)
   * The size of the internal vectors(Eigen) is always D + 1
   */
  std::vector<std::vector<Eigen::VectorXd>> project(const std::vector<int> &freeDimensions,
                                                    const Eigen::MatrixXd &limits);

  /**
   * Apply the tree other on every leaf, although it is 'optimized', it
   * might result in a dramatic increase of the tree size
   * limits might be used to select the space on which the 'other' tree is
   * projected
   */
  void avgTree(const RegressionTree &other, double otherWeight);
  void avgTree(const RegressionTree &other, double otherWeight, const Eigen::MatrixXd &limits);

  static std::unique_ptr<RegressionTree> avgTrees(const RegressionTree &t1, const RegressionTree &t2, double w1,
                                                  double w2, const Eigen::MatrixXd &limits);

  std::unique_ptr<RegressionTree> project(const Eigen::MatrixXd &limits) const;
};
}
}

std::ostream &operator<<(std::ostream &out, const Math::RegressionTree::RegressionTree &tree);
