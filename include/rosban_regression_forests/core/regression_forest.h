#pragma once

#include "rosban_regression_forests/core/regression_tree.h"

namespace Math
{
namespace RegressionTree
{
class RegressionForest
{
private:
  std::vector<std::unique_ptr<RegressionTree>> trees;

public:
  RegressionForest();
  RegressionForest(const RegressionForest &other) = delete;
  virtual ~RegressionForest();

  size_t nbTrees() const;
  const RegressionTree &getTree(size_t treeId) const;

  /**
   * transfers ownership of t to the RegressionForest
   */
  void push(std::unique_ptr<RegressionTree> t);

  size_t maxSplitDim() const;

  double getValue(const Eigen::VectorXd &input) const;

  // maxLeafs is used to avoid growing an oversized tree. It is not activated by
  // default (0 value)
  // When preFilter is activated, each tree is projected before being merged
  // in the final result
  std::unique_ptr<RegressionTree> unifiedProjectedTree(const Eigen::MatrixXd &limits, size_t maxLeafs = 0,
                                                       bool preFilter = false, bool parallelMerge = false);

  void save(const std::string &path) const;
  static std::unique_ptr<RegressionForest> loadFile(const std::string &path);
};
}
}

std::ostream &operator<<(std::ostream &out, const Math::RegressionTree::RegressionForest &forest);
