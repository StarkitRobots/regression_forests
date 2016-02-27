#pragma once

#include "rosban_regression_forests/core/tree.h"

namespace regression_forests
{
class Forest
{
private:
  std::vector<std::unique_ptr<Tree>> trees;

public:
  Forest();
  Forest(const Forest &other) = delete;
  virtual ~Forest();

  size_t nbTrees() const;
  const Tree &getTree(size_t treeId) const;

  /// transfers ownership of t to the Forest
  void push(std::unique_ptr<Tree> t);

  size_t maxSplitDim() const;

  double getValue(const Eigen::VectorXd &input) const;

  // maxLeafs is used to avoid growing an oversized tree. It is not activated by
  // default (0 value)
  // When preFilter is activated, each tree is projected before being merged
  // in the final result
  std::unique_ptr<Tree> unifiedProjectedTree(const Eigen::MatrixXd &limits, size_t maxLeafs = 0);

  void save(const std::string &path) const;
  static std::unique_ptr<Forest> loadFile(const std::string &path);

  /// Apply the given function on every node of every tree
  void apply(Eigen::MatrixXd &limits, Node::Function f);

  /// Apply the given function on every leaf of every tree
  void applyOnLeafs(Eigen::MatrixXd &limits, Node::Function f);
};
}

std::ostream &operator<<(std::ostream &out, const regression_forests::Forest &forest);
