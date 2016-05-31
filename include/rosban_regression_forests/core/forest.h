#pragma once

#include "rosban_regression_forests/core/tree.h"

#include <random>

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

  /// Return the average value for the given input
  double getValue(const Eigen::VectorXd &input) const;

  /// Return the variance of the estimation according to the trees for the given input
  double getVar(const Eigen::VectorXd &input) const;

  /// Return the average gradient at the given input
  Eigen::VectorXd getGradient(const Eigen::VectorXd & input) const;

  /// Return a randomized value based on the confidence interval
  double getRandomizedValue(const Eigen::VectorXd &input,
                            std::default_random_engine &engine) const;

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
