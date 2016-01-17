#pragma once

#include "rosban_regression_forests/approximations/approximation_type.h"
#include "rosban_regression_forests/core/training_set.h"
#include "rosban_regression_forests/core/regression_tree.h"
#include "rosban_regression_forests/core/regression_forest.h"

/**
 * Extension of 'Extremely Randomized Trees' (Geurts06) to a situation where
 * the tree is not grown from an example but from a black box
 */
namespace regression_forests
{
namespace BB2Tree
{
typedef std::function<double(const Eigen::VectorXd &)> EvalFunc;

class SplitEntry
{
public:
  RegressionNode *node;
  double gain;
  OrthogonalSplit split;
  TrainingSet::Subset samples;
  Eigen::MatrixXd space;

  bool operator<(const SplitEntry &other) const;
};

class BB2TreeConfig
{
public:
  ApproximationType apprType;  // Which approximation for leafs?
  int k;                       // Number of dimensions used at each split choice
  // End condition
  double minPotGain;
  size_t maxLeafs;
  // Avoiding undersampling
  int nMin;
  double minDensity;
  // From input to output (no need to know the successor state)
  EvalFunc eval;
  // Overall Space of the problem
  Eigen::MatrixXd space;
  // Only for forests
  int nbTrees;

  BB2TreeConfig();
  std::vector<std::string> names() const;
  std::vector<std::string> values() const;
  void load(const std::vector<std::string> &names, const std::vector<std::string> &values);
};

std::unique_ptr<RegressionTree> bb2Tree(const BB2TreeConfig &config);
std::unique_ptr<RegressionForest> bb2Forest(const BB2TreeConfig &config);
}
}

bool operator<(const regression_forests::BB2Tree::SplitEntry &se1,
               const regression_forests::BB2Tree::SplitEntry &se2);
