#pragma once

#include "rhoban_regression_forests/core/training_set.h"
#include "rhoban_regression_forests/core/tree.h"
#include "rhoban_regression_forests/core/forest.h"

/**
 * Extension of 'Extremely Randomized Trees' (Geurts06) to a situation where
 * the tree is not grown from an example but from a black box
 */
namespace regression_forests
{
class BB2Tree
{
public:
  typedef std::function<double(const Eigen::VectorXd &)> EvalFunc;

  class SplitEntry
  {
  public:
    Node *node;
    double gain;
    OrthogonalSplit split;
    TrainingSet::Subset samples;
    Eigen::MatrixXd space;

    bool operator<(const SplitEntry &other) const;
  };

  class BB2TreeConfig
  {
  public:
    Approximation::ID apprType;  // Which approximation for leafs?
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

  static std::unique_ptr<Tree> bb2Tree(const BB2TreeConfig &config);
  static std::unique_ptr<Forest> bb2Forest(const BB2TreeConfig &config);
};
}

bool operator<(const regression_forests::BB2Tree::SplitEntry &se1,
               const regression_forests::BB2Tree::SplitEntry &se2);
