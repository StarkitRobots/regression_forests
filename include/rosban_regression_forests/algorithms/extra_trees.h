#pragma once

#include "rosban_regression_forests/approximations/approximation_type.h"
#include "rosban_regression_forests/core/training_set.h"
#include "rosban_regression_forests/core/tree.h"
#include "rosban_regression_forests/core/forest.h"

/**
 * Based on 'Extremely Randomized Trees' (Geurts06)
 */
namespace regression_forests
{
class ExtraTrees
{
public:
  class Config
  {
  public:
    /// nb_trees: the number of trees to grow
    size_t nb_trees;
    /// k: the number of dimensions used for randomCut
    size_t k;
    /// n_min: the minimal number of samples per leaf
    size_t n_min;
    /// min_var: if variance is lower than the given threshold, do not split any further
    double min_var;
    /// appr_type: which types of approximation should be used for the leafs
    ApproximationType appr_type;

    Config();
    std::vector<std::string> names() const;
    std::vector<std::string> values() const;
    void load(const std::vector<std::string> &names, const std::vector<std::string> &values);
  };

  Config conf;

  static double evalSplitScore(const TrainingSet &ls,
                               const TrainingSet::Subset &samples,
                               const OrthogonalSplit &split,
                               enum ApproximationType appr_type);

  std::unique_ptr<Tree> solveTree(const TrainingSet &ts);
  std::unique_ptr<Forest> solve(const TrainingSet &ts);
};

}
