#pragma once

#include "rosban_regression_forests/approximations/approximation_type.h"
#include "rosban_regression_forests/core/training_set.h"
#include "rosban_regression_forests/core/tree.h"
#include "rosban_regression_forests/core/forest.h"

#include <rosban_utils/serializable.h>

/**
 * Based on 'Extremely Randomized Trees' (Geurts06)
 */
namespace regression_forests
{
class ExtraTrees
{
public:

  typedef std::function<Approximation *(const TrainingSet::Subset &,
                                        const Eigen::MatrixXd &)> Approximator;

  class Config : public rosban_utils::Serializable
  {
  public:
    /// nb_trees: the number of trees to grow
    size_t nb_trees;
    /// k: the number of dimensions used for randomCut
    size_t k;
    /// n_min: the minimal number of samples per leaf
    size_t n_min;
    /// min_var: if variance is lower than the given threshold, do not split any further
    /// TODO: apply min_var after normalization
    double min_var;
    /// appr_type: which types of approximation should be used for the leafs
    ApproximationType appr_type;
    /// val_max: Approximations are not allowed to send a value above
    double val_max;
    /// val_min: Approximations are not allowed to send a value below
    double val_min;
    /// nb_threads: Number of threads used to compute the regression forest
    int nb_threads;

    Config();

    // XML stuff
    virtual std::string class_name() const override;
    virtual void to_xml(std::ostream &out) const override;
    virtual void from_xml(TiXmlNode *node) override;

    static Config generateAuto(const Eigen::MatrixXd &space_limits,
                               int nb_samples,
                               ApproximationType appr_type);
  };

  Config conf;

  static double evalSplitScore(const TrainingSet &ls,
                               const TrainingSet::Subset &samples,
                               const OrthogonalSplit &split,
                               Approximator approximator,
                               const Eigen::MatrixXd &limits);

  std::unique_ptr<Tree> solveTree(const TrainingSet &ts,
                                  const Eigen::MatrixXd &limits);
  std::unique_ptr<Forest> solve(const TrainingSet &ts,
                                const Eigen::MatrixXd &limits);
};

}
