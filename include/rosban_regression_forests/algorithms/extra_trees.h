#pragma once

#include "rosban_regression_forests/core/training_set.h"
#include "rosban_regression_forests/core/tree.h"
#include "rosban_regression_forests/core/forest.h"

#include "rosban_gp/gradient_ascent/randomized_rprop.h"

#include <rosban_utils/serializable.h>

/**
 * Based on 'Extremely Randomized Trees' (Geurts06)
 */
namespace regression_forests
{
class ExtraTrees
{
public:

  typedef std::function<std::unique_ptr<Approximation> (const TrainingSet::Subset &,
                                                        const Eigen::MatrixXd &)> Approximator;

  class Config : public rosban_utils::Serializable
  {
  public:
    /// nb_trees: the number of trees to grow
    int nb_trees;
    /// k: the number of dimensions used for randomCut
    int k;
    /// n_min: the minimal number of samples per leaf
    int n_min;
    /// max_samples: the maximal number of samples considered when applying a split
    int max_samples;
    /// min_var: if variance is lower than the given threshold, do not split any further
    /// TODO: apply min_var after normalization
    double min_var;
    /// appr_type: which types of approximation should be used for the leafs
    Approximation::ID appr_type;
    /// nb_threads: Number of threads used to compute the regression forest
    int nb_threads;
    /// gp_conf: Parameters for auto-tuning of Gaussian Processes when used
    rosban_gp::RandomizedRProp::Config gp_conf;

    Config();

    // XML stuff
    virtual std::string class_name() const override;
    virtual void to_xml(std::ostream &out) const override;
    virtual void from_xml(TiXmlNode *node) override;

    static Config generateAuto(const Eigen::MatrixXd &space_limits,
                               int nb_samples,
                               Approximation::ID appr_type);
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
