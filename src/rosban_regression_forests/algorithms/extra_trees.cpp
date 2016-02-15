#include "rosban_regression_forests/algorithms/extra_trees.h"

#include "rosban_regression_forests/approximations/pwc_approximation.h"
#include "rosban_regression_forests/approximations/pwl_approximation.h"

#include "rosban_regression_forests/tools/random.h"
#include "rosban_regression_forests/tools/statistics.h"

#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stack>
#include <string>

namespace regression_forests
{
ExtraTrees::Config::Config()
{
  k = 1;
  n_min = 1;
  nb_trees = 1;
  min_var = 0;
  appr_type = ApproximationType::PWC;
  val_min = std::numeric_limits<double>::lowest();
  val_max = std::numeric_limits<double>::max();
}

std::vector<std::string> ExtraTrees::Config::names() const
{
  return {"k", "n_min", "nb_trees", "min_var", "appr_type"};
}

std::vector<std::string> ExtraTrees::Config::values() const
{
  std::vector<std::string> result;
  result.push_back(std::to_string(k));
  result.push_back(std::to_string(n_min));
  result.push_back(std::to_string(nb_trees));
  // Custom behavior for min_var due to rounding
  std::ostringstream min_var_oss;
  min_var_oss << std::setprecision(6) << min_var;
  result.push_back(min_var_oss.str());
  result.push_back(to_string(appr_type));
  return result;
}

void ExtraTrees::Config::load(const std::vector<std::string> &col_names,
                              const std::vector<std::string> &col_values)
{
  const std::vector<std::string> &expected_names = names();
  if (col_names.size() != expected_names.size())
  {
    throw std::runtime_error("Failed to load extraTreesConfig, mismatch of vector size");
  }
  for (size_t col_no = 0; col_no < col_names.size(); col_no++)
  {
    auto given_name = col_names[col_no];
    auto expected_name = expected_names[col_no];
    if (given_name.find(expected_name) == std::string::npos)
    {
      throw std::runtime_error("Given name '" + given_name + "' does not match '"
                               + expected_name + "'");
    }
  }
  k         = std::stoi(col_values[0]);
  n_min     = std::stoi(col_values[1]);
  nb_trees  = std::stoi(col_values[2]);
  min_var   = std::stod(col_values[3]);
  appr_type = loadApproximationType(col_values[4]);
}

std::string ExtraTrees::Config::class_name() const
{
  return "ExtraTreesConfig";
}

void ExtraTrees::Config::to_xml(std::ostream &out) const
{
  rosban_utils::xml_tools::write<int>("k", k, out);
  rosban_utils::xml_tools::write<int>("n_min", n_min, out);
  rosban_utils::xml_tools::write<int>("nb_trees", nb_trees, out);
  rosban_utils::xml_tools::write<double>("min_var", min_var, out);
  rosban_utils::xml_tools::write<double>("val_min", val_min, out);
  rosban_utils::xml_tools::write<double>("val_max", val_max, out);
  rosban_utils::xml_tools::write<std::string>("appr_type", to_string(appr_type), out);
}

void ExtraTrees::Config::from_xml(TiXmlNode *node)
{
  k         = rosban_utils::xml_tools::read<int>(node, "k");
  n_min     = rosban_utils::xml_tools::read<int>(node, "n_min");
  nb_trees  = rosban_utils::xml_tools::read<int>(node, "nb_trees");
  min_var   = rosban_utils::xml_tools::read<double>(node, "min_var");
  val_min   = rosban_utils::xml_tools::read<double>(node, "val_min");
  val_max   = rosban_utils::xml_tools::read<double>(node, "val_max");
  std::string appr_type_str;
  appr_type_str =rosban_utils::xml_tools::read<std::string>(node, "appr_type");
  appr_type = loadApproximationType(appr_type_str);
}


static double avgSquaredErrors(const TrainingSet &ts,
                               const TrainingSet::Subset &samples,
                               Approximation * a)
{
  std::vector<Eigen::VectorXd> inputs = ts.inputs(samples);
  std::vector<double> outputs = ts.values(samples);
  double sumSquaredError = 0;
  for (size_t i = 0; i < inputs.size(); i++)
  {
    double error = a->eval(inputs[i]) - outputs[i];
    sumSquaredError += error * error;
  }
  return sumSquaredError / inputs.size();
}

double ExtraTrees::evalSplitScore(const TrainingSet &ts,
                                  const TrainingSet::Subset &samples,
                                  const OrthogonalSplit &split,
                                  Approximator approximator,
                                  const Eigen::MatrixXd &limits)
{
  std::vector<int> samples_upper, samples_lower;
  ts.applySplit(split, samples, samples_lower, samples_upper);
  // Computing limits
  Eigen::MatrixXd limits_lower(limits), limits_upper(limits);
  limits_lower(split.dim, 1) = split.val;
  limits_upper(split.dim, 0) = split.val;
  // Computing approximations
  Approximation * approx       = approximator(samples      , limits      );
  Approximation * approx_lower = approximator(samples_lower, limits_lower);
  Approximation * approx_upper = approximator(samples_upper, limits_upper);
  // Computing variances
  double nb_samples = samples.size();
  double varAll   = avgSquaredErrors(ts, samples      , approx);
  double varLower = avgSquaredErrors(ts, samples_lower, approx_lower);
  double varUpper = avgSquaredErrors(ts, samples_upper, approx_upper);
  double weightedNewVar = (varLower * samples_lower.size()
                           + varUpper * samples_upper.size()) / nb_samples;
  return (varAll - weightedNewVar) / varAll;
}

std::unique_ptr<Tree> ExtraTrees::solveTree(const TrainingSet &ts,
                                            const Eigen::MatrixXd &space)
{
  std::function<Approximation *(const TrainingSet::Subset &,
                                const Eigen::MatrixXd &)> approximateSamples;
  switch (conf.appr_type)
  {
    case ApproximationType::PWC:
      approximateSamples = [&ts](const TrainingSet::Subset &samples,
                                 const Eigen::MatrixXd &limits)
      {
        (void) limits;
        return new PWCApproximation(Statistics::mean(ts.values(samples)));
      };
      break;
    case ApproximationType::PWL:
      approximateSamples = [this,&ts](const TrainingSet::Subset &samples,
                                      const Eigen::MatrixXd &limits)
      {
        PWLApproximation * pwl_app = new PWLApproximation(ts.inputs(samples), ts.values(samples));
        /// if PWL is valid, return it
        if (pwl_app->getMaxPair(limits).first <= this->conf.val_max &&
            pwl_app->getMinPair(limits).first >= this->conf.val_min)
        {
          return (Approximation*)pwl_app;
        }
        /// If not, delete it and return a PWC approximation
        delete(pwl_app);
        return (Approximation*)new PWCApproximation(Statistics::mean(ts.values(samples)));
      };
      break;
    default:
      throw std::runtime_error("Unknown approximation type in ExtraTrees::solveTree()");
  }

  std::unique_ptr<Tree> t(new Tree);
  auto generator = regression_forests::get_random_engine();
  // All along the resolution, we will stack samples
  std::stack<TrainingSet::Subset> samples_stack;
  std::stack<Node *> nodes_stack;
  std::stack<Eigen::MatrixXd> limits_stack;
  t->root = new Node(NULL);
  // If splitting is not allowed, end directly the process
  if (ts.size() < 2 * conf.n_min)
  {
    t->root->a = approximateSamples(ts.wholeSubset(), space);
    return t;
  }
  samples_stack.push(ts.wholeSubset());
  nodes_stack.push(t->root);
  limits_stack.push(space);
  // While there is still nodes to explore
  while (nodes_stack.size() != 0)
  {
    Node *node = nodes_stack.top();
    TrainingSet::Subset samples = samples_stack.top();
    Eigen::MatrixXd limits = limits_stack.top();
    nodes_stack.pop();
    samples_stack.pop();
    limits_stack.pop();
    // Test if there is enough sample to split, the approximate the node
    if (samples.size() < 2 * conf.n_min || Statistics::variance(ts.values(samples)) < conf.min_var)
    {
      node->a = approximateSamples(samples, limits);
      continue;
    }
    // Find split candidates
    std::vector<size_t> dim_candidates;
    dim_candidates = regression_forests::getKDistinctFromN(conf.k, ts.getInputDim(), &generator);
    std::vector<OrthogonalSplit> split_candidates;
    split_candidates.reserve(conf.k);
    for (size_t i = 0; i < conf.k; i++)
    {
      size_t dim = dim_candidates[i];
      ts.sortSubset(samples, dim);
      double s_val_min = ts(samples[conf.n_min - 1]).getInput(dim);
      double s_val_max = ts(samples[samples.size() - conf.n_min]).getInput(dim);
      if (s_val_min == s_val_max)
      {
        // If we cut on s_val_min, that would lead to having less than nmin
        // candidates on one node, therefore, we do not use this dimension
        // as a split candidate
        continue;
      }
      std::uniform_real_distribution<double> distribution(s_val_min, s_val_max);
      double split_value = distribution(generator);
      // This should never happen according to the definition of uniform_real_distribution,
      // because split_value is supposed to be in [min,max[, but in practice, it happens.
      // Therefore this extra condition is required to ensure that there won't be any empty
      // square
      if (split_value == s_val_max) continue;
      split_candidates.push_back(OrthogonalSplit(dim, split_value));
    }
    // If no splits are available do not split node
    if (split_candidates.size() == 0)
    {
      // could happen if s_val_min == s_val_max for all dimensions in this case,
      // do not split node any further, therefore, add an Estimation!!!
      node->a = approximateSamples(samples, limits);
      continue;
    }
    // Find best split candidate
    size_t best_split_idx = 0;
    double best_split_score = evalSplitScore(ts, samples, split_candidates[0],
                                             approximateSamples, limits);
    for (size_t split_idx = 1; split_idx < split_candidates.size(); split_idx++)
    {
      double splitScore = evalSplitScore(ts, samples, split_candidates[split_idx],
                                         approximateSamples, limits);
      if (splitScore > best_split_score)
      {
        best_split_score = splitScore;
        best_split_idx = split_idx;
      }
    }
    // Apply best split
    node->s = split_candidates[best_split_idx];
    TrainingSet::Subset lower_samples, upper_samples;
    ts.applySplit(node->s, samples, lower_samples, upper_samples);
    // Compute limits for childs
    Eigen::MatrixXd upper_limits(limits), lower_limits(limits);
    upper_limits(node->s.dim,0) = node->s.val;
    lower_limits(node->s.dim,1) = node->s.val;
    // UpperChild
    node->upperChild = new Node(node);
    if (upper_samples.size() >= 2 * conf.n_min)
    {
      nodes_stack.push(node->upperChild);
      samples_stack.push(upper_samples);
      limits_stack.push(upper_limits);
    }
    else
    {
      node->upperChild->a = approximateSamples(upper_samples, upper_limits);
    }
    // LowerChild
    node->lowerChild = new Node(node);
    if (lower_samples.size() >= 2 * conf.n_min)
    {
      nodes_stack.push(node->lowerChild);
      samples_stack.push(lower_samples);
      limits_stack.push(lower_limits);
    }
    else
    {
      node->lowerChild->a = approximateSamples(lower_samples, lower_limits);
    }
  }
  return t;
}

std::unique_ptr<Forest> ExtraTrees::solve(const TrainingSet &ts,
                                          const Eigen::MatrixXd &limits)
{
  std::unique_ptr<Forest> f(new Forest);
  for (size_t i = 0; i < conf.nb_trees; i++)
  {
    f->push(solveTree(ts, limits));
  }
  return f;
}

}
