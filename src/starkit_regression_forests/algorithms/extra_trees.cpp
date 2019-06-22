#include "starkit_regression_forests/algorithms/extra_trees.h"

#ifdef STARKIT_RF_USES_GP
#include "starkit_regression_forests/approximations/gp_approximation.h"
#endif

#include "starkit_regression_forests/approximations/pwc_approximation.h"
#include "starkit_regression_forests/approximations/pwl_approximation.h"

#include "starkit_random/tools.h"
#include "starkit_regression_forests/tools/statistics.h"

#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stack>
#include <string>
#include <thread>

namespace regression_forests
{
ExtraTrees::Config::Config()
{
  k = 1;
  n_min = 1;
  max_samples = std::numeric_limits<int>::max();
  nb_trees = 1;
  nb_threads = 1;
  min_var = 0;
  appr_type = Approximation::ID::PWC;
}

std::string ExtraTrees::Config::getClassName() const
{
  return "ExtraTreesConfig";
}

Json::Value ExtraTrees::Config::toJson() const
{
  Json::Value v;
  v["k"] = k;
  v["n_min"] = n_min;
  v["max_samples"] = max_samples;
  v["nb_trees"] = nb_trees;
  v["nb_threads"] = nb_threads;
  v["min_var"] = min_var;
  v["appr_type"] = Approximation::idToString(appr_type);
#ifdef STARKIT_RF_USES_GP
  if (appr_type == Approximation::ID::GP)
  {
    v["gp_conf"] = gp_conf.toJson();
  }
#endif
  return v;
}

void ExtraTrees::Config::fromJson(const Json::Value& v, const std::string& dir_name)
{
  (void)dir_name;
  std::string appr_type_str;
  starkit_utils::tryRead(v, "k", &k);
  starkit_utils::tryRead(v, "n_min", &n_min);
  starkit_utils::tryRead(v, "max_samples", &max_samples);
  starkit_utils::tryRead(v, "nb_trees", &nb_trees);
  starkit_utils::tryRead(v, "nb_threads", &nb_threads);
  starkit_utils::tryRead(v, "min_var", &min_var);
  starkit_utils::tryRead(v, "appr_type", &appr_type_str);
  if (appr_type_str != "")
  {
    appr_type = Approximation::loadID(appr_type_str);
  }
#ifdef STARKIT_RF_USES_GP
  if (appr_type == Approximation::ID::GP)
  {
    gp_conf.read(v, "gp_conf");
  }
#endif
}

ExtraTrees::Config ExtraTrees::Config::generateAuto(const Eigen::MatrixXd& space_limits, int nb_samples,
                                                    Approximation::ID appr_type)
{
  // Forbid PWL if nb_samples < 1 + space_limits.rows
  if (nb_samples < 1 + space_limits.rows())
  {
    appr_type = Approximation::ID::PWC;
  }

  ExtraTrees::Config conf;
  conf.nb_trees = 25;  // Widely accepted as high enough to bring satisfying results
  // conf.k = (int)std::sqrt(space_limits.rows());// Usual heuristic proposed in Ernst05
  conf.k = space_limits.rows();
  int n_min_base = std::max(1, (int)std::log(nb_samples));
  switch (appr_type)
  {
    case Approximation::ID::PWC:
      conf.n_min = n_min_base;
      break;
    case Approximation::ID::PWL:
      conf.n_min = n_min_base * space_limits.rows();
      // PWL requires a number of sample strictly greater than space dimension
      conf.n_min = std::max((int)(space_limits.rows() + 1), conf.n_min);
      break;
#ifdef STARKIT_RF_USES_GP
    case Approximation::ID::GP:
      conf.n_min = std::ceil(std::log2(nb_samples));
      // Setting a minimal value for conf.n_min when using Gaussian Processes
      conf.n_min = std::max(5, conf.n_min);
      break;
#endif
  }
  // conf.max_samples = 4 * conf.n_min;
  conf.max_samples = std::numeric_limits<int>::max();
  conf.appr_type = appr_type;
  return conf;
}

static double avgSquaredErrors(const TrainingSet& ts, const TrainingSet::Subset& samples,
                               std::shared_ptr<const Approximation> a)
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

// Here, scorecould be simplified: no real need to use varAll
// no need to normalize by number of samples, we only want the lowest squaredError split
double ExtraTrees::evalSplitScore(const TrainingSet& ts, const TrainingSet::Subset& samples,
                                  const OrthogonalSplit& split, Approximator approximator,
                                  const Eigen::MatrixXd& limits)
{
  std::vector<int> samples_upper, samples_lower;
  ts.applySplit(split, samples, samples_lower, samples_upper);
  // Computing limits
  Eigen::MatrixXd limits_lower(limits), limits_upper(limits);
  limits_lower(split.dim, 1) = split.val;
  limits_upper(split.dim, 0) = split.val;
  // Computing approximations
  std::shared_ptr<Approximation> approx = approximator(samples, limits);
  std::shared_ptr<Approximation> approx_lower = approximator(samples_lower, limits_lower);
  std::shared_ptr<Approximation> approx_upper = approximator(samples_upper, limits_upper);
  // Computing variances
  double nb_samples = samples.size();
  double varAll = avgSquaredErrors(ts, samples, approx);
  double varLower = avgSquaredErrors(ts, samples_lower, approx_lower);
  double varUpper = avgSquaredErrors(ts, samples_upper, approx_upper);
  double weightedNewVar = (varLower * samples_lower.size() + varUpper * samples_upper.size()) / nb_samples;
  return (varAll - weightedNewVar) / varAll;
}

std::unique_ptr<Tree> ExtraTrees::solveTree(const TrainingSet& ts, const Eigen::MatrixXd& space)
{
  Approximator approximateSamples;
  switch (conf.appr_type)
  {
    case Approximation::ID::PWC:
      approximateSamples = [&ts](const TrainingSet::Subset& samples, const Eigen::MatrixXd& limits) {
        (void)limits;
        double mean = Statistics::mean(ts.values(samples));
        return std::unique_ptr<Approximation>(new PWCApproximation(mean));
      };
      break;
    case Approximation::ID::PWL:
      approximateSamples = [this, &ts](const TrainingSet::Subset& samples, const Eigen::MatrixXd& limits) {
        return std::unique_ptr<Approximation>(new PWLApproximation(ts.inputs(samples), ts.values(samples)));
      };
      break;
#ifdef STARKIT_RF_USES_GP
    case Approximation::ID::GP:
      approximateSamples = [this, &ts](const TrainingSet::Subset& samples, const Eigen::MatrixXd& limits) {
        (void)limits;
        if (samples.size() > 2 * this->conf.n_min)
        {
          std::ostringstream oss;
          oss << "Warning: large number of samples used for approximation: " << samples.size()
              << " samples with n_min = " << this->conf.n_min << std::endl;
          std::cout << oss.str();
        }
        return std::unique_ptr<Approximation>(
            new GPApproximation(ts.inputs(samples), ts.values(samples), this->conf.gp_conf));
      };
      break;
#endif
    default:
      throw std::runtime_error("Unknown approximation type in ExtraTrees::solveTree()");
  }

  std::unique_ptr<Tree> t(new Tree);
  auto generator = starkit_random::getRandomEngine();
  // All along the resolution, we will stack samples
  std::stack<TrainingSet::Subset> samples_stack;
  std::stack<Node*> nodes_stack;
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
    Node* node = nodes_stack.top();
    TrainingSet::Subset samples = samples_stack.top();
    Eigen::MatrixXd limits = limits_stack.top();
    nodes_stack.pop();
    samples_stack.pop();
    limits_stack.pop();
    // Test if there is not enough samples to split or if variance is too low,
    // then approximate the node
    // TODO: here we check variances of samples, before making use of max_samples,
    //       this makes max_samples pretty useless
    if (samples.size() < 2 * conf.n_min || Statistics::variance(ts.values(samples)) <= conf.min_var)
    {
      node->a = approximateSamples(samples, limits);
      continue;
    }
    // If there is too much samples, only use a subset
    TrainingSet::Subset split_samples;
    if (samples.size() > conf.max_samples)
    {
      std::vector<size_t> used_indices = starkit_random::getKDistinctFromN(conf.max_samples, samples.size(), &generator);
      split_samples.reserve(conf.max_samples);
      for (size_t idx : used_indices)
      {
        split_samples.push_back(samples[idx]);
      }
    }
    else
    {
      split_samples = samples;
    }
    // Find split candidates
    std::vector<size_t> dim_candidates;
    dim_candidates = starkit_random::getKDistinctFromN(conf.k, ts.getInputDim(), &generator);
    std::vector<OrthogonalSplit> split_candidates;
    split_candidates.reserve(conf.k);
    for (size_t i = 0; i < conf.k; i++)
    {
      size_t dim = dim_candidates[i];
      ts.sortSubset(split_samples, dim);
      double s_val_min = ts(split_samples[conf.n_min - 1]).getInput(dim);
      double s_val_max = ts(split_samples[split_samples.size() - conf.n_min]).getInput(dim);
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
      if (split_value == s_val_max)
        continue;
      split_candidates.push_back(OrthogonalSplit(dim, split_value));

#ifdef STARKIT_RF_USES_GP
      // When using GP approximations, use the first available split
      if (conf.appr_type == Approximation::ID::GP)
        break;
#endif
    }
    // If no splits are available do not split node
    if (split_candidates.size() == 0)
    {
      // could happen if s_val_min == s_val_max for all dimensions in this case,
      // do not split node any further, therefore, add an Estimation!!!
      node->a = approximateSamples(samples, limits);
      continue;
    }
    // Find best split candidate (using only subset of all the samples)
    size_t best_split_idx = 0;
    double best_split_score = 0;
#ifdef STARKIT_RF_USES_GP
    // GP only uses 1 random split yet so no need to score them
    if (conf.appr_type != Approximation::ID::GP)
    {
      best_split_score = evalSplitScore(ts, split_samples, split_candidates[0], approximateSamples, limits);
    }
#endif
    for (size_t split_idx = 1; split_idx < split_candidates.size(); split_idx++)
    {
#ifdef STARKIT_RF_USES_GP
      if (conf.appr_type == Approximation::ID::GP)
      {
        throw std::runtime_error("Running ExtraTrees with GaussianProcesses and k > 1");
      }
#endif
      double splitScore = evalSplitScore(ts, split_samples, split_candidates[split_idx], approximateSamples, limits);
      if (splitScore > best_split_score)
      {
        best_split_score = splitScore;
        best_split_idx = split_idx;
      }
    }
    // Apply best split (this time use all the samples)
    node->s = split_candidates[best_split_idx];
    TrainingSet::Subset lower_samples, upper_samples;
    ts.applySplit(node->s, samples, lower_samples, upper_samples);
    // Compute limits for childs
    Eigen::MatrixXd upper_limits(limits), lower_limits(limits);
    upper_limits(node->s.dim, 0) = node->s.val;
    lower_limits(node->s.dim, 1) = node->s.val;
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

std::unique_ptr<Forest> ExtraTrees::solve(const TrainingSet& ts, const Eigen::MatrixXd& limits)
{
  std::unique_ptr<Forest> f(new Forest);
  std::vector<std::thread> threads;
  std::mutex forest_mutex;
  auto solver = [this, &forest_mutex, &f, &ts, &limits](int nb_trees) {
    double solving_time = 0;
    double waiting_time = 0;
    double pushing_time = 0;
    for (int i = 0; i < nb_trees; i++)
    {
      std::unique_ptr<Tree> tree = this->solveTree(ts, limits);
      // Ensuring thread-safety when accessing the forest
      forest_mutex.lock();
      f->push(std::move(tree));
      forest_mutex.unlock();
    }
  };

  int nb_threads = std::min(conf.nb_threads, conf.nb_trees);

  double trees_by_thread = conf.nb_trees / (double)nb_threads;
  for (size_t thread_no = 0; thread_no < nb_threads; thread_no++)
  {
    // Compute trees in [start, end[
    int start = std::floor(thread_no * trees_by_thread);
    int end = std::floor((thread_no + 1) * trees_by_thread);
    int nb_trees = end - start;
    threads.push_back(std::thread(solver, nb_trees));
  }
  for (std::thread& t : threads)
  {
    t.join();
  }

  return f;
}

}  // namespace regression_forests
