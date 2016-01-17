#include "rosban_regression_forests/algorithms/randomized_tree.h"

#include "rosban_regression_forests/approximations/pwc_approximation.h"
#include "rosban_regression_forests/approximations/pwl_approximation.h"

#include "rosban_regression_forests/tools/random.h"
#include "rosban_regression_forests/tools/statistics.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stack>
#include <string>

namespace regression_forests
{
namespace RandomizedTree
{
ExtraTreesConfig::ExtraTreesConfig()
{
  k = 1;
  nMin = 1;
  nbTrees = 1;
  minVar = 0;
  bootstrap = false;
  apprType = ApproximationType::PWC;
}

std::vector<std::string> ExtraTreesConfig::names() const
{
  return {"K", "NMin", "NbTrees", "MinVar", "Bootstrap", "ApprType"};
}

std::vector<std::string> ExtraTreesConfig::values() const
{
  std::vector<std::string> result;
  result.push_back(std::to_string(k));
  result.push_back(std::to_string(nMin));
  result.push_back(std::to_string(nbTrees));
  // Custom behavior for minVar due to rounding
  std::ostringstream min_var_oss;
  min_var_oss << std::setprecision(6) << minVar;
  result.push_back(min_var_oss.str());
  result.push_back(std::to_string(bootstrap));
  result.push_back(to_string(apprType));
  return result;
}

void ExtraTreesConfig::load(const std::vector<std::string> &colNames, const std::vector<std::string> &colValues)
{
  auto expectedNames = names();
  if (colNames.size() != expectedNames.size())
  {
    throw std::runtime_error("Failed to load extraTreesConfig, mismatch of vector size");
  }
  for (size_t colNo = 0; colNo < colNames.size(); colNo++)
  {
    auto givenName = colNames[colNo];
    auto expectedName = expectedNames[colNo];
    if (givenName.find(expectedName) == std::string::npos)
    {
      throw std::runtime_error("Given name '" + givenName + "' does not match '" + expectedName + "'");
    }
  }
  k = std::stoi(colValues[0]);
  nMin = std::stoi(colValues[1]);
  nbTrees = std::stoi(colValues[2]);
  minVar = std::stod(colValues[3]);
  bootstrap = std::stoi(colValues[4]);
  apprType = loadApproximationType(colValues[5]);
}

double avgSquaredErrors(const TrainingSet &ls, const TrainingSet::Subset &samples, enum ApproximationType apprType)
{
  switch (apprType)
  {
    case PWC:
      return Statistics::variance(ls.values(samples));
    case PWL:
    {
      std::vector<Eigen::VectorXd> inputs = ls.inputs(samples);
      std::vector<double> outputs = ls.values(samples);
      PWLApproximation a(inputs, outputs);
      double sumSquaredError = 0;
      for (size_t i = 0; i < inputs.size(); i++)
      {
        double error = a.eval(inputs[i]) - outputs[i];
        sumSquaredError += error * error;
      }
      return sumSquaredError / inputs.size();
    }
  }
  throw std::runtime_error("Unknown ApprType");
}

double evalSplitScore(const TrainingSet &ls, const TrainingSet::Subset &samples, const OrthogonalSplit &split,
                      enum ApproximationType apprType)
{
  std::vector<int> samplesUpper, samplesLower;
  ls.applySplit(split, samples, samplesLower, samplesUpper);
  double nbSamples = samples.size();
  // This happened once, but no idea why
  if (samplesLower.size() == 0 || samplesUpper.size() == 0)
  {
    std::ostringstream oss;
    oss << "One of the sample Set is empty, this should never happen" << std::endl;
    oss << "Split: (" << split.dim << "," << split.val << ")" << std::endl;
    oss << "Samples:" << std::endl;
    std::vector<double> dimValues = ls.inputs(samples, split.dim);
    std::sort(dimValues.begin(), dimValues.end());
    for (double v : dimValues)
    {
      std::cout << v << std::endl;
    }
  }
  double varAll = avgSquaredErrors(ls, samples, apprType);
  double varLower = avgSquaredErrors(ls, samplesLower, apprType);
  double varUpper = avgSquaredErrors(ls, samplesUpper, apprType);
  double weightedNewVar = (varLower * samplesLower.size() + varUpper * samplesUpper.size()) / nbSamples;
  return (varAll - weightedNewVar) / varAll;
}

std::unique_ptr<Tree> learn(const TrainingSet &ls, size_t k, size_t nmin, double minVariance,
                                      enum ApproximationType apprType)
{
  std::function<Approximation *(const TrainingSet::Subset &)> approximateSamples;
  switch (apprType)
  {
    case PWC:
      approximateSamples = [&ls](const TrainingSet::Subset &samples)
      {
        return new PWCApproximation(Statistics::mean(ls.values(samples)));
      };
      break;
    case PWL:
      approximateSamples = [&ls](const TrainingSet::Subset &samples)
      {
        return new PWLApproximation(ls.inputs(samples), ls.values(samples));
      };
      break;
  }

  std::unique_ptr<Tree> t(new Tree);
  auto generator = regression_forests::get_random_engine();
  // All along the resolution, we will stack samples
  std::stack<TrainingSet::Subset> samplesStack;
  std::stack<Node *> nodesStack;
  t->root = new Node(NULL);
  // If splitting is not allowed, end directly the process
  if (ls.size() < 2 * nmin)
  {
    t->root->a = approximateSamples(ls.wholeSubset());
    return t;
  }
  samplesStack.push(ls.wholeSubset());
  nodesStack.push(t->root);
  // While there is still nodes to explore
  while (nodesStack.size() != 0)
  {
    Node *node = nodesStack.top();
    TrainingSet::Subset samples = samplesStack.top();
    nodesStack.pop();
    samplesStack.pop();
    // Test if there is enough sample to split, the approximate the node
    if (samples.size() < 2 * nmin || Statistics::variance(ls.values(samples)) < minVariance)
    {
      node->a = approximateSamples(samples);
      continue;
    }
    // Find split candidates
    std::vector<size_t> dimCandidates = regression_forests::getKDistinctFromN(k, ls.getInputDim(), &generator);
    std::vector<OrthogonalSplit> splitCandidates;
    splitCandidates.reserve(k);
    for (size_t i = 0; i < k; i++)
    {
      size_t dim = dimCandidates[i];
      ls.sortSubset(samples, dim);
      double sValMin = ls(samples[nmin - 1]).getInput(dim);
      double sValMax = ls(samples[samples.size() - nmin]).getInput(dim);
      if (sValMin == sValMax)
      {
        // If we cut on sValMin, that would lead to having less than nmin
        // candidates on one node, therefore, we do not use this dimension
        // as a split candidate
        continue;
      }
      std::uniform_real_distribution<double> distribution(sValMin, sValMax);
      splitCandidates.push_back(OrthogonalSplit(dim, distribution(generator)));
    }
    // If no splits are available do not split node
    if (splitCandidates.size() == 0)
    {
      // could happen if sValMin == sValMax for all dimensions in this case,
      // do not split node any further, therefore, add an Estimation!!!
      node->a = approximateSamples(samples);
      continue;
    }
    // Find best split candidate
    size_t bestSplitIdx = 0;
    double bestSplitScore = evalSplitScore(ls, samples, splitCandidates[0], apprType);
    for (size_t splitIdx = 1; splitIdx < splitCandidates.size(); splitIdx++)
    {
      double splitScore = evalSplitScore(ls, samples, splitCandidates[splitIdx], apprType);
      if (splitScore > bestSplitScore)
      {
        bestSplitScore = splitScore;
        bestSplitIdx = splitIdx;
      }
    }
    // Apply best split
    node->s = splitCandidates[bestSplitIdx];
    TrainingSet::Subset lowerSamples, upperSamples;
    ls.applySplit(node->s, samples, lowerSamples, upperSamples);
    // UpperChild
    node->upperChild = new Node(node);
    if (upperSamples.size() >= 2 * nmin)
    {
      nodesStack.push(node->upperChild);
      samplesStack.push(upperSamples);
    }
    else
    {
      node->upperChild->a = approximateSamples(upperSamples);
    }
    // LowerChild
    node->lowerChild = new Node(node);
    if (lowerSamples.size() >= 2 * nmin)
    {
      nodesStack.push(node->lowerChild);
      samplesStack.push(lowerSamples);
    }
    else
    {
      node->lowerChild->a = approximateSamples(lowerSamples);
    }
  }
  return t;
}

std::unique_ptr<Forest> extraTrees(const TrainingSet &ls, size_t k, size_t nmin, size_t nbTrees,
                                             double minVariance, bool bootstrap, enum ApproximationType apprType)
{
  std::unique_ptr<Forest> f(new Forest);
  for (size_t i = 0; i < nbTrees; i++)
  {
    if (bootstrap)
    {
      f->push(learn(ls.buildBootstrap(), k, nmin, minVariance, apprType));
    }
    else
    {
      f->push(learn(ls, k, nmin, minVariance, apprType));
    }
  }
  return f;
}
}
}
