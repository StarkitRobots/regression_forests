#include "rosban_regression_forests/tools/parser.h"

#include "rosban_regression_forests/approximations/composite_approximation.h"
#include "rosban_regression_forests/approximations/pwc_approximation.h"
#include "rosban_regression_forests/approximations/pwl_approximation.h"

namespace Math
{
namespace RegressionTree
{
namespace Parser
{
Approximation *approximation(const std::string &s, size_t *index)
{
  Approximation *result = NULL;
  size_t idx = 0;
  if (index != NULL)
  {
    idx = *index;
  }
  if (s[idx] != 'a')
  {
    throw std::runtime_error("RegressionTree parser: expecting a for approximation");
  }
  idx++;
  // Read PWCApproximation
  if (s.compare(idx, 3, "pwc") == 0)
  {
    idx += 3;
    int endVal = s.find("$", idx);
    double val = stod(s.substr(idx, endVal - idx));
    idx = endVal + 1;
    result = new PWCApproximation(val);
  }
  // Read PWLApproximation
  else if (s.compare(idx, 3, "pwl") == 0)
  {
    idx += 3;
    // Read number of dimensions
    int endDim = s.find("|", idx);
    size_t dim = stoi(s.substr(idx, endDim - idx));
    idx = endDim + 1;
    // Read factors
    Eigen::VectorXd factors(dim);
    for (size_t d = 0; d < dim; d++)
    {
      int endFac = s.find("|", idx);
      factors(d) = stod(s.substr(idx, endFac - idx));
      idx = endFac + 1;
    }
    // Expecting a $ to signal end of approximation
    if (idx >= s.size() || s[idx] != '$')
    {
      throw std::runtime_error("Expecting a '$' at end of PWLApproximation");
    }
    idx++;
    // Create PWLApproximation
    result = new PWLApproximation(factors);
  }
  // Read CompositeApproximation
  else if (s.compare(idx, 1, "c") == 0)
  {
    CompositeApproximation *ca = new CompositeApproximation();
    idx++;
    while (idx < s.size() && s[idx] == 'a')
    {
      ca->push(approximation(s, &idx));
    }
    if (idx >= s.size())
    {
      delete (ca);
      throw std::runtime_error("Reached end of string while parsing a compositeApproximation");
    }
    if (s[idx] != '$')
    {
      delete (ca);
      throw std::runtime_error("Expecting a '$' at end of compositeApproximation");
    }
    idx++;
    result = ca;
  }
  else
  {
    throw std::runtime_error("Unexpected description of action near: " + s.substr(idx, 5));
  }
  if (index != NULL)
  {
    *index = idx;
  }
  return result;
}

OrthogonalSplit orthogonalSplit(const std::string &s, size_t *index)
{
  OrthogonalSplit result;
  size_t idx = 0;
  if (index != NULL)
  {
    idx = *index;
  }
  if (s.compare(idx, 2, "sd") != 0)
  {
    throw std::runtime_error("Invalid format while parsing split, expecting sd");
  }
  idx += 2;
  int endDim = s.find("$", idx);
  result.dim = std::stoi(s.substr(idx, endDim - idx));
  idx = endDim + 1;
  if (s[idx] != 'v')
  {
    throw std::runtime_error("Invalid format while parsing split, expecting v");
  }
  idx++;
  int endVal = s.find("$", idx);
  result.val = std::stod(s.substr(idx, endVal - idx));
  idx = endVal + 1;
  if (s[idx] != '$')
  {
    throw std::runtime_error("Invalid format while parsing split, expecting $");
  }
  idx++;
  if (index != NULL)
  {
    *index = idx;
  }
  return result;
}

RegressionNode *regressionNode(const std::string &s, size_t *index)
{
  RegressionNode *result = NULL;
  size_t idx = 0;
  if (index != NULL)
  {
    idx = *index;
  }
  if (s[idx] != 'n')
  {
    throw std::runtime_error("Expecting a 'n' for the beginning of a regressionNode");
  }
  idx++;
  result = new RegressionNode();
  // Leaf case:
  if (s[idx] == 'a')
  {
    result->a = approximation(s, &idx);
  }
  // Splitted node case:
  else if (s[idx] == 's')
  {
    result->s = orthogonalSplit(s, &idx);
    result->lowerChild = regressionNode(s, &idx);
    result->upperChild = regressionNode(s, &idx);
  }
  else
  {
    throw std::runtime_error("Invalid char at the beginning of node");
  }
  if (s[idx] != '$')
  {
    throw std::runtime_error("Expecting a $ at end of node");
  }
  idx++;
  if (index != NULL)
  {
    *index = idx;
  }
  return result;
}

std::unique_ptr<RegressionTree> regressionTree(const std::string &s, size_t *index)
{
  std::unique_ptr<RegressionTree> tree(new RegressionTree);
  tree->root = regressionNode(s, index);
  return tree;
}

std::unique_ptr<RegressionForest> regressionForest(const std::string &s, size_t *index)
{
  std::unique_ptr<RegressionForest> forest(new RegressionForest);
  size_t idx = 0;
  if (index != NULL)
  {
    idx = *index;
  }
  if (s[idx] != 'f')
  {
    throw std::runtime_error("Expecting a 'f' at the beginning of forest");
  }
  idx++;
  while (idx < s.size() && s[idx] != '$')
  {
    forest->push(regressionTree(s, &idx));
  }
  if (idx >= s.size() || s[idx] != '$')
  {
    throw std::runtime_error("Expecting a '$' at the end of forest");
  }
  idx++;
  if (index != NULL)
  {
    *index = idx;
  }
  return forest;
}
}
}
}
