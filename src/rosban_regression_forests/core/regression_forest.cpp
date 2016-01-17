#include "rosban_regression_forests/core/regression_forest.h"

#include "rosban_regression_forests/tools/parser.h"
#include "rosban_regression_forests/core/pruning.h"

#include <fstream>

namespace regression_forests
{
RegressionForest::RegressionForest()
{
}

RegressionForest::~RegressionForest()
{
}

size_t RegressionForest::nbTrees() const
{
  return trees.size();
}

const RegressionTree &RegressionForest::getTree(size_t treeId) const
{
  return *trees[treeId];
}

void RegressionForest::push(std::unique_ptr<RegressionTree> t)
{
  trees.push_back(std::move(t));
}

size_t RegressionForest::maxSplitDim() const
{
  size_t max = 0;
  for (const auto &t : trees)
  {
    max = std::max(max, t->maxSplitDim());
  }
  return max;
}

double RegressionForest::getValue(const Eigen::VectorXd &input) const
{
  double sum = 0.0;
  for (const auto &t : trees)
  {
    sum += t->getValue(input);
  }
  return sum / trees.size();
}

std::unique_ptr<RegressionTree> RegressionForest::unifiedProjectedTree(const Eigen::MatrixXd &limits, size_t maxLeafs,
                                                                       bool preFilter, bool parallelMerge)
{
  std::unique_ptr<RegressionTree> result;
  if (trees.size() == 0)
    return result;
  result = trees[0]->project(limits);
  if (maxLeafs != 0)
  {
    result = pruneTree(std::move(result), limits, maxLeafs);
  }
  for (size_t treeId = 1; treeId < trees.size(); treeId++)
  {
    std::unique_ptr<RegressionTree> tree;
    if (preFilter)
    {
      tree = trees[treeId]->project(limits);
    }
    else
    {
      tree = std::move(trees[treeId]);
    }
    if (parallelMerge)
    {
      result = RegressionTree::avgTrees(*result, *tree, treeId, 1, limits);
    }
    else
    {
      result->avgTree(*tree, 1.0 / treeId, limits);
    }
    // Give back property to the vector
    if (!preFilter)
    {
      trees[treeId] = std::move(tree);
    }
    if (maxLeafs != 0)
    {
      result = pruneTree(std::move(result), limits, maxLeafs);
    }
  }
  return result;
}

void RegressionForest::save(const std::string &path) const
{
  std::ofstream ofs;
  ofs.open(path);
  ofs << *this;
  ofs.close();
}

std::unique_ptr<RegressionForest> RegressionForest::loadFile(const std::string &path)
{
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (in)
  {
    std::string contents;
    in.seekg(0, std::ios::end);
    contents.resize(in.tellg());
    in.seekg(0, std::ios::beg);
    in.read(&contents[0], contents.size());
    in.close();
    return Parser::regressionForest(contents);
  }
  throw std::runtime_error("Failed to open file '" + path + "'");
}
}

std::ostream &operator<<(std::ostream &out, const regression_forests::RegressionForest &forest)
{
  out << 'f';
  for (size_t treeId = 0; treeId < forest.nbTrees(); treeId++)
  {
    out << forest.getTree(treeId);
  }
  out << '$';
  return out;
}
