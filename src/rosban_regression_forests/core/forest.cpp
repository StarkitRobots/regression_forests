#include "rosban_regression_forests/core/forest.h"

#include "rosban_regression_forests/tools/parser.h"
#include "rosban_regression_forests/core/pruning.h"

#include <fstream>

namespace regression_forests
{
Forest::Forest()
{
}

Forest::~Forest()
{
}

size_t Forest::nbTrees() const
{
  return trees.size();
}

const Tree &Forest::getTree(size_t treeId) const
{
  return *trees[treeId];
}

void Forest::push(std::unique_ptr<Tree> t)
{
  trees.push_back(std::move(t));
}

size_t Forest::maxSplitDim() const
{
  size_t max = 0;
  for (const auto &t : trees)
  {
    max = std::max(max, t->maxSplitDim());
  }
  return max;
}

double Forest::getValue(const Eigen::VectorXd &input) const
{
  double sum = 0.0;
  for (const auto &t : trees)
  {
    sum += t->getValue(input);
  }
  return sum / trees.size();
}

std::unique_ptr<Tree> Forest::unifiedProjectedTree(const Eigen::MatrixXd &limits, size_t maxLeafs)
{
  std::unique_ptr<Tree> result;
  if (trees.size() == 0)
    return result;
  result = trees[0]->project(limits);
  if (maxLeafs != 0)
  {
    result = pruneTree(std::move(result), limits, maxLeafs);
  }
  for (size_t treeId = 1; treeId < trees.size(); treeId++)
  {
    std::unique_ptr<Tree> tree;
    tree = std::move(trees[treeId]);
    result = Tree::avgTrees(*result, *tree, treeId, 1, limits);
    // Give back property to the vector
    trees[treeId] = std::move(tree);
    if (maxLeafs != 0)
    {
      result = pruneTree(std::move(result), limits, maxLeafs);
    }
  }
  return result;
}

void Forest::save(const std::string &path) const
{
  std::ofstream ofs;
  ofs.open(path);
  ofs << *this;
  ofs.close();
}

std::unique_ptr<Forest> Forest::loadFile(const std::string &path)
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

void Forest::apply(Eigen::MatrixXd &limits, Node::Function f)
{
  for (size_t i = 0; i < trees.size(); i++)
  {
    trees[i]->apply(limits, f);
  }
}

void Forest::applyOnLeafs(Eigen::MatrixXd &limits, Node::Function f)
{
  for (size_t i = 0; i < trees.size(); i++)
  {
    trees[i]->applyOnLeafs(limits, f);
  }
}

}

std::ostream &operator<<(std::ostream &out, const regression_forests::Forest &forest)
{
  out << 'f';
  for (size_t treeId = 0; treeId < forest.nbTrees(); treeId++)
  {
    out << forest.getTree(treeId);
  }
  out << '$';
  return out;
}
