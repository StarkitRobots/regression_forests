#include "rosban_regression_forests/core/forest.h"

#include "rosban_regression_forests/tools/parser.h"
#include "rosban_regression_forests/core/pruning.h"

#include "rosban_utils/io_tools.h"

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

double Forest::getVar(const Eigen::VectorXd &input) const
{
  // Excluding case where forest is empty
  if (trees.size() == 0) return 0;

  double mean = getValue(input);
  double var = 0;
  for (const auto &t : trees)
  {
    double diff = t->getValue(input) - mean;
    var += diff * diff;
  }
  return var / trees.size();
}

Eigen::VectorXd Forest::getGradient(const Eigen::VectorXd &input) const
{
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(input.rows());

  // Excluding case where forest is empty
  if (trees.size() == 0) return grad;

  for (const auto & tree : trees)
  {
    grad += tree->getGrad(input);
  }
  return grad / trees.size();
}

/// Return a randomized value based on the confidence interval
double Forest::getRandomizedValue(const Eigen::VectorXd &input,
                                  std::default_random_engine &engine) const
{
  double mean = getValue(input);
  double var = 0;
  for (const auto &t : trees)
  {
    double diff = t->getValue(input) - mean;
    var += diff * diff;
  }
  var /= trees.size();
  double std_dev = std::sqrt(var);
  // Special case which causes issue with the normal distribution
  if (std_dev == 0) return mean;

  std::normal_distribution<double> distrib(mean, std_dev);
  return distrib(engine);
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
    result = Tree::avgTrees(*result, *(trees[treeId]), treeId, 1, limits);
    if (maxLeafs != 0)
    {
      result = pruneTree(std::move(result), limits, maxLeafs);
    }
  }
  return result;
}

void Forest::save(const std::string &path) const
{
  std::ofstream ofs(path, std::ios::binary);
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

int Forest::write(std::ostream & out) const
{
  int bytes_written = 0;
  bytes_written += rosban_utils::write<int>(out, trees.size());
  for (int i = 0; i < trees.size(); i++) {
    bytes_written += trees[i]->write(out);
  }
  return bytes_written;  
}

int Forest::read(std::istream & in)
{
  // Fist clean used ressources
  trees.clear();
  // Then read
  int bytes_read = 0;
  int nb_trees;
  bytes_read += rosban_utils::read<int>(in, &nb_trees);
  for (int i = 0; i < nb_trees; i++) {
    std::unique_ptr<Tree> tree(new Tree);
    bytes_read += tree->read(in);
    trees.push_back(std::move(tree));
  }
  return bytes_read;
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
