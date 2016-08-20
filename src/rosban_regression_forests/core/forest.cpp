#include "rosban_regression_forests/core/forest.h"

#include "rosban_regression_forests/core/pruning.h"

#include "rosban_utils/io_tools.h"

#include <algorithm>
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

double Forest::getValue(const Eigen::VectorXd &input,
                        AggregationMethod method) const
{
  double sum = 0;
  std::vector<double> values = getValues(input, method);
  for (double val : values)
  {
    sum += val;
  }
  return sum / values.size();
}

std::vector<double> Forest::getValues(const Eigen::VectorXd &input,
                                      AggregationMethod method) const
{
  std::vector<double> raw_values;
  raw_values.reserve(trees.size());
  for (const auto &t : trees)
  {
    raw_values.push_back(t->getValue(input));
  }
  switch (method)
  {
    case AggregationMethod::All: return raw_values;
    case AggregationMethod::AutoCut:
    {
      int tail_size = std::ceil(trees.size() * 0.1);
      std::sort(raw_values.begin(), raw_values.end());
      std::vector<double> filtered_values;
      filtered_values.reserve(trees.size() - 2 * tail_size);
      for (int i = tail_size; i < raw_values.size() - tail_size; i++) {
        filtered_values.push_back(raw_values[i]);
      }
      return filtered_values;
    }
    default:
      throw std::logic_error("Forest::getAggregatedValues: unknown aggregation method");
  }
}


double Forest::getVar(const Eigen::VectorXd &input,
                      AggregationMethod method) const
{
  double sum = 0;
  std::vector<double> values = getValues(input, method);
  for (double val : values)
  {
    sum += val;
  }
  double mean = sum / values.size();
  double var = 0;
  for (double val : values)
  {
    double diff = val - mean;
    var += diff * diff;
  }
  return var / values.size();
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

double Forest::getRandomizedValue(const Eigen::VectorXd &input,
                                  std::default_random_engine &engine) const
{
  double mean = getValue(input);
  double std_dev = std::sqrt(getVar(input));
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

int Forest::getClassID() const { return 0; }

int Forest::writeInternal(std::ostream & out) const
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

Forest * Forest::clone() const
{
  Forest * copy = new Forest();
  for (int i = 0; i < trees.size(); i++) {
    std::unique_ptr<Tree> tree_copy(trees[i]->clone());
    copy->push(std::move(tree_copy));
  }
  return copy;
}

Forest::AggregationMethod loadAggregationMethod(const std::string & str)
{
  if (str == "All") return Forest::AggregationMethod::All;
  if (str == "AutoCut") return Forest::AggregationMethod::AutoCut;
  std::ostringstream oss;
  oss << "In loadAggregationMethod: unknown aggregation method: " << str;
  throw std::runtime_error(oss.str());
}

std::string aggregationMethod2Str(Forest::AggregationMethod method)
{
  switch(method)
  {
    case Forest::AggregationMethod::All: return "All";
    case Forest::AggregationMethod::AutoCut: return "AutoCut";
    default:
    {
      std::ostringstream oss;
      oss << "In aggregationMethod2Str: unknown aggregation method: " << method;
      throw std::runtime_error(oss.str());
    }
  }
}

}
