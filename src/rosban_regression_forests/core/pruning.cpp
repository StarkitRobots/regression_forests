#include "rosban_regression_forests/core/pruning.h"

#include "rosban_regression_forests/approximations/composite_approximation.h"

#include <iostream>
#include <list>
#include <map>

namespace regression_forests
{
typedef std::pair<RegressionNode *, double> EvaluatedNode;

static double spaceSize(const Eigen::MatrixXd &space)
{
  if (space.cols() != 2)
  {
    throw std::runtime_error("Asking for space size in a matrix which has not 2 columns");
  }
  double size = 1.0;
  for (int r = 0; r < space.rows(); r++)
  {
    size = size * (space(r, 1) - space(r, 0));
  }
  return size;
}

bool isLastSplit(RegressionNode *node)
{
  return node->lowerChild->isLeaf() && node->upperChild->isLeaf();
}

void pushLastSplitNodes(RegressionNode *node, std::list<RegressionNode *> &splitNodes)
{
  if (node->isLeaf())
  {
    return;
  }
  // If both childs of the nodes are leaf, we are on a lastSplit, add it
  if (isLastSplit(node))
  {
    splitNodes.push_back(node);
  }
  // Add lastSplit from childs
  else
  {
    pushLastSplitNodes(node->lowerChild, splitNodes);
    pushLastSplitNodes(node->upperChild, splitNodes);
  }
}

std::pair<Approximation *, double> getSplitData(RegressionNode *node, const Eigen::MatrixXd &limits)
{
  std::pair<Approximation *, double> result;
  Eigen::MatrixXd nodeSpace = node->getSubSpace(limits);
  size_t sDim = node->s.dim;
  double sVal = node->s.val;
  double nodeSize = spaceSize(limits);
  double sDimSize = nodeSpace(sDim, 1) - nodeSpace(sDim, 0);
  double lowerRatio = (sVal - nodeSpace(sDim, 0)) / sDimSize;
  double upperRatio = (nodeSpace(sDim, 1) - sVal) / sDimSize;
  ;
  result.first = CompositeApproximation::weightedMerge(node->lowerChild->a->clone(), lowerRatio,
                                                       node->upperChild->a->clone(), upperRatio);
  double lowerSize = upperRatio * nodeSize;
  double upperSize = upperRatio * nodeSize;
  double upperDiff = std::fabs(CompositeApproximation::difference(node->upperChild->a, result.first));
  double lowerDiff = std::fabs(CompositeApproximation::difference(node->upperChild->a, result.first));
  result.second = lowerSize * lowerDiff + upperSize * upperDiff;
  return result;
}

std::unique_ptr<RegressionTree> pruneTree(std::unique_ptr<RegressionTree> tree, const Eigen::MatrixXd &limits,
                                          size_t maxLeafs)
{
  // 1. count leafs and add preLeafs
  size_t nbLeafs = tree->nbLeafs();
  if (nbLeafs <= maxLeafs)
  {
    return tree;
  }
  std::list<RegressionNode *> splitNodes;
  pushLastSplitNodes(tree->root, splitNodes);
  auto nodeComp = [](const EvaluatedNode &a, const EvaluatedNode &b)
  {
    if (a.second == b.second)
    {
      return a.first < b.first;
    }
    return a.second < b.second;
  };
  std::map<EvaluatedNode, Approximation *, decltype(nodeComp)> splits(nodeComp);
  for (RegressionNode *node : splitNodes)
  {
    auto splitData = getSplitData(node, limits);
    EvaluatedNode key(node, splitData.second);
    splits[key] = splitData.first;
  }
  // 3. While not enough leafs have been removed, remove worst leaf
  while (nbLeafs > maxLeafs)
  {
    // Retrieving node which bring the lowest quality improvement
    RegressionNode *current = splits.begin()->first.first;
    Approximation *app = splits.begin()->second;
    auto second = ++splits.begin();
    splits.erase(splits.begin(), second);
    // Split node effectively
    if (current->a != NULL)
    {
      delete (current->a);
    }
    current->a = app;
    delete (current->lowerChild);
    delete (current->upperChild);
    current->lowerChild = NULL;
    current->upperChild = NULL;
    nbLeafs--;
    // If father is now a lastSplit, add it to the splitNodes
    RegressionNode *father = current->father;
    if (father != NULL && isLastSplit(father))
    {
      auto splitData = getSplitData(father, limits);
      EvaluatedNode key(father, splitData.second);
      splits[key] = splitData.first;
    }
  }
  for (auto &entry : splits)
  {
    delete (entry.second);
  }
  return tree;
}
}
