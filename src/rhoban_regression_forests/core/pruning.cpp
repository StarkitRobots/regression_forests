#include "rhoban_regression_forests/core/pruning.h"

#include "rhoban_regression_forests/approximations/composite_approximation.h"

#include <iostream>
#include <list>
#include <map>

namespace regression_forests
{
typedef std::pair<Node *, double> EvaluatedNode;
typedef std::pair<std::shared_ptr<const Approximation>, double> EvaluatedApproximation;

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

bool isLastSplit(Node *node)
{
  return node->lowerChild->isLeaf() && node->upperChild->isLeaf();
}

void pushLastSplitNodes(Node *node, std::list<Node *> &splitNodes)
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

EvaluatedApproximation getSplitData(Node *node, const Eigen::MatrixXd &limits)
{
  EvaluatedApproximation result;
  Eigen::MatrixXd nodeSpace = node->getSubSpace(limits);
  size_t sDim = node->s.dim;
  double sVal = node->s.val;
  double nodeSize = spaceSize(limits);
  double sDimSize = nodeSpace(sDim, 1) - nodeSpace(sDim, 0);
  double lowerRatio = (sVal - nodeSpace(sDim, 0)) / sDimSize;
  double upperRatio = (nodeSpace(sDim, 1) - sVal) / sDimSize;
  result.first = CompositeApproximation::weightedMerge(node->lowerChild->a->clone(), lowerRatio,
                                                       node->upperChild->a->clone(), upperRatio);
  double lowerSize = upperRatio * nodeSize;
  double upperSize = upperRatio * nodeSize;
  double upperDiff = CompositeApproximation::avgDifference(node->upperChild->a, result.first,
                                                           nodeSpace);
  double lowerDiff = CompositeApproximation::avgDifference(node->lowerChild->a, result.first,
                                                           nodeSpace);
  result.second = lowerSize * lowerDiff + upperSize * upperDiff;
  return result;
}

std::unique_ptr<Tree> pruneTree(std::unique_ptr<Tree> tree, const Eigen::MatrixXd &limits,
                                          size_t maxLeafs)
{
  // 1. count leafs and add preLeafs
  size_t nbLeafs = tree->nbLeafs();
  if (nbLeafs <= maxLeafs)
  {
    return tree;
  }
  std::list<Node *> splitNodes;
  pushLastSplitNodes(tree->root, splitNodes);
  auto nodeComp = [](const EvaluatedNode &a, const EvaluatedNode &b)
  {
    if (a.second == b.second)
    {
      return a.first < b.first;
    }
    return a.second < b.second;
  };
  std::map<EvaluatedNode, std::shared_ptr<const Approximation>, decltype(nodeComp)> splits(nodeComp);
  for (Node *node : splitNodes)
  {
    auto splitData = getSplitData(node, limits);
    EvaluatedNode key(node, splitData.second);
    splits[key] = splitData.first;
  }
  // 3. While not enough leafs have been removed, remove worst leaf
  while (nbLeafs > maxLeafs)
  {
    // Retrieving node which bring the lowest quality improvement
    Node *current = splits.begin()->first.first;
    std::shared_ptr<const Approximation> app = splits.begin()->second;
    auto second = ++splits.begin();
    splits.erase(splits.begin(), second);
    // Destroy children effectively
    current->a = app;
    delete (current->lowerChild);
    delete (current->upperChild);
    current->lowerChild = NULL;
    current->upperChild = NULL;
    nbLeafs--;
    // If father is now a lastSplit, add it to the splitNodes
    Node *father = current->father;
    if (father != NULL && isLastSplit(father))
    {
      auto splitData = getSplitData(father, limits);
      EvaluatedNode key(father, splitData.second);
      splits[key] = splitData.first;
    }
  }
  return tree;
}
}
