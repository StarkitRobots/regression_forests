#include "rosban_regression_forests/core/regression_node.h"

#include "rosban_regression_forests/approximations/composite_approximation.h"

namespace Math
{
namespace RegressionTree
{
RegressionNode::RegressionNode() : a(NULL), father(NULL), upperChild(NULL), lowerChild(NULL)
{
}

RegressionNode::RegressionNode(RegressionNode *father_) : a(NULL), father(father_), upperChild(NULL), lowerChild(NULL)
{
}

RegressionNode::RegressionNode(RegressionNode *father_, Approximation *a_)
  : a(a_), father(father_), upperChild(NULL), lowerChild(NULL)
{
}

RegressionNode::~RegressionNode()
{
  if (upperChild != NULL)
  {
    delete (upperChild);
  }
  if (lowerChild != NULL)
  {
    delete (lowerChild);
  }
  if (a != NULL)
  {
    delete (a);
  }
}

size_t RegressionNode::maxSplitDim() const
{
  if (isLeaf())
    return 0;
  int childMaxDim = std::max(lowerChild->maxSplitDim(), upperChild->maxSplitDim());
  return std::max(s.dim, childMaxDim);
}

size_t RegressionNode::nbLeafs() const
{
  if (isLeaf())
  {
    return 1;
  }
  return lowerChild->nbLeafs() + upperChild->nbLeafs();
}

bool RegressionNode::isLeaf() const
{
  return (lowerChild == NULL && upperChild == NULL);
}

const RegressionNode *RegressionNode::getLeaf(const Eigen::VectorXd &state) const
{
  if (isLeaf())
    return this;
  if (s.isLower(state))
  {
    return lowerChild->getLeaf(state);
  }
  return upperChild->getLeaf(state);
}

Eigen::MatrixXd RegressionNode::getSubSpace(const Eigen::MatrixXd &space) const
{
  Eigen::MatrixXd subSpace = space;
  const RegressionNode *current = this;
  while (current->father != NULL)
  {
    size_t sDim = current->father->s.dim;
    double sVal = current->father->s.val;
    // If current node is upperChild
    if (current->father->upperChild == current)
    {
      if (subSpace(sDim, 0) < sVal)
      {
        subSpace(sDim, 0) = sVal;
      }
    }
    // If current node is lowerChild
    else if (current->father->lowerChild == current)
    {
      if (subSpace(sDim, 1) > sVal)
      {
        subSpace(sDim, 1) = sVal;
      }
    }
    else
    {
      throw std::runtime_error("Inconsistency detected in RegressionNode");
    }
    // Jump to father
    current = current->father;
  }
  return subSpace;
}

void RegressionNode::addApproximation(const Approximation *newApproximation, double newWeight)
{
  if (!isLeaf())
  {
    lowerChild->addApproximation(newApproximation, newWeight);
    upperChild->addApproximation(newApproximation, newWeight);
  }
  if (a == NULL)
  {
    a = newApproximation->clone();
  }
  else
  {
    a = CompositeApproximation::weightedMerge(a, 1, newApproximation->clone(), newWeight);
  }
}

double RegressionNode::getValue(const Eigen::VectorXd &state) const
{
  if (!isLeaf())
  {
    const RegressionNode *leaf = getLeaf(state);
    return leaf->getValue(state);
  }
  return a->eval(state);
}

double RegressionNode::getMax(Eigen::MatrixXd &limits) const
{
  std::pair<double, Eigen::VectorXd> best;
  best.first = std::numeric_limits<double>::lowest();
  return getMaxPair(limits).first;
}

Eigen::VectorXd RegressionNode::getArgMax(Eigen::MatrixXd &limits) const
{
  return getMaxPair(limits).second;
}

std::pair<double, Eigen::VectorXd> RegressionNode::getMaxPair(Eigen::MatrixXd &limits) const
{
  std::pair<double, Eigen::VectorXd> best;
  best.first = std::numeric_limits<double>::lowest();
  best.second = Eigen::VectorXd(limits.rows());
  updateMaxPair(limits, best);
  return best;
}

void RegressionNode::updateMaxPair(Eigen::MatrixXd &limits, std::pair<double, Eigen::VectorXd> &best) const
{
  if (isLeaf())
  {
    return a->updateMaxPair(limits, best);
  }
  double oldMin = limits(s.dim, 0);
  double oldMax = limits(s.dim, 1);
  // If split is above limits min, search in lower child
  if (oldMin <= s.val)
  {
    limits(s.dim, 1) = s.val;
    lowerChild->updateMaxPair(limits, best);
    limits(s.dim, 1) = oldMax;
  }
  // If split is above limits min, search in lower child
  if (oldMax > s.val)
  {
    limits(s.dim, 0) = s.val;
    upperChild->updateMaxPair(limits, best);
    limits(s.dim, 0) = oldMin;
  }
}

std::vector<Eigen::VectorXd> RegressionNode::project(const std::vector<int> &freeDimensions,
                                                     const Eigen::MatrixXd &limits)
{
  if (a == NULL)
  {
    throw std::runtime_error("RegressionNode::project: no approximation "
                             "available on the current node");
  }
  int D = freeDimensions.size();
  int nbCorners = std::pow(2, D);
  std::vector<Eigen::VectorXd> result;
  result.reserve(nbCorners);
  Eigen::VectorXd min = limits.block(0, 0, limits.rows(), 1);
  Eigen::VectorXd max = limits.block(0, 1, limits.rows(), 1);
  Eigen::VectorXd center = (min + max) / 2;
  for (int i = 0; i < nbCorners; i++)
  {
    Eigen::VectorXd corner(D + 1);
    Eigen::VectorXd cornerState = center;
    for (int d = 0; d < D; d++)
    {
      bool lower = (i / (int)std::pow(2, d)) % 2 == 0;
      double val = limits(freeDimensions[d], lower ? 0 : 1);
      corner(d) = val;
      cornerState(freeDimensions[d]) = val;
    }
    corner(D) = a->eval(cornerState);
    result.push_back(corner);
  }
  return result;
}

RegressionNode *RegressionNode::clone() const
{
  RegressionNode *copy = softClone();
  copy->copyContent(this);
  if (lowerChild != NULL)
  {
    copy->lowerChild = lowerChild->clone();
    copy->lowerChild->father = copy;
  }
  if (upperChild != NULL)
  {
    copy->upperChild = upperChild->clone();
    copy->upperChild->father = copy;
  }
  return copy;
}

void RegressionNode::copyContent(const RegressionNode *other)
{
  if (a != NULL)
  {
    delete (a);
  }
  if (other->a != NULL)
  {
    a = other->a->clone();
  }
  else
  {
    a = NULL;
  }
  s = other->s;
}

RegressionNode *RegressionNode::softClone() const
{
  return new RegressionNode();
}

RegressionNode *RegressionNode::subTreeCopy(const Eigen::MatrixXd &limits) const
{
  // End of recursion on leaf
  if (isLeaf())
  {
    return clone();
  }
  RegressionNode *result = NULL;
  // If splitVal is above limits return a copy of the lower child
  if (s.val > limits(s.dim, 1))
  {
    result = lowerChild->subTreeCopy(limits);
  }
  // If splitVal is strictly under limits return a copy of the upper child
  else if (s.val <= limits(s.dim, 0))
  {
    result = upperChild->subTreeCopy(limits);
  }
  else
  {
    result = softClone();
    result->copyContent(this);
    result->lowerChild = lowerChild->subTreeCopy(limits);
    result->lowerChild->father = result;
    result->upperChild = upperChild->subTreeCopy(limits);
    result->upperChild->father = result;
  }
  return result;
}

void RegressionNode::parallelMerge(RegressionNode &node, const RegressionNode &t1, const RegressionNode &t2, double w1,
                                   double w2, Eigen::MatrixXd &limits)
{
  // 1. T1 is a leaf
  if (t1.isLeaf())
  {
    // 1.1 Both nodes to merge are leafs, just merge their approximations
    if (t2.isLeaf())
    {
      node.a = CompositeApproximation::weightedMerge(t1.a->clone(), w1, t2.a->clone(), w2);
    }
    // 1.2 Only t1 is a leaf, merge the subtree t2 with t1 approximation
    else
    {
      parallelMerge(node, t2, t1, w2, w1, limits);
    }
  }
  // 2. T1 is a split node
  else
  {
    size_t sDim = t1.s.dim;
    double sVal = t1.s.val;
    double min = limits(sDim, 0);
    double max = limits(sDim, 1);
    // 2.1 limits forces to choose the lower child (max is higher or equal to
    // splitVal)
    if (max <= sVal)
    {
      parallelMerge(node, t2, *t1.lowerChild, w2, w1, limits);
    }
    // 2.2 limits forces to choose the upper child (min is strictly lower than
    // splitVal)
    else if (min > sVal)
    {
      parallelMerge(node, t2, *t1.upperChild, w2, w1, limits);
    }
    // 2.3 sVal is inside limits, t2 has to be merged on both childs
    else
    {
      // Setting split
      node.s = t1.s;
      // Handling lowerChild
      node.lowerChild = new RegressionNode(&node);
      limits(sDim, 1) = sVal;
      parallelMerge(*node.lowerChild, t2, *t1.lowerChild, w2, w1, limits);
      limits(sDim, 1) = max;  // Restore limit
      // Handling upperChild
      node.upperChild = new RegressionNode(&node);
      limits(sDim, 0) = sVal;
      parallelMerge(*node.upperChild, t2, *t1.upperChild, w2, w1, limits);
      limits(sDim, 0) = min;  // Restore limit
    }
  }
}
}
}

std::ostream &operator<<(std::ostream &out, const Math::RegressionTree::RegressionNode &node)
{
  out << "n";
  if (node.isLeaf())
  {
    out << *node.a;
  }
  else
  {
    out << node.s << *node.lowerChild << *node.upperChild;
  }
  out << "$";
  return out;
}
