#include "rhoban_regression_forests/core/node.h"

#include "rhoban_regression_forests/approximations/approximation_factory.h"
#include "rhoban_regression_forests/approximations/composite_approximation.h"

#include "rhoban_utils/io_tools.h"

static regression_forests::ApproximationFactory approximation_factory;

namespace regression_forests
{
Node::Node() : father(NULL), upperChild(NULL), lowerChild(NULL)
{
}

Node::Node(Node* father_) : father(father_), upperChild(NULL), lowerChild(NULL)
{
}

Node::Node(Node* father_, std::shared_ptr<const Approximation> a_)
  : a(a_), father(father_), upperChild(NULL), lowerChild(NULL)
{
}

Node::~Node()
{
  if (upperChild != NULL)
  {
    delete (upperChild);
  }
  if (lowerChild != NULL)
  {
    delete (lowerChild);
  }
}

size_t Node::maxSplitDim() const
{
  if (isLeaf())
    return 0;
  int childMaxDim = std::max(lowerChild->maxSplitDim(), upperChild->maxSplitDim());
  return std::max(s.dim, childMaxDim);
}

size_t Node::nbLeafs() const
{
  if (isLeaf())
  {
    return 1;
  }
  return lowerChild->nbLeafs() + upperChild->nbLeafs();
}

bool Node::isLeaf() const
{
  return (lowerChild == NULL && upperChild == NULL);
}

const Node* Node::getLeaf(const Eigen::VectorXd& state) const
{
  if (isLeaf())
    return this;
  if (s.isLower(state))
  {
    return lowerChild->getLeaf(state);
  }
  return upperChild->getLeaf(state);
}

Eigen::MatrixXd Node::getSubSpace(const Eigen::MatrixXd& space) const
{
  Eigen::MatrixXd subSpace = space;
  const Node* current = this;
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
      throw std::runtime_error("Inconsistency detected in Node");
    }
    // Jump to father
    current = current->father;
  }
  return subSpace;
}

void Node::addApproximation(std::shared_ptr<const Approximation> newApproximation, double newWeight)
{
  if (!isLeaf())
  {
    lowerChild->addApproximation(newApproximation, newWeight);
    upperChild->addApproximation(newApproximation, newWeight);
  }
  if (!a)
  {
    a = newApproximation;
  }
  else
  {
    a = CompositeApproximation::weightedMerge(a, 1, newApproximation, newWeight);
  }
}

double Node::getValue(const Eigen::VectorXd& state) const
{
  if (!isLeaf())
  {
    const Node* leaf = getLeaf(state);
    return leaf->getValue(state);
  }
  return a->eval(state);
}

Eigen::VectorXd Node::getGrad(const Eigen::VectorXd& input) const
{
  if (!isLeaf())
  {
    const Node* leaf = getLeaf(input);
    return leaf->getGrad(input);
  }
  return a->getGrad(input);
}

double Node::getMax(Eigen::MatrixXd& limits) const
{
  std::pair<double, Eigen::VectorXd> best;
  best.first = std::numeric_limits<double>::lowest();
  return getMaxPair(limits).first;
}

Eigen::VectorXd Node::getArgMax(Eigen::MatrixXd& limits) const
{
  return getMaxPair(limits).second;
}

std::pair<double, Eigen::VectorXd> Node::getMaxPair(Eigen::MatrixXd& limits) const
{
  std::pair<double, Eigen::VectorXd> best;
  best.first = std::numeric_limits<double>::lowest();
  best.second = Eigen::VectorXd(limits.rows());
  updateMaxPair(limits, best);
  return best;
}

void Node::updateMaxPair(Eigen::MatrixXd& limits, std::pair<double, Eigen::VectorXd>& best) const
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

std::pair<double, Eigen::VectorXd> Node::getMinPair(Eigen::MatrixXd& limits) const
{
  std::pair<double, Eigen::VectorXd> best;
  best.first = std::numeric_limits<double>::max();
  best.second = Eigen::VectorXd(limits.rows());
  updateMinPair(limits, best);
  return best;
}

void Node::updateMinPair(Eigen::MatrixXd& limits, std::pair<double, Eigen::VectorXd>& best) const
{
  if (isLeaf())
  {
    return a->updateMinPair(limits, best);
  }
  double oldMin = limits(s.dim, 0);
  double oldMax = limits(s.dim, 1);
  // If split is above limits min, search in lower child
  if (oldMin <= s.val)
  {
    limits(s.dim, 1) = s.val;
    lowerChild->updateMinPair(limits, best);
    limits(s.dim, 1) = oldMax;
  }
  // If split is below limits max, search in upper child
  if (oldMax > s.val)
  {
    limits(s.dim, 0) = s.val;
    upperChild->updateMinPair(limits, best);
    limits(s.dim, 0) = oldMin;
  }
}

std::vector<Eigen::VectorXd> Node::project(const std::vector<int>& freeDimensions, const Eigen::MatrixXd& limits)
{
  if (!a)
  {
    throw std::runtime_error("Node::project: no approximation "
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

void Node::apply(Eigen::MatrixXd& limits, Function f)
{
  // Start by applying function
  f(this, limits);
  // If we reached a leaf, return
  if (isLeaf())
    return;
  // Otherwise carry on to the childs
  double oldMin = limits(s.dim, 0);
  double oldMax = limits(s.dim, 1);
  // If split is above limits min, apply on lower child
  if (oldMin <= s.val)
  {
    limits(s.dim, 1) = s.val;
    lowerChild->apply(limits, f);
    limits(s.dim, 1) = oldMax;
  }
  // If split is below limits max, apply on upper child
  if (oldMax > s.val)
  {
    limits(s.dim, 0) = s.val;
    upperChild->apply(limits, f);
    limits(s.dim, 0) = oldMin;
  }
}

void Node::applyOnLeafs(Eigen::MatrixXd& limits, Function f)
{
  Function new_f = [f](Node* node, const Eigen::MatrixXd& limits) {
    if (node->isLeaf())
      f(node, limits);
  };
  apply(limits, new_f);
}

Node* Node::clone() const
{
  Node* copy = softClone();
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

void Node::copyContent(const Node* other)
{
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

Node* Node::softClone() const
{
  return new Node();
}

Node* Node::subTreeCopy(const Eigen::MatrixXd& limits) const
{
  // End of recursion on leaf
  if (isLeaf())
  {
    return clone();
  }
  Node* result = NULL;
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

void Node::parallelMerge(Node& node, const Node& t1, const Node& t2, double w1, double w2, Eigen::MatrixXd& limits)
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
      node.lowerChild = new Node(&node);
      limits(sDim, 1) = sVal;
      parallelMerge(*node.lowerChild, t2, *t1.lowerChild, w2, w1, limits);
      limits(sDim, 1) = max;  // Restore limit
      // Handling upperChild
      node.upperChild = new Node(&node);
      limits(sDim, 0) = sVal;
      parallelMerge(*node.upperChild, t2, *t1.upperChild, w2, w1, limits);
      limits(sDim, 0) = min;  // Restore limit
    }
  }
}

int Node::write(std::ostream& out) const
{
  int bytes_written = 0;
  if (isLeaf())
  {
    bytes_written += rhoban_utils::write<char>(out, 1);
    bytes_written += a->write(out);
  }
  else
  {
    bytes_written += rhoban_utils::write<char>(out, 0);
    bytes_written += rhoban_utils::write<int>(out, s.dim);
    bytes_written += rhoban_utils::write<double>(out, s.val);
    bytes_written += lowerChild->write(out);
    bytes_written += upperChild->write(out);
  }
  return bytes_written;
}

int Node::read(std::istream& in)
{
  // start by removing eventual subtree / approximation
  a.reset();
  if (upperChild != nullptr)
    delete (upperChild);
  if (lowerChild != nullptr)
    delete (lowerChild);
  // Then read data
  int bytes_read = 0;
  char is_leaf;
  bytes_read += rhoban_utils::read<char>(in, &is_leaf);
  if (is_leaf == 1)
  {
    std::unique_ptr<Approximation> approximation;
    bytes_read += approximation_factory.read(in, approximation);
    a = std::move(approximation);
  }
  else if (is_leaf == 0)
  {
    // Read split
    bytes_read += rhoban_utils::read<int>(in, &s.dim);
    bytes_read += rhoban_utils::read<double>(in, &s.val);
    // Read childs
    lowerChild = new Node();
    bytes_read += lowerChild->read(in);
    upperChild = new Node();
    bytes_read += upperChild->read(in);
  }
  else
  {
    throw std::runtime_error("Unexpected byte value when reading regression_forests::Node");
  }
  return bytes_read;
}

}  // namespace regression_forests
