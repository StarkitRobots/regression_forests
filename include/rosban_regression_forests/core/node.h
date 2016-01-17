#pragma once

#include "rosban_regression_forests/approximations/approximation.h"
#include "rosban_regression_forests/core/orthogonal_split.h"

namespace regression_forests
{
class Node
{
public:
  Approximation *a;
  Node *father, *upperChild, *lowerChild;
  OrthogonalSplit s;

  /**
   * Regression Node takes in charge the destruction of both childs and of
   * the approximation function but not of the father
   */
  Node();
  Node(const Node &other) = delete;
  Node(Node *father);
  Node(Node *father, Approximation *a);
  virtual ~Node();

  size_t maxSplitDim() const;

  size_t nbLeafs() const;
  bool isLeaf() const;
  const Node *getLeaf(const Eigen::VectorXd &state) const;

  // Return the intersection between provided space and space associated to
  // the node by iterating on all the father
  Eigen::MatrixXd getSubSpace(const Eigen::MatrixXd &space) const;

  // Add a cloned version of the approximation on all the leafs with a given
  // weight
  void addApproximation(const Approximation *a, double weight);

  double getValue(const Eigen::VectorXd &state) const;

  /**
   * Return max over all leafs
   */
  double getMax(Eigen::MatrixXd &limits) const;
  Eigen::VectorXd getArgMax(Eigen::MatrixXd &limits) const;

  std::pair<double, Eigen::VectorXd> getMaxPair(Eigen::MatrixXd &limits) const;
  void updateMaxPair(Eigen::MatrixXd &limits, std::pair<double, Eigen::VectorXd> &best) const;

  /**
   * Return a vector of size 2^|freeDimensions| containing the 'corners' of
   * the freeDimensions inside limits.
   * Should only be called on root (recursion is not handled here)
   * It returns a Vector v of size |freeDimensions| + 1 where
   * - v(i) is the value of the dimension freeDimensions[i] for all i <
   * |freeDimensions|
   * - v(|freeDimensions|) is the evaluation of the value at this point
   */
  std::vector<Eigen::VectorXd> project(const std::vector<int> &freeDimensions, const Eigen::MatrixXd &limits);

  /**
   * Return a deepCopy of the current regressionNode.
   * The link to the father is set to NULL.
   */
  virtual Node *clone() const;

  /**
   * Copy the content of the node 'other' inside the current node
   * default is action copy and split copy, behavior might be extended
   * for child classes
   */
  virtual void copyContent(const Node *other);

  /**
   * Return an empty node of the same class as the parameter
   */
  virtual Node *softClone() const;

  /**
   * Return a deepCopy of the subTree rooted on the given node.
   * The deepCopy will automatically remove the split which do not belong
   * to the limits given as parameter.
   */
  Node *subTreeCopy(const Eigen::MatrixXd &limits) const;

  /**
   * Merge parallely t1 and t2 at n, using given weights and limits.
   * limits are modified during the process but its final content is
   * the same as the original
   */
  static void parallelMerge(Node &node, const Node &t1, const Node &t2, double w1,
                            double w2, Eigen::MatrixXd &limits);
};
}

std::ostream &operator<<(std::ostream &out, const regression_forests::Node &node);
