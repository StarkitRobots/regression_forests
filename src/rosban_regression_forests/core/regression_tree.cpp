#include "RegressionTree.hpp"

#include <limits>
#include <iostream>

namespace Math {
  namespace RegressionTree {

    RegressionTree::RegressionTree()
      : root(NULL)
    {
    }

    RegressionTree::~RegressionTree()
    {
      if (root != NULL) {
        delete(root);
      }
    }

    size_t RegressionTree::nbLeafs() const
    {
      return root->nbLeafs();
    }

    size_t RegressionTree::maxSplitDim() const
    {
      return root->maxSplitDim();
    }

    double RegressionTree::getValue(const Eigen::VectorXd& input) const
    {
      return root->getValue(input);
    }

    double RegressionTree::getMax(const Eigen::MatrixXd& limits) const
    {
      return getMaxPair(limits).first;
    }

    Eigen::VectorXd RegressionTree::getArgMax(const Eigen::MatrixXd& limits) const
    {
      return getMaxPair(limits).second;
    }

    std::pair<double, Eigen::VectorXd> 
    RegressionTree::getMaxPair(const Eigen::MatrixXd& limits) const
    {
      Eigen::MatrixXd localLimits = limits;
      return root->getMaxPair(localLimits);
    }

    void RegressionTree::fillProjection(std::vector<std::vector<Eigen::VectorXd>>& out,
                                        RegressionNode * currentNode,
                                        const std::vector<int>& freeDimensions,
                                        Eigen::MatrixXd& limits)
    {
      // If leaf, fill vector and return
      if (currentNode->isLeaf()) {
        out.push_back(currentNode->project(freeDimensions, limits));
        return;
      }
      double splitDim = currentNode->s.dim;
      double splitVal = currentNode->s.val;
      double oldMin = limits(splitDim, 0);
      double oldMax = limits(splitDim, 1);
      // If lowerChild shares a common space with limits
      if (oldMin <=  splitVal) {
        // set a new limit if necessary
        limits(splitDim, 1) = std::min(splitVal, oldMax);
        // Explore subtree
        fillProjection(out, currentNode->lowerChild, freeDimensions, limits);
        // Restor old limit
        limits(splitDim, 1) = oldMax;
      }
      // If upperChild shares a common space with limits
      if (oldMax >  splitVal) {
        // set a new limit if necessary
        limits(splitDim, 0) = std::max(splitVal, oldMin);
        // Explore subtree
        fillProjection(out, currentNode->upperChild, freeDimensions, limits);
        // Restor old limit
        limits(splitDim, 0) = oldMin;
      }
    }

    std::vector<std::vector<Eigen::VectorXd>>
    RegressionTree::project(const std::vector<int>& freeDimensions,
                            const Eigen::MatrixXd& limits)
    {
      std::vector<std::vector<Eigen::VectorXd>> out;
      Eigen::MatrixXd localLimits = limits;// Copying to allow modification
      fillProjection(out, root, freeDimensions, localLimits);
      return out;
    }

    void RegressionTree::addSubTree(RegressionNode * node,
                                    Eigen::MatrixXd& limits,
                                    const RegressionTree& other,
                                    double otherWeight)
    {
      if (otherWeight == 0) {
        return;
      }
      if (node->isLeaf()) {
        // Deep copy of the 'other' subTree corresponding to limits
        RegressionNode * subTreeRoot = other.root->subTreeCopy(limits);
        // If the subTree is void, then nothing needs to be done
        if (subTreeRoot == NULL) { return; }
        // Add current approximation to all leafs of the subTree and then delete it
        if (node->a != NULL) {
          subTreeRoot->addApproximation(node->a, 1.0 / otherWeight);
          delete(node->a);
        }
        // Import approximation and childs with subTreeRoot
        node->a = subTreeRoot->a;
        node->s = subTreeRoot->s;
        node->lowerChild = subTreeRoot->lowerChild;
        node->upperChild = subTreeRoot->upperChild;
        // Replace the father of subTreeRoot childs if necessary
        if (!node->isLeaf()) {
          node->lowerChild->father = node;
          node->upperChild->father = node;
        }
        // Set pointers to NULL and delete subTreeRoot
        subTreeRoot->a = NULL;
        subTreeRoot->lowerChild = NULL;
        subTreeRoot->upperChild = NULL;
        delete(subTreeRoot);
        return;
      }
      double sDim = node->s.dim;
      double sVal = node->s.val;
      double oldMin = limits(sDim, 0);
      double oldMax = limits(sDim, 1);
      // Add subTree to lowerchild (if lowerchild is in limits)
      if (limits(sDim, 0) <= sVal) {
        limits(sDim, 1) = sVal;
        addSubTree(node->lowerChild, limits, other, otherWeight);
        limits(sDim, 1) = oldMax;
      }
      // Add subTree to upperchild (if upperchild is in limits)
      if(limits(sDim, 1) > sVal) {
        limits(sDim, 0) = sVal;
        addSubTree(node->upperChild, limits, other, otherWeight);
        limits(sDim, 0) = oldMin;
      }
    }

    void RegressionTree::avgTree(const RegressionTree& other, double otherWeight)
    {
      size_t dim = 1 + std::max(maxSplitDim(), other.maxSplitDim());
      Eigen::MatrixXd limits(dim,2);
      for (size_t d = 0; d < dim; d++) {
        limits(d,0) = std::numeric_limits<double>::lowest();
        limits(d,1) = std::numeric_limits<double>::max();
      }
      avgTree(other, otherWeight, limits);
    }

    void RegressionTree::avgTree(const RegressionTree& other, double otherWeight,
                                 const Eigen::MatrixXd& limits)
    {
      Eigen::MatrixXd localLimits = limits;
      addSubTree(root, localLimits, other, otherWeight);
    }
     
    std::unique_ptr<RegressionTree>
    RegressionTree::avgTrees(const RegressionTree& t1,
                             const RegressionTree& t2,
                             double w1,
                             double w2,
                             const Eigen::MatrixXd& limits)
    {
      Eigen::MatrixXd localLimits = limits;
      std::unique_ptr<RegressionTree> result(new RegressionTree());
      result->root = new RegressionNode();
      RegressionNode::parallelMerge(*result->root,
                                    *t1.root, *t2.root, w1, w2,
                                    localLimits);
      return result;
    }

    std::unique_ptr<RegressionTree>
    RegressionTree::project(const Eigen::MatrixXd& limits) const
    {
      std::unique_ptr<RegressionTree> t(new RegressionTree);
      t->root = new RegressionNode(NULL, NULL);
      Eigen::MatrixXd localLimits = limits;
      t->addSubTree(t->root, localLimits, *this, 1.0);
      return t;
    }

  }
}

std::ostream& operator<<(std::ostream& out,
                         const Math::RegressionTree::RegressionTree& tree)
{
  if (tree.root != NULL) {
    return out << *tree.root;
  }
  return out;
}
