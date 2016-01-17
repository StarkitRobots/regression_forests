#pragma once

#include "Approximation.hpp"
#include "OrthogonalSplit.hpp"

namespace Math {
  namespace RegressionTree {

    class RegressionNode {
    public:
      Approximation * a;
      RegressionNode *father, *upperChild, *lowerChild;
      OrthogonalSplit s;

      /**
       * Regression Node takes in charge the destruction of both childs and of
       * the approximation function but not of the father
       */
      RegressionNode();
      RegressionNode(const RegressionNode& other) = delete;
      RegressionNode(RegressionNode * father);
      RegressionNode(RegressionNode * father, Approximation * a);
      virtual ~RegressionNode();

      size_t maxSplitDim() const;

      size_t nbLeafs() const;
      bool isLeaf() const;
      const RegressionNode * getLeaf(const Eigen::VectorXd& state) const;

      // Return the intersection between provided space and space associated to
      // the node by iterating on all the father
      Eigen::MatrixXd getSubSpace(const Eigen::MatrixXd& space) const;

      // Add a cloned version of the approximation on all the leafs with a given weight
      void addApproximation(const Approximation * a, double weight);

      double getValue(const Eigen::VectorXd& state) const;

      /**
       * Return max over all leafs
       */
      double getMax(Eigen::MatrixXd& limits) const;      
      Eigen::VectorXd getArgMax(Eigen::MatrixXd& limits) const;
      
      std::pair<double, Eigen::VectorXd>
      getMaxPair(Eigen::MatrixXd& limits) const;
      void updateMaxPair(Eigen::MatrixXd& limits,
                         std::pair<double, Eigen::VectorXd>& best) const;

      /**
       * Return a vector of size 2^|freeDimensions| containing the 'corners' of
       * the freeDimensions inside limits.
       * Should only be called on root (recursion is not handled here)
       * It returns a Vector v of size |freeDimensions| + 1 where
       * - v(i) is the value of the dimension freeDimensions[i] for all i < |freeDimensions|
       * - v(|freeDimensions|) is the evaluation of the value at this point
       */
      std::vector<Eigen::VectorXd>
      project(const std::vector<int>& freeDimensions,
              const Eigen::MatrixXd& limits);

      /**
       * Return a deepCopy of the current regressionNode.
       * The link to the father is set to NULL.
       */
      virtual RegressionNode * clone() const;

      /**
       * Copy the content of the node 'other' inside the current node
       * default is action copy and split copy, behavior might be extended
       * for child classes
       */
      virtual void copyContent(const RegressionNode * other);

      /**
       * Return an empty node of the same class as the parameter
       */
      virtual RegressionNode * softClone() const;

      /**
       * Return a deepCopy of the subTree rooted on the given node.
       * The deepCopy will automatically remove the split which do not belong
       * to the limits given as parameter.
       */
      RegressionNode * subTreeCopy(const Eigen::MatrixXd& limits) const;

      /**
       * Merge parallely t1 and t2 at n, using given weights and limits.
       * limits are modified during the process but its final content is
       * the same as the original
       */
      static void parallelMerge(RegressionNode& node,
                                const RegressionNode& t1,
                                const RegressionNode& t2,
                                double w1,
                                double w2,
                                Eigen::MatrixXd& limits);
                                
    };
  }
}

std::ostream& operator<<(std::ostream& out,
                         const Math::RegressionTree::RegressionNode& node);
