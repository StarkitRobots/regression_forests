#include "RegressionTree.hpp"

namespace Math {
  namespace RegressionTree {

    std::unique_ptr<RegressionTree>
    pruneTree(std::unique_ptr<RegressionTree> tree,
              const Eigen::MatrixXd& limits,
              size_t maxNode);
  }
}
