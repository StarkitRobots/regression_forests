#include "rosban_regression_forests/core/regression_tree.h"

namespace Math
{
namespace RegressionTree
{
std::unique_ptr<RegressionTree> pruneTree(std::unique_ptr<RegressionTree> tree, const Eigen::MatrixXd &limits,
                                          size_t maxNode);
}
}
