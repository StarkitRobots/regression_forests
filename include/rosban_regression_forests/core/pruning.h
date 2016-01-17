#include "rosban_regression_forests/core/regression_tree.h"

namespace regression_forests
{
std::unique_ptr<Tree> pruneTree(std::unique_ptr<Tree> tree,
                                          const Eigen::MatrixXd &limits,
                                          size_t maxNode);
}
