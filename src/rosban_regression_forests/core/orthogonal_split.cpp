#include "OrthogonalSplit.hpp"

namespace Math {
  namespace RegressionTree {

    OrthogonalSplit::OrthogonalSplit()
      : dim(-1), val(0)
    {
    }

    OrthogonalSplit::OrthogonalSplit(int dim_, double val_)
      : dim(dim_), val(val_)
    {
    }

    bool OrthogonalSplit::isLower(const Eigen::VectorXd& input) const
    {
      return input(dim) <= val;
    }
  }
}

std::ostream& operator<<(std::ostream& out,
                         const Math::RegressionTree::OrthogonalSplit& split)
{
  return out << "sd" << split.dim << "$v" << split.val << "$$";
}
