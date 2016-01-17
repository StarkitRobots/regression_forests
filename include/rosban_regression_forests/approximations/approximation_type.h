#pragma once

#include <string>
#include <ostream>

namespace Math {
  namespace RegressionTree {

    enum ApproximationType{
      PWC, PWL
    };

    ApproximationType loadApproximationType(const std::string& s);
    std::string approximationType2String(Math::RegressionTree::ApproximationType at);
  }
}
