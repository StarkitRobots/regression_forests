#pragma once

#include <string>
#include <ostream>

namespace regression_forests
{
enum ApproximationType
{
  PWC,
  PWL
};

ApproximationType loadApproximationType(const std::string &s);
std::string to_string(Math::RegressionTree::ApproximationType at);
}
}
