#pragma once

#include <string>
#include <ostream>

namespace Math
{
namespace RegressionTree
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
