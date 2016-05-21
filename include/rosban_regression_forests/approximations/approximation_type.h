#pragma once

#include <string>
#include <ostream>

namespace regression_forests
{
enum class ApproximationType
{
  PWC, PWL, GP
};

ApproximationType loadApproximationType(const std::string &s);
std::string to_string(ApproximationType at);
}
