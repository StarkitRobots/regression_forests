#pragma once

#include <vector>

namespace regression_forests
{
namespace Statistics
{
double mean(const std::vector<double>& vec);
double median(const std::vector<double>& vec);
double stdDev(const std::vector<double>& vec);
double variance(const std::vector<double>& vec);

// result: {1stQuartile, median, 3rd Quartile}
// The implemented method follows the method 3 at
// - https://en.wikipedia.org/wiki/Quartile
// There is no universal agreement on the computation of the quartiles
std::vector<double> getQuartiles(const std::vector<double>& vec);
}  // namespace Statistics
}  // namespace regression_forests
