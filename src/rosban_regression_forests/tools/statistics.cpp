#include "Basics.hpp"

#include <algorithm>
#include <stdexcept>

namespace Math {
  namespace Statistics {

    double mean(const std::vector<double> & vec)
    {
      if (vec.size() == 0)
        throw std::runtime_error("Impossible to compute mean from an empty set");
      double total = 0;
      for (double e : vec) {
        total += e;
      }
      return total / vec.size();
    }

    double median(const std::vector<double> & vec)
    {
      if (vec.size() == 0)
        throw std::runtime_error("Impossible to compute median from an empty set");
      std::vector<double> tmp = vec;
      std::sort(tmp.begin(), tmp.end());
      if (tmp.size() % 2 == 1) {
        return tmp[tmp.size() / 2];
      }
      double val1 = tmp[tmp.size() / 2 - 1];
      double val2 = tmp[tmp.size() / 2    ];
      return (val1 + val2) / 2;
    }

    double stdDev(const std::vector<double> & vec)
    {
      if (vec.size() == 0)
        throw std::runtime_error("Impossible to compute standard deviation from an empty set");
      return std::sqrt(variance(vec));
    }

    double variance(const std::vector<double> & vec)
    {
      if (vec.size() == 0)
        throw std::runtime_error("Impossible to compute variance from an empty set");
      double m = mean(vec);
      double total = 0;
      for (double e : vec) {
        double diff = e - m;
        total += diff * diff;
      }
      return total / vec.size();
    }

    std::vector<double> getQuartiles(const std::vector<double> & vec)
    {
      if (vec.size() == 0)
        throw std::runtime_error("Impossible to compute median from an empty set");
      std::vector<double> tmp = vec;
      std::sort(tmp.begin(), tmp.end());
      std::vector<double> result(3);
      size_t N = tmp.size();
      size_t n = N / 4;
      // Median only depends on N % 2
      switch(N % 2) {
      case 0:
        result[1] = (tmp[N/2] + tmp[N/2+1])/2;
        break;
      case 1:
        result[1] = tmp[N/2+1];
      }
      // Quartiles depends on N % 4
      switch (N % 4){
      case 0:
        result[0] = (tmp[  n] + tmp[  n+1])/2;
        result[2] = (tmp[3*n] + tmp[3*n+1])/2;
        break;
      case 2:
        result[0] = tmp[  n+1];
        result[2] = tmp[3*n+2];
        break;
      case 1:
        result[0] = 0.25 * tmp[  n  ] + 0.75 * tmp[  n+1];
        result[2] = 0.75 * tmp[3*n+1] + 0.25 * tmp[3*n+2];
        break;
      case 3:
        result[0] = 0.75 * tmp[  n+1] + 0.25 * tmp[n+2];
        result[2] = 0.25 * tmp[3*n+2] + 0.75 * tmp[3*n+3];
        break;
      }        
      return result;
    }
  }
}
