#include "ApproximationType.hpp"

#include <stdexcept>

using Math::RegressionTree::ApproximationType;

namespace Math {
  namespace RegressionTree {

    ApproximationType loadApproximationType(const std::string& s)
    {
      if (s == "PWC") {
        return ApproximationType::PWC;
      }
      if (s == "PWL") {
        return ApproximationType::PWL;
      }
      throw std::runtime_error("Unknown approximation description '" + s + "'");
    }

    std::string approximationType2String(Math::RegressionTree::ApproximationType at)
    {
      switch(at){
      case ApproximationType::PWC:
        return "PWC";
      case ApproximationType::PWL:
        return "PWL";
      }
      throw std::runtime_error("Unkown ApproximationType");
    }


  }
}
