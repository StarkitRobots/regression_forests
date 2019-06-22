#pragma once

#include "starkit_regression_forests/approximations/approximation.h"

#include "starkit_utils/serialization/factory.h"

namespace regression_forests
{
class ApproximationFactory : public starkit_utils::Factory<Approximation>
{
public:
  ApproximationFactory();
};

}  // namespace regression_forests
