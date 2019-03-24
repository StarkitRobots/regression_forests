#pragma once

#include "rhoban_regression_forests/approximations/approximation.h"

#include "rhoban_utils/serialization/factory.h"

namespace regression_forests
{
class ApproximationFactory : public rhoban_utils::Factory<Approximation>
{
public:
  ApproximationFactory();
};

}  // namespace regression_forests
