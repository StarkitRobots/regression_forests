#pragma once

#include "rosban_regression_forests/approximations/approximation.h"

#include "rosban_utils/factory.h"

namespace regression_forests
{

class ApproximationFactory : public rosban_utils::Factory<Approximation>
{
public:
  ApproximationFactory();
};

}
