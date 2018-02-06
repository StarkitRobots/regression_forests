#include "rhoban_regression_forests/approximations/approximation_factory.h"

#include "rhoban_regression_forests/approximations/gp_approximation.h"
#include "rhoban_regression_forests/approximations/pwc_approximation.h"
#include "rhoban_regression_forests/approximations/pwl_approximation.h"

namespace regression_forests
{

ApproximationFactory::ApproximationFactory()
{
  registerBuilder(Approximation::GP,
                  [](){return std::unique_ptr<Approximation>(new GPApproximation);});
  registerBuilder(Approximation::PWC,
                  [](){return std::unique_ptr<Approximation>(new PWCApproximation);});
  registerBuilder(Approximation::PWL,
                  [](){return std::unique_ptr<Approximation>(new PWLApproximation);});
}

}
