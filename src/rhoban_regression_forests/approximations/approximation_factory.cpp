#include "rhoban_regression_forests/approximations/approximation_factory.h"

#include "rhoban_regression_forests/approximations/pwc_approximation.h"
#include "rhoban_regression_forests/approximations/pwl_approximation.h"

#ifdef RHOBAN_RF_USES_GP
#include "rhoban_regression_forests/approximations/gp_approximation.h"
#endif

namespace regression_forests
{

ApproximationFactory::ApproximationFactory()
{
  registerBuilder(Approximation::PWC,
                  [](){return std::unique_ptr<Approximation>(new PWCApproximation);});
  registerBuilder(Approximation::PWL,
                  [](){return std::unique_ptr<Approximation>(new PWLApproximation);});
#ifdef RHOBAN_RF_USES_GP
  registerBuilder(Approximation::GP,
                  [](){return std::unique_ptr<Approximation>(new GPApproximation);});
#endif
}

}
