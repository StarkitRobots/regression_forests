#include "starkit_regression_forests/approximations/approximation_factory.h"

#include "starkit_regression_forests/approximations/pwc_approximation.h"
#include "starkit_regression_forests/approximations/pwl_approximation.h"

#ifdef STARKIT_RF_USES_GP
#include "starkit_regression_forests/approximations/gp_approximation.h"
#endif

namespace regression_forests
{
ApproximationFactory::ApproximationFactory()
{
  registerBuilder(Approximation::PWC, []() { return std::unique_ptr<Approximation>(new PWCApproximation); });
  registerBuilder(Approximation::PWL, []() { return std::unique_ptr<Approximation>(new PWLApproximation); });
#ifdef STARKIT_RF_USES_GP
  registerBuilder(Approximation::GP, []() { return std::unique_ptr<Approximation>(new GPApproximation); });
#endif
}

}  // namespace regression_forests
