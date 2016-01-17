#include "rosban_regression_forests/tools/random.h"

namespace regression_forests
{

std::default_random_engine get_random_engine()
{
  unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
  return std::default_random_engine(seed);
}

std::vector<size_t> getKDistinctFromN(size_t k, size_t n,
                                      std::default_random_engine * engine)
{
  bool cleanAtEnd = false;
  if (engine == NULL) {
    cleanAtEnd = true;
    unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
    engine = new std::default_random_engine(seed);
  }
  // preparing structure
  std::vector<size_t> chosenIndex;
  std::vector<size_t> availableIndex;
  chosenIndex.reserve(k);
  availableIndex.reserve(n);
  for (size_t i = 0; i < n; i++) {
    availableIndex.push_back(i);
  }
  while (chosenIndex.size() < k) {
    int max = n - chosenIndex.size() - 1;
    std::uniform_int_distribution<size_t> distribution(0, max);
    int idx = distribution(*engine);
    chosenIndex.push_back(availableIndex[idx]);
    // availableIndex[max] will not be available at next iteration, therfore
    // we add it at the location where the index was taken
    availableIndex[idx] = availableIndex[max];
  }
  if (cleanAtEnd) {
    delete(engine);
  }
  return chosenIndex;
}

std::vector<Eigen::VectorXd> getUniformSamples(const Eigen::MatrixXd& limits,
                                                       size_t nbSamples)
{
  std::vector<Eigen::VectorXd> result;
  result.reserve(nbSamples);
  auto generator = get_random_engine();
  std::vector<std::uniform_real_distribution<double>> distribs;
  for (int i = 0; i < limits.rows(); i++)
  {
    std::uniform_real_distribution<double> d(limits(i,0), limits(i,1));
    distribs.push_back(d);
  }
  for (size_t sId = 0; sId < nbSamples; sId++) {
    Eigen::VectorXd sample(limits.rows());
    for (size_t dim = 0; dim < distribs.size(); dim++) {
      sample(dim) = distribs[dim](generator);
    }
    result.push_back(sample);
  }
  return result;
}

}
