#pragma once

#include <Eigen/Core>

#include <random>

namespace regression_forests
{

std::default_random_engine get_random_engine();

/// Create its own engine if no engine is provided
std::vector<size_t> getKDistinctFromN(size_t k, size_t n,
                                      std::default_random_engine * engine = NULL);

/// Create its own engine if no engine is provided
std::vector<Eigen::VectorXd> getUniformSamples(const Eigen::MatrixXd& limits,
                                               size_t nbSamples,
                                               std::default_random_engine * engine = NULL);

}
