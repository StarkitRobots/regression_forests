#pragma once

namespace regression_forests
{

std::default_random_engine get_random_engine();

std::vector<size_t> getKDistinctFromN(size_t k, size_t n,
                                      std::default_random_engine * engine);

std::vector<Eigen::VectorXd> getUniformSamples(const Eigen::MatrixXd& limits,
                                               size_t nbSamples);

}
