#pragma once

#include "rosban_regression_forests/core/orthogonal_split.h"
#include "rosban_regression_forests/core/sample.h"

#include <vector>

namespace regression_forests
{
class TrainingSet
{
private:
  int inputDim;
  std::vector<Sample> experiments;

public:
  TrainingSet(int inputDim);

  /// For inputs:
  /// - Each row is a different dimension
  /// - Each col is a different entries
  /// For outputs:
  /// - Each row is a different output
  /// f(inputs.col(i)) = outputs(i)
  TrainingSet(const Eigen::MatrixXd & inputs,
              const Eigen::VectorXd & outputs);

  void push(const Sample &s);

  size_t size() const;

  size_t getInputDim() const;

  /**
   * throws an std::out_of_range exception if idx is invalid
   */
  const Sample &operator()(size_t idx) const;

  /**
   * It is possible to use a vector of index in order to avoid copying all
   * the data concerning input.
   */
  typedef std::vector<int> Subset;

  Subset wholeSubset() const;

  /**
   * Sort the subset provided as argument along the dimension requested
   */
  void sortSubset(Subset &s, size_t dim) const;

  /**
   * Build the lower and upper subsets obtained by applying the given split
   * to the given subset
   */
  void applySplit(const OrthogonalSplit &split, const Subset &subset, Subset &lowerSet, Subset &upperSet) const;

  std::vector<double> values(const Subset &s) const;
  std::vector<Eigen::VectorXd> inputs(const Subset &s) const;
  std::vector<double> inputs(const Subset &s, size_t dim) const;

  TrainingSet buildBootstrap() const;
  TrainingSet buildBootstrap(size_t nbSamples) const;
};
}
