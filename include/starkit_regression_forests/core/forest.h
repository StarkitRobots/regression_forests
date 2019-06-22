#pragma once

#include "starkit_regression_forests/core/tree.h"

#include "starkit_utils/serialization/stream_serializable.h"

#include <random>

namespace regression_forests
{
class Forest : public starkit_utils::StreamSerializable
{
private:
  std::vector<std::unique_ptr<Tree>> trees;

public:
  /// All: values from all the trees are considered
  /// AutoCut: the 10 % tail and 10 % bottom of the distribution are removed
  enum AggregationMethod
  {
    All,
    AutoCut
  };

  Forest();
  Forest(const Forest& other) = delete;
  virtual ~Forest();

  size_t nbTrees() const;
  const Tree& getTree(size_t treeId) const;

  /// transfers ownership of t to the Forest
  void push(std::unique_ptr<Tree> t);

  size_t maxSplitDim() const;

  /// Return the average value for the given input
  double getValue(const Eigen::VectorXd& input, AggregationMethod method = AggregationMethod::All) const;

  /// Return the aggregated values provided by each tree
  std::vector<double> getValues(const Eigen::VectorXd& input, AggregationMethod method = AggregationMethod::All) const;

  /// Return the variance of the estimation according to the trees for the given input
  double getVar(const Eigen::VectorXd& input, AggregationMethod method = AggregationMethod::All) const;

  /// Return the average gradient at the given input
  Eigen::VectorXd getGradient(const Eigen::VectorXd& input) const;

  /// Return a randomized value based on the standard deviation
  double getRandomizedValue(const Eigen::VectorXd& input, std::default_random_engine& engine) const;

  // maxLeafs is used to avoid growing an oversized tree. It is not activated by
  // default (0 value)
  // When preFilter is activated, each tree is projected before being merged
  // in the final result
  std::unique_ptr<Tree> unifiedProjectedTree(const Eigen::MatrixXd& limits, size_t maxLeafs = 0);

  /// Apply the given function on every node of every tree
  void apply(Eigen::MatrixXd& limits, Node::Function f);

  /// Apply the given function on every leaf of every tree
  void applyOnLeafs(Eigen::MatrixXd& limits, Node::Function f);

  /// Return 0 since there is no need to have several types in the same factory
  int getClassID() const override;
  /// Write a binary stream saving the configuration of the node and all its children
  /// Return the number of bytes written
  int writeInternal(std::ostream& out) const override;
  /// Read the configuration of the node and all its children from the provided binary stream
  /// Return the number of bytes read
  int read(std::istream& in) override;

  Forest* clone() const;
};

Forest::AggregationMethod loadAggregationMethod(const std::string& str);
std::string aggregationMethod2Str(Forest::AggregationMethod method);

}  // namespace regression_forests
