#include "starkit_regression_forests/core/training_set.h"

#include "starkit_random/tools.h"

#include <stdexcept>

namespace regression_forests
{
TrainingSet::TrainingSet(int inputDim_) : inputDim(inputDim_)
{
}

TrainingSet::TrainingSet(const Eigen::MatrixXd& inputs, const Eigen::VectorXd& outputs) : inputDim(inputs.rows())
{
  if (inputs.cols() != outputs.rows())
  {
    throw std::runtime_error("TrainingSet creation: inputs.cols() != outputs.rows()");
  }
  for (int sample_id = 0; sample_id < inputs.cols(); sample_id++)
  {
    push(Sample(inputs.col(sample_id), outputs(sample_id)));
  }
}

void TrainingSet::push(const Sample& s)
{
  experiments.push_back(s);
}

size_t TrainingSet::size() const
{
  return experiments.size();
}

size_t TrainingSet::getInputDim() const
{
  return inputDim;
}

const Sample& TrainingSet::operator()(size_t idx) const
{
  return experiments.at(idx);
}

TrainingSet::Subset TrainingSet::wholeSubset() const
{
  Subset s;
  s.reserve(experiments.size());
  for (size_t i = 0; i < experiments.size(); i++)
  {
    s.push_back(i);
  }
  return s;
}

void TrainingSet::sortSubset(Subset& s, size_t dim) const
{
  std::sort(s.begin(), s.end(),
            [this, dim](size_t i1, size_t i2) { return (*this)(i1).getInput(dim) < (*this)(i2).getInput(dim); });
}

void TrainingSet::applySplit(const OrthogonalSplit& split, const Subset& subset, Subset& lowerSet,
                             Subset& upperSet) const
{
  lowerSet.clear();
  upperSet.clear();
  for (size_t idx : subset)
  {
    if ((*this)(idx).getInput(split.dim) <= split.val)
    {
      lowerSet.push_back(idx);
    }
    else
    {
      upperSet.push_back(idx);
    }
  }
}

std::vector<double> TrainingSet::values(const Subset& s) const
{
  std::vector<double> result;
  result.reserve(s.size());
  for (size_t idx : s)
  {
    result.push_back((*this)(idx).getOutput());
  }
  return result;
}

std::vector<Eigen::VectorXd> TrainingSet::inputs(const Subset& s) const
{
  std::vector<Eigen::VectorXd> result;
  result.reserve(s.size());
  for (size_t idx : s)
  {
    result.push_back((*this)(idx).getInput());
  }
  return result;
}

std::vector<double> TrainingSet::inputs(const Subset& s, size_t dim) const
{
  std::vector<double> result;
  result.reserve(s.size());
  for (size_t idx : s)
  {
    result.push_back((*this)(idx).getInput(dim));
  }
  return result;
}

TrainingSet TrainingSet::buildBootstrap() const
{
  return buildBootstrap(size());
}

TrainingSet TrainingSet::buildBootstrap(size_t nbSamples) const
{
  TrainingSet bootstrap(inputDim);
  auto engine = starkit_random::getRandomEngine();
  std::uniform_int_distribution<size_t> indexDistrib(0, size() - 1);
  for (size_t i = 0; i < nbSamples; i++)
  {
    size_t idx = indexDistrib(engine);
    bootstrap.push((*this)(idx));
  }
  return bootstrap;
}
}  // namespace regression_forests
