#pragma once

#include "rosban_regression_forests/approximations/approximation.h"
#include "rosban_regression_forests/core/orthogonal_split.h"
#include "rosban_regression_forests/core/regression_forest.h"

namespace regression_forests
{
namespace Parser
{
/**
 * Build an approximation from string and update index
 * index is placed after the $ ending the 'a...$' string describing the
 * approximation
 */
Approximation *approximation(const std::string &s, size_t *index = NULL);

OrthogonalSplit orthogonalSplit(const std::string &s, size_t *index = NULL);

Node *regressionNode(const std::string &s, size_t *index = NULL);

std::unique_ptr<Tree> regressionTree(const std::string &s, size_t *index = NULL);

std::unique_ptr<Forest> regressionForest(const std::string &s, size_t *index = NULL);
}
}
