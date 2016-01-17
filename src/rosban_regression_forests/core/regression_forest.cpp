#include "RegressionForest.hpp"

#include "Parser.hpp"
#include "Pruning.hpp"

#include <timing/Benchmark.hpp>
#include <util.h>

#include <fstream>

using Utils::Timing::Benchmark;

namespace Math {
  namespace RegressionTree {

    RegressionForest::RegressionForest()
    {
    }

    RegressionForest::~RegressionForest()
    {
    }

    size_t RegressionForest::nbTrees() const
    {
      return trees.size();
    }

    const RegressionTree& RegressionForest::getTree(size_t treeId) const
    {
      return *trees[treeId];
    }

    void RegressionForest::push(std::unique_ptr<RegressionTree> t)
    {
      trees.push_back(std::move(t));
    }

    size_t RegressionForest::maxSplitDim() const
    {
      size_t max = 0;
      for (const auto& t : trees) {
        max = std::max(max, t->maxSplitDim());
      }
      return max;
    }

    double RegressionForest::getValue(const Eigen::VectorXd& input) const
    {
      double sum = 0.0;
      for (const auto& t : trees) {
        sum += t->getValue(input);
      }
      return sum / trees.size();
    }

    std::unique_ptr<RegressionTree>
    RegressionForest::unifiedProjectedTree(const Eigen::MatrixXd& limits,
                                           size_t maxLeafs,
                                           bool preFilter,
                                           bool parallelMerge)
    {
      //Benchmark::open("unifiedProjectedTree");
      std::unique_ptr<RegressionTree> result;
      if (trees.size() == 0) return result;
      //Benchmark::open("FirstTree");
      result = trees[0]->project(limits);
      if (maxLeafs != 0) {
        result = pruneTree(std::move(result), limits, maxLeafs);
      }
      //Benchmark::close();//FirstTree
      for (size_t treeId = 1; treeId < trees.size(); treeId++) {
        std::unique_ptr<RegressionTree> tree;
        if (preFilter) {
          //Benchmark::open("PreFiltering");
          tree =  trees[treeId]->project(limits);
          //Benchmark::close();
        }
        else {
          tree = std::move(trees[treeId]);
        }
        //Benchmark::open("AvgTree");
        if (parallelMerge) {
          result = RegressionTree::avgTrees(*result,*tree, treeId, 1, limits);
        }
        else {
          result->avgTree(*tree, 1.0 / treeId, limits);
        }
        //Benchmark::close();//AvgTree
        // Give back property to the vector
        if (!preFilter) {
          trees[treeId] = std::move(tree);
        }
        if (maxLeafs != 0) {
          //Benchmark::open("Pruning");
          result = pruneTree(std::move(result), limits, maxLeafs);
          //Benchmark::close();//Pruning
        }
      }
      //Benchmark::close();
      return result; 
    }

    void RegressionForest::save(const std::string& path) const
    {
      std::ofstream ofs;
      ofs.open(path);
      ofs << *this;
      ofs.close();
    }

    std::unique_ptr<RegressionForest>
    RegressionForest::loadFile(const std::string& path)
    {
      return Parser::regressionForest(slurpFile(path));
    }

  }
}

std::ostream& operator<<(std::ostream& out,
                         const Math::RegressionTree::RegressionForest& forest)
{
  out << 'f';
  for (size_t treeId = 0; treeId < forest.nbTrees(); treeId++) {
    out << forest.getTree(treeId);
  }
  out << '$';
  return out;
}
