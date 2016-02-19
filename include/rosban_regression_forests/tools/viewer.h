#include "rosban_regression_forests/core/forest.h"

#include "rosban_viewer/viewer.h"

namespace regression_forests
{

class Viewer : public rosban_viewer::Viewer {
private:
  std::unique_ptr<Forest> forest;

  // The dimension on which the user is currently acting
  int currentDim;//currentDim = -1: nothing selected
  size_t inputDim;

  // Values used for projection
  Eigen::VectorXd lockValues;
  std::vector<bool> locked;

  // Global limits for dimensions (including value)
  std::vector<std::string> dimNames;
  Eigen::MatrixXd limits;

  // Display Tiles
  std::vector<std::vector<Eigen::VectorXd>> cornersPos;
  std::vector<std::vector<rosban_viewer::Color>> cornersColor;

protected:

  // Rescale rawValue in [0,1] from [min[dim], max[dim]]
  double rescaleValue(double rawValue, int dim);

  void increaseValue(double ratio);
  void valueToMax();
  void valueToMin();
  void lock();
  void unlock();

  void updateCorners();

  void drawTiles();

  std::vector<int> freeDimensions();

  Eigen::MatrixXd getLocalLimits();

public:
  /**
   * forestFile should be readable by the Math::RegressionTree:Parser
   * configFile should be a csv with n columns and 3 rows
   * 1. headers (including value)
   * 2. mins
   * 3. maxs
   */
  Viewer(const std::string& forestFile,
         const std::string& configFile,
         unsigned int width = 800, unsigned int height = 600);

  void updateStatus();
  virtual bool update() override;

  /**
   * Jump to previous index if shift is pressed, next otherwise 
   */
  void navigate();
      
};
}
