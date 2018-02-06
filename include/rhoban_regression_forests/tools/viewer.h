#include "rhoban_regression_forests/core/forest.h"

#include "rhoban_viewer/viewer.h"

namespace regression_forests
{

class Viewer : public rhoban_viewer::Viewer {
private:
  /// The forest which is being visualized
  std::unique_ptr<Forest> forest;

  /// The dimension on which the user is currently acting
  /// -1: no dimension selected
  int dim_index;

  /// Internal index of the dimension:
  /// -1: Focus on the dimension itself
  ///  0: Focus on min
  ///  1: Focus on max
  int sub_dim_index;

  /// Dimension names
  std::vector<std::string> dim_names;

  /// Global limits for dimensions (including value)
  Eigen::MatrixXd space_limits;

  /// Current range of values for input/output
  Eigen::MatrixXd current_limits;

  /// For input, if dim is locked, it has a specific value
  /// For output, if dim is locked, it is considered as automatically chosen
  std::vector<bool> locked;

  /// Position of corners (in 0,1)
  std::vector<std::vector<Eigen::VectorXd>> corners_pos;
  /// Color of corners
  std::vector<std::vector<rhoban_viewer::Color>> corners_color;

protected:

  /// Rescale rawValue from current limits to [0,1]
  double rescaleValue(double rawValue, int dim);

  /// Increase value of the selected item by ratio * (space_max - space_min)
  /// Note: In case this would result in min > max on the chosen dimension,
  ///       the value is set to the other bound value
  void increaseValue(double ratio);

  /// Set current value to space_max
  /// Note: If selected index is min, then current_min is set to current_max
  void valueToMax();

  /// Set current value to space_min
  /// Note: If selected index is max, then current_max is set to current_min
  void valueToMin();

  /// Switch a dimension status between free and locked
  void toggle();

  /// Update corners according to current limits
  void updateCorners();

  void drawTiles();

  /// Return a list of the dimensions which are considered as 'free'
  std::vector<int> freeDimensions();

  /// Accessors to current limits
  const Eigen::MatrixXd & getCurrentLimits() const;

  /// Append dimension description to a stream
  void appendDim(int dim, std::ostream &out) const;

  /// Append dimension limits to a stream
  void appendLimits(int dim, std::ostream &out) const;

public:
  /// ForestFile should be readable by the Math::RegressionTree:Parser
  /// ConfigFile should be a csv with n columns and 3 rows
  /// 1. headers (including value)
  /// 2. mins
  /// 3. maxs
  Viewer(const std::string& forest_path,
         const std::string& config_path,
         unsigned int width = 800, unsigned int height = 600);

  /// Update status message
  void updateStatus();
  virtual bool update() override;

  /// Number of dimensions of input
  int inputSize() const;

  /**
   * Jump to previous index if shift is pressed, next otherwise 
   */
  void navigate();
      
};
}
