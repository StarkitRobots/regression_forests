#include "rosban_regression_forests/tools/viewer.h"

#include <rosban_utils/string_tools.h>

#include <fstream>
#include <iostream>

#include <SFML/OpenGL.hpp>

using rosban_viewer::Color;

namespace regression_forests {

Viewer::Viewer(const std::string& forest_path,
               const std::string& config_path,
               unsigned int width,
               unsigned int height)
  : rosban_viewer::Viewer(width, height),
    forest(new Forest()),
    dim_index(-1),
    sub_dim_index(-1)
{
  std::cout << "Loading file '" << forest_path << "' as a forest" << std::endl;
  forest->load(forest_path);

  // Read config
  std::ifstream configStream;
  configStream.open(config_path);
  std::vector<std::string> lines;
  while(configStream.good()) {
    std::string line;
    getline(configStream, line);
    lines.push_back(line);
  }
  configStream.close();
  if (lines.size() != 3) {
    throw std::runtime_error("Expecting 3 lines in config file: (header, mins, maxs)");
  }
  std::vector<std::string> mins, maxs;
  dim_names = rosban_utils::split_string(lines[0], ',');
  mins     = rosban_utils::split_string(lines[1], ',');
  maxs     = rosban_utils::split_string(lines[2], ',');
  if (dim_names.size() != mins.size() || dim_names.size() != maxs.size()) {
    throw std::runtime_error("Inconsistent config file");
  }
  space_limits = Eigen::MatrixXd(dim_names.size(), 2);
  for (size_t dim = 0; dim < dim_names.size(); dim++) {
    space_limits(dim, 0) = std::stod(mins[dim]);
    space_limits(dim, 1) = std::stod(maxs[dim]);
  }

  if (forest->maxSplitDim() > inputSize()) {
    throw std::runtime_error("config has not enough columns");
  }

  // Initially, all dimensions are locked on the middle value (special behavior for output)
  current_limits = Eigen::MatrixXd(dim_names.size(), 2);
  locked.clear();
  for (size_t dim = 0; dim <= inputSize(); dim++) {
    locked.push_back(true);
    double mid_val = (space_limits(dim, 0) + space_limits(dim, 1)) / 2;
    for (int sub_dim : {0,1})
    {
      if (dim < inputSize())
        current_limits(dim, sub_dim) = mid_val;
      else
        current_limits(dim, sub_dim) = space_limits(dim, sub_dim);
    }
  }

  // Initializing keyboard callbacks
  onKeyPress[sf::Keyboard::Tab].push_back([this](){ this->navigate(); });
  onKeyPress[sf::Keyboard::PageUp].push_back([this]()
                                             {
                                               this->increaseValue(0.01);
                                             });
  onKeyPress[sf::Keyboard::PageDown].push_back([this]()
                                               {
                                                 this->increaseValue(-0.01);
                                               });
  onKeyPress[sf::Keyboard::Home].push_back([this]()
                                           { this->valueToMax(); });
  onKeyPress[sf::Keyboard::End].push_back([this]()
                                          { this->valueToMin(); });
  onKeyPress[sf::Keyboard::T].push_back([this]()
                                        { this->toggle(); });

  // Default messages
  font.loadFromFile("monkey.ttf");//Follow it!
  status = sf::Text("Status", font);
  status.setCharacterSize(30);
  status.setColor(sf::Color::Red);

  // Update ground
  for(int i : {0,1}){
    groundLimits(i,0) = 0;
    groundLimits(i,1) = 1;
  }
}

int Viewer::inputSize() const
{
  return dim_names.size() - 1;
}

double Viewer::rescaleValue(double rawValue, int dim)
{
  double min = current_limits(dim, 0);
  double max = current_limits(dim, 1);
  double delta = max - min;
  return (rawValue - min) / delta;
}

void Viewer::increaseValue(double ratio)
{
  // If no dimension is selected, do nothing
  if (dim_index == -1) return;
  // Applying eventual modifiers
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift)) { ratio *= 10; }
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::LControl)) { ratio /= 10; }
  // Getting basic properties
  double old_min = current_limits(dim_index,0);
  double old_max = current_limits(dim_index,1);
  double min = space_limits(dim_index, 0);
  double max = space_limits(dim_index, 1);
  double delta = max - min;
  // Updating value
  if (sub_dim_index < 0)
  {
    for (int sub_dim : {0,1})
    {
      current_limits(dim_index, sub_dim) += delta * ratio;
    }
  }
  else
  {
    current_limits(dim_index, sub_dim_index) += delta * ratio;
  }
  // Ensuring that current_limits is a subspace of space_limits
  if (current_limits(dim_index, 0) < min) current_limits(dim_index, 0) = min;
  if (current_limits(dim_index, 1) > max) current_limits(dim_index, 1) = max;
  // Ensure min <= max
  if (current_limits(dim_index,0) > current_limits(dim_index,1))
  {
    if (sub_dim_index == 0)
      current_limits(dim_index, 0) = current_limits(dim_index, 1);
    else
      current_limits(dim_index, 1) = current_limits(dim_index, 0);
  }
  // If there was effective change update corners
  if (current_limits(dim_index, 0) != old_min ||
      current_limits(dim_index, 1) != old_max)
  {
    updateCorners();
  }
}

void Viewer::valueToMax()
{
  if (dim_index == -1) return;
  double max = space_limits(dim_index, 1);
  switch(sub_dim_index)
  {
    case -1:
      current_limits(dim_index, 0) = max;
      current_limits(dim_index, 1) = max;
      break;
    case 0:
      current_limits(dim_index, 0) = current_limits(dim_index,1);
      break;
    case 1:
      current_limits(dim_index, 1) = max;
      break;
    default:
      throw std::logic_error("Invalid value for dim_index");
  }
  updateCorners();
}

void Viewer::valueToMin()
{
  if (dim_index == -1) return;
  double min = space_limits(dim_index, 0);
  switch(sub_dim_index)
  {
    case -1:
      current_limits(dim_index, 0) = min;
      current_limits(dim_index, 1) = min;
      break;
    case 0:
      current_limits(dim_index, 0) = min;
      break;
    case 1:
      current_limits(dim_index, 1) = current_limits(dim_index, 0);
      break;
    default:
      throw std::logic_error("Invalid value for dim_index");
  }
  updateCorners();
}

void Viewer::toggle()
{
  // If no dim selected, do nothing
  if (dim_index == -1) return;
  // toggle locked status
  locked[dim_index] = !locked[dim_index];
  // If locked, set min and max to the middle value
  if (locked[dim_index])
  {
    double mid_val = (space_limits(dim_index, 0) + space_limits(dim_index, 1)) / 2;
    for (int sub_dim : {0,1})
    {
      current_limits(dim_index, sub_dim) = mid_val;
    }
    // Min and max cannot be selected since dim is locked
    sub_dim_index = -1;
  }
  // If unlocked, set min and max to extremum values
  else
  {
    for (int sub_dim : {0,1})
    {
      current_limits(dim_index, sub_dim) = space_limits(dim_index, sub_dim);
    }
  }
  updateCorners();
}

std::vector<int> Viewer::freeDimensions()
{
  std::vector<int> result;
  for (size_t d = 0; d < inputSize(); d++) {
    if (!locked[d]) {
      result.push_back(d);
    }
  }
  return result;
}

const Eigen::MatrixXd & Viewer::getCurrentLimits() const
{
  return current_limits;
}

void Viewer::updateCorners()
{
  corners_pos.clear();
  corners_color.clear();

  // Local limits
  Eigen::MatrixXd input_limits = getCurrentLimits().block(0,0,inputSize(),2);

  std::vector<int> freeDims = freeDimensions();

  if (freeDims.size() > 2) { return;}
  if (freeDims.size() == 0) { // Special mode, print value of each tree
    Eigen::VectorXd input = input_limits.col(0);
    std::cout << "Input: "  << input                   << std::endl;
    std::cout << "Output: " << forest->getValue(input) << std::endl;
    for (size_t treeId = 0; treeId < forest->nbTrees(); treeId++)
    {
      double val = forest->getTree(treeId).getValue(input);
      std::cout << treeId << ": " << val << std::endl;
    }
    return;
  }

  size_t maxLeafs = 0;//200000;

  std::unique_ptr<Tree> projectedTree;
  std::cout << "Unifying projectTree" << std::endl;
  projectedTree = forest->unifiedProjectedTree(input_limits, maxLeafs);
  std::cout << "Finding max:" << std::endl;
  std::pair<double, Eigen::VectorXd> projectionMax = projectedTree->getMaxPair(input_limits);
  std::pair<double, Eigen::VectorXd> projectionMin = projectedTree->getMinPair(input_limits);
  std::cout << "\tMin: " << projectionMin.first << " at "
            << projectionMin.second.transpose() << std::endl;
  std::cout << "\tMax: " << projectionMax.first << " at "
            << projectionMax.second.transpose() << std::endl;
  std::cout << "Projecting projectTree" << std::endl;
  std::vector<std::vector<Eigen::VectorXd>> projectionTiles;
  projectionTiles = projectedTree->project(freeDims, input_limits);
  std::cout << "Filling structures" << std::endl;

  // Auto mode, update current_limits for value according to content
  if (locked[inputSize()])
  {
    current_limits(inputSize(), 0) = std::max(projectionMin.first, space_limits(inputSize(),0));
    current_limits(inputSize(), 1) = std::min(projectionMax.first, space_limits(inputSize(),1));
  }
  // Normalize and add color
  for (size_t tileId = 0; tileId < projectionTiles.size(); tileId++) {
    std::vector<Eigen::VectorXd>& projectedTile = projectionTiles[tileId];
    std::vector<Eigen::VectorXd> tileCornersPos;
    std::vector<Color> tileCornersColor;
    for (size_t cornerId = 0; cornerId < projectedTile.size(); cornerId++) {
      Eigen::VectorXd corner(3);
      // Rescale x,y values
      for (size_t corner_dim = 0; corner_dim < freeDims.size(); corner_dim++) {
        int free_dim = freeDims[corner_dim];
        double rawVal = projectedTile[cornerId](corner_dim);
        // If min != max, just rescale values
        if (input_limits(free_dim,0) != input_limits(free_dim,1))
        {
          corner(corner_dim) = rescaleValue(rawVal, free_dim);
        }
        // If min == max, select 0 or 1 depending on cornerId
        else
        {
          bool is_min = (cornerId / (int)std::pow(2, corner_dim)) % 2;
          corner(corner_dim) = is_min ? 0 : 1;
        }
      }
      // Rescale z value (output)
      double rawOutput = projectedTile[cornerId](freeDims.size());
      // Bounding raw Output to acceptable values
      if (!locked[inputSize()])
      {
        if (rawOutput > current_limits(inputSize(),1))
          rawOutput = current_limits(inputSize(),1);
        if (rawOutput < current_limits(inputSize(),0))
          rawOutput = current_limits(inputSize(),0);
      }
      double output = 0.5;
      // If minValue != maxValue, update output
      if (current_limits(inputSize(), 0) != current_limits(inputSize(), 1))
      {
        output = rescaleValue(rawOutput, inputSize());
      }
      corner(2) = output;
      Color color;
      double altColor = 1.0 - 2 * std::fabs(output - 0.5);
      if (output > 0.5) {
        color = Color(1, altColor, altColor);
      }
      else {
        color = Color(altColor, altColor, 1);
      }
      // if freeDim is 1: add a variable custom Y
      if (freeDims.size() == 1) {
        for (double y : {0,1}) {
          Eigen::VectorXd fakeCorner = corner;
          fakeCorner(1) = y;
          tileCornersPos.push_back(fakeCorner);
          tileCornersColor.push_back(color);
        }
      }
      else {
        tileCornersPos.push_back(corner);
        tileCornersColor.push_back(color);
      }
    }
    corners_pos.push_back(tileCornersPos);
    corners_color.push_back(tileCornersColor);
  }
}

void Viewer::navigate()
{
  //Shift-tab not handled in sfml 2.0
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::LControl)) {
    // If we can jump to previous sub_dim, it is enough
    if (sub_dim_index > -1)
    {
      sub_dim_index--;
    }
    // Jump to previous dimension
    else
    {
      dim_index--;
      // Circular list
      if (dim_index == -2) {
        dim_index = inputSize();
      }
      // If final dimension is not locked, use last sub_dim_index
      if (dim_index >= 0 && !locked[dim_index])
        sub_dim_index = 1;
      else
        sub_dim_index = -1;
    }
  }
  // Forward case
  else {
    // If we can jump to next sub_dim, it is enough
    if (dim_index >= 0 && !locked[dim_index] && sub_dim_index < 1)
    {
      sub_dim_index++;
    }
    // Jump to next dim
    else
    {
      dim_index++;
      // circular list
      if (dim_index > inputSize()) {
        dim_index = -1;
      }
      sub_dim_index = -1;
    }
  }
}

void Viewer::drawTiles()
{
  glBegin(GL_QUADS);
  for (size_t tileId = 0; tileId < corners_pos.size(); tileId++) {
    const std::vector<Eigen::VectorXd>& points = corners_pos[tileId];
    const std::vector<Color>& colors = corners_color[tileId];
    for (size_t pId : {0,1,3,2}) {//Ordering matters for opengl
      const Eigen::VectorXd& p = points[pId];
      const Color& c = colors[pId];
      glColor3f(c.r, c.g, c.b);
      glVertex3f(p(0), p(1), p(2));
    }
  }
  glEnd();
}

void Viewer::appendLimits(int dim, std::ostream &out) const
{
  std::vector<std::pair<std::string,int>> names = {{"min", 0}, {"max", 1}};
  for (const auto & entry : names)
  {
    // Print selection indicator
    if (dim == dim_index && sub_dim_index == entry.second)
      out << "->    ";
    else
      out << "      ";
    // Print limits itself
    out << entry.first << ": " << current_limits(dim, entry.second) << std::endl;
  }     
}

void Viewer::appendDim(int dim, std::ostream &out) const
{
  bool input = dim != inputSize();
  // Printing selection indicator
  if (dim_index == dim && sub_dim_index == -1)
    out << "->";
  else
    out << "  ";
  // Printing name and limits
  out << dim_names[dim] << ": ";//Padding would be nice
  out << " [" << space_limits(dim,0) << "," << space_limits(dim,1) << "] ";
  // Status dependant message
  if (locked[dim])
  {
    // Just print information on lock value
    if (input)
    {
      out << "Locked at " << current_limits(dim,0) << ": " << std::endl;
    }
    // Print used bounds for output
    else
    {
      out << "Auto" << std::endl;
      appendLimits(dim, out);
    }
  }
  // Dimension is free
  else
  {
    // Just print information on lock value
    out << "Manual" << std::endl;
    appendLimits(dim, out);
  }
}

void Viewer::updateStatus()
{
  std::ostringstream oss;
  // Input:
  for (size_t dim = 0; dim <= inputSize(); dim++)
  {
    appendDim(dim, oss);
  }
  rosban_viewer::Viewer::updateStatus(oss.str());
}

bool Viewer::update()
{
  updateStatus();
  drawTiles();
  return rosban_viewer::Viewer::update();
}

}
