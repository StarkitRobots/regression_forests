#include "rosban_regression_forests/tools/viewer.h"

#include "rosban_regression_forests/tools/parser.h"

#include <rosban_utils/string_tools.h>

#include <fstream>
#include <iostream>

#include <SFML/OpenGL.hpp>

using rosban_viewer::Color;

namespace regression_forests {

Viewer::Viewer(const std::string& forestFile,
               const std::string& configFile,
               unsigned int width, unsigned int height)
  : rosban_viewer::Viewer(width, height),
    currentDim(-1)
{
  std::cout << "Loading file '" << forestFile << "' as a forestFile" << std::endl;
  forest = Forest::loadFile(forestFile);

  // Read config
  std::ifstream configStream;
  configStream.open(configFile);
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
  dimNames = rosban_utils::split_string(lines[0], ',');
  mins     = rosban_utils::split_string(lines[1], ',');
  maxs     = rosban_utils::split_string(lines[2], ',');
  if (dimNames.size() != mins.size() || dimNames.size() != maxs.size()) {
    throw std::runtime_error("Inconsistent config file");
  }
  limits = Eigen::MatrixXd(dimNames.size(), 2);
  for (size_t dim = 0; dim < dimNames.size(); dim++) {
    limits(dim, 0) = std::stod(mins[dim]);
    limits(dim, 1) = std::stod(maxs[dim]);
  }
  inputDim = dimNames.size() - 1;

  if (forest->maxSplitDim() > inputDim) {
    throw std::runtime_error("configFile has not enough columns");
  }

  lockValues = Eigen::VectorXd(inputDim);
  locked.clear();
  for (size_t dimIdx = 0; dimIdx < inputDim; dimIdx++) {
    locked.push_back(true);
    lockValues(dimIdx) = (limits(dimIdx, 0) + limits(dimIdx, 1)) / 2;
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
  onKeyPress[sf::Keyboard::L].push_back([this]()
                                        { this->lock(); });
  onKeyPress[sf::Keyboard::F].push_back([this]()
                                        { this->unlock(); });

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

double Viewer::rescaleValue(double rawValue, int dim)
{
  double min = limits(dim, 0);
  double max = limits(dim, 1);
  double delta = max - min;
  return (rawValue - min) / delta;
}

void Viewer::increaseValue(double ratio)
{
  if (currentDim == -1) return;
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift)) { ratio *= 10; }
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::LControl)) { ratio /= 10; }
  double oldVal = lockValues(currentDim);
  double min = limits(currentDim, 0);
  double max = limits(currentDim, 1);
  double delta = max - min;
  double newVal = oldVal + delta * ratio;
  if (newVal > max) { newVal -= delta; }
  if (newVal < min) { newVal += delta; }
  lockValues(currentDim) = newVal;
  updateCorners();
}

void Viewer::valueToMax()
{
  if (currentDim == -1) return;
  lockValues(currentDim) = limits(currentDim, 1);
  updateCorners();
}

void Viewer::valueToMin()
{
  if (currentDim == -1) return;
  lockValues(currentDim) = limits(currentDim, 0);
  updateCorners();
}

void Viewer::lock()
{
  if (currentDim == -1) return;
  locked[currentDim] = true;
  updateCorners();
}

void Viewer::unlock()
{
  if (currentDim == -1) return;
  locked[currentDim] = false;
  updateCorners();
}

std::vector<int> Viewer::freeDimensions()
{
  std::vector<int> result;
  for (size_t d = 0; d < inputDim; d++) {
    if (!locked[d]) {
      result.push_back(d);
    }
  }
  return result;
}

Eigen::MatrixXd Viewer::getLocalLimits()
{
  Eigen::MatrixXd localLimits = limits.block(0,0,inputDim,2);
  for (size_t dim = 0; dim < inputDim; dim++) {
    if (locked[dim]) {
      localLimits(dim,0) = lockValues[dim];
      localLimits(dim,1) = lockValues[dim];
    }
  }
  return localLimits;
}

void Viewer::updateCorners()
{
  cornersPos.clear();
  cornersColor.clear();

  // Local limits
  Eigen::MatrixXd localLimits = getLocalLimits();

  std::vector<int> freeDims = freeDimensions();

  if (freeDims.size() > 2) { return;}
  if (freeDims.size() == 0) { // Special mode, print value of each tree
    Eigen::VectorXd input = localLimits.col(0);
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
  projectedTree = forest->unifiedProjectedTree(localLimits, maxLeafs);
  std::cout << "Finding max:" << std::endl;
  std::pair<double, Eigen::VectorXd> projectionMax = projectedTree->getMaxPair(localLimits);
  std::cout << "\tMax: " << projectionMax.first << " at "
            << projectionMax.second.transpose() << std::endl;
  std::cout << "Projecting projectTree" << std::endl;
  std::vector<std::vector<Eigen::VectorXd>> projectionTiles;
  projectionTiles = projectedTree->project(freeDims, localLimits);
  std::cout << "Filling structures" << std::endl;
  // Normalize and add color
  for (size_t tileId = 0; tileId < projectionTiles.size(); tileId++) {
    std::vector<Eigen::VectorXd>& projectedTile = projectionTiles[tileId];
    std::vector<Eigen::VectorXd> tileCornersPos;
    std::vector<Color> tileCornersColor;
    for (size_t cornerId = 0; cornerId < projectedTile.size(); cornerId++) {
      Eigen::VectorXd corner(3);
      // Rescale x,y values
      for (size_t freeDim = 0; freeDim < freeDims.size(); freeDim++) {
        double rawVal = projectedTile[cornerId](freeDim);
        corner(freeDim) = rescaleValue(rawVal, freeDims[freeDim]);
      }
      // Rescale z value (output)
      double rawOutput = projectedTile[cornerId](freeDims.size());
      double output = rescaleValue(rawOutput, inputDim);
      corner(2) = output;
      Color color;
      double altColor = 1 - 2 * abs(output - 0.5);
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
    cornersPos.push_back(tileCornersPos);
    cornersColor.push_back(tileCornersColor);
  }
}

void Viewer::navigate()
{
  //Shift-tab not handled in sfml 2.0
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::LControl)) {
    currentDim--;
    if (currentDim == -2) {
      currentDim = inputDim - 1;
    }
  }
  else {
    currentDim++;
    if ((size_t)currentDim == inputDim) {
      currentDim = -1;
    }
  }
}

void Viewer::drawTiles()
{
  glBegin(GL_QUADS);
  for (size_t tileId = 0; tileId < cornersPos.size(); tileId++) {
    const std::vector<Eigen::VectorXd>& points = cornersPos[tileId];
    const std::vector<Color>& colors = cornersColor[tileId];
    for (size_t pId : {0,1,3,2}) {//Ordering matters for opengl
      const Eigen::VectorXd& p = points[pId];
      const Color& c = colors[pId];
      glColor3f(c.r, c.g, c.b);
      glVertex3f(p(0), p(1), p(2));
    }
  }
  glEnd();
}

void Viewer::updateStatus()
{
  std::ostringstream oss;
  for (size_t dim = 0; dim < inputDim; dim++) {
    if (currentDim == (int) dim) {
      oss << "->";
    }
    else {
      oss << "  ";
    }
    if (locked[dim]) {
      oss << "Locked at " << lockValues[dim] << ": "; 
    }
    else {
      oss << "Free  : ";
    }
    oss << dimNames[dim];
    oss << " [" << limits(dim,0) << "," << limits(dim,1) << "]";
    oss << std::endl;
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
