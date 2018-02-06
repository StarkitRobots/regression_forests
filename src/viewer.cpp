#include "rhoban_regression_forests/tools/viewer.h"

#include <iostream>

int main(int argc, char ** argv)
{
  if (argc < 3) {
    std::ostringstream oss;
    oss << "Usage: " << argv[0] << " <forestFile> <configFile>";
    throw std::runtime_error(oss.str());
  }

  regression_forests::Viewer viewer(argv[1], argv[2],1920,1080);

  while(viewer.update()) {
  }

}
