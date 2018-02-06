#include "rhoban_regression_forests/approximations/gp_approximation.h"
#include "rhoban_regression_forests/approximations/pwc_approximation.h"
#include "rhoban_regression_forests/approximations/pwl_approximation.h"

#include "rhoban_regression_forests/approximations/approximation_factory.h"

#include <iostream>

using namespace regression_forests;

void testApproximations()
{
  int nb_cells = 5;
  int nb_cells2 = nb_cells * nb_cells;
  std::vector<Eigen::VectorXd> inputs;
  std::vector<double> observations;
  // Creating sample from x - 2y = z
  int sample = 0;
  for (int x = 0; x < nb_cells; x++) {
    for (int y = 0; y < nb_cells; y++) {
      Eigen::VectorXd input(2);
      input << x , y;
      double z = x - 2 * y;
      inputs.push_back(input);
      observations.push_back(z);
      sample++;
    }
  }
  std::vector<std::unique_ptr<Approximation>> approximations;
  approximations.push_back(std::unique_ptr<Approximation>(new GPApproximation(inputs,
                                                                              observations)));
  approximations.push_back(std::unique_ptr<Approximation>(new PWCApproximation(-3.5)));
  approximations.push_back(std::unique_ptr<Approximation>(new PWLApproximation(inputs,
                                                                               observations)));
  ApproximationFactory approximation_factory;
  for (int i = 0; i < approximations.size(); i++)
  {
    std::cout << "-----------------------------------------" << std::endl
              << "Testing approximation " << i << std::endl;
    
    std::string filename("/tmp/approximation.bin");
    std::string copy_filename("/tmp/copy_approximation.bin");

    std::unique_ptr<Approximation> copy;

    int original_bytes_written = approximations[i]->save(filename);
    std::cout << "Original bytes written: " << original_bytes_written << std::endl;

    // Reading from original
    int original_bytes_read = approximation_factory.loadFromFile(filename, copy);
    std::cout << "Original bytes read   : " << original_bytes_read    << std::endl;

    // writing copy
    int copy_bytes_written = copy->save(copy_filename);
    std::cout << "Copy bytes written    : " << copy_bytes_written     << std::endl;

    // Used to test point
    Eigen::VectorXd test_input(2);
    test_input << 1.5, 2.5;

    double original_prediction, copy_prediction;
    original_prediction = approximations[i]->eval(test_input);
    copy_prediction = copy->eval(test_input);

    // Outputting some messages:
    std::cout << "For test input: " << test_input.transpose() << std::endl
              << "\toriginal prediction : " << original_prediction << std::endl
              << "\tcopy prediction     : " << copy_prediction     << std::endl;
  }
}

int main()
{
  testApproximations();
}
