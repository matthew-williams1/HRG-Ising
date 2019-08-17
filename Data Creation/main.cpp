// Lattice.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <time.h>
#include <fstream>
#include <chrono>
#include <math.h>
#include "Lattice.h"

#define SAVEFOLDER = "/Users/alexbeaudin/Library/Caches/AppCode2019.2/DerivedData/Ising-gztwancwhhgzhcajhscmdygpsunw/Build/Products/Debug"

int main(int argc, const char * argv[]) {

	srand(time(NULL));

	int numData = 1000;
	unsigned long latticeSize = 40;

	std::vector<double> saveTemps{0.3, 0.65, 1.0, 1.3, 1.65, 2.0, 2.3, 2.65, 3.0, 3.3, 3.65, 4.0, 4.3, 4.65, 5.0};
	//Set up the files which will store the data.

    std::string data = "/Users/alexbeaudin/Documents/C++/Ising/train/data.csv";
    std::ofstream cooling (data);
    std::string labels = "/Users/alexbeaudin/Documents/C++/Ising/train/labels.csv";
    std::ofstream field (labels);

	for (int k = 0; k < numData; k++) {

		Lattice lattice(latticeSize, 5.0, saveTemps, true);


		while (lattice.getTemperature() > 0.03)
		{
			if (lattice.isSaveTemp())
			{
				lattice.saveLattice(cooling);
			}
			lattice.monteCarlo(1);
		}

		lattice.saveField(field);
	}

    std::string valData = "/Users/alexbeaudin/Documents/C++/Ising/validate/data.csv";
    std::ofstream valCooling (valData);
    std::string valLabels = "/Users/alexbeaudin/Documents/C++/Ising/validate/labels.csv";
    std::ofstream valField (valLabels);

	for (int k = 0; k < 0.2 * numData; k++) {

		Lattice lattice(latticeSize, 5.0, saveTemps, true);


		while (lattice.getTemperature() > 0.03)
		{
			if (lattice.isSaveTemp())
			{
				lattice.saveLattice(valCooling);
			}
			lattice.monteCarlo(1);
		}

		lattice.saveField(valField);
	}

	return 0;
}
