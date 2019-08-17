//
// Created by Alex Beaudin on 2019-08-06.
// Copyright (c) 2019 ME. All rights reserved.
//

#ifndef ISING_LATTICE_H
#define ISING_LATTICE_H

#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>

class Lattice {
private:
    unsigned long m_dimensions;
    double m_temperature;
    std::vector <std::vector <int>> m_lattice;
    std::vector <std::vector <float>> m_field;
    std::vector <double> m_saveTemps;

    bool m_constant;
    std::vector<std::vector <float>> makeField(unsigned long dimensions, bool constantStrength);
public:

    Lattice(unsigned long dimension = 128, double temperature = 5.0, std::vector<double> saveTemps = {0.5, 1.0, 1.5, 2.0}, bool constantStrength = false);

    void print();

    void monteCarlo(int steps);

    void saveLattice(std::ofstream &file);

    void saveField(std::ofstream &file);

    double getTemperature();

    bool isSaveTemp();
};

#endif //ISING_LATTICE_H
