//
// Created by Alex Beaudin on 2019-08-06.
// Copyright (c) 2019 ME. All rights reserved.
//

#include "Lattice.h"

std::vector<std::vector <float>> Lattice::makeField(unsigned long dimensions, bool constantStrength)
{
    //Initialize the value which the magnetic field will have.
    float strength;
    if (constantStrength)
        strength = 5.0 * (float) rand() / RAND_MAX;
    else
        strength = 5.0 * (float) rand() / RAND_MAX;

    std::vector <float> temp(dimensions, 0);
    std::vector <std::vector <float>> field(dimensions, temp); //The field which will be filled with values.

    if (constantStrength) {

        std::vector <std::vector <unsigned long>> active; // The active array, which will keep track of which
        // points can still grow.

        //Pick out an initial index to start the shape, and the value of the field at that index to 1.
        unsigned long xIndex = rand() % dimensions;
        unsigned long yIndex = rand() % dimensions;
        field[xIndex][yIndex] = strength;
        active.push_back(std::vector<unsigned long> {xIndex, yIndex});

        // Add points around the central point for some arbitrary number of iterations. Set probability of adjacent
        // values becoming active to some number which will decrease after each iteration.

        int chunkSize = dimensions / 3;
        int iterations =  rand() % chunkSize + chunkSize;
        float prob = 0.8;
        float probDelta = prob / (float) iterations;
        for (int iteration = 0; iteration < iterations; iteration++) {
            std::vector<std::vector<unsigned long>> nextActive;

            for (int index = 0; index < active.size(); index++) {
                //Check each tile around the indices in the Active array. If there is no field there, and
                // some random probability is satisfied, then add a field in that tile.
                if ((float) rand() / RAND_MAX < prob && !field[(active[index][0] - 1) % dimensions][active[index][1]]) {
                    field[(active[index][0] - 1) % dimensions][active[index][1]] = strength;
                    nextActive.push_back(std::vector<unsigned long>{(active[index][0] - 1) % dimensions, active[index][1]});
                }
                if ((float) rand() / RAND_MAX < prob && !field[(active[index][0] + 1) % dimensions][active[index][1]]) {
                    field[(active[index][0] + 1) % dimensions][active[index][1]] = strength;
                    nextActive.push_back(std::vector<unsigned long>{(active[index][0] + 1) % dimensions, active[index][1]});
                }
                if ((float) rand() / RAND_MAX < prob && !field[active[index][0]][(active[index][1] - 1) % dimensions]) {
                    field[(active[index][0])][(active[index][1] - 1) % dimensions] = strength;
                    nextActive.push_back(std::vector<unsigned long>{active[index][0], (active[index][1] - 1) % dimensions});
                }
                if ((float) rand() / RAND_MAX < prob && !field[active[index][0]][(active[index][1] + 1) % dimensions]) {
                    field[(active[index][0])][(active[index][1] + 1) % dimensions] = strength;
                    nextActive.push_back(std::vector<unsigned long>{active[index][0], (active[index][1] + 1) % dimensions});
                }
            }
            active = nextActive;
            prob -= probDelta;
        }
    }
    else {
        for (int col = 0; col < dimensions / 2; col++) {
            std::fill (field[col].begin(), field[col].end(), strength);
        }
    }

    return field;
}

Lattice::Lattice(unsigned long dimension, double temperature, std::vector<double> saveTemps, bool constantStrength)
{
    m_dimensions = dimension;
    m_temperature = temperature;
    m_field = makeField(dimension, constantStrength);
    m_saveTemps = saveTemps;

    for (int col = 0; col < m_dimensions; col++)
    {
        std::vector<int> temp;

        for (int row = 0; row < m_dimensions; row++)
        {
            temp.push_back(2 * (rand() % 2) - 1);
        }
        m_lattice.push_back(temp);
    }
}

void Lattice::print()
{
    for (int col = 0; col < m_dimensions; col++)
    {
        for (int row = 0; row < m_dimensions; row++)
        {
            std::cout << m_lattice[col][row] << ' ';
        }
        std::cout << '\n';
    }
}

void Lattice::monteCarlo(int steps)
{
    for (int i = 0; i < steps; i++)
    {
        for (int col = 0; col < m_dimensions; col++) {
            for (int row = 0; row < m_dimensions; row++) {
                //Calculate the sum of the spin states of adjacent particles
                int adjacent = m_lattice[col % m_dimensions][(row + m_dimensions - 1) % m_dimensions] + m_lattice[col % m_dimensions][(row + 1) % m_dimensions]
                               + m_lattice[(col + m_dimensions - 1) % m_dimensions][row % m_dimensions] + m_lattice[(col + 1) % m_dimensions][row % m_dimensions];

                //Calculate energy change. (E1 - E2)
                float delta = (float) m_lattice[col][row] * ((float) adjacent - m_field[col][row]);

                if (delta >= 0)
                    m_lattice[col][row] = -m_lattice[col][row];
                else if ((float) rand() / RAND_MAX < exp(2 * (float) delta / m_temperature))
                    m_lattice[col][row] = -m_lattice[col][row];
            }
        }

        if (m_temperature > 0.01)
            m_temperature -= 0.01;
    }
}

void Lattice::saveLattice(std::ofstream &file)
{
    for (int col = 0; col < m_dimensions; col++)
    {
        for (int row = 0; row < m_dimensions; row++)
        {
            file << m_lattice[col][row] << ',';
        }
        file << std::endl;
    }
}

void Lattice::saveField(std::ofstream &file)
{
    for (int col = 0; col < m_dimensions; col++)
    {
        for (int row = 0; row < m_dimensions; row++)
        {
            file << m_field[col][row] << ',';
        }
        file << std::endl;
    }
}

double Lattice::getTemperature()
{
    return m_temperature;
}

bool Lattice::isSaveTemp() {
    if (static_cast<float>(m_temperature) == static_cast<float>(m_saveTemps[m_saveTemps.size() - 1]))
    {
        m_saveTemps.pop_back();
        return true;
    }

    return false;
}
