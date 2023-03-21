/**
 * @file LineMandelCalculator.h
 * @author Vojtěch Bůbela <xbubel08@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date 14.11.2022
 */

#ifndef LINEMANDELCALCULATOR_H
#define LINEMANDELCALCULATOR_H

#include <BaseMandelCalculator.h>

class LineMandelCalculator : public BaseMandelCalculator
{
public:
    LineMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~LineMandelCalculator();
    int *calculateMandelbrot();

private:
    int* data;
    float* m_line;
    int* line_data;
};

#endif
