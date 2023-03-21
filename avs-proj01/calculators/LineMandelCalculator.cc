/**
 * @file LineMandelCalculator.cc
 * @author Vojtěch Bůbela <xbubel08@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date 14.11.2022
 */
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdlib.h>

#include "LineMandelCalculator.h"

LineMandelCalculator::LineMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
	data = (int *)(aligned_alloc(64 ,height * width * sizeof(int)));
	m_line = (float *)(aligned_alloc(64, width * 2 * sizeof(float)));
}

LineMandelCalculator::~LineMandelCalculator() {
	free(data);
	data = NULL;
	free(m_line);
}

int * LineMandelCalculator::calculateMandelbrot () {
	int *pdata = data;
	float *p_matrix_line = m_line;
	int n_height = height / 2;
	const int n_width = width;
	const int n_limit = limit;

	#pragma omp simd
	for (int l = 0; l < width * height; l++) {
		pdata[l] = limit;
	}

	for (int i = 0; i < n_height; i++) {
		int cp_ptr = (height - i - 1) * width;
		float y_base = y_start + i * dy;
		const int d_width = 2 * width;

		int c = 0;
		#pragma omp simd reduction(+: c)
		for (int prep = 0; prep < d_width; prep = prep + 2) {
			p_matrix_line[prep] = y_base;
			p_matrix_line[prep + 1] = x_start + c * dx;
			c++;
 		}

		int offset = i * width;
		int write_counter = 0;
		for (int j = 0; j < n_limit; ++j) {
		
			int line_cnt = 0;
			#pragma omp simd reduction(+: line_cnt)
			for (int k = 0; k < n_width; k++) {
				float y = p_matrix_line[line_cnt];
				float x = p_matrix_line[line_cnt + 1];

				float r2 = x * x;
				float i2 = y * y;

				/*
			 	* When the limit for the point is found store it
			 	*/
				if (r2 + i2 > 4.0f && pdata[offset + k] == limit) {
					pdata[offset + k] = j;
				} else {
					p_matrix_line[line_cnt] = 2.0f * x * y + y_base;
					p_matrix_line[line_cnt + 1] = r2 - i2 + (x_start + k * dx); 
				}

				line_cnt = line_cnt + 2;
			}
		}

		#pragma omp simd 
		for (int r = 0; r < width; r++) {
			pdata[cp_ptr + r] = pdata[width * i + r];
		}
	}


	return data;
}
