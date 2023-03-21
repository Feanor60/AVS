/**
 * @file BatchMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date DATE
 */

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <stdexcept>

#include "BatchMandelCalculator.h"

#define batch_len 64

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
	data = (int *)(aligned_alloc(64, width * height * sizeof(int)));
	batch_data = (float *)(aligned_alloc(64, batch_len * 2 * sizeof(float)));
}

BatchMandelCalculator::~BatchMandelCalculator() {
	free(data);
	data = NULL;
	free(batch_data);
}

int * BatchMandelCalculator::calculateMandelbrot () {
	const int n_height = height / 2;
	int *p_data = data;
	float *p_batch_data = batch_data;

	#pragma omp simd
	for (int prep = 0; prep < width * height; prep++) {
			p_data[prep] = limit;
	}

	for (int i = 0; i < n_height; i++) {
		int cp_ptr = (height - i - 1) * width;
		float y_base = y_start + i * dy;
		

		/*
		 * for each segment of line - @batch_len floats long
		 */
		for (int j = 0; j < width; j = j + batch_len) {
			int c = j;
			int offset = i * width + j;

			#pragma omp simd
			for (int prep = 0; prep < batch_len * 2; prep = prep + 2) {
				p_batch_data[prep] = x_start + c * dx;
				p_batch_data[prep + 1] = y_base;
				c++;
			}

			/*
			 * For limit
			 */
			for (int k = 0; k < limit; ++k) {
			int counter = 0;
			int line_ptr = j;	
				/*
				 * For each pixel in a batch 
				 */
				#pragma omp simd reduction(+: counter)
				for (int l = 0; l < batch_len; l++) {
					float x = batch_data[counter];
					float y = batch_data[counter + 1];
					
					float r2 = x * x;
					float i2 = y * y;

					if (r2 + i2 > 4.0f && p_data[offset + l] == limit){
						p_data[offset + l] = k;
					} else {
						batch_data[counter] = r2 - i2 + (x_start + line_ptr * dx);
						batch_data[counter + 1] = 2.0f * x * y + y_base; 
					}
					counter = counter + 2;
					line_ptr++;
				}
			}
		}

		#pragma omp simd
		for (int r = 0; r < width; r++) {
			p_data[cp_ptr + r] = p_data[width * i + r];
		}

	}
	return data;
}
