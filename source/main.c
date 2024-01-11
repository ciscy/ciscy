#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "neural.h"

#if defined(_WIN32) || defined(_WIN64)
#define ENDL "\r\n"
#elif defined(__unix__) || defined(__unix) || defined(__linux__)
#define ENDL "\n"
#elif defined(__APPLE__)
#define ENDL "\n"
#endif

#define length(array) (sizeof(array) / sizeof((array)[0]))

int main(int argc, const char *argv[])
{
	double xor_inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	double xor_targets[] = {0, 1, 1, 0};

	const int input_size = 2;
	const int hidden_size = 2;
	const int output_size = 1;
	const double learning_rate = 0.01;
	const int epochs = 1;

	neural_t neural;
	neural_init(&neural, input_size, hidden_size, output_size);

	for(int epoch = 0; epoch < epochs; ++epoch)
	{
		for(int i = 0; i < length(xor_targets); ++i)
		{
			neural_train(&neural, xor_inputs[i], &xor_targets[i], learning_rate);
		}
	}

	printf("Testing XOR neural network:" ENDL);
	for (int i = 0; i < length(xor_targets); ++i)
	{
		double output;
		neural_feedforward(&neural, xor_inputs[i], &output);
		printf("Input: %d %d, Output: %lf, Target: %lf" ENDL, (int)xor_inputs[i][0], (int)xor_inputs[i][1], output, xor_targets[i]);
	}

	neural_free(&neural);

	return 0;
}