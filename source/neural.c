#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "neural.h"

int neural_init(neural_t *neural, int input_size, int hidden_size, int output_size)
{
	neural = malloc(sizeof(neural_t));
	neural->input_size = input_size;
	neural->hidden_size = hidden_size;
	neural->output_size = output_size;

	neural->weights_input_hidden = (double**)malloc(hidden_size * sizeof(double*));
	for(int i = 0; i < hidden_size; ++i)
	{
		neural->weights_input_hidden[i] = (double*)malloc(input_size * sizeof(double));
		for(int j = 0; j < input_size; ++j)
		{
			neural->weights_input_hidden[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
		}
	}

	neural->weights_hidden_output = (double**)malloc(output_size * sizeof(double*));
	for(int i = 0; i < output_size; ++i)
	{
		neural->weights_hidden_output[i] = (double*)malloc(hidden_size * sizeof(double));
		for(int j = 0; j < hidden_size; ++j)
		{
			neural->weights_hidden_output[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
		}
	}

	neural->bias_hidden = (double*)malloc(hidden_size * sizeof(double));
	neural->bias_output = (double*)malloc(output_size * sizeof(double));

	for(int i = 0; i < hidden_size; ++i)
	{
		neural->bias_hidden[i] = 0.0;
	}

	for (int i = 0; i < output_size; ++i)
	{
		neural->bias_output[i] = 0.0;
	}
}

void neural_free(neural_t* neural)
{
	for (int i = 0; i < neural->hidden_size; ++i)
	{
		free(neural->weights_input_hidden[i]);
	}

	free(neural->weights_input_hidden);

	for (int i = 0; i < neural->output_size; ++i)
	{
		free(neural->weights_hidden_output[i]);
	}

	free(neural->weights_hidden_output);

	free(neural->bias_hidden);
	free(neural->bias_output);
}