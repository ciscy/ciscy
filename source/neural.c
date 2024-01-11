#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "neural.h"

/**
 *	@brief Initialize a neural network structure.
 *	@param neural The pointer to the neural network structure.
 *	@param input_size The amount of inputs for the neural network.
 *	@param hidden_size The amount of neurons in the hidden layer.
 *	@param output_size The amount of outputs for the neural network.
 */
void neural_init(neural_t *neural, int input_size, int hidden_size, int output_size)
{
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

/**
 *	@brief De-initialize a neural network structure.
 *	@param neural The neural network structure to de-initialize.
 */
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

/**
 *	@brief Perform a feedforward pass through the neural network.
 *	@param neural Pointer to the neural network structure.
 *	@param input Array of input values.
 *	@param output Array to store the output values after the feedforward pass.
 */
void neural_feedforward(neural_t *neural, double* input, double* output)
{
	double* hidden_layer = (double*)malloc(neural->hidden_size * sizeof(double));

	for(int i = 0; i < neural->hidden_size; ++i)
	{
		hidden_layer[i] = 0.0;

		for(int j = 0; j < neural->input_size; ++j)
		{
			hidden_layer[i] += neural->weights_input_hidden[i][j] * input[j];
		}

		hidden_layer[i] += neural->bias_hidden[i];
		hidden_layer[i] = tanh(hidden_layer[i]);
	}

	for(int i = 0; i < neural->output_size; ++i)
	{
		output[i] = 0.0;

		for(int j = 0; j < neural->hidden_size; ++j)
		{
			output[i] += neural->weights_hidden_output[i][j] * hidden_layer[j];
		}

		output[i] += neural->bias_output[i];
		output[i] = 1.0 / (1.0 + exp(-output[i]));
	}

	free(hidden_layer);
}

/**
 *	@brief Train the neural network using backpropagation.
 *	@param neural Pointer to the neural network structure.
 *	@param input Array of input values.
 *	@param target Array of target values for training.
 *	@param learning_rate The learning rate for adjusting weights during training.
 */
void neural_train(neural_t *neural, double* input, double* target, double learning_rate)
{
	double* output = (double*)malloc(neural->output_size * sizeof(double));
	neural_feedforward(neural, input, output);

	double* output_errors = (double*)malloc(neural->output_size * sizeof(double));
	double* output_deltas = (double*)malloc(neural->output_size * sizeof(double));
	for (int i = 0; i < neural->output_size; ++i)
	{
		output_errors[i] = target[i] - output[i];
		output_deltas[i] = output_errors[i] * output[i] * (1.0 - output[i]);
	}

	for (int i = 0; i < neural->output_size; ++i)
	{
		for (int j = 0; j < neural->hidden_size; ++j)
		{
			neural->weights_hidden_output[i][j] += learning_rate * output_deltas[i] * output[j];
		}
		neural->bias_output[i] += learning_rate * output_deltas[i];
	}

	double* hidden_errors = (double*)malloc(neural->hidden_size * sizeof(double));
	double* hidden_deltas = (double*)malloc(neural->hidden_size * sizeof(double));
	for (int i = 0; i < neural->hidden_size; ++i)
	{
		hidden_errors[i] = 0.0;
		for (int j = 0; j < neural->output_size; ++j)
		{
			hidden_errors[i] += output_deltas[j] * neural->weights_hidden_output[j][i];
		}
		hidden_deltas[i] = hidden_errors[i] * (output[i] > 0 ? 1 : 0);
	}

	for (int i = 0; i < neural->hidden_size; ++i)
	{
		for (int j = 0; j < neural->input_size; ++j)
		{
			neural->weights_input_hidden[i][j] += learning_rate * hidden_deltas[i] * input[j];
		}
		neural->bias_hidden[i] += learning_rate * hidden_deltas[i];
	}

	free(output);
	free(output_errors);
	free(output_deltas);
	free(hidden_errors);
	free(hidden_deltas);
}