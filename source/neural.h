#pragma once

#include <math.h>

static inline double sigmoid(double value)
{
	return 1.0 / (1.0 + exp(-value));
}

typedef struct
{
	int input_size;
	int hidden_size;
	int output_size;
	double** weights_input_hidden;
	double** weights_hidden_output;
	double* bias_hidden;
	double* bias_output;
} neural_t;

void neural_init(neural_t *neural, int input_size, int hidden_size, int output_size);
void neural_free(neural_t* neural);
void neural_feedforward(neural_t *neural, double* input, double* output);
void neural_train(neural_t* neural, double* input, double* target, double learning_rate);