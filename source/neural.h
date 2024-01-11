#pragma once

#include <math.h>

static inline double sigmoid(double value)
{
	return 1.0 / (1.0 + exp(-value));
}