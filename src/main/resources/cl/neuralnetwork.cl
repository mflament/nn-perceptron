void print(int rows, int cols, int offset, __global const float* m) {
	for (int r=0 ; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			printf("%.3f ", m[(offset + c) * rows + r]);
		}
		printf("\n");
	}
}

const float sigmoid(const float v) {
	return 1.0 / (1.0 + exp(-v));
}

/*
 * z = weight . input + bias
 * activation = sigmoid(z)
 */
__kernel void forward(const int neurons, 
											const int features, 
											__global const float* weights,
										  __global const float* biases,
											const int inputsOffset,
											__global const float* inputs,
											__global float* z,
											__global float* activation) {
	const int tc = get_global_id(0); // sample 
	const int tr = get_global_id(1); // neuron

	float sum = 0;
	for (int f = 0; f < features; f++) {
		sum += weights[f * neurons + tr] * inputs[(inputsOffset + tc) * features + f];
	}
	sum += biases[tr];
	const int offset = tc * neurons + tr;
	z[offset] = sum;
	activation[offset] = sigmoid(sum);

#ifdef DEBUG
	if (tc == 0 && tr == 0) {
		printf("--------------forward--------------\n");
		printf("weights\n");
		print(neurons, features, 0, weights);
		printf("biases\n");
		print(neurons, 1, 0, biases);
		printf("inputs\n");
		print(neurons, 4, inputsOffset, inputs);
		printf("z\n");
		print(neurons, 4, 0, z);
		printf("activation\n");
		print(neurons, 4, 0, activation);
	}
#endif	
}

/**
 * indices[tc] = index of max value activation[c]
 */
__kernel void indices(const int outputs, 
											__global const float* activation, 
											const int indicesOffset,
										  __global int* indices) {
	const int tc = get_global_id(0); // sample
	const int colOffset = tc * outputs;
	int maxIndex = 0;
	for (int r = 1; r < outputs; r++) {
		if (activation[colOffset + r] > activation[colOffset + maxIndex]) {
			maxIndex = r;
		}
	}
	indices[indicesOffset + tc] = maxIndex;
}

/**
 * activation = activation - expected
 */
__kernel void cost(const int neurons, 
								  __global float* activation,
									const int expectedOffset,
									__global const float* expected) {
	const int tc = get_global_id(0); // sample
	const int tr = get_global_id(1); // neuron (network output)
	activation[tc * neurons + tr] -= expected[(expectedOffset + tc) * neurons + tr];

	#ifdef DEBUG
		barrier(CLK_GLOBAL_MEM_FENCE );
		if (tc == 0 && tr == 0) {
			printf("--------------cost(%d)--------------\n", neurons);
			printf("expected\n");
			print(neurons, 4, expectedOffset, expected);
			printf("activation\n");
			print(neurons, 4, 0, activation);
		}
	#endif	
}

/**
 * activation = activation * sigmoid_prime(z)
 */
__kernel void sigmoid_prime(const int neurons,
											 			__global const float* z,
											 			__global float* activation) {
	const int tc = get_global_id(0);
	const int tr = get_global_id(1);
	const int offset = tc * neurons + tr;
	float s = sigmoid(z[offset]);
	activation[offset] *= s * (1.0 - s);

	#ifdef DEBUG
		barrier(CLK_GLOBAL_MEM_FENCE );
		if (tc == 0 && tr == 0) {
			printf("--------------sigmoid_prime(%d)--------------\n", neurons);

			printf("activation\n");
			print(neurons, 4, 0, activation);
		}
	#endif	
}

/**
 * bgrad = sum(activation)
 * wgrad = activation . T(inputs)
 */
__kernel void dot_transpose(const int neurons,	
														const int features,	
														const int samples,	
														__global float* activation,
														const int inputsOffset,
														__global float* inputs,
														__global float* wgrads,
														__global float* bgrads) {
	const int tc = get_global_id(0); // feature
	const int tr = get_global_id(1); // neuron

	float wgrad = 0;
	float bgrad = 0;
	for (int c = 0; c < samples; c++) {
		bgrad += activation[c * neurons + tr];
		wgrad += activation[c * neurons + tr] * inputs[(inputsOffset + c) * features + tc];
	}
	wgrads[tc * neurons + tr] = wgrad;
	bgrads[tr] = bgrad;
	#ifdef DEBUG
		barrier(CLK_GLOBAL_MEM_FENCE );
		if (tc == 0 && tr == 0) {
			printf("--------------dot_transpose(%d,%d,%d)--------------\n", neurons, features, samples);

			printf("activation\n");
			print(neurons, samples, 0, activation);

			printf("inputs(%d)\n",  inputsOffset);
			print(neurons, samples, inputsOffset, inputs);

			printf("wgrads\n");
			print(neurons, features, 0, wgrads);

			printf("bgrads\n");
			print(neurons, 1, 0, bgrads);
		}
	#endif	
}

/**
 * res = T(weights) . activation
 */
__kernel void transpose_dot(const int neurons,	
														const int features,	
													  __global const float* weights,
														__global const float* activation,
														__global float* res) {
	const int tc = get_global_id(0); // sample
	const int tr = get_global_id(1); // neuron[layer-1] = feature[layer]
	float sum = 0;	
	for (int r = 0; r < neurons; r++) {
		sum += weights[tr * neurons + r] * activation[tc * neurons + r];
	}
	res[tc * features + tr] = sum;

	#ifdef DEBUG
	barrier(CLK_GLOBAL_MEM_FENCE );
	if (tc == 0 && tr == 0) {
			printf("--------------transpose_dot(%d,%d)--------------\n", neurons, features);

			printf("res\n");
			print(features, 4, 0, res);
	}
	#endif	
}

/**
 *  w = w - lr * wgrad
 *  b = b - lr * bgrad
 *  lr = (learningRate/batchSize)
 */
__kernel void update_layer(const int neurons,
													 const int samples,
													__global const float* wgrads,
													__global const float* bgrads,
													__global float* weights,
													__global float* biases,
													const float lr) {
	const int tc = get_global_id(0); // feature
	const int tr = get_global_id(1); // neuron
	weights[tc * neurons + tr] -= lr * wgrads[tc * neurons + tr];
	if (tc == 0) {
		biases[tr] -= lr * bgrads[tr];
	}
}
