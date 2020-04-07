// #DEFINE LAYERS x (layers count)
// #DEFINE TYPE (float|double)

typedef TYPE T;

typedef struct Network {
	int inputs;
	int layers[LAYERS];
	int totalWeights;
	int totalNeurons;
} Network;

int outputs(const Network* n) {
	return n->layers[LAYERS - 1];
}

const float sigmoid(const T v) {
	return 1.0 / (1.0 + exp(-v));
}

const float sigmoid_prime(const T v) {
	T s = sigmoid(v);
	return s * (1.0 - s);
}

T layer_forward(const int layer, const int neurons, const int features, const T input,
						 global const T** weight, global const T** bias,
						 local T* partial, local T** z, local T** activation) {
	int neuron = get_local_id(0);	
	int feature = get_local_id(1);
	int maxNeurons = get_local_size(0);
	int maxFeatures = get_local_size(1);
	int group_id = get_group_id(1);

	int partialOffset = feature * maxNeurons + neuron;
	// partial = weight * input
	if (neuron < neurons && feature < features) {
		T w = (*weight)[feature * neurons + neuron];
		partial[partialOffset] = input * w;
		if (group_id == 0)
			printf("weight;%d;%d;%d;%.3f\n", layer, neuron, feature, w);
		printf("input;%d;%d;%d;%d;%.3f\n", layer, group_id, neuron, feature, input);
		printf("partial;%d;%d;%d;%d;%.3f\n", layer, group_id, neuron, feature, partial[partialOffset]);
	} else {
		partial[partialOffset] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// partial[neuron][0] = sum(partials[neuron])
	for(unsigned int s = 1; s < maxFeatures; s *= 2) {
		int f = 2 * s * feature;
		if (f < maxFeatures) {
			partial[f * maxNeurons + neuron] += partial[(f + s) * maxNeurons + neuron];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// partial[neuron][0] = sigmoid(partials[neuron][0] + biases[layer][neuron])
	if (feature == 0) {
		T dot = partial[neuron] + (*bias)[neuron];
		T sig = sigmoid(dot);
		partial[neuron] = sig;
		if (activation) { 
			(*z)[neuron] = dot; 
			(*activation)[neuron] = sig;
		}
		printf("activation;%d;%d;%d;%.3f\n", layer, neuron, group_id, partial[neuron]);
	}
	*weight += features * neurons;
	*bias += neurons;
	if (activation) {
		*z += neurons;
		*activation += neurons;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	return partial[feature];
}

void forward(const Network* network, 
						 global const T* weights,
						 global const T* biases,
						 int samplesOffset, global const T* samples,
						 local T* partial) {
	int feature = get_local_id(1);
	int groupId = get_group_id(1);

	int sampleIndex = (samplesOffset + groupId) * network->inputs + feature;
	T input = samples[sampleIndex];
	global T* weight = weights;
	global T* bias = biases;
	int features = network->inputs, neurons;
	for (int layer = 0; layer < LAYERS; layer++) {
		neurons = network->layers[layer];
		input = layer_forward(layer, neurons, features, input, &weight, &bias, partial, NULL, NULL);
		int neuron = get_local_id(0);
		features = neurons;
	}
}

void maxIndex(int outputs, local T* partial,  local int* partialIndices) {
  int neuron = get_local_id(0);	
	int feature = get_local_id(1);

	partialIndices[neuron] = neuron < outputs ? neuron : -1;
	barrier(CLK_LOCAL_MEM_FENCE);

	int maxNeurons = get_local_size(0);
	// max output resolution
 	for(unsigned int s = 1; s < maxNeurons; s *= 2) {
			int n = 2 * s * neuron;
			if (feature == 0 && n < maxNeurons) {
				int index = partialIndices[n];
				int otherIndex = partialIndices[n + s];
				if (index >= 0 && otherIndex >= 0) {
					if (partial[index] < partial[otherIndex]) {
							partialIndices[n] = otherIndex;
					}
				}
	 		}
	    barrier(CLK_LOCAL_MEM_FENCE);
	}
}

void	countMatched(int samplesOffset,
									 global const int* expectedIndices,
									 local int* partialIndices, 
									 global int* matchedCount) {
	int groupId = get_group_id(1);
	int actual = partialIndices[0];
	int expected = expectedIndices[samplesOffset + groupId];
	matchedCount[groupId] = actual == expected ? 1 : 0;
}

/**
 * layerSizes: layer sizes [0] is network inputs, [1 .. LAYERS-1] neurons per layer
 * weights: weights (column major) matrix [feature][neuron] per layer
 * biases: biases per neuron vector per layer
 * samplesOffset: offset of sample in samples
 * samples: vector of samplesCount samples (size = layerSizes[0])
 * outputIndices: vector of int receiving sample max output index (size must be samplesCount)
 * partial: storage used for suming dot product, size must be max(neurons) * max(features) (and max(features) % 2 == 0)
 * partialIndices: int vector used for max sample output resolution (size must be network outputs)  
 */
kernel void propagate(Network network, 
											global const T* weights,
											global const T* biases,
											int samplesOffset, 
											global const T* samples,
											global int* outputIndices,
											local T* partial, local int* partialIndices) {
	forward(&network, weights, biases, samplesOffset, samples, partial);
	maxIndex(outputs(&network), partial, partialIndices);
	if (get_local_id(0) == 0 && get_local_id(1) == 0) {
		outputIndices[get_group_id(1)] = partialIndices[0];
	}
}

/**
 * Network: the network
 * weights: weights matrix [neuron][feature] per layer
 * biases: biases per neuron vector per layer
 * samplesOffset: offset of sample in samples
 * samples: vector of samplesCount samples (size = layerSizes[0])
 * result: vector of matched count 
 * partial: storage used for suming dot product, size must be max(neurons) * max(features) (and max(features) % 2 == 0)
 * activation: results of last layer, size must be max(neurons)
 * partialIndices: int vector used for max sample output resolution (size must be network outputs)  
 */
kernel void evaluate0(Network network, 
											global const T* weights,
											global const T* biases,
											int samplesOffset, 
											global const T* samples,
											global const int* expectedIndices,
											global int* matchedCount,
											local T* partial, local int* partialIndices) {
	forward(&network, weights, biases, samplesOffset, samples, partial);
	maxIndex(outputs(&network), partial, partialIndices);
	if (get_local_id(0) == 0 && get_local_id(1) == 0) {
		countMatched(samplesOffset, expectedIndices, partialIndices, matchedCount);
	}
}

kernel void evaluate1(Network network, 
											global const T* weights,
											global const T* biases,
											int samplesOffset, 
											global const T* samples,
											global const int* expectedIndices,
											global int* matchedCount,
											global int* outputIndices,
											local T* partial, local int* partialIndices) {
	forward(&network, weights, biases, samplesOffset, samples, partial);
	maxIndex(outputs(&network), partial, partialIndices);
	if (get_local_id(0) == 0 && get_local_id(1) == 0) {
		countMatched(samplesOffset, expectedIndices, partialIndices, matchedCount);
		outputIndices[get_group_id(1)] = partialIndices[0];
	}
}

kernel void sum_matched(const int size,
												global int* matchedCount,
												local int* partial) {
  int local_id = get_local_id(0);
	int local_size = get_local_size(0);
	int index = get_global_id(0);
	partial[local_id] = index < size ? matchedCount[index] : 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	int max = index + size - index;
	for(unsigned int s = 1; s < local_size; s *= 2) {
		int f = 2 * s * local_id;
		if (f < local_size && (f+s)<max) {
			partial[f] += partial[f + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (local_id == 0) {
		int groupId = get_group_id(0);
		matchedCount[groupId] = partial[0];
	}
}	


/**
 * Network: the network
 * weights: weights matrix [neuron][feature] per layer
 * biases: biases per neuron vector per layer
 * samplesOffset: offset of sample in samples and expectedIndices
 * samples: vector of samplesCount samples (stride = layerSizes[0])
 * expectedIndices: vector of expected output indices per sample
 * wgrads: array of matrices [neurons][features] per layer per sample. (size = batch samples count = get_num_group)
 * bgrads: array of vectors [neurons] per layer per sample.
 * partial: storage used for suming dot product, size must be max(neurons) * max(features) (and max(features) % 2 == 0)
 * zs: storage used for keeping forward weigth . input (size must be network.totalNeurons)
 * activations: storage used for keeping forward activations (size must be network.totalNeurons)
 */
kernel void train(Network network, 
									global const T* weights,
									global const T* biases,
									int samplesOffset,
									global const T* samples,
									global const int* expectedIndices,
									global T* wgrads,
									global T* bgrads,
									local T* partial, local T* zs, local T* activations) {
  int neuron = get_local_id(0);	
	int feature = get_local_id(1);
	int maxNeurons = get_local_size(0);
	int maxFeatures = get_local_size(1);
	int groupId = get_group_id(1);
	
	global const T* weight = weights;
	global const T* bias = biases;

	local T* activation = activations;
	local T* z = zs;

	int sampleIndex = (samplesOffset + groupId) * network.inputs + feature;
	T sampleInput = samples[sampleIndex];
	T input = sampleInput;
	// forward 
	int layer, neurons, features = network.inputs;
	for (layer = 0; layer < LAYERS; layer++) {
		neurons = network.layers[layer];
		input = layer_forward(layer, neurons, features, input, &weight, &bias, partial, &z, &activation);
		features = neurons;
	}

	//move back to last layer
	layer--;
	features = network.layers[layer - 1];
	weight -= neurons * features;
	activation -= neurons;
	z -= neurons;

	// cost derivative = actual - expected = activation - (1 or 0 depending of expected index)
	if (neuron < neurons && feature == 0) {
		T expectedValue = expectedIndices[samplesOffset + groupId] == neuron ? 1.0 : 0.0;
		activation[neuron] -= expectedValue;
	}
  barrier(CLK_LOCAL_MEM_FENCE);

	if (feature < features) {
		input = (activation - features)[feature];
	} else {
		input = 0;
	}

	global T* wgrad = wgrads + (groupId + 1) * network.totalWeights - neurons * features;
	global T* bgrad = bgrads + (groupId + 1) * network.totalNeurons - neurons;

	// backward propagation
	for (; layer >= 0; --layer) {
		features = network.layers[layer - 1];

		// activation = activation * sigmoid_prime(z)
		if (neuron < neurons && feature == 0) {
			activation[neuron] *= sigmoid_prime(z[neuron]);
			bgrad[neuron] = activation[neuron];
		}
	  barrier(CLK_LOCAL_MEM_FENCE);


		// wgrad = activation . T(inputs) 
		if (neuron < neurons && feature < features) {
			wgrad[feature * neurons + neuron] = input * activation[neuron];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		if (layer > 0) {
			// activation[layer-1] (next inputs) = T(weight[layer]) . activation
			int partialOffset = feature * maxNeurons + neuron;
			if (feature < neurons && neuron < features) {
				partial[partialOffset] = weight[neuron * neurons + feature] * activation[feature];
			} else{
				partial[partialOffset] = 0;
			}

			// sum partials columns (features)
			for(unsigned int s = 1; s < maxFeatures; s *= 2) {
				int f = 2 * s * feature;
				if (f < features) {
					partial[f * maxNeurons + neuron] += partial[(f + s) * maxNeurons + neuron];
				}
				barrier(CLK_LOCAL_MEM_FENCE);
			}

			local T* inputs = activation - neurons;
			if (feature == 0) {
				inputs[neuron] = partial[neuron];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			if (layer > 1)
				input = feature < features ? (inputs - features)[feature] : 0;
			else 
				input = feature < network.inputs ? samples[(samplesOffset + groupId) * network.inputs + feature] : 0;
		}

		weight -= neurons * features;
		wgrad -= neurons * features;
		bgrad -= neurons;
		activation -= neurons;
		z -= neurons;

		neurons = features;
	}
}

kernel void sum_grads(Network network,
				const int samplesCount,
				global T* wgrads,
				global T* bgrads,
				local T* wpartial, local T* bpartial) {
	int sample = get_global_id(0);
	int neuron = get_global_id(1);
	int group_id = get_group_id(0);
	int group_size = get_local_size(0);
	int lid = get_local_id(0);

	//printf("neuron: %d , sample %d , group_id: %d, lid: %d\n", neuron, sample, group_id, lid);

	if (sample < samplesCount) {
		wpartial[lid] = wgrads[sample * network.totalWeights + neuron];
		if (neuron < network.totalNeurons) {
			bpartial[lid] = bgrads[sample * network.totalNeurons + neuron];
		} else
			bpartial[lid] = 0;
	} else {
		wpartial[lid] = bpartial[lid] = 0;
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	for(unsigned int s = 1; s < group_size; s *= 2) {
		int index = 2 * s * lid;
		if (index < group_size) {
			wpartial[index] += wpartial[index + s];
			bpartial[index] += bpartial[index + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (lid == 0) {
		wgrads[group_id * network.totalWeights + neuron] = wpartial[0];
		printf("wgrads;%d;%.3f\n", group_id * network.totalWeights + neuron, wgrads[group_id * network.totalWeights + neuron]);
		if (neuron < network.totalNeurons) {
			bgrads[group_id * network.totalNeurons + neuron] = bpartial[0];
			printf("bgrads;%d;%.3f\n", group_id * network.totalNeurons + neuron, bgrads[group_id * network.totalNeurons + neuron]);
		}
	}
}

kernel void update_network(Network network,
				const T lr,
				global T* weights,
				global T* biases,
				global const T* wgrads,
				global const T* bgrads) {
  int neuron = get_global_id(0);

	T t = weights[neuron];
	T g = wgrads[neuron];
	weights[neuron] = t - lr * g;
	if (neuron < network.totalNeurons) {
		t = biases[neuron];
		g = bgrads[neuron];
		biases[neuron] = t - lr * g;
	}
}