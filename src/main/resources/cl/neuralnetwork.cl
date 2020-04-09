// #DEFINE LAYERS x (layers count)
// #DEFINE TYPE (float|double)

typedef TYPE T;

/****************************** Activation ******************************/
const float sigmoid(const T v) {
	return 1.0 / (1.0 + exp(-v));
}

const float sigmoid_prime(const T v) {
	T s = sigmoid(v);
	return s * (1.0 - s);
}

/****************************** Network  ******************************/
typedef struct Network {
	int inputs;
	int layers[LAYERS];
	int totalWeights;
	int totalNeurons;
} Network;


/****************************** Layer  ******************************/

typedef struct Layer {
	const Network* network;
	int index;
	bool training;
	int neurons;
	int features;

	global const T* weights; // [neurons x features]
	global const T* biases; // [neurons]

	local T* partial; // [maxNeurons x maxFeatures]
	local int* partialIndices; // [maxNeurons]

	local T* activations; // [neurons]
	local T* zs; // [neurons]

	global T* wgrads; // [neurons x features]
	global T* bgrads; // [neurons]
} Layer;

int weights_stride(Layer* layer) {
	return layer->neurons * layer->features + layer->neurons;
}

void init_layer(const Network* network, 
	global const T* data,
	local T* partial, local int* partialIndices,
	Layer* layer) {
	layer->network = network;
	layer->index = 0;
	layer->training = false;
	layer->features = network->inputs;
	layer->neurons = network->layers[0];
	layer->weights = data;
	layer->biases = data + layer->neurons * layer->features;
	layer->partial = partial;
	layer->partialIndices = partialIndices;
	layer->activations = layer->zs = 0;
	layer->wgrads = layer->bgrads = 0;
}

void init_training_layer(const Network* network, 
	global const T* data,
	local T* partial, local T* activations, local T* zs,
	global T* grads,
	Layer* layer) {
	layer->network = network;
	layer->index = 0;
	layer->training = true;
	layer->features = network->inputs;
	layer->neurons = network->layers[0];
	layer->weights = data;
	layer->partial = partial;
	layer->partialIndices = 0;
	layer->activations = activations;
	layer->zs = zs;

	int sample = get_group_id(1);
	layer->wgrads = grads + sample * (network->totalWeights + network->totalNeurons);

	int weights_cnt = layer->neurons * layer->features;
	layer->biases = layer->weights + weights_cnt;
	layer->bgrads = layer->wgrads + weights_cnt;
}

void next_layer(Layer* layer) {
	int stride = weights_stride(layer);
	layer->weights += stride;
	layer->activations += layer->neurons;
	layer->zs += layer->neurons;
	layer->wgrads += stride;

	layer->index++;
	layer->features = layer->network->layers[layer->index - 1];
	layer->neurons = layer->network->layers[layer->index];

	int weights_cnt = layer->neurons * layer->features;
	layer->biases = layer->weights + weights_cnt;
	layer->bgrads = layer->wgrads + weights_cnt;
}

void prev_layer(Layer* layer) {
	layer->index--;
	layer->features = layer->network->layers[layer->index - 1];
	layer->neurons = layer->network->layers[layer->index];

	int stride = weights_stride(layer);
	layer->weights -= stride;
	layer->activations -= layer->neurons;
	layer->zs -= layer->neurons;
	layer->wgrads -= stride;

	int weights_cnt = layer->neurons * layer->features;
	layer->biases = layer->weights + weights_cnt;
	layer->bgrads = layer->wgrads + weights_cnt;
}

bool in_layer(const Layer* layer, int neuron, int feature) {
	return neuron < layer->neurons && feature < layer->features;
}

T layer_weight(const Layer* layer, int neuron, int feature) {
	return in_layer(layer, neuron, feature) ? layer->weights[feature * layer->neurons + neuron] : 0;
}

local T get_layer_partial(const Layer* layer, int neuron, int feature) {
	return layer->partial[feature * get_local_size(0) + neuron];
}

local void set_layer_partial(const Layer* layer, int neuron, int feature, T v) {
	layer->partial[feature * get_local_size(0) + neuron] = v;
}

local void add_layer_partial(const Layer* layer, int neuron, int feature, T v) {
	layer->partial[feature * get_local_size(0) + neuron] += v;
}

void set_layer_wgrad(const Layer* layer, int neuron, int feature, T value) {
	layer->wgrads[feature * layer->neurons + neuron] = value; 
}

T get_layer_wgrad(const Layer* layer, int neuron, int feature)  {
	return layer->wgrads[feature * layer->neurons + neuron];
}

T sum_rows(const Layer* layer) {
	// partial[neuron][0] = sum(partials[neuron][f])
	int maxNeurons = get_local_size(0);
	int maxFeatures = get_local_size(1);
	int neuron = get_local_id(0);
	int feature = get_local_id(1);
	for(unsigned int s = 1; s < maxFeatures; s *= 2) {
		int f = 2 * s * feature;
		if (f < maxFeatures) {
			add_layer_partial(layer, neuron, f, get_layer_partial(layer, neuron, f + s));
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	return get_layer_partial(layer, neuron, 0);
}

T sum_columns(const Layer* layer) {
	// partial[0][feature] = sum(partials[n][feature])
	int maxNeurons = get_local_size(0);
	int maxFeatures = get_local_size(1);
	int neuron = get_local_id(0);
	int feature = get_local_id(1);
	for(unsigned int s = 1; s < maxNeurons; s *= 2) {
		int n = 2 * s * neuron;
		if (n < maxNeurons) {
			add_layer_partial(layer, n, feature, get_layer_partial(layer, n +s, feature));
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	return get_layer_partial(layer, 0, feature);
}

/****************************** Forward  ******************************/

T layer_forward(const Layer* layer, const T input) {
	int neuron = get_local_id(0);	
	int feature = get_local_id(1);

	// partial = weight . input
	T weight = layer_weight(layer, neuron, feature);
	int sample = get_group_id(1);
	set_layer_partial(layer, neuron, feature, weight * input);
	barrier(CLK_LOCAL_MEM_FENCE);

	T sum = sum_rows(layer);
	if (neuron < layer->neurons && feature == 0) {
		T z = sum + layer->biases[neuron];
		T a = sigmoid(z); 
		set_layer_partial(layer, neuron, 0, a);
		if (layer->training) {
			layer->zs[neuron] = z;
			layer->activations[neuron] = a;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	return feature < layer->neurons ? get_layer_partial(layer, feature, 0) : 0;
}

int index_layer_outputs(Layer* layer) {
  int neuron = get_local_id(0);	
	int feature = get_local_id(1);

	int outputs = layer->network->layers[LAYERS - 1];
	layer->partialIndices[neuron] = neuron < outputs ? neuron : -1;
	barrier(CLK_LOCAL_MEM_FENCE);

	int maxNeurons = get_local_size(0);
	// max output resolution
 	for(unsigned int s = 1; s < maxNeurons; s *= 2) {
			int n = 2 * s * neuron;
			if (feature == 0 && n < maxNeurons) {
				int index = layer->partialIndices[n];
				int otherIndex = layer->partialIndices[n + s];
				if (index >= 0 && otherIndex >= 0) {
					if (get_layer_partial(layer, index, 0) < get_layer_partial(layer, otherIndex, 0)) {
							layer->partialIndices[n] = otherIndex;
					}
				}
	 		}
	    barrier(CLK_LOCAL_MEM_FENCE);
	}
	return layer->partialIndices[0]; 
}

kernel void propagate(Network network, global const T* data,
					int samplesOffset, global const T* samples, global const int* expectedIndices,
					global int* outputIndices, global int* matchedCount,
					local T* partial, local int* partialIndices) {
	int neuron = get_local_id(1);
	int feature = get_local_id(1);

	Layer layer;
	init_layer(&network, data, partial, partialIndices, &layer);

	int batchSampleIndex = get_group_id(1);
	int globalSampleIndex = samplesOffset + batchSampleIndex;
	T sample = samples[globalSampleIndex * network.inputs + feature];
	T input = sample;
	for (; layer.index < LAYERS; next_layer(&layer)) {
		input = layer_forward(&layer, input);
	}

	int outputIndex = index_layer_outputs(&layer);
	if (neuron == 0 && feature == 0) {
		if (outputIndices)
			outputIndices[batchSampleIndex] = outputIndex;
		if (expectedIndices) {
			int expected = expectedIndices[globalSampleIndex];
			matchedCount[batchSampleIndex] = outputIndex == expected ? 1 : 0;
		}
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

kernel void train(Network network, global const T* data,
									int samplesOffset, global const T* samples, global const int* expectedIndices,
									global T* grads,
									local T* partial, local T* zs, local T* activations) {
	int neuron = get_local_id(0);
	int feature = get_local_id(1);

	Layer layer;
	init_training_layer(&network, data, 
		partial, activations, zs, 
		grads, 
		&layer);

	int batchSampleIndex = get_group_id(1);
	int globalSampleIndex = samplesOffset + batchSampleIndex;
	local T* activation_inputs;
	T sample ,input , activation, temp;
	input = sample = samples[globalSampleIndex * network.inputs + feature];
	for (; layer.index < LAYERS; next_layer(&layer)) {
		input = layer_forward(&layer, input);
	}
	prev_layer(&layer);
	
	if (neuron < layer.neurons && feature == 0) {
			activation = layer.activations[neuron];
			// cost derivative = actual - expected = activation - (1 or 0 depending of expected index)
			T expectedValue = expectedIndices[globalSampleIndex] == neuron ? 1.0 : 0.0;
			layer.activations[neuron] -= expectedValue;
	}
  barrier(CLK_LOCAL_MEM_FENCE);
	
	// backward propagation
	for (; layer.index >= 0; prev_layer(&layer)) {
		// activation = activation * sigmoid_prime(z)
		if (neuron < layer.neurons && feature == 0) {
			activation = layer.activations[neuron];
			activation *= sigmoid_prime(layer.zs[neuron]);
			layer.activations[neuron] = activation;
			layer.bgrads[neuron] = activation; 
		}
	  barrier(CLK_LOCAL_MEM_FENCE);
		activation = layer.activations[neuron];

		// wgrad = activation . T(inputs) 
		if (neuron < layer.neurons && feature < layer.features) {
			temp = layer.index > 0 ? (layer.activations - layer.features)[feature] : sample;
			set_layer_wgrad(&layer, neuron, feature, activation * temp);
		}

		if (layer.index > 0) { // prepare next activation
			// activation[layer-1] (next inputs) = T(weight[layer]) . activation
			temp = layer_weight(&layer, neuron, feature) * activation;
			set_layer_partial(&layer, neuron, feature, temp);
			barrier(CLK_LOCAL_MEM_FENCE);
			temp = sum_columns(&layer);
			if (neuron == 0 && feature < layer.features)
				(layer.activations - layer.features)[feature] = temp;
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}
}

kernel void sum_grads(Network network,
				const int samplesCount,
				global T* grads,
				local T* partial) {
	int sample = get_global_id(0);
	int row = get_global_id(1);
	int group_id = get_group_id(0);
	int group_size = get_local_size(0);
	int lid = get_local_id(0);

	int gradIndex = sample * (network.totalWeights + network.totalNeurons) + row;
	if (sample < samplesCount) {
		partial[lid] = grads[gradIndex];
	} else {
		partial[lid] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for(unsigned int s = 1; s < group_size; s *= 2) {
		int index = 2 * s * lid;
		if (index < group_size) {
			partial[index] += partial[index + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (lid == 0) {
		grads[gradIndex] = partial[0];
	}
}

kernel void update_network(Network network,
				const T lr,
				global T* data,
				global const T* grads) {
  int index = get_global_id(0);
	data[index] -= grads[index] * lr;
}
