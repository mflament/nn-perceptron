// neuralnetwork.cpp : Defines the entry point for the application.
//

#include "org_yah_tests_perceptron_jni_NativeNetwork.h"
#include <iostream>
#include <math.h>

void deleteMatrix(double*& ptr) {
	if (ptr) {
		delete ptr;
		ptr = NULL;
	}
}

class Network {
	int layerCount = 0;
	int* layerSizes = 0;
	double** weights = 0;
	double** biases = 0;

	int batchSize = 0;
	double** zs = NULL;
	double** activations = NULL;
	double** bgrads = NULL;
	double** wgrads = NULL;

public:

	/**
	* Network:
	*  int: layers (count)
	*  int: features (count)
	*  [int] x layers: layer sizes (last layer is outputs count)
	*  for each layer
	*	 double[neurons[layer]][features[layer]]: layer weights
	*	 double[neurons[layer]: layer biases
	*/
	Network(void* buffer) {
		int* intBuffer = (int*)buffer;
		layerCount = intBuffer[0];
		layerSizes = intBuffer + 1;
		weights = new double* [layerCount];
		biases = new double* [layerCount];
		double* pos = (double*)(intBuffer + 2 + layerCount);
		for (int layer = 0; layer < layerCount; layer++)
		{
			weights[layer] = pos;
			pos += (size_t) neurons(layer) * features(layer);
			biases[layer] = pos;
			pos += neurons(layer);
		}
	}

	~Network() {
		deleteMatrices(zs);
		deleteMatrices(activations);
		deleteMatrices(bgrads);
		deleteMatrices(wgrads);
		deleteArray(weights);
		deleteArray(biases);
		layerCount = 0;
		layerSizes = 0;
	}

	/*
	*
	* Batch:
	*  int: samples count
	*  double[features][samples] batch inputs
	*  double[outputs][samples] expected outputs
	*/
	void train(void* buffer, double learningRate) {
		double* inputs, * expected;
		int samples = readBatch(buffer, inputs, expected);
		double* activation = inputs;
		for (int layer = 0; layer < layerCount; layer++) {
			activation = forward(layer, activation);
		}

		cost_derivative(expected);
		for (int layer = layerCount - 1; layer > 0; layer--)
		{
			backward(layer, activations[layer - 1]);
		}
		backward(0, inputs);

		for (int layer = 0; layer < layerCount; layer++) {
			updateNetwork(layer, learningRate);
		}
	}
	
	void propagate(void* inputsBuffer, void* outputsBuffer) {
		double* inputs, * expected;
		int samples = readBatch(inputsBuffer, inputs, expected);
		double* activation = inputs;
		for (int layer = 0; layer < layerCount - 1; layer++) {
			activation = forward(layer, activation);
		}
		forward(layerCount - 1, activation, (double*)outputsBuffer);
	}

	int readBatch(void* buffer, double*& inputs, double*& expecteds) {
		int samples = *((int*)buffer);
		if (samples == 0) return 0;
		inputs = (double*)((int*)buffer + 1);
		expecteds = inputs + (size_t)features() * samples;
		updateBatchSize(samples);
		return samples;
	}

	inline int features() {
		return layerSizes[0];
	}

	inline int features(int layer) {
		return layerSizes[layer];
	}

	inline int neurons(int layer) {
		return layerSizes[layer + 1];
	}

	inline int outputs() {
		return layerSizes[layerCount];
	}

	inline double weight(int layer, int neuron, int feature) {
		return *(weights[layer] + (size_t)neuron * features(layer) + feature);
	}

	inline double bias(int layer, int neuron) {
		return *(biases[layer] + neuron);
	}

private:
	double* forward(int layer, double* inputs) {
		inputs = propagate(layer, inputs);
		return activate(layer, inputs, activations[layer]);
	}

	double* forward(int layer, double* inputs, double* outputs) {
		inputs = propagate(layer, inputs);
		return activate(layer, inputs, outputs);
	}

	double* propagate(int layer, double* inputs) {
		int rows = neurons(layer);
		int cols = batchSize;
		int inputsRows = features(layer);
		double* output = zs[layer];
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				*output = 0;
				for (int ir = 0; ir < inputsRows; ir++) {
					*output += weight(layer, row, ir) * inputs[ir * batchSize + col];
				}
				output++;
			}
		}
		return zs[layer];
	}

	double* activate(int layer, double* inputs, double* outputs) {
		int count = neurons(layer) * batchSize;
		for (int i = 0; i < count; i++) {
			outputs[i] = sigmoid(zs[layer][i]);
		}
		return outputs;
	}

	void backward(int layer, double* inputs) {
		int rows = neurons(layer);
		int cols = batchSize;
		int index = 0;
		for (int r = 0; r < rows; r++) {
			bgrads[layer][r] = 0;
			for (int c = 0; c < cols; c++) {
				activations[layer][index] *= sigmoid_prime(zs[layer][index]);
				bgrads[layer][r] += activations[layer][index];
				index++;
			}
		}

		size_t size = (size_t)rows * features(layer) * sizeof(double);
		memset(wgrads[layer], 0, size);
		double* wgrad = wgrads[layer];
		// delta . T(inputs)
		double* activation = activations[layer];
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				for (int ir = 0; ir < rows; ir++) {
					wgrad[ir] += *activation * inputs[ir * batchSize + c];
				}
				activation++;
			}
			wgrad += cols;
		}

		if (layer > 0) {
			// delta = T(W[layer+1]) . delta
			double* nextActivation = activations[layer - 1];
			size = (size_t)neurons(layer - 1) * batchSize * sizeof(double);
			memset(nextActivation, 0, size);

			double* weight = weights[layer];
			cols = features(layer);
			for (int c = 0; c < cols; c++) {
				index = 0;
				for (int r = 0; r < rows; r++) {
					for (int dc = 0; dc < batchSize; dc++) {
						// activation[c][dc] += m[r][c] * delta[r][dc];
						nextActivation[dc] += weight[r * cols + c] * activations[layer][index++];
					}
				}
				nextActivation += batchSize;
			}
		}
	}

	void updateNetwork(int layer, double learningRate) {
		int rows = neurons(layer);
		int cols = features(layer);
		double lr = learningRate / batchSize;
		int index = 0;
		for (int r = 0; r < rows; r++) {
			biases[layer][r] -= lr * bgrads[layer][r];
			for (int c = 0; c < cols; c++) {
				weights[layer][index] -= lr * wgrads[layer][index];
				index++;
			}
		}
	}

	void cost_derivative(double* expected) {
		int count = outputs() * batchSize;
		double* activation = activations[layerCount - 1];
		for (size_t i = 0; i < count; i++)
		{
			activation[i] -= expected[i];
		}
	}

	inline double sigmoid(double v) {
		return  1.0 / (1.0 + exp(-v));
	}

	inline double sigmoid_prime(double v) {
		double sv = sigmoid(v);
		return sv * (1.0 - sv);
	}

	void deleteMatrices(double**& matrices) {
		if (!matrices)
			return;
		for (size_t layer = 0; layer < layerCount; layer++)
		{
			delete matrices[layer];
		}
		deleteArray(matrices);
	}

	void deleteArray(double**& array) {
		if (!array)
			return;
		delete[]array;
		array = NULL;
	}

	void updateBatchSize(int samples) {
		if (batchSize != samples) {
			deleteMatrices(zs);
			deleteMatrices(activations);
			zs = new double* [layerCount];
			activations = new double* [layerCount];
			for (int layer = 0; layer < layerCount; layer++)
			{
				size_t size = (size_t)neurons(layer) * samples;
				zs[layer] = new double[size];
				activations[layer] = new double[size];
				bgrads[layer] = new double[neurons(layer)];
				wgrads[layer] = new double[(size_t)neurons(layer) * features(layer)];
			}
			batchSize = samples;
		}
	}

};

thread_local Network* network = 0;

void throwError(JNIEnv* env, char* message) {
	env->ThrowNew(env->FindClass("java/lang/Exception"), message);
}

/*
	* Class:     org_yah_tests_perceptron_jni_NativeNetwork
	* Method:    network
	* Signature: (Ljava/nio/ByteBuffer;)J
	*/
JNIEXPORT void JNICALL Java_org_yah_tests_perceptron_jni_NativeNetwork_network(JNIEnv* env, jclass, jobject byteBuffer) {
	if (network) {
		delete network;
		network = 0;
	}
	void* buffer = env->GetDirectBufferAddress(byteBuffer);
	network = new Network(buffer);
}

/*
 * Class:     org_yah_tests_perceptron_jni_NativeNetwork
 * Method:    train
 * Signature: (Ljava/nio/ByteBuffer;D)V
 */
JNIEXPORT void JNICALL Java_org_yah_tests_perceptron_jni_NativeNetwork_train(JNIEnv* env, jclass, jobject byteBuffer, jdouble learningRate) {
	if (!network) throwError(env, "Network not set");
	void* buffer = env->GetDirectBufferAddress(byteBuffer);
	network->train(buffer, learningRate);
}

/*
 * Class:     org_yah_tests_perceptron_jni_NativeNetwork
 * Method:    propagate
 * Signature: (Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_org_yah_tests_perceptron_jni_NativeNetwork_propagate(JNIEnv* env, jclass, jobject inputsByteBuffer, jobject outputsByteBuffer) {
	if (!network) throwError(env, "Network not set");
	void* inputsBuffer = env->GetDirectBufferAddress(inputsByteBuffer);
	void* outputsBuffer = env->GetDirectBufferAddress(outputsByteBuffer);
	network->propagate(inputsBuffer, outputsBuffer);
}