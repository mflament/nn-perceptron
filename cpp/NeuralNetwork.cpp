// neuralnetwork.cpp : Defines the entry point for the application.
//

#include "jni/org_yah_tests_perceptron_jni_NativeNeuralNetwork.h"
#include <iostream>
#include <math.h>
#include <algorithm>
#include "TrainingSamples.h"
#include "StreamBuffer.h"

double sigmoid(double v) {
	return 1.0 / (1.0 + exp(-v));
}

double sigmoid_prime(double v) {
	double s = sigmoid(v);
	return s * (1.0 - s);
}

double* newMatrix(int rows, int columns) {
	double* res = new double[rows * columns];
	memset(res, 0, rows * columns * sizeof(double));
	return res;
}

int maxIndex(double* values, int count) {
	int res = 0;
	for (int i = 1; i < count; i++) {
		if (values[i] > values[res])
			res = i;
	}
	return res;
}

class NeuralNetwork {
private:
	int layersCount = 0;
	int* layerSizes = 0;
	double** weights = 0;
	double** biases = 0;

	int capacity = 0;
	double** zs = 0;
	double** activations = 0;

	double** wgrads = 0;
	double** bgrads = 0;
public:
	NeuralNetwork(int, int*, double* state);
	~NeuralNetwork();

	inline int layers() const { return layersCount; }
	inline int features() const { return layerSizes[0]; }
	inline int outputs() const { return layerSizes[layersCount]; }
	inline int neurons(int layer) const { return layerSizes[layer + 1]; }
	inline int features(int layer) const { return layerSizes[layer]; }

	void propagate(const TrainingSamples& samples, int* outputs);
	double evaluate(const TrainingSamples& samples, int* outputs);
	void train(const TrainingSamples& samples, double learningRate);

private:
	void forward(TrainingBatch& batch);

	void ensureCapacity(int capacity);

	int indexOutputs(TrainingBatch& batch, int* outputs);

	void train(TrainingBatch& batch, double learningRate);

	void backward(int layer, const double* inputs, int samples);
	void updateNetwork(int layer, double lr);

};

NeuralNetwork::NeuralNetwork(int _layersCount, int* _layerSizes, double* state)
	: layersCount(_layersCount), layerSizes(_layerSizes) {
	weights = new double* [layersCount];
	biases = new double* [layersCount];
	zs = new double* [layersCount];
	activations = new double* [layersCount];
	wgrads = new double* [layersCount];
	bgrads = new double* [layersCount];

	int stateOffset = 0;
	for (int layer = 0; layer < layersCount; layer++)
	{
		weights[layer] = state + stateOffset;
		stateOffset += neurons(layer) * features(layer);

		wgrads[layer] = newMatrix(neurons(layer), features(layer));
		bgrads[layer] = newMatrix(neurons(layer), 1);
	}
	for (int layer = 0; layer < layersCount; layer++)
	{
		biases[layer] = state + stateOffset;
		stateOffset += neurons(layer);
	}
}

NeuralNetwork::~NeuralNetwork() {
	for (int layer = 0; layer < layers(); layer++)
	{
		delete[]zs[layer];
		delete[]activations[layer];
		delete[]wgrads[layer];
		delete[]bgrads[layer];
	}
	delete[]zs;
	delete[]activations;
	delete[]wgrads;
	delete[]bgrads;
	delete[]weights;
	delete[]biases;
}

void NeuralNetwork::ensureCapacity(int newCapacity) {
	if (capacity < newCapacity) {
		if (capacity) {
			for (int layer = 0; layer < layers(); layer++)
			{
				delete[]zs[layer];
				delete[]activations[layer];
			}
		}
		for (int layer = 0; layer < layers(); layer++)
		{
			zs[layer] = newMatrix(neurons(layer), newCapacity);
			activations[layer] = newMatrix(neurons(layer), newCapacity);
		}
	}
}


void NeuralNetwork::propagate(const TrainingSamples& samples, int* outputs) {
	TrainingBatch batch(samples);
	while (batch.hasNext()) {
		forward(batch);
		indexOutputs(batch, outputs);
		batch.next();
	}
}

double NeuralNetwork::evaluate(const TrainingSamples& samples, int* outputs) {
	TrainingBatch batch(samples);
	int matched = 0;
	while (batch.hasNext()) {
		forward(batch);
		matched += indexOutputs(batch, outputs);
		batch.next();
	}
	return matched / (double)samples.size;
}

void NeuralNetwork::train(const TrainingSamples& samples, double learningRate) {
	TrainingBatch batch(samples);
	while (batch.hasNext()) {
		train(batch, learningRate);
		batch.next();
	}
}

void NeuralNetwork::forward(TrainingBatch& batch) {
	ensureCapacity(batch.size);
	const double* inputs = batch.inputs();
	for (int layer = 0; layer < layers(); layer++)
	{
		// weight . inputs + bias
		int neurons = this->neurons(layer);
		int features = this->features(layer);
		const double* w = weights[layer];
		const double* b = biases[layer];
		double* z = zs[layer];
		double* a = activations[layer];
		for (int neuron = 0; neuron < neurons; neuron++) {
			for (int sample = 0; sample < batch.size; sample++) {
				double s = 0;
				for (int feature = 0; feature < features; feature++) {
					s += w[feature * neurons + neuron] * inputs[sample * features + feature];
				}
				s += b[neuron];
				z[sample * features + neuron] = s;
				a[sample * features + neuron] = sigmoid(s);
			}
		}
		inputs = a;
	}
}

void NeuralNetwork::train(TrainingBatch& batch, double learningRate) {
	// forward propagation
	forward(batch);

	//cost derivative
	double* a = activations[layersCount - 1];
	int neurons = outputs();
	for (int sample = 0; sample < batch.size; sample++)
	{
		for (int neuron = 0; neuron < neurons; neuron++) {
			a[sample * neurons + neuron] -= batch.expectedIndex(sample) == neuron ? 1 : 0;
		}
	}

	// back propagation
	int layer = layersCount - 1;
	for (layer = layersCount - 1; layer > 0; layer--)
	{
		backward(layer, activations[layer - 1], batch.size);
	}
	backward(0, batch.inputs(), batch.size);

	// model update
	double lr = learningRate / batch.size;
	for (int layer = 0; layer < layersCount; layer++)
	{
		updateNetwork(layer, lr);
	}
}

int NeuralNetwork::indexOutputs(TrainingBatch& batch, int* outputsIndices) {
	int matched = 0;
	double* networkOutputs = activations[layersCount - 1];
	int outputsCount = outputs();
	for (int sample = 0; sample < batch.size; sample++)
	{
		int outputIndex = maxIndex(networkOutputs - (size_t) sample * outputsCount, outputsCount);
		if (outputIndex == batch.expectedIndex(sample))
			matched++;
		if (outputsIndices)
			outputsIndices[batch.offset + sample] = outputIndex;
	}
	return matched;
}

void NeuralNetwork::backward(int layer, const double* inputs, int samples) {
	int neurons = this->neurons(layer);
	int features = this->features(layer);
	double* z = zs[layer];
	double* a = activations[layer];
	double* bgrad = bgrads[layer];
	double* wgrad = wgrads[layer];

	// delta = activation * sigmoid_prime(z) 
	for (int sample = 0; sample < samples; sample++) {
		for (int neuron = 0; neuron < neurons; neuron++) {
			int offset = sample * neurons + neuron;
			a[offset] *= sigmoid_prime(z[offset]);
		}
	}

	// bgrad = sum(activations[r])
	for (int neuron = 0; neuron < neurons; neuron++) {
		double s = 0;
		for (int sample = 0; sample < samples; sample++) {
			s += a[sample * neurons + neuron];
		}
		bgrad[neuron] = s;
	}
	
	// wgrad = a . T(inputs)
	for (int neuron = 0; neuron < neurons; neuron++) {
		for (int feature = 0; feature < features; feature++) {
			double s = 0;
			for (int sample = 0; sample < samples; sample++) {
				s += a[sample * neurons + neuron] * inputs[sample * features + neuron];
			}
			wgrad[feature * neurons + neuron] = s;
		}
	}

	if (layer > 0) {
		// activations[layer - 1] = T(weights[layer]) . a 
		double* nexta = activations[layer - 1];
		double* weight = weights[layer];
		for (int feature = 0; feature < features; feature++)
		{
			for (int sample = 0; sample < samples; sample++) {
				double s = 0;
				for (int neuron = 0; neuron < neurons; neuron++) {
					s += weight[feature * neurons + neuron] * a[sample * neurons + neuron];
				}
				nexta[feature * features + sample] = s;
			}
		}
	}
}

void NeuralNetwork::updateNetwork(int layer, double lr) {
	// w = w - (learningRate/batchSize) * wgrad
	int neurons = this->neurons(layer);
	int features = this->features(layer);
	double* weight = weights[layer];
	double* wgrad = wgrads[layer];
	for (int feature = 0; feature < features; feature++) {
		for (int neuron = 0; neuron < neurons; neuron++) {
			int offset = feature * neurons + neuron;
			weight[offset] -= lr * wgrad[offset];
		}
	}
	// b = b - (learningRate/batchSize) * bgrad
	double* bias = biases[layer];
	double* bgrad = bgrads[layer];
	for (int neuron = 0; neuron < neurons; neuron++) {
		bias[neuron] -= bgrad[neuron];
	}
}

/*
 * Class:     org_yah_tests_perceptron_jni_NativeNeuralNetwork
 * Method:    create
 * Signature: ([I)J
 */
JNIEXPORT jlong JNICALL Java_org_yah_tests_perceptron_jni_NativeNeuralNetwork_create(JNIEnv* env, jclass, jobject _stateBuffer) {
	StreamBuffer sb(env, _stateBuffer);
	int layersCount;
	if (!sb.next(layersCount))
		return 0;
	int* layerSizes;
	if (!sb.array(layerSizes, layersCount + 1))
		return 0;
	return (jlong) new NeuralNetwork(layersCount, layerSizes, (double*) sb.address());
}

/*
 * Class:     org_yah_tests_perceptron_jni_NativeNeuralNetwork
 * Method:    delete
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_yah_tests_perceptron_jni_NativeNeuralNetwork_delete(JNIEnv*, jobject, jlong networkReference) {
	delete ((NeuralNetwork*)networkReference);
}

JNIEXPORT void JNICALL Java_org_yah_tests_perceptron_jni_NativeNeuralNetwork_propagate(JNIEnv* env, jclass, jlong networkReference, jobject _samplesBuffer, jobject _outputsBuffer) {
	NeuralNetwork* network = (NeuralNetwork*)networkReference;
	TrainingSamples* samples = (TrainingSamples*)env->GetDirectBufferAddress(_samplesBuffer);
	int* outputs = (int*)env->GetDirectBufferAddress(_outputsBuffer);
	network->propagate(*samples, outputs);
}

JNIEXPORT jdouble JNICALL Java_org_yah_tests_perceptron_jni_NativeNeuralNetwork_evaluate(JNIEnv* env, jclass, jlong networkReference, jobject _samplesBuffer, jobject _outputsBuffer) {
	NeuralNetwork* network = (NeuralNetwork*)networkReference;
	TrainingSamples* samples = (TrainingSamples*)env->GetDirectBufferAddress(_samplesBuffer);
	int* outputs = _outputsBuffer ? (int*)env->GetDirectBufferAddress(_outputsBuffer) : 0;
	return network->evaluate(*samples, outputs);
}

/*
 * Class:     org_yah_tests_perceptron_jni_NativeNeuralNetwork
 * Method:    train
 * Signature: (JJD)V
 */
JNIEXPORT void JNICALL Java_org_yah_tests_perceptron_jni_NativeNeuralNetwork_train(JNIEnv* env, jclass, jlong networkReference, jobject _samplesBuffer, jdouble learningRate) {
	NeuralNetwork* network = (NeuralNetwork*)networkReference;
	TrainingSamples* samples = (TrainingSamples*)env->GetDirectBufferAddress(_samplesBuffer);
	network->train(*samples, learningRate);
}