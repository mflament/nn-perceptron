// neuralnetwork.cpp : Defines the entry point for the application.
//

#include "jni/org_yah_tests_perceptron_jni_NativeNeuralNetwork.h"
#include <iostream>
#include <math.h>
#include <random>
#include "Matrix.h"
#include "TrainingSamples.h"


std::default_random_engine generator;
std::normal_distribution<double> distribution(0, 1.0);

double sigmoid(double v) {
	return 1.0 / (1.0 + exp(-v));
}

double sigmoid_prime(double v) {
	double s = sigmoid(v);
	return s * (1.0 - s);
}

void randomize(Matrix& m, double s) {
	for (int c = 0; c < m.columns; c++) {
		double* col = m.column(c);
		for (int r = 0; r < m.rows; r++) {
			col[r] = distribution(generator) * s;
		}
	}
}

class TrainingContext;

class NeuralNetwork {
private:
	int layersCount = 0;
	int* layerSizes = 0;
	Matrix* weights = 0;
	Matrix* biases = 0;

	TrainingContext* trainingContext;

public:
	NeuralNetwork(int, int*);
	~NeuralNetwork();

	inline int layers() const { return layersCount; }
	inline int features() const { return layerSizes[0]; }
	inline int outputs() const { return layerSizes[layersCount]; }
	inline int neurons(int layer) const { return layerSizes[layer + 1]; }
	inline int features(int layer) const { return layerSizes[layer]; }

	inline Matrix* weight(int layer) { return weights + layer; }
	inline Matrix* bias(int layer) { return biases + layer; }

	void propagate(TrainingSamples* samples, int* outputs) const;
	double evaluate(TrainingSamples* samples, int* outputs) const;
	void train(TrainingSamples* samples, double learningRate);
};

class TrainingContext {
private:
	NeuralNetwork* network;
	int batchSize = 0;
	Matrix* zs = 0;
	Matrix* activations = 0;

	Matrix* wgrads = 0;
	Matrix* bgrads = 0;
public:
	TrainingContext(NeuralNetwork* network);
	~TrainingContext();

	inline int layers() const { return network->layers(); }
	inline Matrix* activation(int layer) {
		if (layer < 0) return activations + network->layers() + layer;
		return activations + layer;
	}
	void setBatchSize(int);

	void forward(Matrix& inputs);
	void forward(int layer, Matrix& inputs);
	void backward(int layer, Matrix& inputs);
	void updateNetwork(int layer, double learningRate);

	void getOutputIndices(int* outputs) const;
	int getOutputIndices(Matrix& expecteds, int* outputs) const;
};

NeuralNetwork::NeuralNetwork(int _layersCount, int* _layerSizes) : layersCount(_layersCount) {
	layerSizes = _layerSizes;
	weights = new Matrix[layersCount];
	biases = new Matrix[layersCount];

	for (int layer = 0; layer < layersCount; layer++) {
		int rows = neurons(layer);
		int columns = features(layer);
		weights[layer].create(rows, columns);
		randomize(weights[layer], columns);
		biases[layer].create(rows, 1);
	}
	trainingContext = new TrainingContext(this);
}

NeuralNetwork::~NeuralNetwork() {
	if (layersCount) {
		for (int layer = 0; layer < layersCount; layer++) {
			weights[layer].free();
			biases[layer].free();
		}
		delete trainingContext;
		trainingContext = 0;
		delete[]weights;
		delete[]biases;
		delete[]layerSizes;
		weights = biases = 0;
		layerSizes = 0;
		layersCount = 0;
	}
}

void NeuralNetwork::propagate(TrainingSamples* samples, int* outputs) const {
	int batchSize;
	samples->rewind();
	int size = samples->size();
	for (int batchOffset = 0; batchOffset < size; batchOffset += batchSize)
	{
		batchSize = samples->inputs.slide(batchOffset, samples->batchSize);
		trainingContext->setBatchSize(batchSize);
		trainingContext->forward(samples->inputs);
		trainingContext->getOutputIndices(outputs);
	}
}

double NeuralNetwork::evaluate(TrainingSamples* samples, int* outputs) const {
	int batchSize;
	double total = 0;
	int matched = 0;
	samples->rewind();
	int size = samples->size();
	for (int batchOffset = 0; batchOffset < size; batchOffset += batchSize)
	{
		batchSize = samples->slide(batchOffset, samples->batchSize);
		trainingContext->setBatchSize(batchSize);
		trainingContext->forward(samples->inputs);
		matched += trainingContext->getOutputIndices(samples->expectedIndices, outputs);
	}
	return matched / (double)size;
}

void NeuralNetwork::train(TrainingSamples* samples, double learningRate) {
	int batchSize;
	samples->rewind();
	int size = samples->size();
	for (int batchOffset = 0; batchOffset < size; batchOffset += batchSize)
	{
		batchSize = samples->slide(batchOffset, samples->batchSize);
		trainingContext->setBatchSize(batchSize);

		// forward propagation
		trainingContext->forward(samples->inputs);

		//cost derivative
		trainingContext->activation(-1)->sub(samples->expectedOutputs);

		// back propagation
		trainingContext->backward(layersCount - 1, *trainingContext->activation(-2));
		for (int layer = layersCount - 2; layer > 0; layer--)
		{
			trainingContext->backward(layer, *trainingContext->activation(layer - 1));
		}
		trainingContext->backward(0, samples->inputs);

		// model update
		double lr = learningRate / batchSize;
		for (int layer = 0; layer < layersCount; layer++)
		{
			trainingContext->updateNetwork(layer, lr);
		}
	}
}


TrainingContext::TrainingContext(NeuralNetwork* _network) {
	network = _network;
	int layers = network->layers();
	zs = new Matrix[layers];
	activations = new Matrix[layers];
	wgrads = new Matrix[layers];
	bgrads = new Matrix[layers];
	for (int layer = 0; layer < layers; layer++)
	{
		wgrads[layer].create(network->neurons(layer), network->features(layer));
		bgrads[layer].create(network->neurons(layer), 1);
	}
}

TrainingContext::~TrainingContext() {
	for (int layer = 0; layer < layers(); layer++)
	{
		zs[layer].free();
		activations[layer].free();
		wgrads[layer].free();
		bgrads[layer].free();
	}
	delete[]zs;
	delete[]activations;
	delete[]wgrads;
	delete[]bgrads;
	network = 0;
}

void TrainingContext::setBatchSize(int _batchSize) {
	if (batchSize != _batchSize) {
		if (batchSize < _batchSize) {
			for (int layer = 0; layer < layers(); layer++)
			{
				int neurons = network->neurons(layer);
				zs[layer].free();
				zs[layer].create(neurons, _batchSize);
				activations[layer].free();
				activations[layer].create(neurons, _batchSize);
			}
		}
		else {
			for (int layer = 0; layer < layers(); layer++)
			{
				zs[layer].columns = _batchSize;
				activations[layer].columns = _batchSize;
			}
		}
		batchSize = _batchSize;
	}
}

void TrainingContext::forward(Matrix& inputs) {
	forward(0, inputs);
	for (int layer = 1; layer < layers(); layer++)
	{
		forward(layer, activations[layer - 1]);
	}
}

void TrainingContext::forward(int layer, Matrix& inputs) {
	// weight . inputs + bias
	Matrix* weight = network->weight(layer);
	Matrix* bias = network->bias(layer);
	for (int tr = 0; tr < zs[layer].rows; tr++) {
		for (int tc = 0; tc < zs[layer].columns; tc++) {
			double s = 0;
			double* icol = inputs.column(tc);
			for (int c = 0; c < weight->columns; c++) {
				s += weight->get(tr, c) * icol[c];
			}
			s += bias->get(tr, 0);
			zs[layer].set(tr, tc, s);
			activations[layer].set(tr, tc, sigmoid(s));
		}
	}
}

void TrainingContext::backward(int layer, Matrix& inputs) {
	Matrix z = zs[layer];
	Matrix a = activations[layer];
	Matrix bgrad = bgrads[layer];
	Matrix wgrad = wgrads[layer];

	// delta = activation * sigmoid_prime(z) , bgrad = sum(activations[r])
	bgrad.zero();
	for (int c = 0; c < z.columns; c++)
	{
		double* zcolumn = z.column(c);
		double* acolumn = a.column(c);
		for (int r = 0; r < z.rows; r++)
		{
			acolumn[r] *= sigmoid_prime(zcolumn[r]);
			bgrad.data[r] += acolumn[r];
		}
	}

	// wgrad = delta . T(inputs)
	for (int r = 0; r < wgrad.rows; r++)
	{
		for (int c = 0; c < wgrad.columns; c++) {
			double s = 0;
			for (int dc = 0; dc < a.columns; dc++) {
				s += a.get(r, dc) * inputs.get(c, dc);
			}
			wgrad.set(r, c, s);
		}
	}

	if (layer > 0) {
		// delta[layer-1] = T(W[layer]) . delta 
		Matrix* next = activation(layer - 1);
		Matrix* weight = network->weight(layer);
		for (int r = 0; r < next->rows; r++)
		{
			double* wcol = weight->column(r);
			for (int c = 0; c < next->columns; c++) {
				double s = 0;
				double* dcol = a.column(c);
				for (int wr = 0; wr < weight->rows; wr++) {
					s += wcol[wr] * dcol[wr];
				}
				next->set(r, c, s);
			}
		}
	}
}

void TrainingContext::updateNetwork(int layer, double lr) {
	// w = w - (learningRate/batchSize) * wgrad
	wgrads[layer].mul(lr);
	network->weight(layer)->sub(wgrads[layer]);
	// b = b - (learningRate/batchSize) * bgrad
	bgrads[layer].mul(lr);
	network->bias(layer)->sub(bgrads[layer]);
}

void TrainingContext::getOutputIndices(int* outputs) const {
	Matrix* activation = activations + network->layers() - 1;
	for (int s = 0; s < batchSize; s++)
	{
		int index = activation->maxRowIndex(s);
		outputs[s] = index;
	}
}

int TrainingContext::getOutputIndices(Matrix& expecteds, int* outputs) const {
	Matrix* activation = activations + layers() - 1;
	int matched = 0;
	for (int s = 0; s < batchSize; s++)
	{
		int index = activation->maxRowIndex(s);
		if (expecteds.get(0, s) == index) matched++;
		if (outputs) outputs[s] = index;
	}
	return matched;
}

/*
 * Class:     org_yah_tests_perceptron_jni_NativeNeuralNetwork
 * Method:    create
 * Signature: ([I)J
 */
 //JNIEXPORT jlong JNICALL Java_org_yah_tests_perceptron_jni_NativeNeuralNetwork_create(JNIEnv*, jclass, jintArray);
JNIEXPORT jlong JNICALL Java_org_yah_tests_perceptron_jni_NativeNeuralNetwork_create(JNIEnv* env, jclass, jintArray _layerSizes) {
	int sizeCount = env->GetArrayLength(_layerSizes);
	if (sizeCount < 2) return 0;

	int* layerSizes = new int[sizeCount];
	env->GetIntArrayRegion(_layerSizes, 0, sizeCount, (jint*)layerSizes);
	return (jlong)new NeuralNetwork(sizeCount - 1, layerSizes);
}

/*
 * Class:     org_yah_tests_perceptron_jni_NativeNeuralNetwork
 * Method:    delete
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_yah_tests_perceptron_jni_NativeNeuralNetwork_delete(JNIEnv*, jobject, jlong networkReference) {
	delete ((NeuralNetwork*)networkReference);
}

/*
 * Class:     org_yah_tests_perceptron_jni_NativeNeuralNetwork
 * Method:    layers
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_yah_tests_perceptron_jni_NativeNeuralNetwork_layers(JNIEnv*, jclass, jlong networkReference) {
	return ((NeuralNetwork*)networkReference)->layers();
}


/*
 * Class:     org_yah_tests_perceptron_jni_NativeNeuralNetwork
 * Method:    features
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_yah_tests_perceptron_jni_NativeNeuralNetwork_features__J(JNIEnv*, jclass, jlong networkReference) {
	return ((NeuralNetwork*)networkReference)->features();
}


/*
 * Class:     org_yah_tests_perceptron_jni_NativeNeuralNetwork
 * Method:    outputs
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_yah_tests_perceptron_jni_NativeNeuralNetwork_outputs(JNIEnv*, jclass, jlong networkReference) {
	return ((NeuralNetwork*)networkReference)->outputs();
}


/*
 * Class:     org_yah_tests_perceptron_jni_NativeNeuralNetwork
 * Method:    features
 * Signature: (JI)I
 */
JNIEXPORT jint JNICALL Java_org_yah_tests_perceptron_jni_NativeNeuralNetwork_features__JI(JNIEnv*, jclass, jlong networkReference, jint layer) {
	return ((NeuralNetwork*)networkReference)->features(layer);
}


/*
 * Class:     org_yah_tests_perceptron_jni_NativeNeuralNetwork
 * Method:    neurons
 * Signature: (JI)I
 */
JNIEXPORT jint JNICALL Java_org_yah_tests_perceptron_jni_NativeNeuralNetwork_neurons(JNIEnv*, jclass, jlong networkReference, jint layer) {
	return ((NeuralNetwork*)networkReference)->neurons(layer);
}

/*
 * Class:     org_yah_tests_perceptron_jni_NativeNeuralNetwork
 * Method:    propagate
 * Signature: (JJLjava/nio/IntBuffer;)V
 */
JNIEXPORT void JNICALL Java_org_yah_tests_perceptron_jni_NativeNeuralNetwork_propagate(JNIEnv* env, jclass, jlong networkReference, jlong samplesReference, jobject javaOutputs) {
	NeuralNetwork* network = (NeuralNetwork*)networkReference;
	TrainingSamples* samples = (TrainingSamples*)samplesReference;
	int* outputs = (int*)env->GetDirectBufferAddress(javaOutputs);
	network->propagate(samples, outputs);
}


/*
 * Class:     org_yah_tests_perceptron_jni_NativeNeuralNetwork
 * Method:    evaluate
 * Signature: (JJLjava/nio/IntBuffer;)D
 */
JNIEXPORT jdouble JNICALL Java_org_yah_tests_perceptron_jni_NativeNeuralNetwork_evaluate(JNIEnv* env, jclass, jlong networkReference, jlong samplesReference, jobject javaOutputs) {
	NeuralNetwork* network = (NeuralNetwork*)networkReference;
	TrainingSamples* samples = (TrainingSamples*)samplesReference;
	int* outputs = javaOutputs ? (int*)env->GetDirectBufferAddress(javaOutputs) : 0;
	return network->evaluate(samples, outputs);
}


/*
 * Class:     org_yah_tests_perceptron_jni_NativeNeuralNetwork
 * Method:    train
 * Signature: (JJD)V
 */
JNIEXPORT void JNICALL Java_org_yah_tests_perceptron_jni_NativeNeuralNetwork_train(JNIEnv*, jclass, jlong networkReference, jlong samplesReference, jdouble learningRate) {
	NeuralNetwork* network = (NeuralNetwork*)networkReference;
	TrainingSamples* samples = (TrainingSamples*)samplesReference;
	network->train(samples, learningRate);
}

