/*
 * Network.h

 *
 *  Created on: 24/11/2015
 *      Author: ger
 */
#include <iostream>
#include <vector>
#include <math.h>
#include <eigen3/Eigen/Dense>

#ifndef NETWORK_H_
#define NETWORK_H_

using namespace std;
using namespace Eigen;

typedef struct {
	vector<VectorXf> deltaNabla_b;
	vector<MatrixXf> deltaNabla_w;
}nablas_t;

class Network {
public:
	Network(vector<int> sizes);
	/*
	 * Recibe los features en una matriz de x_train y la clasificacion de forma Ej: {0,0,1,0} en y_train
	 * Se usa el mismo formato para el set de datos para validacion
	 */
	void SGD(MatrixXf* x_train, VectorXi* y_train, MatrixXf* x_test, VectorXi* y_test, int epochs, int miniBatchSize,
			float learningRate, float regularizationFactor);

	virtual ~Network();

private:
	float regularizationFactor, learningRate;
	int numLayers, miniBatchSize, epochs, featuresSize, n;
	vector<int> sizes;
	vector<MatrixXf> weights;
	vector<VectorXf> biases;

	void defaultWeightInitializer();

	void updateMiniBatch(MatrixXf* miniBatch_x, VectorXi* miniBatch_y);

	nablas_t backPropagation(VectorXf* x, int y);

	int accuracy(MatrixXf* x, VectorXi* y);

	VectorXf* feedfordward(VectorXf* a);

//	VectorXf costFunction(VectorXf* estimatedResults, VectorXi* y);

	VectorXf costDelta(VectorXf* estimatedResults, VectorXi* y);

	VectorXf softmax(VectorXf* z);

	VectorXf relu(VectorXf* z);

	VectorXf reluPrime(VectorXf* z);

	int argmax(VectorXf* v);

	VectorXi yToVector(int y);
};

#endif /* NETWORK_H_ */
