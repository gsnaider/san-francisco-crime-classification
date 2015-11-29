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
	Network(const vector<int>& inputLayers);

	/*
	 * Recibe los features en una matriz de x_train y la clasificacion de forma Ej: {0,0,1,0} en y_train
	 * Se usa el mismo formato para el set de datos para validacion
	 */
	void SGD(const MatrixXf& x_train, const VectorXf& y_train, const MatrixXf& x_test, const VectorXf& y_test, const int epochs, const int miniBatchSize,
			const float learningRate, const float regularizationFactor);

	MatrixXf evaluate(const MatrixXf& x) const;

	int accuracy(const MatrixXf& x, const VectorXf& y) const;

	virtual ~Network();

private:
	int numLayers;
	vector<int> layers;
	vector<MatrixXf> weights;
	vector<VectorXf> biases;

	void defaultWeightInitializer();

	void updateMiniBatch(const MatrixXf& miniBatch_x, const VectorXf& miniBatch_y,
			const float learningRate, const float regularizationFactor, const int dataSize);

	nablas_t backPropagation(const VectorXf& x, const int y);

	VectorXf feedfordward(const VectorXf& row) const;

//	VectorXf costFunction(VectorXf* estimatedResults, VectorXf* y);

	VectorXf costDelta(const VectorXf& estimatedResults, const VectorXf& y) const;

	VectorXf softmax(const VectorXf& z) const;

	VectorXf relu(const VectorXf& z) const;

	VectorXf reluPrime(const VectorXf& z) const;

	int argmax(const VectorXf& v) const;

	VectorXf yToVector(int y) const;
};

#endif /* NETWORK_H_ */
