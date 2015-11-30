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
#include <random>

#ifndef NETWORK_H_
#define NETWORK_H_

using namespace std;
using namespace Eigen;

typedef struct {
	vector<VectorXd> deltaNabla_b;
	vector<MatrixXd> deltaNabla_w;
}nablas_t;

class Network {
public:

	Network(const vector<int>& inputLayers, const vector<VectorXd>& biases, const vector<MatrixXd>& weights);

	Network(const vector<int>& inputLayers);

	/*
	 * Recibe los features en una matriz de x_train y la clasificacion de forma Ej: {0,0,1,0} en y_train
	 * Se usa el mismo formato para el set de datos para validacion
	 */
	void SGD(const MatrixXd& x_train, const VectorXd& y_train, const MatrixXd& x_test, const VectorXd& y_test, const int epochs, const int miniBatchSize,
			const double learningRate, const double regularizationFactor);

	MatrixXd evaluate(const MatrixXd& x) const;

	int accuracy(const MatrixXd& x, const VectorXd& y) const;

	vector<VectorXd>* getBiases();

	vector<MatrixXd>* getWeights();


	virtual ~Network();

private:
	int numLayers;
	vector<int> layers;
	vector<MatrixXd> weights;
	vector<VectorXd> biases;

	void defaultWeightInitializer();

	void updateMiniBatch(const MatrixXd& miniBatch_x, const VectorXd& miniBatch_y,
			const double learningRate, const double regularizationFactor, const int dataSize);

	nablas_t backPropagation(const VectorXd& x, const int y);

	VectorXd feedfordward(const VectorXd& row) const;

//	VectorXd costFunction(VectorXd* estimatedResults, VectorXd* y);

	VectorXd costDelta(const VectorXd& estimatedResults, const VectorXd& y) const;

	//double costLogloss(const VectorXd& estimatedResults, const VectorXd& y) const;

	VectorXd softmax(const VectorXd& z) const;

	VectorXd relu(const VectorXd& z) const;

	VectorXd reluPrime(const VectorXd& z) const;

	int argmax(const VectorXd& v) const;

	VectorXd yToVector(int y) const;
};

#endif /* NETWORK_H_ */
