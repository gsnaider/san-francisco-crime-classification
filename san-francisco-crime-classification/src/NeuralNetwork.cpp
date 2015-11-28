//============================================================================
// Name        : NeuralNetwork.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <math.h>
#include "Network.h"

#include "CsvReader.h"
#include "CsvWriter.h"
#include <fstream>
#include <iomanip>

using namespace std;
using namespace Eigen;

int main() {

	CsvReader reader;
	//CsvWriter writer;
	MatrixXf matrix = reader.csvReadToMatrix("../data/parsed_train.csv");

	if (matrix.rows() == 0 && matrix.cols() == 0) {
		printf("ERROR\n");
		return -1;
	} else {
		cout << "cantidad de features: " << (matrix.cols() - 1) << endl << endl;
	}

	cout << "features: " << matrix.block(0, 0, 10, matrix.cols() - 1) << '\n' << '\n';

	cout << "clase: " << matrix.block(0, matrix.cols() - 1, 10, 1) << '\n' << '\n';

	//TODO HASTA ACA ESTA PROBADO QUE ANDA

	//shuflear toda la matrix
	PermutationMatrix<Dynamic, Dynamic> permutacionFilasRandom(matrix.rows());
	permutacionFilasRandom.setIdentity();

	srand(time(0));
	random_shuffle(permutacionFilasRandom.indices().data(),
			permutacionFilasRandom.indices().data() + permutacionFilasRandom.indices().size());

	matrix = permutacionFilasRandom * (matrix);

	//crear matrix_train 80% , matrix_test 20% (probar hacer delete de matrix)
	int ultimo_indice_train = round(matrix.rows() * 0.8);

	MatrixXf matrix_train;
	MatrixXf matrix_test;
	matrix_train = matrix.block(0, 0, ultimo_indice_train, matrix.cols());

	matrix_test = matrix.block(ultimo_indice_train, 0, matrix.rows(), matrix.cols());


	MatrixXf x_train;
	VectorXd y_train;
	MatrixXf x_test;
	VectorXd y_test;
	//separar matrix_train en x_train, y_train
	//separar matrix_test en x_test, y_test
	x_train = matrix_train.block(0, 0, matrix_train.rows(), matrix_train.cols() - 1);
	//y_train = (matrix_train.block(0, matrix_train.cols() - 1, matrix_train.rows(), 1)).cast<int>(); //me dice que puse different types
	x_test = matrix_test.block(0, 0, matrix_test.rows(), matrix_test.cols() - 1);
	//y_test = matrix_test.block(0, matrix_test.cols() - 1, matrix_test.rows(), 1).cast<int>();

//writer.makeSubmitWithMatrix("../data/submit.csv",matrix);

	//TODO HASTA ACA LO QUE NO ESTA PROBADO
	//TODO hacer el submit con el feedforward que no esta implementado! usar writer.makeSubmitWithMatrix;

	int input_dim = x_train.cols();
	int output_dim = 39;

	vector<int> sizes;
	sizes.push_back(input_dim);
	sizes.push_back(8);
	sizes.push_back(output_dim);

	int epochs = 30;
	int miniBatchSize = 5;
	float learningRate = 0.5;
	float regularizationFactor = 0.1;

	Network* net = new Network(sizes);
	net->SGD(&x_train, &y_train, &x_test, &y_test, epochs, miniBatchSize, learningRate, regularizationFactor);

	delete net;
	return 0;
}

