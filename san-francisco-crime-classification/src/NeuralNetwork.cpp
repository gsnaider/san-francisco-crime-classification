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

typedef struct {
	MatrixXf x_train;
	VectorXf y_train;
	MatrixXf x_test;
	VectorXf y_test;
	MatrixXf x_validation;
	VectorXf y_validation;
}inputData_t;

const int OUTPUT_SIZE = 39;

inputData_t* generateInputData(){
	CsvReader reader;
	MatrixXf matrix = reader.csvReadToMatrix("data/parsed_train.csv");

	if (matrix.rows() == 0 && matrix.cols() == 0) {
		printf("Error leyendo data.\n");
		return NULL;
	} else {
		cout << "cantidad de features: " << (matrix.cols() - 1) << endl << endl;
	}

	//shuflear toda la matrix
	PermutationMatrix<Dynamic, Dynamic> permutacionFilasRandom(matrix.rows());
	permutacionFilasRandom.setIdentity();

	srand(time(0));
	random_shuffle(permutacionFilasRandom.indices().data(),
			permutacionFilasRandom.indices().data()
					+ permutacionFilasRandom.indices().size());

	matrix = permutacionFilasRandom * (matrix);
	int ultimo_indice_train = round(matrix.rows() * 0.8);
	int ultimo_indice_test = round(matrix.rows() * 0.9);

	MatrixXf matrix_train;
	MatrixXf matrix_test;
	MatrixXf matrix_validation;

	matrix_train = matrix.block(0, 0, ultimo_indice_train, matrix.cols());

	matrix_test = matrix.block(ultimo_indice_train, 0,
			ultimo_indice_test - ultimo_indice_train, matrix_train.cols());

	matrix_validation = matrix.block(ultimo_indice_test, 0,
			matrix.rows() - ultimo_indice_test, matrix_train.cols());

	matrix.resize(0, 0);

	MatrixXf x_train;
	VectorXf y_train;
	MatrixXf x_test;
	VectorXf y_test;
	MatrixXf x_validation;
	VectorXf y_validation;

	//separar matrix_train en x_train, y_train
	//separar matrix_test en x_test, y_test
	x_train = matrix_train.block(0, 0, matrix_train.rows(),
			matrix_train.cols() - 1);
	y_train = (matrix_train.block(0, matrix_train.cols() - 1,
			matrix_train.rows(), 1)); //me dice que puse different types
	x_test = matrix_test.block(0, 0, matrix_test.rows(),
			matrix_test.cols() - 1);
	y_test = matrix_test.block(0, matrix_test.cols() - 1, matrix_test.rows(),
			1);
	x_validation = matrix_validation.block(0, 0, matrix_validation.rows(),
			matrix_validation.cols() - 1);
	y_validation = matrix_validation.block(0, matrix_validation.cols() - 1,
			matrix_validation.rows(), 1);

	matrix_train.resize(0, 0);
	matrix_test.resize(0, 0);

	cout << "Train x: " << x_train.rows() << "x" << x_train.cols() << "\n";
	cout << "Train y: " << y_train.rows() << "x" << y_train.cols() << "\n";
	cout << "Test x: " << x_test.rows() << "x" << x_test.cols() << "\n";
	cout << "Test y: " << y_test.rows() << "x" << y_test.cols() << "\n";
	cout << "Validation x: " << x_validation.rows() << "x"
			<< x_validation.cols() << "\n";
	cout << "Validation y: " << y_validation.rows() << "x"
			<< x_validation.cols() << "\n";


	inputData_t *inputData = (inputData_t*) malloc(sizeof(inputData_t));
	inputData->x_train = x_train;
	inputData->y_train = y_train;
	inputData->x_test = x_test;
	inputData->y_test = y_test;
	inputData->x_validation = x_validation;
	inputData->y_validation = y_validation;

	return inputData;

}

Network* trainNetWithParsedTrainData(vector<int> hiddenLayers, int epochs,
		int miniBatchSize, float learningRate, float regularizationFactor) {

	inputData_t* inputData = generateInputData();

	MatrixXf x_train = inputData->x_train;
	VectorXf y_train = inputData->y_train;
	MatrixXf x_test = inputData->x_test;
	VectorXf y_test = inputData->y_test;
	MatrixXf x_validation = inputData->x_validation;
	VectorXf y_validation = inputData->y_validation;

	delete inputData;

	int input_dim = x_train.cols();
	int output_dim = OUTPUT_SIZE;

	vector<int> layers;
	layers.push_back(input_dim);
	layers.insert(layers.end(), hiddenLayers.begin(), hiddenLayers.end()); //inserta todos los elementos de hiddenLayers
	layers.push_back(output_dim);

	Network* net = new Network(layers);

	cout << "Arranca train" << endl;

	net->SGD(&x_train, &y_train, &x_test, &y_test, epochs, miniBatchSize,
			learningRate, regularizationFactor);

	int validationResult = net->accuracy(&x_validation, &y_validation);
	cout << "Validation results: " <<  validationResult << " / " << y_validation.rows() << endl;

	return net;
}

void evaluateTestData(Network* net){
	CsvReader reader;
	CsvWriter writer;

	MatrixXf testData = reader.csvReadToMatrix("data/parsed_test.csv");
	if (testData.rows() == 0 && testData.cols() == 0) {
		printf("Error leyendo test data.\n");
		return;
	}
	cout << "cantidad de features: " << (testData.cols() - 1) << endl << endl;
	MatrixXf results = net->evaluate(&testData);

	writer.makeSubmitWithMatrix("data/submit.csv", results);
}

int main() {

	vector<int> hiddenLayers;
	hiddenLayers.push_back(40);

	int epochs = 10;
	int miniBatchSize = 600;
	float learningRate = 0.6;
	float regularizationFactor = 0.01;


	Network* net = trainNetWithParsedTrainData(hiddenLayers, epochs, miniBatchSize, learningRate, regularizationFactor);


//	Usar para probar si anda el evaluateTestData
//	vector<int> layers;
//	layers.push_back(43);
//	layers.insert(layers.end(), hiddenLayers.begin(), hiddenLayers.end());
//	layers.push_back(OUTPUT_SIZE);
//	Network* net = new Network(layers);


	if (net){
		//TODO Deberiamos guardar los datos de la red en un archivo, sino se pierde despues de correr el prog!
		evaluateTestData(net);
		delete net;
	}

	return 0;
}

