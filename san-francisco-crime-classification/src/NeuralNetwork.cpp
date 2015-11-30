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
	MatrixXd x_train;
	VectorXd y_train;
	MatrixXd x_test;
	VectorXd y_test;
	MatrixXd x_validation;
	VectorXd y_validation;
} inputData_t;

const int OUTPUT_SIZE = 39;
const double DATA_SIZE = 0.1;
const bool LOAD_NET = false;
const bool TRAIN_NET = true;
const bool EVALUATE_TEST_DATA = false;

const string TRAIN_DATA_FILE = "data/parsed_train.csv";
const string TEST_DATA_FILE = "data/parsed_test.csv";
const string WEIGHTS_BASE_LOAD_PATH = "nets/net-temp-weights";
const string BIASES_LOAD_PATH = "nets/net-temp-biases.csv";
const string WEIGHTS_BASE_STORAGE_FILE = "nets/net-temp-weights";
const string BIASES_STORAGE_FILE = "nets/net-temp-biases.csv";
const string SUBMIT_FILE = "submits/submit-temp.csv";


inputData_t generateInputData() {
	CsvReader reader;

	MatrixXd matrix = reader.csvReadToMatrix(TRAIN_DATA_FILE);

	if (matrix.rows() == 0 && matrix.cols() == 0) {
		printf("Error leyendo data.\n");
		exit(-1);
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

//	Recuce el tamanio de la matriz para poder testear.
//	TODO Para correr con todos los datos cambiar DATA_SIZE = 1.
	int nuevo_ultimo_indice = round(matrix.rows() * DATA_SIZE);
	matrix.conservativeResize(nuevo_ultimo_indice, matrix.cols());

	int ultimo_indice_train = round(matrix.rows() * 0.8);
	int ultimo_indice_test = round(matrix.rows() * 0.9);

	MatrixXd matrix_train = matrix.block(0, 0, ultimo_indice_train, matrix.cols());

	MatrixXd matrix_test = matrix.block(ultimo_indice_train, 0,
			ultimo_indice_test - ultimo_indice_train, matrix_train.cols());

	MatrixXd matrix_validation = matrix.block(ultimo_indice_test, 0,
			matrix.rows() - ultimo_indice_test, matrix_train.cols());

	matrix.resize(0, 0);

	inputData_t data;

	//separar matrix_train en x_train, y_train
	//separar matrix_test en x_test, y_test
	data.x_train = matrix_train.block(0, 0, matrix_train.rows(),
			matrix_train.cols() - 1);
	data.y_train = (matrix_train.block(0, matrix_train.cols() - 1,
			matrix_train.rows(), 1)); //me dice que puse different types
	data.x_test = matrix_test.block(0, 0, matrix_test.rows(),
			matrix_test.cols() - 1);
	data.y_test = matrix_test.block(0, matrix_test.cols() - 1, matrix_test.rows(),
			1);
	data.x_validation = matrix_validation.block(0, 0, matrix_validation.rows(),
			matrix_validation.cols() - 1);
	data.y_validation = matrix_validation.block(0, matrix_validation.cols() - 1,
			matrix_validation.rows(), 1);

	matrix_train.resize(0, 0);
	matrix_test.resize(0, 0);

	cout << "Train x: " << data.x_train.rows() << "x" << data.x_train.cols() << "\n";
	cout << "Train y: " << data.y_train.rows() << "x" << data.y_train.cols() << "\n";
	cout << "Test x: " << data.x_test.rows() << "x" << data.x_test.cols() << "\n";
	cout << "Test y: " << data.y_test.rows() << "x" << data.y_test.cols() << "\n";
	cout << "Validation x: " << data.x_validation.rows() << "x"
			<< data.x_validation.cols() << "\n";
	cout << "Validation y: " << data.y_validation.rows() << "x"
			<< data.y_validation.cols() << "\n";

	return data;

}

Network trainNetWithParsedTrainData(vector<int> hiddenLayers, int epochs,
		int miniBatchSize, double learningRate, double regularizationFactor) {

	inputData_t data = generateInputData();

	int input_dim = data.x_train.cols();
	int output_dim = OUTPUT_SIZE;

	vector<int> layers;
	layers.push_back(input_dim);
	layers.insert(layers.end(), hiddenLayers.begin(), hiddenLayers.end()); //inserta todos los elementos de hiddenLayers
	layers.push_back(output_dim);

	Network net(layers);
	if (LOAD_NET){
		CsvReader reader;
		vector<MatrixXd> weights = reader.readWheights(WEIGHTS_BASE_LOAD_PATH,2);
		vector<VectorXd> biases = reader.readBiases(BIASES_LOAD_PATH);
		net = Network(layers,biases,weights);
	}
	if (TRAIN_NET){
		cout << "Arranca train" << endl;
		net.SGD(data.x_train, data.y_train, data.x_test, data.y_test, epochs, miniBatchSize,
				learningRate, regularizationFactor);
	}

	int validationResult = net.accuracy(data.x_validation, data.y_validation);
	double cost = net.totalCost(data.x_validation, data.y_validation);
	cout << "---------------------------" << endl;
	cout << "Validation results: " << validationResult << " / "
			<< data.y_validation.rows() << endl;
	cout << "---------------------------" << endl;
	cout << "Validation cost: " << cost << endl;
	cout << "---------------------------" << endl;
	return net;
}
void evaluateTestData(const Network& net) {
	CsvReader reader;
	CsvWriter writer;
	cout << "Leyendo test data" << endl;
	MatrixXd testData = reader.csvReadToMatrix(TEST_DATA_FILE);
	if (testData.rows() == 0 && testData.cols() == 0) {
		printf("Error leyendo test data.\n");
		return;
	}
	cout << "cantidad de features: " << (testData.cols() - 1) << endl << endl;
	cout << "Evaluando test data" << endl;
	MatrixXd results = net.evaluate(testData);
	writer.makeSubmitWithMatrix(SUBMIT_FILE, results);
}


int main() {

	CsvReader reader;
	CsvWriter writer;

	vector<int> hiddenLayers;
	hiddenLayers.push_back(90);
	hiddenLayers.push_back(60);

	int epochs = 15;
	int miniBatchSize = 100;
	double learningRate = 0.05;
	double regularizationFactor = 0.01;

	Network net = trainNetWithParsedTrainData(hiddenLayers, epochs,
			miniBatchSize, learningRate, regularizationFactor);

	writer.storeWeights(WEIGHTS_BASE_STORAGE_FILE,net.getWeights());
	writer.storeBiases(BIASES_STORAGE_FILE,net.getBiases());

	if (EVALUATE_TEST_DATA){
		evaluateTestData(net);
	}

	return 0;
}

