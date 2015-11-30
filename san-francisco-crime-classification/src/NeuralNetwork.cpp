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

//Esto es para testeo.
//TODO Para correr con todos los datos tiene que ser DATA_SIZE = 1!!!
const double DATA_SIZE = 0.1;

inputData_t generateInputData() {
	CsvReader reader;

	MatrixXd matrix = reader.csvReadToMatrix("data/parsed_train.csv");

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
		int miniBatchSize, double learningRate, double regularizationFactor, bool load) {

	inputData_t data = generateInputData();

	int input_dim = data.x_train.cols();
	int output_dim = OUTPUT_SIZE;

	vector<int> layers;
	layers.push_back(input_dim);
	layers.insert(layers.end(), hiddenLayers.begin(), hiddenLayers.end()); //inserta todos los elementos de hiddenLayers
	layers.push_back(output_dim);

	Network net(layers);
	if (load){
		CsvReader reader;
		vector<MatrixXd> weights = reader.readWheights("Weights",2);
		vector<VectorXd> biases = reader.readBiases("Biases.csv");
		net = Network(layers,biases,weights);
	}

	cout << "Arranca train" << endl;

	net.SGD(data.x_train, data.y_train, data.x_test, data.y_test, epochs, miniBatchSize,
			learningRate, regularizationFactor);

	int validationResult = net.accuracy(data.x_validation, data.y_validation);
	cout << "Validation results: " << validationResult << " / "
			<< data.y_validation.rows() << endl;

	return net;
}
void evaluateTestData(const Network& net) {
	CsvReader reader;
	CsvWriter writer;
	cout << "Leyendo test data" << endl;
	MatrixXd testData = reader.csvReadToMatrix("data/parsed_test.csv");
	if (testData.rows() == 0 && testData.cols() == 0) {
		printf("Error leyendo test data.\n");
		return;
	}
	cout << "cantidad de features: " << (testData.cols() - 1) << endl << endl;
	cout << "Evaluando test data" << endl;
	MatrixXd results = net.evaluate(testData);
	writer.makeSubmitWithMatrix("data/submit.csv", results);
}


int main() {

	CsvReader reader;
				CsvWriter writer;

	vector<int> hiddenLayers;
	hiddenLayers.push_back(90);
	hiddenLayers.push_back(60);

	int epochs = 5;
	int miniBatchSize = 100;
	double learningRate = 0.01;
	double regularizationFactor = 0.01;

	Network net = trainNetWithParsedTrainData(hiddenLayers, epochs,
			miniBatchSize, learningRate, regularizationFactor, false);

	writer.storeWeights("net-1-weights",net.getWeights());
	writer.storeBiases("net-1-biases.csv",net.getBiases());

	evaluateTestData(net);

	return 0;
}



//VectorXd yToVecto(int y) {
//	int output_size = 39;
//	VectorXd v = VectorXd::Zero(output_size, 1);
//	v[y] = 1;
//	return v;
//}

//VectorXd rel(const VectorXd& z) {
//	VectorXd result(z.size(), 1);
//	for (int i = 0; i < z.size(); i++) {
//		result[i] = max(0.0f, z(i));
//	}
//	return result;
//}
//
//VectorXd relPrime(const VectorXd& z) {
//	VectorXd result(z.size(), 1);
//	for (int i = 0; i < z.size(); i++) {
//		result[i] = (z[i] > 0) ? 1 : 0;
//	}
//	return result;
//}
//
//VectorXd softma(const VectorXd& z) {
//	VectorXd z_exp(z.size(), 1);
//	for (int i = 0; i < z.size(); i++) {
//		double elem_i = z[i];
//		z_exp[i] = exp(elem_i);
//	}
//	return (z_exp / z_exp.sum());
//}
//
//int argma(const VectorXd& v){
//	double max = 0;
//	int max_idx;
//	for (int i = 0; i < v.size(); i++){
//		if (v[i] > max){
//			max = v[i];
//			max_idx = i;
//		}
//	}
//	return max_idx;
//}
//
//VectorXd costDelt(const VectorXd& estimatedResults, const VectorXd& y){
//	return (estimatedResults - y);
//}


//int main(){
//
//	VectorXd a(7);
//	VectorXd b(7);
//
//	a << 1, -2, 3, 0, -3, 6, 0;
//	b << 0, 0, 0, 0, 0, 1, 0;
//
////	cout << rel(a).transpose() << endl;
////	cout << relPrime(a).transpose() << endl;
////	cout << softma(a).transpose() << endl;
////	cout << argma(a) << endl;
////	cout << costDelt(a, b).transpose() << endl;
//
//	cout << yToVecto(5);


//	MatrixXd* m = new MatrixXd;
//	MatrixXd* m2 = new MatrixXd;
//
//	*m << 1, 2,
//	     4, 3;
//	*m2 << 3, 4,
//		     2, 0;
//
//
//
//	*m = MatrixXd::Random(2, 3);
//	*m2 = MatrixXd::Random(2, 3);
//
//
//	cout << *m << endl;
//
//	delete m;
//
//	VectorXd a(3);
//	VectorXd b(3);
//
//	a << 1, 2, 3;
//	b << 4, 1, 0;
//
//	VectorXd c = a.array() * b.array();
//
//	double d = a.dot(b);
//	cout << c << endl << d;
//
//	MatrixXd a(2,2);
//	MatrixXd b(2,2);
//
//	a << 1, 2,
//		3, 5;
//	b << 4, 1,
//		0, 3;
//
//	cout << a * b;
//
//	vector<int> a;
//	a.push_back(2);
//	a.push_back(4);
//	a[1] = 8;
//	for (int i = 0; i < a.size(); i++){
//		cout << i << " " << a[i] << endl;
//	}


//	return 0;
//}

