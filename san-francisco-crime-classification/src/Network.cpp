/*
 * Network.cpp
 *
 *  Created on: 24/11/2015
 *      Author: ger
 */

#include "Network.h"

void printSample(const MatrixXd& matrix, const string& texto){
	cout << texto << "\n"<< matrix.block(0,0,4,matrix.cols()) << "\n";
}
void printSample(const VectorXd& matrix, const string& texto){
	cout << texto << "\n"<< matrix.block(0,0,4,matrix.cols()) << "\n";
}
void printSample(const VectorXi& matrix, const string& texto){
	cout << texto << "\n"<< matrix.block(0,0,4,matrix.cols()) << "\n";
}

Network::Network(const vector<int>& inputLayers) {

	layers = inputLayers;
	numLayers = layers.size();
	defaultWeightInitializer();
}

Network::Network(const vector<int>& inputLayers, const vector<VectorXd>& biases, const vector<MatrixXd>& weights) {

	layers = inputLayers;
	numLayers = layers.size();
	this->biases = biases;
	this->weights = weights;
}

vector<VectorXd>* Network::getBiases(){
	return &biases;
}

vector<MatrixXd>* Network::getWeights(){
	return &weights;
}

double gaussian(double dummy)
{
  static mt19937 rng;
  static normal_distribution<> nd;
  return nd(rng);
}

void Network::defaultWeightInitializer() {

	if (numLayers < 2) {
		cout << "Debe haber al menos una layer para el input y otra para el output" << endl;
		return;
	}
	for (int i = 1; i < this->numLayers; i++) {
		VectorXd bias = VectorXd::Zero(this->layers[i], 1).unaryExpr(ptr_fun(gaussian));
		biases.push_back(bias);

//		srand(time(0)); //para que cambie el random de cada matriz
		MatrixXd weight = MatrixXd::Zero(layers[i], layers[i - 1]).unaryExpr(ptr_fun(gaussian))  / sqrt(layers[i - 1]); //glorot unifrom
		weights.push_back(weight);
	}

	for (size_t i = 0; i < biases.size(); i++){
		cout << "Weights[" << i <<"]: " << weights[i] << '\n' << '\n';
		cout << "Biases[" << i <<"]: " << biases[i].transpose() << '\n' << '\n';
	}
}

void Network::SGD(const MatrixXd& x_train, const VectorXd& y_train, const MatrixXd& x_test, const VectorXd& y_test, const int epochs, const int miniBatchSize,
			const double learningRate, const double regularizationFactor) {

	vector<int> results;

	int dataSize = x_train.rows();
	int featuresSize = x_train.cols();

	PermutationMatrix<Dynamic, Dynamic> permutacionFilasRandom(dataSize);
	permutacionFilasRandom.setIdentity();
	for (int i = 0; i < epochs; i++) {
		cout << "Iniciando train de epoch " << i << endl;
//		srand(time(0));
		random_shuffle(permutacionFilasRandom.indices().data(),
				permutacionFilasRandom.indices().data() + permutacionFilasRandom.indices().size());
		MatrixXd x_train_shuffled = permutacionFilasRandom * x_train;
		VectorXd y_train_shuffled = permutacionFilasRandom * y_train;
//		printSample(&x_train_shuffled, "x_train_shuffled");
//		printSample(&y_train_shuffled, "y_train_shuffled");
//		Shuffle esta OK.
		for (int j = 0; j < (dataSize - miniBatchSize); j += miniBatchSize) {
			MatrixXd miniBatch_x = x_train_shuffled.block(j, 0, miniBatchSize, featuresSize);
			VectorXd miniBatch_y = y_train_shuffled.segment(j, miniBatchSize);
//			printSample(&miniBatch_x, "MiniBatch_X");
//			printSample(&miniBatch_y, "MiniBatch_Y");
//			MiniBatches OK.
			updateMiniBatch(miniBatch_x, miniBatch_y, learningRate, regularizationFactor, dataSize);
		}
		cout << "Finalizado train de epoch " << i << endl;
		cout << "Iniciando test de epoch " << i << endl;
		int result = accuracy(x_test, y_test);
		cout << endl << "-------------------------------------------------------" << endl;
		cout << "Epoch " <<i <<  " test results: " << result << " / " << y_test.size()  << endl;
		cout << endl << "-------------------------------------------------------" << endl;
		results.push_back(result);
	}
	cout << "Results" << endl;
	for (size_t i = 0; i < results.size(); i++){
		cout << results[i] << endl;
	}

}

void Network::updateMiniBatch(const MatrixXd& miniBatch_x, const VectorXd& miniBatch_y,
			const double learningRate, const double regularizationFactor, const int dataSize) {
	vector<VectorXd> nabla_b;
	vector<MatrixXd> nabla_w;

	for (size_t i = 0; i < biases.size(); i++) {
		nabla_b.push_back(VectorXd::Zero(biases[i].size(), 1));
	}

	for (size_t i = 0; i < weights.size(); i++) {
		nabla_w.push_back(MatrixXd::Zero(weights[i].rows(), weights[i].cols()));
	}

	for (int i = 0; i < miniBatch_y.size(); i++) {
		const VectorXd& x = miniBatch_x.row(i);
		const int y = miniBatch_y[i];
		nablas_t nablas = backPropagation(x, y);
		for (size_t j = 0; j < nabla_b.size(); j++) {
			nabla_b[j] = nabla_b[j] + (nablas.deltaNabla_b)[j];
		}
		for (size_t j = 0; j < nabla_w.size(); j++) {
			nabla_w[j] = nabla_w[j] + (nablas.deltaNabla_w)[j];
		}
//		if (i < 3){
//			for (size_t j = 0; j < biases.size(); j++){
//				cout << "DELTA_NABLA_B[" << j <<"]: " << (nablas.deltaNabla_b)[j] << '\n' << '\n';
//			}
//			for (size_t j = 0; j < biases.size(); j++){
//				cout << "DELTA_NABLA_W[" << j <<"]: " << (nablas.deltaNabla_w)[j] << '\n' << '\n';
//			}
//			for (size_t j = 0; j < biases.size(); j++){
//				cout << "NABLA_B[" << j <<"]: " << nabla_b[j] << '\n' << '\n';
//			}
//			for (size_t j = 0; j < biases.size(); j++){
//				cout << "NABLA_W[" << j <<"]: " << nabla_w[j] << '\n' << '\n';
//			}
//		}
		for (size_t j = 0; j < weights.size(); j++) {
			weights[j] = weights[j] * (1 - learningRate * (regularizationFactor / dataSize))
					- nabla_w[j] * (learningRate / miniBatch_y.size());
		}
		for (size_t j = 0; j < biases.size(); j++) {
			biases[j] = biases[j] - nabla_b[j] * (learningRate / miniBatch_y.size());
		}
	}
}
nablas_t Network::backPropagation(const VectorXd& x, const int y) {

	vector<VectorXd> nabla_b;
	vector<MatrixXd> nabla_w;
	vector<VectorXd> activations; //vector to store all the activations, layer by layer
	vector<VectorXd> zs; //vector to store all the z vectors, layer by layer

	VectorXd z;

	for (size_t i = 0; i < biases.size(); i++) {
		//nabla_b is filled with elements so that we can fill it later from the end.
		VectorXd aux;
		nabla_b.push_back(aux);
	}

	for (size_t i = 0; i < weights.size(); i++) {
		//nabla_w is filled with elements so that we can fill it later from the end.
		MatrixXd aux;
		nabla_w.push_back(aux);
	}

	VectorXd y_vector = yToVector(y);

	VectorXd activation = x;

	activations.push_back(activation);

	for (size_t i = 0; (i < (biases.size() - 1)); i++) {
		z = weights[i] * activation + biases[i];
		zs.push_back(z);
		activation = relu(z);
		activations.push_back(activation);
	}
	int lastIdx_weights = weights.size() - 1;
	int lastIdx_biases = biases.size() - 1;
	z = weights[lastIdx_weights] * activation + biases[lastIdx_biases];
	zs.push_back(z);
	activation = softmax(z);
	activations.push_back(activation);

	//HASTA ACA PARECERIA ESTAR BIEN

	VectorXd delta = costDelta(activations[activations.size() - 1] , y_vector);
	nabla_b[nabla_b.size() - 1] = delta;
	nabla_w[nabla_w.size() - 1] = delta * (activations[activations.size() - 2].transpose());

	for (int i = 2; i < numLayers; i++) {
		z = zs[zs.size() - i];
		VectorXd rp = reluPrime(z);

		VectorXd aux =( (weights[weights.size() - i + 1]).transpose()) * delta;

		delta = aux.array() * rp.array(); //ACA HICE MULTIPLICACION PUNTO A PUNTO

		nabla_b[nabla_b.size() - i] = delta;
		nabla_w[nabla_w.size() - i] = delta * (activations[activations.size() - i - 1].transpose());
	}

	nablas_t nablas;
	nablas.deltaNabla_b = nabla_b;
	nablas.deltaNabla_w = nabla_w;
	return nablas;
//TODO ver delete
}

VectorXd Network::relu(const VectorXd& z) const{
	VectorXd result(z.size(), 1);
	for (int i = 0; i < z.size(); i++) {
		result[i] = max(0.0, z(i));
	}
	return result;
}

VectorXd Network::reluPrime(const VectorXd& z) const{
	VectorXd result(z.size(), 1);
	for (int i = 0; i < z.size(); i++) {
		result[i] = (z[i] > 0) ? 1 : 0;
	}
	return result;
}

VectorXd Network::softmax(const VectorXd& z)const {
	VectorXd z_exp(z.size(), 1);
	for (int i = 0; i < z.size(); i++) {
		double elem_i = z[i];
		z_exp[i] = exp(elem_i);
	}
	return (z_exp / z_exp.sum());
}

VectorXd Network::yToVector(int y) const {
	int output_size = layers[layers.size() - 1];
	VectorXd v = VectorXd::Zero(output_size, 1);
	v[y] = 1;
	return v;
}

int Network::argmax(const VectorXd& v) const{
	double max = 0;
	int max_idx;
	for (int i = 0; i < v.size(); i++){
		if (v[i] > max){
			max = v[i];
			max_idx = i;
		}
	}
	return max_idx;
}

VectorXd Network::costDelta(const VectorXd& estimatedResults, const VectorXd& y) const{
	return (estimatedResults - y);
}
//double Network::costLogloss(const VectorXd& estimatedResults, const VectorXd& y) const{
//	VectorXd (-y*np.log(a))
//	return (estimatedResults - y);
//}

int Network::accuracy(const MatrixXd& x, const VectorXd& y) const {

	int correct_predictions = 0;
//	for (size_t i = 0; i < biases.size(); i++){
//		cout << "Weights[" << i <<"]: " << weights[i] << '\n' << '\n';
//		cout << "Biases[" << i <<"]: " << biases[i].transpose() << '\n' << '\n';
//	}

	for (int i = 0; i < x.rows(); i++){
//	for (int i = 0; i < 3; i++){
		VectorXd result = feedfordward(x.row(i));
		int estimated_result = argmax(result);
//		if (i < 3){
//			cout << "result: " <<result.transpose() << endl << endl;
//		}
		if (estimated_result == y[i]){
			correct_predictions++;
		}
	}
	return correct_predictions;
}

VectorXd Network::feedfordward(const VectorXd& row) const {

//	cout << "Row: " << row.transpose() << endl << endl;
	VectorXd aux = row;
//	cout << "Aux (inic): " << aux.transpose() << endl << endl;
	for (size_t i = 0; i < (biases.size() - 1); i++) {
		VectorXd z = weights[i] * aux + biases[i];
//		cout << "z: " << z.transpose() << endl << endl;
		aux = relu(z);
//		cout << "Aux: " << aux.transpose() << endl << endl;
	}
//	cout << "aux(fin for): " << aux.transpose() << endl << endl;

	int last_idx = biases.size() - 1;
	VectorXd z = weights[last_idx] * aux + biases[last_idx];
//	cout << "z: " << z.transpose() << endl;
	return softmax(z);
}

MatrixXd Network::evaluate(const MatrixXd& x) const {
	MatrixXd results(x.rows(), layers[layers.size() - 1]);
	for (int i = 0; i < x.rows(); i++) {
		//Para generar resultados con probabilidades
		VectorXd result = feedfordward(x.row(i));
		for (int j = 0; j < result.size(); j++){
			results(i,j) = result[j];
		}

//		if (i < 3){
//			const MatrixXd& result_t = result.transpose();
//			cout << result_t << endl;
//		}

	}
	return results;




}

Network::~Network() {
}
