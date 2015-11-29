/*
 * Network.cpp
 *
 *  Created on: 24/11/2015
 *      Author: ger
 */

#include "Network.h"

void printSample(const MatrixXf& matrix, const string& texto){
	cout << texto << "\n"<< matrix.block(0,0,4,matrix.cols()) << "\n";
}
void printSample(const VectorXf& matrix, const string& texto){
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
void Network::defaultWeightInitializer() {

	if (numLayers < 2) {
		cout << "Debe haber al menos una layer para el input y otra para el output" << endl;
		return;
	}
	for (int i = 1; i < this->numLayers; i++) {
		VectorXf bias = VectorXf::Random(this->layers[i], 1);
		biases.push_back(bias);

//		srand(time(0)); //para que cambie el random de cada matriz
		MatrixXf weight = MatrixXf::Random(layers[i], layers[i - 1]) / sqrt(layers[i - 1]); //glorot unifrom
		weights.push_back(weight);
	}

//	for (size_t i = 0; i < biases.size(); i++){
//		cout << "Weights[" << i <<"]: " << weights[i] << '\n' << '\n';
//		cout << "Biases[" << i <<"]: " << biases[i].transpose() << '\n' << '\n';
//	}
}

void Network::SGD(const MatrixXf& x_train, const VectorXf& y_train, const MatrixXf& x_test, const VectorXf& y_test, const int epochs, const int miniBatchSize,
			const float learningRate, const float regularizationFactor) {

	vector<int> results;

	int dataSize = x_train.rows();
	int featuresSize = x_train.cols();

	PermutationMatrix<Dynamic, Dynamic> permutacionFilasRandom(dataSize);
	permutacionFilasRandom.setIdentity();
	for (int i = 0; i < epochs; i++) {

//		srand(time(0));
		random_shuffle(permutacionFilasRandom.indices().data(),
				permutacionFilasRandom.indices().data() + permutacionFilasRandom.indices().size());
		MatrixXf x_train_shuffled = permutacionFilasRandom * x_train;
		VectorXf y_train_shuffled = permutacionFilasRandom * y_train;
//		printSample(&x_train_shuffled, "x_train_shuffled");
//		printSample(&y_train_shuffled, "y_train_shuffled");
//		Shuffle esta OK.
		for (int j = 0; j < (dataSize - miniBatchSize); j += miniBatchSize) {
			MatrixXf miniBatch_x = x_train_shuffled.block(j, 0, miniBatchSize, featuresSize);
			VectorXf miniBatch_y = y_train_shuffled.segment(j, miniBatchSize);
//			printSample(&miniBatch_x, "MiniBatch_X");
//			printSample(&miniBatch_y, "MiniBatch_Y");
//			MiniBatches OK.
			updateMiniBatch(miniBatch_x, miniBatch_y, learningRate, regularizationFactor, dataSize);
		}
		int result = accuracy(x_test, y_test);
		cout << result << " / " << y_test.size() << endl;
		results.push_back(result);
	}
	cout << "Results" << endl;
	for (size_t i = 0; i < results.size(); i++){
		cout << results[i] << endl;
	}

}

void Network::updateMiniBatch(const MatrixXf& miniBatch_x, const VectorXf& miniBatch_y,
			const float learningRate, const float regularizationFactor, const int dataSize) {
	vector<VectorXf> nabla_b;
	vector<MatrixXf> nabla_w;

	for (size_t i = 0; i < biases.size(); i++) {
		nabla_b.push_back(VectorXf::Zero(biases[i].size(), 1));
	}

	for (size_t i = 0; i < weights.size(); i++) {
		nabla_w.push_back(MatrixXf::Zero(weights[i].rows(), weights[i].cols()));
	}

	for (int i = 0; i < miniBatch_y.size(); i++) {
		const VectorXf& x = miniBatch_x.row(i);
		const int y = miniBatch_y[i];
		nablas_t nablas = backPropagation(x, y);
		for (size_t i = 0; i < nabla_b.size(); i++) {
			nabla_b[i] = nabla_b[i] + nablas.deltaNabla_b[i];
		}
		for (size_t i = 0; i < nabla_w.size(); i++) {
			nabla_w[i] = nabla_w[i] + nablas.deltaNabla_w[i];
		}
		for (size_t i = 0; i < weights.size(); i++) {
			weights[i] = weights[i] * (1 - learningRate * (regularizationFactor / dataSize))
					- nabla_w[i] * (learningRate / miniBatch_y.size());
		}
		for (size_t i = 0; i < biases.size(); i++) {
			biases[i] = biases[i] - nabla_b[i] * (learningRate / miniBatch_y.size());
		}

	}
}
nablas_t Network::backPropagation(const VectorXf& x, const int y) {

	vector<VectorXf> nabla_b;
	vector<MatrixXf> nabla_w;
	vector<VectorXf> activations; //vector to store all the activations, layer by layer
	vector<VectorXf> zs; //vector to store all the z vectors, layer by layer

	VectorXf z;

	for (size_t i = 0; i < biases.size(); i++) {
		//nabla_b is filled with elements so that we can fill it later from the end.
		VectorXf aux;
		nabla_b.push_back(aux);
	}

	for (size_t i = 0; i < weights.size(); i++) {
		//nabla_w is filled with elements so that we can fill it later from the end.
		VectorXf aux;
		nabla_w.push_back(aux);
	}

	VectorXf y_vector = yToVector(y);

	VectorXf activation = x;

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

	VectorXf delta = costDelta(activations[activations.size() - 1] , y_vector);

	nabla_b[nabla_b.size() - 1] = delta;
	nabla_w[nabla_w.size() - 1] = delta * (activations[activations.size() - 2].transpose());

	for (int i = 2; i < numLayers; i++) {
		z = zs[zs.size() - i];
		VectorXf rp = reluPrime(z);

		VectorXf aux =( (weights[weights.size() - i + 1]).transpose()) * delta;

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

VectorXf Network::relu(const VectorXf& z) const{
	VectorXf result(z.size(), 1);
	for (int i = 0; i < z.size(); i++) {
		result[i] = max(0.0f, z(i));
	}
	return result;
}

VectorXf Network::reluPrime(const VectorXf& z) const{
	VectorXf result(z.size(), 1);
	for (int i = 0; i < z.size(); i++) {
		result[i] = (z[i] >= 0) ? 1 : 0;
	}
	return result;
}

VectorXf Network::softmax(const VectorXf& z)const {
	VectorXf z_exp(z.size(), 1);
	for (int i = 0; i < z.size(); i++) {
		float elem_i = z[i];
		z_exp[i] = exp(elem_i);
	}
	return (z_exp / z_exp.sum());
}

VectorXf Network::yToVector(int y) const {
	int output_size = layers[layers.size() - 1];
	VectorXf v = VectorXf::Zero(output_size, 1);
	v[y] = 1;
	return v;
}

int Network::argmax(const VectorXf& v) const{
	float max = 0;
	int max_idx;
	for (int i = 0; i < v.size(); i++){
		if (v[i] > max){
			max = v[i];
			max_idx = i;
		}
	}
	return max_idx;
}

VectorXf Network::costDelta(const VectorXf& estimatedResults, const VectorXf& y) const{
	return (estimatedResults - y);
}

int Network::accuracy(const MatrixXf& x, const VectorXf& y) const {

	int correct_predictions = 0;
//	for (size_t i = 0; i < biases.size(); i++){
//		cout << "Weights[" << i <<"]: " << weights[i] << '\n' << '\n';
//		cout << "Biases[" << i <<"]: " << biases[i].transpose() << '\n' << '\n';
//	}

	for (int i = 0; i < x.rows(); i++){
//	for (int i = 0; i < 3; i++){
		VectorXf result = feedfordward(x.row(i));
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

VectorXf Network::feedfordward(const VectorXf& row) const {

//	cout << "Row: " << row.transpose() << endl << endl;
	VectorXf aux = row;
//	cout << "Aux (inic): " << aux.transpose() << endl << endl;
	for (size_t i = 0; i < (biases.size() - 1); i++) {
		VectorXf z = weights[i] * aux + biases[i];
//		cout << "z: " << z.transpose() << endl << endl;
		aux = relu(z);
//		cout << "Aux: " << aux.transpose() << endl << endl;
	}
//	cout << "aux(fin for): " << aux.transpose() << endl << endl;

	int last_idx = biases.size() - 1;
	VectorXf z = weights[last_idx] * aux + biases[last_idx];
//	cout << "z: " << z.transpose() << endl;
	return softmax(z);
}

MatrixXf Network::evaluate(const MatrixXf& x) const {
	MatrixXf results(x.rows(), layers[layers.size() - 1]);
	for (int i = 0; i < x.rows(); i++) {
		//Para generar resultados con probabilidades
//		VectorXf result = feedfordward(x.row(i));
//		for (int j = 0; j < result.size(); j++){
//			results(i,j) = result[j];
//		}

		//Para generar resultados binarios
		VectorXf result = feedfordward(x.row(i));
		int estimated_result = argmax(result);
		VectorXi result_binario = VectorXi::Zero(result.rows(), 1);
		result_binario[estimated_result] = 1;
		for (int j = 0; j < result_binario.size(); j++) {
			int result_j = result_binario[j];
			results(i, j) = result_j;
		}

//		if (i < 3){
//			const MatrixXf& result_t = result.transpose();
//			cout << result_t << endl;
//		}

	}
	return results;




}

Network::~Network() {
}
