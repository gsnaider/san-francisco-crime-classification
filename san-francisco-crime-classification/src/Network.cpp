/*
 * Network.cpp
 *
 *  Created on: 24/11/2015
 *      Author: ger
 */

#include "Network.h"

void printSample(MatrixXf *matrix, string texto) {
	cout << texto << "\n" << matrix->block(0, 0, 4, matrix->cols()) << "\n";
}
void printSample(VectorXf *matrix, string texto) {
	cout << texto << "\n" << matrix->block(0, 0, 4, matrix->cols()) << "\n";
}
void printSample(VectorXi *matrix, string texto) {
	cout << texto << "\n" << matrix->block(0, 0, 4, matrix->cols()) << "\n";
}

Network::Network(vector<int> layers) {

	this->layers = layers;
	numLayers = layers.size();
	defaultWeightInitializer();
}
void Network::defaultWeightInitializer() {

	if (numLayers < 2) {
		cout
				<< "Debe haber al menos una layer para el input y otra para el output"
				<< endl;
		return;
	}
	for (int i = 1; i < this->numLayers; i++) {
		VectorXf bias = VectorXf::Random(this->layers[i], 1).cwiseAbs();
		this->biases.push_back(bias);

		srand(time(0)); //para que cambie el random de cada matriz
		MatrixXf weight = MatrixXf::Random(layers[i], layers[i - 1]).cwiseAbs()
				/ sqrt(layers[i - 1]); //glorot unifrom

		weights.push_back(weight);
	}
}

void Network::SGD(MatrixXf* x_train, VectorXf* y_train, MatrixXf* x_test,
		VectorXf* y_test, int epochs, int miniBatchSize, float learningRate,
		float regularizationFactor) {

	MatrixXf x_train_shuffled;
	VectorXf y_train_shuffled;
	MatrixXf miniBatch_x;
	VectorXf miniBatch_y;
	vector<int> results;
	int result;

	int dataSize = x_train->rows();
	int featuresSize = x_train->cols();

	PermutationMatrix<Dynamic, Dynamic> permutacionFilasRandom(dataSize);
	permutacionFilasRandom.setIdentity();
	for (int i = 0; i < epochs; i++) {

		srand(time(0));
		random_shuffle(permutacionFilasRandom.indices().data(),
				permutacionFilasRandom.indices().data()
						+ permutacionFilasRandom.indices().size());
		x_train_shuffled = permutacionFilasRandom * (*x_train);
		y_train_shuffled = permutacionFilasRandom * (*y_train);
//		printSample(&x_train_shuffled, "x_train_shuffled");
//		printSample(&y_train_shuffled, "y_train_shuffled");
//		Shuffle esta OK.
		for (int j = 0; j < (dataSize - miniBatchSize); j += miniBatchSize) {
			miniBatch_x = x_train_shuffled.block(j, 0, miniBatchSize,
					featuresSize);
			miniBatch_y = y_train_shuffled.segment(j, miniBatchSize);
//			printSample(&miniBatch_x, "MiniBatch_X");
//			printSample(&miniBatch_y, "MiniBatch_Y");
//			MiniBatches OK.
			updateMiniBatch(&miniBatch_x, &miniBatch_y, learningRate,
					regularizationFactor, dataSize);
		}
		result = accuracy(x_test, y_test);
		cout << result << " / " << y_test->size() << endl;
		results.push_back(result);
	}
	cout << "Results" << endl;
	for (size_t i = 0; i < results.size(); i++) {
		cout << results[i] << endl;
	}

}

void Network::updateMiniBatch(MatrixXf* miniBatch_x, VectorXf* miniBatch_y,
		float learningRate, float regularizationFactor, int dataSize) {
	vector<VectorXf> nabla_b;
	vector<MatrixXf> nabla_w;

	for (size_t i = 0; i < biases.size(); i++) {
		nabla_b.push_back(VectorXf::Zero(biases[i].size(), 1));
	}

	for (size_t i = 0; i < weights.size(); i++) {
		nabla_w.push_back(MatrixXf::Zero(weights[i].rows(), weights[i].cols()));
	}
	for (int i = 0; i < miniBatch_y->size(); i++) {
		VectorXf x = miniBatch_x->row(i);
		int y = (*miniBatch_y)[i];
		nablas_t nablas = backPropagation(&x, y);
		for (size_t i = 0; i < nabla_b.size(); i++) {
			nabla_b[i] = nabla_b[i] + nablas.deltaNabla_b[i];
		}
		for (size_t i = 0; i < nabla_w.size(); i++) {
			MatrixXf nabla_w_i = nabla_w[i];
			MatrixXf delta_nabla_w_i = nablas.deltaNabla_w[i];
			nabla_w[i] = nabla_w_i + delta_nabla_w_i;
		}
		for (size_t i = 0; i < weights.size(); i++) {
			weights[i] = weights[i]
					* (1 - learningRate * (regularizationFactor / dataSize))
					- nabla_w[i] * (learningRate / miniBatch_y->size());
		}
		for (size_t i = 0; i < biases.size(); i++) {
			biases[i] = biases[i]
					- nabla_b[i] * (learningRate / miniBatch_y->size());
		}

	}
}
nablas_t Network::backPropagation(VectorXf* x, int y) {

	vector<VectorXf> nabla_b;
	vector<MatrixXf> nabla_w;
	vector<VectorXf> activations; //vector to store all the activations, layer by layer
	vector<VectorXf> zs; //vector to store all the z vectors, layer by layer

	VectorXf activation;
	VectorXf b;
	VectorXf z;
	VectorXf rp;
	MatrixXf w;
	VectorXf delta;

	for (size_t i = 0; i < biases.size(); i++) {
		nabla_b.push_back(VectorXf::Zero(biases[i].rows(), 1));
	}

	for (size_t i = 0; i < weights.size(); i++) {
		nabla_w.push_back(MatrixXf::Zero(weights[i].rows(), weights[i].cols()));
	}

	VectorXf y_vector = yToVector(y);

	activation = *x;

	activations.push_back(activation);

	for (size_t i = 0; (i < (biases.size() - 1)); i++) {

		z = weights[i] * activation + biases[i];

		zs.push_back(z);
		activation = relu(&z);
		activations.push_back(activation);
	}

	int lastIdx_weights = weights.size() - 1;
	int lastIdx_biases = biases.size() - 1;

	z = weights[lastIdx_weights] * activation + biases[lastIdx_biases];

	zs.push_back(z);
	activation = softmax(&z);

	activations.push_back(activation);

	delta = costDelta(&activations[activations.size() - 1], &y_vector);

	nabla_b[nabla_b.size() - 1] = delta;
	nabla_w[nabla_w.size() - 1] = delta
			* (activations[activations.size() - 2].transpose());

	for (int l = 2; l < numLayers; l++) {
		z = zs[zs.size() - l];
		rp = reluPrime(&z);

		VectorXf aux = ((weights[weights.size() - l + 1]).transpose()) * delta;

		delta = aux.array() * rp.array(); //ACA HICE MULTIPLICACION PUNTO A PUNTO

		nabla_b[nabla_b.size() - l] = delta;
		nabla_w[nabla_w.size() - l] = delta
				* (activations[activations.size() - l - 1].transpose());
	}

	nablas_t nablas;
	nablas.deltaNabla_b = nabla_b;
	nablas.deltaNabla_w = nabla_w;
	return nablas;
//TODO ver delete
}

VectorXf Network::relu(VectorXf* z) {
	VectorXf* a = new VectorXf;

	VectorXf result = VectorXf::Zero(z->size(), 1);
	for (int i = 0; i < z->size(); i++) {
		result[i] = (*z)[i] * ((*z)[i] > 0);
	}
	return result;
}

VectorXf Network::reluPrime(VectorXf* z) {
	VectorXf result = VectorXf::Zero(z->size(), 1);
	for (int i = 0; i < z->size(); i++) {
		result[i] = ((*z)[i] >= 0);
	}
	return result;
}

VectorXf Network::softmax(VectorXf* z) {
	VectorXf result = VectorXf::Zero(z->size(), 1);
	VectorXf z_exp = VectorXf::Zero(z->size(), 1);
	for (int i = 0; i < z->size(); i++) {
		float elem_i = (*z)[i];
		z_exp[i] = exp(elem_i);
	}
	result = (z_exp / z_exp.sum());
	return result;
}

VectorXf Network::yToVector(int y) {
	int output_size = layers[layers.size() - 1];
	VectorXf v = VectorXf::Zero(output_size, 1);
	v[y] = 1;
	return v;
}

int Network::argmax(VectorXf* v) {
	float max = 0;
	int max_idx;
	for (int i = 0; i < v->size(); i++) {
		if ((*v)[i] > max) {
			max = (*v)[i];
			max_idx = i;
		}
	}
	return max_idx;
}

VectorXf Network::costDelta(VectorXf* estimatedResults, VectorXf* y) {
	return ((*estimatedResults) - (*y));
}

int Network::accuracy(MatrixXf* x, VectorXf* y) {

	int correct_predictions = 0;
	for (int i = 0; i < x->rows(); i++) {
		VectorXf x_i = x->row(i);
		int estimated_result = argmax(feedfordward(&x_i));
		//printf("%d \n", estimated_result);
		if (estimated_result == (*y)[i]) {
			correct_predictions++;
		}
	}
	return correct_predictions;
}

VectorXf* Network::feedfordward(VectorXf* row) {
	VectorXf z;
	MatrixXf w;
	VectorXf b;
	for (size_t i = 0; i < (biases.size() - 1); i++) {
		w = weights[i];
		b = biases[i];
		z = w * (*row) + b;
		*row = relu(&z);
	}
	w = weights[weights.size() - 1];
	b = biases[biases.size() - 1];
	z = w * (*row) + b;
	*row = softmax(&z);
	return row;
}

MatrixXi Network::evaluate(MatrixXf* x) {
	MatrixXi results = MatrixXi::Zero(x->rows(), layers[layers.size() - 1]);
	for (int i = 0; i < x->rows(); i++) {
		VectorXf x_i = x->row(i);
		VectorXf* result = feedfordward(&x_i);

		int estimated_result = argmax(result);

		VectorXi result_binario = VectorXi::Zero(result->rows(),1);

		result_binario[estimated_result] = 1;

		for (int j = 0; j < result_binario.size(); j++) {
			int result_j = result_binario[j];
			results(i, j) = result_j;
		}
	}
	return results;
}

Network::~Network() {
}

