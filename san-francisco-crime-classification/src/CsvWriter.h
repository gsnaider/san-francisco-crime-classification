/*
 * CsvWriter.h
 *
 *  Created on: Nov 25, 2015
 *      Author: tobias
 */
#include <iostream>
#include <vector>
#include <memory>
#include <eigen3/Eigen/Core>
#include <string>
#include <fstream>
#include <chrono>
#ifndef CSVWRITER_H_
#define CSVWRITER_H_
using namespace std;
using namespace Eigen;
class CsvWriter {
public:
	CsvWriter();
	void makeSubmitWithMatrix(std::string path, MatrixXf matrix);
	void storeWeights(string path_base, vector<MatrixXf>* weights);
	void storeBiases(string path,vector<VectorXf>* biases);
	void writeMatrixIn(ofstream &file, MatrixXf &matrix);
	virtual ~CsvWriter();
};

#endif /* CSVWRITER_H_ */
