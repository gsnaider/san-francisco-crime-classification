/*
 * CsvReader.h
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
#include <numeric>
#include "CsvRow.h"


#ifndef SRC_CSVREADER_H_
#define SRC_CSVREADER_H_
using namespace std;
using namespace Eigen;

class CsvReader {
public:
	CsvReader();
	vector<vector<string>> csvReadToString(string pathCsv);
	MatrixXf csvReadToMatrix(string pathCsv);
	virtual ~CsvReader();
private:
	vector<string> parseLine(string line);
};

#endif /* SRC_CSVREADER_H_ */
