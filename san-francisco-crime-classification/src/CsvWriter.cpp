/*
 * CsvWriter.cpp
 *
 *  Created on: Nov 25, 2015
 *      Author: tobias
 */

#include "CsvWriter.h"

CsvWriter::CsvWriter() {
	// TODO Auto-generated constructor stub

}

void CsvWriter::writeMatrixIn(ofstream &file, MatrixXd &matrix){
	int rows = matrix.rows();
	int cols = matrix.cols();
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols ; j++){
			//add data
			file << matrix(i,j) << ",";
		}
		file << endl;
	}
	file.close();

}

void CsvWriter::makeSubmitWithMatrix(string path,MatrixXd matrix){

	ofstream file(path);
	int rows = matrix.rows();
	int cols = matrix.cols();
	auto start = chrono::steady_clock::now();

	//groups
	vector<string> titulos;
	string campos = "Id,ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS\n"; //o lo hago leendo el submit example????*/

	file << campos;


	//data
	for (int i = 0; i < rows; i++){
		//put id
		file << i;
		for (int j = 0; j < cols ; j++){
			//add data
			file << ',' << matrix(i,j);
		}
		file << endl;
	}
	file.close();

	auto end = chrono::steady_clock::now();
	auto diff = end-start;
	cout << "Tiempo escritura matriz : " << chrono::duration <double, milli> (diff).count() << " ms\n";

}

void CsvWriter::storeWeights(string path_base, vector<MatrixXd>* weights){
	vector<MatrixXd>::iterator it =weights->begin();
	int idx = 0;
	for(; it != weights->end(); it++){
		//path
		string path_location = path_base + to_string(idx) + ".csv";

		//abro archivo y escribo
		ofstream file(path_location);
		this->writeMatrixIn(file,(*it));
		file.close();
		cout << "Escribio matriz de " << (*it).rows() <<"x"<< (*it).cols() <<endl;

		idx++;
	}

}

void CsvWriter::storeBiases(string path,vector<VectorXd>* biases){

	vector<VectorXd>::iterator it = biases->begin();
	ofstream file(path);
	for (; it != biases->end(); it++) {

		int cols = (*it).rows();
		cout << cols << " elementos" << endl;
		for (int j = 0; j < cols; j++) {
			//add data
			file << (*it)(j) << ",";
		}
		file << endl;

	}
	file.close();

}

CsvWriter::~CsvWriter() {
	// TODO Auto-generated destructor stub
}

