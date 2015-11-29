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

void CsvWriter::makeSubmitWithMatrix(string path,MatrixXf matrix){

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

CsvWriter::~CsvWriter() {
	// TODO Auto-generated destructor stub
}

