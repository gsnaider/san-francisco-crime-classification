/*
 * CsvReader.cpp
 *
 *  Created on: Nov 25, 2015
 *      Author: tobias
 */

#include "CsvReader.h"

CsvReader::CsvReader() {
	// TODO Auto-generated constructor stub

}

istream& operator>>(std::istream& str,CsvRow& data){
    data.readNextRow(str);
    return str;
}
int countLines(string path){
	//se podria optimizar
	auto start = chrono::steady_clock::now();
	ifstream file(path);
	string line;
	int i;
	for ( i = 0; std::getline(file, line); ++i);
	auto end = chrono::steady_clock::now();
	auto diff = end-start;
	cout << "Obtener total lineas: " << chrono::duration <double, milli> (diff).count() << " ms\n";
	return i;
}
vector<vector<string> > CsvReader::csvReadToString(string path){
	auto start = chrono::steady_clock::now();
	vector< vector<string> > text;
	ifstream file(path);
	int linea = 0;
	CsvRow row;
	while (file >> row) {
		linea++;
		text.push_back(row.returnData());
	}
	file.close();
	auto end = chrono::steady_clock::now();
	auto diff = end-start;
	cout << "Leer y meter todo en strings: " << chrono::duration <double, milli> (diff).count() << " ms\n";
	printf("Lineas leidas %d\n",linea);
	return text;
}

MatrixXf CsvReader::csvReadToMatrix(string path){

	//toma la cantidad de elementos en la primera linea para setear columnas
	//la primera linea solo tiene que tener comas separando datos
	auto start = chrono::steady_clock::now();
	ifstream file(path);
	CsvRow row;
	int rows = 1;
	int columns;
	int lines = countLines(path);
	cout << "Lineas: " << lines << "\n";
	MatrixXf matrix;

	//hago do while para fijarme la cantidad de elementos del csv
	if (!file.eof()){
		file >> row;
	}else{
		printf("Archivo vacio\n");
		return matrix;
	}

	//fijo cantidad de columnas
	columns = row.returnData().size();
	matrix.resize(lines,columns);

	do {

		int column = 0;
		vector<string> data = row.returnData();
		vector<string>::iterator it = data.begin();

		//matrix.conservativeResize(rows,NoChange);

		//por cada elemento leido meto en la row de matriz
		for (;it!=data.end();it++){

			matrix(rows-1,column) = (strtof((*it).c_str(),NULL));
			//por cada dato avanzo una columna en la matriz
			column++;

		}
		//cheque que tengan el mismo tamanio, podria sacarlo si estoy seguro q esta bien formateado
		if (column != columns ){
			printf("Error en formato csv, lineas de distintos tamanios. Linea %d\n",rows);
			printf("Tam base: %d, tam original %d\n",columns,column);
			MatrixXf vacio;
			return vacio;

		}

		//aumento cantidad de rows que es lo mismo que cantidad de lineas
		rows++;

	} while (file >> row);

	file.close();

	auto end = chrono::steady_clock::now();
	auto diff = end-start;

	cout << "Tiempo carga de matriz: " << chrono::duration <double, milli> (diff).count() << " ms\n";
	cout << "Proceso " << rows-1 << "lineas\n";

	return matrix;
}

vector<MatrixXf> CsvReader::readWheights(string path_base, int layers){
	vector<MatrixXf> wheights;

	for (int layer = 0; layer < layers ; layer++){
		string path_lectura = path_base + to_string(layer) + ".csv";
		MatrixXf matrix = this->csvReadToMatrix(path_lectura);
		wheights.push_back(matrix);
		cout << "Levanto una matriz de " << matrix.rows()<<"x"<< matrix.cols() << endl;
	}

	return wheights;
}

vector<VectorXf> CsvReader::readBiases(string path){
	auto start = chrono::steady_clock::now();
	ifstream file(path);
	CsvRow row;
	vector<VectorXf> biases;

	while (file >> row){
		vector<string> data = row.returnData();

		VectorXf bias(data.size());
		int idx = 0;
		vector<string>::iterator it = data.begin();
		for (; it != data.end(); it++ ){
			bias(idx) = strtof((*it).c_str(),NULL);
			idx++;
		}
		cout << "Levanto un vector de " << bias.rows() << endl;
		biases.push_back(bias);
	}

	return biases;
}

CsvReader::~CsvReader() {
	// TODO Auto-generated destructor stub
}

