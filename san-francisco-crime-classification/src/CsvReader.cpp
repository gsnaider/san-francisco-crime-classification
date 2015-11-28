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
	vector<vector<float>> matrixBase;
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

			//ESTE IF PUEDE VOLAR SI NO HAY COMAS ENTRE " "
			if ((*it).at(0)=='"'){

				string partido = (*it);

				do{
					it++;
					partido+=(*it);
				}while ((*(*it).rbegin())!='"');
				//si, ese choclo devuelve el ultimo char leido

				matrix(rows-1,column) = strtof((partido).c_str(),NULL);

			}else{

				matrix(rows-1,column) = (strtof((*it).c_str(),NULL));
			}

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

	} while (file >> row || rows < lines+1);
	file.close();
	auto end = chrono::steady_clock::now();
	auto diff = end-start;
	cout << "Tiempo carga de matriz: " << chrono::duration <double, milli> (diff).count() << " ms\n";
	cout << "Proceso " << rows-1 << "lineas\n";
	//cout << "Elemento 2,9: " << matrix(1,8) << "\n";
	//cout << "Elemnto 2,8: " << matrix(1,7) << "\n";
	return matrix;
}

CsvReader::~CsvReader() {
	// TODO Auto-generated destructor stub
}

