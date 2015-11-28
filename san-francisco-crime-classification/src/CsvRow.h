/*
 * CsvRow.h
 *
 *  Created on: Nov 25, 2015
 *      Author: tobias
 */
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#ifndef SRC_CSVROW_H_
#define SRC_CSVROW_H_
using namespace std;

class CsvRow {
public:
	CsvRow();
        vector<string> returnData(){
        	return m_data;
        }
        void readNextRow(istream &str)
        {
            string line;

            //lee linea entera
            getline(str,line);

            //la transforma en stream
            stringstream lineStream(line);
            string cell;

            //limpia data vieja
            m_data.clear();

            //la separa en pedacitos
            while(getline(lineStream,cell,','))
            {
                m_data.push_back(cell);
            }
        }
	virtual ~CsvRow();
private:
    std::vector<std::string>    m_data;
};

#endif /* SRC_CSVROW_H_ */
