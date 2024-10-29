#ifndef PARSE_H
#define PARSE_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>
#include <cassert>

#include "Types.h"

const int seconds_in_day = 60*60*24;

class EventIterator {
	public:
		EventIterator(const std::string& filename) : file(filename), done(false) {
			if (!file.is_open()) {
				done = true;
			} else {
				std::string line;
				std::getline(file, line);	
				std::getline(file, line);	
				readNextLine();
			}
		}

		bool operator!=(const EventIterator& other) const {
			return !done;
		}

		const Event* operator*() const {
			return currentEvent;
		}

		EventIterator& operator++() {
			readNextLine();
			return *this;
		}

	private:
		std::ifstream file;
		std::vector<std::string> currentRow;
		Event *currentEvent = NULL;
		bool done;

		REAL mid_es = 0;
		REAL mid_mes = 0;
		REAL spread_es = 0;
		REAL spread_mes = 0;
		REAL imbalance_es = 0;
		REAL imbalance_mes = 0;
		REAL pressure_es = 0;
		REAL pressure_mes = 0;
		REAL bq_es = 0;
		REAL bq_mes = 0;
		REAL aq_es = 0;
		REAL aq_mes = 0;

		void readNextLine() {
			std::string line;
			if (std::getline(file, line)) {
				std::stringstream lineStream(line);
				std::string cell;
				currentRow.clear();

				while (std::getline(lineStream, cell, '|')) {
					currentRow.push_back(cell);
				}

				//       0|         1|            2|       3|     4|   5|   6|       7|       8|    9|              10|            11|                    12|   13|14|15|16|17
				// seconds|instrument|instrument_id|ts_event|action|size|side|order_id|sequence|price|cancelled_orders|modified_order|modified_order_newsize|price|bq|bp|aq|ap
				// 0.027280861|MES|7114|1720396800027280505|A|3|A|6853508913474|531179|5612750000000|[]|||5612.75|34|5612.5|7|5612.75
				if (currentRow.size() != 18) {
					std::cout << "WARNING: READ " << currentRow.size() << " ITEMS: " << line << std::endl;
					readNextLine();
				}

				long double time = std::stod(currentRow[0]);
				std::string ticker = currentRow[1];
				std::string action = currentRow[4];
				int size = std::stoi(currentRow[5]);
				std::string side = currentRow[6];
				long double price;
				if (currentRow[13]=="") {
					price = 0;

				} else {
					price = std::stod(currentRow[13]);
				}
				int bq = std::stoi(currentRow[14]);
				long double bp = std::stod(currentRow[15]);
				int aq = std::stoi(currentRow[16]);
				long double ap = std::stod(currentRow[17]);

				/*
				std::string ticker = currentRow[0];
				long double ts_event = std::stod(currentRow[1]);
				long double ts_recv = std::stod(currentRow[2]);
				long double time = std::stod(currentRow[3]);
				//order_id
				std::string action = currentRow[5];
				std::string side = currentRow[6];
				int size = std::stoi(currentRow[7]);
				double price = std::stod(currentRow[8]);
				int bq = 0;//std::stoi(currentRow[9]);
				double bp = 0;//std::stod(currentRow[10]);
				int aq = 0;//std::stoi(currentRow[11]);
				double ap = 0;//std::stod(currentRow[12]);

				*/

				//'AB', 'AA', 'CB', 'CA', 'MA', 'MB', 'TA', 'FB', 'TB', 'FA'
				int event_type = -1;
				if (action=="A") {
					event_type = 0;
				} else if (action=="C") {
					event_type = 1;
				} else if (action=="M") {
					event_type = 2;
				} else if (action=="T") {
					event_type = 3;
				}


				if (event_type != -1) {
					event_type *= 2;
					if (ticker=="MES") {
						event_type += 1;

						spread_mes = (ap>0 & bp>0) ? ap-bp : 0;
						mid_mes = bp+spread_mes/2;
						imbalance_mes = bq/(aq+bq);
						pressure_mes = imbalance_mes*spread_mes;
						bq_mes = bq;
						aq_mes = aq;
					} else if (ticker=="ES") {
						spread_es = (ap>0 & bp>0) ? ap-bp : 0;
						mid_es = bp+spread_es/2;
						imbalance_es = bq/(aq+bq);
						pressure_es = imbalance_es*spread_es;
						bq_es = bq;
						aq_es = aq;
					} else if (side=="N") {
						event_type = -1;
					}
				}


				if (event_type != -1) {
					event_type *= 2;
					if (side=="A") {
						event_type += 1;
					} else if (side=="N") {
						event_type = -1;
					}
				}


				if (event_type==-1) {
					readNextLine();
				} else {
					if (currentEvent) {
						delete currentEvent;
					}
					REAL spread=ap-bp;
					REAL mid = bp+spread/2;
					REAL imbalance = bq/(bq+aq);
					REAL weighted_mid = bp+imbalance*spread;

					std::vector<REAL> marks {(mid_es>0 & mid_mes>0) ? mid_es-mid_mes : 0, spread_es, spread_mes, imbalance_es, imbalance_mes, pressure_es, pressure_mes, bq_es, bq_mes, aq_es, aq_mes, (REAL)size, price-mid};
					int num_mark_vars = marks.size();
					Eigen::VectorXd marks_eigen = Eigen::VectorXd::Constant(2*num_mark_vars,0.0);
					for (int i = 0; i < num_mark_vars; i++) {
						marks_eigen.array()[i]               = (double) marks[i];
						marks_eigen.array()[num_mark_vars+i] = (double)(marks[i]*marks[i]);
					}
					currentEvent = new Event(time, event_type, marks_eigen, 1.0);
				}


			} else {
				done = true; // No more lines to read
			}
		}
};

class Realisation {
public:
    Realisation(const std::string& filename) : filename(filename) {}

    EventIterator begin() {
        return EventIterator(filename);
    }

    EventIterator end() {
        return EventIterator("");
    }

private:
    std::string filename;
};

#endif //PARSE_H
