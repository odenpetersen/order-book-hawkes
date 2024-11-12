#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Dense>
#include <iostream>

#define REAL long double
#define VECTOR Eigen::VectorXd
#define MATRIX Eigen::MatrixXd
#define GENERATOR Realisation

struct Event {
	REAL time;
	int event_type;
	Eigen::VectorXd marks;
	REAL weight;

	Event(REAL time, int event_type, Eigen::VectorXd marks, REAL weight) : time(time), event_type(event_type), marks(marks), weight(weight) {}
};

std::ostream& operator<<(std::ostream& os, const Event& e) {
	return os << "Event(" << e.time << "," << e.event_type << "," << e.marks << "," << e.weight << ")";
}

#endif //TYPES_H

