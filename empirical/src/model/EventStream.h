#ifndef EVENTSTREAM_H
#define EVENTSTREAM_H

#include <string>
#include <fstream>
#include <Eigen/Dense>

#include "Types.h"

class EventStream {
	public:
		using value_type = Event;
		using reference = Event&;
		using pointer = Event*;
		using iterator_category = std::forward_iterator_tag;
		using difference_type = std::ptrdiff_t;

		EventStream(std::string filename) : filename(filename), current_event(0.0,0,Eigen::VectorXd({}),0.0) {
			fileStream.open(filename);
			if (fileStream) {
				;
			}
		}

		EventStream(const EventStream& existing) : current_event(*existing), filename(existing.filename), fileStream(existing.fileStream) {}

		explicit operator bool() const {
			return (bool)fileStream;
		}

		reference operator*() { return current_event; }

		pointer operator->() { return &current_event; }

		EventStream& operator++() {
			/*
			std::string line;
		       	std::getline(fileStream, line);
			*/
			return *this;
		}
		EventStream operator++(int) {
			EventStream tmp = *this;
			++(*this);
			return tmp;
		}

	private:
		Event current_event;
		std::string filename;
		std::ifstream fileStream;
};

#endif //EVENTSTREAM_H
