#include <iostream>
#include <string>
#include <nlohmann/json.hpp>

std::string replace(const std::string& json_str, std::string from, std::string to) {
	std::string processed = json_str;
	size_t pos = 0;
	while ((pos = processed.find(from, pos)) != std::string::npos) {
		processed.replace(pos, from.length(), to);
		pos += to.length();
	}
	return processed;
}

nlohmann::json parse_line(std::string line) {
	line = replace(line,"ada::mkt_state:","");
	line = replace(line,"nan","null");
	line = replace(line,"bid","\"bid\"");
	line = replace(line,"\"\"bid\"\"","\"bid\"");
	line = replace(line,"ask","\"ask\"");
	line = replace(line,"\"\"ask\"\"","\"ask\"");
	line = replace(line,"final","\"final\"");
	return nlohmann::json::parse(line);
}

int main() {
	std::string line;
	while (std::getline(std::cin, line)) {
		try {
			// Parse each line as JSON
			nlohmann::json j = parse_line(line);
			// Convert JSON back to a string and print
			std::cout << j.dump(4) << std::endl;
		} catch (const nlohmann::json::parse_error& e) {
			std::cerr << "Parse error: " << e.what() << std::endl;
		}
	}
	return 0;
}
