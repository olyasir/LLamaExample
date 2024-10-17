#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdint>

/**
 * @brief Reads the contents of a file and returns it as a string.
 * 
 * This function opens the file at the specified path, reads its entire contents,
 * and returns it as a std::string. If the file cannot be opened, it throws
 * a std::runtime_error.
 * 
 * @param file_path The path to the file to be read.
 * @return std::string The contents of the file.
 * @throws std::runtime_error If the file cannot be opened.
 */
std::string get_file_contents(const std::string& file_path);

/**
 * @brief Reads the contents of a file and returns it as a vector of bytes.
 * 
 * This function opens the file at the specified path in binary mode, reads its
 * entire contents, and returns it as a std::vector<uint8_t>. Each element in the
 * vector represents a byte from the file.
 * 
 * @param file_path The path to the file to be read.
 * @return std::vector<uint8_t> The contents of the file as a vector of bytes.
 * @throws std::runtime_error If the file cannot be opened or read.
 */
std::vector<uint8_t> get_bytes_from_file(std::string file_path);





