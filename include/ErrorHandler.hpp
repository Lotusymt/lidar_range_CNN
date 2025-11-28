#pragma once

#include <string>
#include <stdexcept>
#include <sstream>

namespace slam {

/**
 * @brief Enhanced exception with context information
 */
class ContextException : public std::runtime_error {
public:
    ContextException(const std::string& message, 
                     const std::string& file = "",
                     int line = 0,
                     const std::string& function = "")
        : std::runtime_error(formatMessage(message, file, line, function))
        , file_(file), line_(line), function_(function) {}

    const std::string& file() const { return file_; }
    int line() const { return line_; }
    const std::string& function() const { return function_; }

private:
    std::string formatMessage(const std::string& msg,
                              const std::string& file,
                              int line,
                              const std::string& func) const {
        std::ostringstream oss;
        oss << msg;
        if (!file.empty() || line > 0 || !func.empty()) {
            oss << " [";
            if (!func.empty()) oss << func << "()";
            if (!file.empty()) {
                if (!func.empty()) oss << " ";
                // Extract just filename from path
                size_t pos = file.find_last_of("\\/");
                oss << (pos == std::string::npos ? file : file.substr(pos + 1));
            }
            if (line > 0) oss << ":" << line;
            oss << "]";
        }
        return oss.str();
    }

    std::string file_;
    int line_;
    std::string function_;
};

/**
 * @brief Print error with context and attempt to show stack trace
 */
void printErrorWithContext(const std::exception& e, const std::string& context = "");

} // namespace slam

// Macro for throwing exceptions with context
#define THROW_CONTEXT(msg) throw slam::ContextException(msg, __FILE__, __LINE__, __FUNCTION__)
#define THROW_CONTEXT_IF(condition, msg) if (condition) THROW_CONTEXT(msg)

