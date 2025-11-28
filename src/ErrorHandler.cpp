#include "ErrorHandler.hpp"
#include <iostream>
#include <iomanip>

namespace slam {

void printErrorWithContext(const std::exception& e, const std::string& context) {
    std::cerr << "\n";
    std::cerr << "═══════════════════════════════════════════════════════════\n";
    std::cerr << "ERROR OCCURRED";
    if (!context.empty()) {
        std::cerr << " in " << context;
    }
    std::cerr << "\n";
    std::cerr << "═══════════════════════════════════════════════════════════\n";
    std::cerr << "Message: " << e.what() << "\n";
    
    // Check if it's a ContextException for additional info
    try {
        const ContextException& ctx_e = dynamic_cast<const ContextException&>(e);
        if (!ctx_e.file().empty() || ctx_e.line() > 0) {
            std::cerr << "\nLocation:\n";
            std::cerr << "  File: " << ctx_e.file() << "\n";
            std::cerr << "  Line: " << ctx_e.line() << "\n";
            if (!ctx_e.function().empty()) {
                std::cerr << "  Function: " << ctx_e.function() << "\n";
            }
        }
    } catch (const std::bad_cast&) {
        // Not a ContextException, that's fine
    }
    
    std::cerr << "\n";
    std::cerr << "To debug:\n";
    std::cerr << "  1. Run in Visual Studio debugger (F5) to see full call stack\n";
    std::cerr << "  2. Or run: gdb build/bin/Debug/lidar_range_CNN.exe\n";
    std::cerr << "  3. Set breakpoints before the error location\n";
    std::cerr << "═══════════════════════════════════════════════════════════\n";
}

} // namespace slam

