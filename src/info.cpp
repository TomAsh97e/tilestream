#include <tilestream/info.h>

#include <tilestream/build_config.h>

#if TILESTREAM_WITH_CUDA
#include "cuda/device_info.h"
#endif

#include <sys/sysinfo.h>
#include <sys/utsname.h>

#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

namespace tilestream {
namespace {

std::string json_escape(const std::string &value) {
    std::ostringstream output;
    for (const char value_character : value) {
        const auto character = static_cast<unsigned char>(value_character);
        switch (character) {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\b':
            output << "\\b";
            break;
        case '\f':
            output << "\\f";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (character < 0x20U) {
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<unsigned int>(character) << std::dec;
            } else {
                output << static_cast<char>(character);
            }
        }
    }
    return output.str();
}

std::string cpu_name() {
    std::ifstream cpu_info("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpu_info, line)) {
        constexpr const char *key = "model name";
        if (line.compare(0, 10, key) != 0) {
            continue;
        }
        const std::size_t separator = line.find(':');
        if (separator == std::string::npos) {
            continue;
        }
        std::string name = line.substr(separator + 1);
        const std::size_t first = name.find_first_not_of(" \t");
        if (first != std::string::npos) {
            name.erase(0, first);
        }
        return name;
    }
    return "unknown";
}

std::string os_name() {
    utsname system{};
    if (uname(&system) != 0) {
        return "unknown";
    }
    return std::string(system.sysname) + " " + system.release + " " + system.machine;
}

std::uint64_t memory_bytes() {
    struct sysinfo information {};
    if (sysinfo(&information) != 0) {
        return 0;
    }
    return static_cast<std::uint64_t>(information.totalram) *
           static_cast<std::uint64_t>(information.mem_unit);
}

} // namespace

const char *version() noexcept {
    return TILESTREAM_VERSION;
}

std::string info_json() {
    std::ostringstream output;
    output << "{\n"
           << "  \"tilestream_version\": \"" << TILESTREAM_VERSION << "\",\n"
           << "  \"build_type\": \"" << json_escape(TILESTREAM_BUILD_TYPE) << "\",\n"
           << "  \"compiler\": \"" << json_escape(TILESTREAM_CXX_COMPILER) << "\",\n"
           << "  \"system\": {\n"
           << "    \"os\": \"" << json_escape(os_name()) << "\",\n"
           << "    \"cpu\": \"" << json_escape(cpu_name()) << "\",\n"
           << "    \"ram_bytes\": " << memory_bytes() << "\n"
           << "  },\n"
           << "  \"cuda\": {\n"
           << "    \"compiled\": " << (TILESTREAM_WITH_CUDA ? "true" : "false") << ",\n"
           << "    \"toolkit_version\": \"" << TILESTREAM_CUDA_VERSION << "\",\n";

#if TILESTREAM_WITH_CUDA
    const detail::CudaInfo cuda = detail::query_cuda_info();
    output << "    \"available\": " << (cuda.available ? "true" : "false") << ",\n"
           << "    \"status\": \"" << json_escape(cuda.status) << "\",\n"
           << "    \"device_count\": " << cuda.device_count << ",\n"
           << "    \"driver_version\": " << cuda.driver_version << ",\n"
           << "    \"runtime_version\": " << cuda.runtime_version << ",\n"
           << "    \"device_name\": \"" << json_escape(cuda.device_name) << "\",\n"
           << "    \"compute_capability\": \"" << cuda.compute_major << '.' << cuda.compute_minor
           << "\",\n"
           << "    \"memory_bytes\": " << cuda.memory_bytes << "\n";
#else
    output << "    \"available\": false,\n"
           << "    \"status\": \"CUDA support was not built\",\n"
           << "    \"device_count\": 0\n";
#endif

    output << "  },\n"
           << "  \"nvcomp_enabled\": " << (TILESTREAM_WITH_NVCOMP ? "true" : "false") << "\n"
           << '}';
    return output.str();
}

} // namespace tilestream
