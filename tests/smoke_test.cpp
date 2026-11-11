#include <tilestream/info.h>

#include <iostream>
#include <string>

int main() {
    const std::string info = tilestream::info_json();
    if (std::string(tilestream::version()).empty()) {
        std::cerr << "TileStream version is empty\n";
        return 1;
    }
    if (info.find("\"tilestream_version\"") == std::string::npos ||
        info.find("\"cuda\"") == std::string::npos) {
        std::cerr << "Runtime information is incomplete\n";
        return 1;
    }
    return 0;
}
