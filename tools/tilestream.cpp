#include <tilestream/info.h>

#include <iostream>
#include <string_view>

namespace {

void print_usage(std::ostream &output) {
    output << "Usage: tilestream info --json\n";
}

} // namespace

int main(int argc, char **argv) {
    if (argc == 3 && std::string_view(argv[1]) == "info" && std::string_view(argv[2]) == "--json") {
        std::cout << tilestream::info_json() << '\n';
        return 0;
    }

    print_usage(std::cerr);
    return 2;
}
