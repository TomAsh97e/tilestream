// Runs a fuzz target over corpus files and directories without libFuzzer, so that CTest replays
// the corpus and every regression input in ordinary builds.
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t *data, std::size_t size);

namespace {

void run(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    LLVMFuzzerTestOneInput(bytes.data(), bytes.size());
}

} // namespace

int main(int argc, char **argv) {
    std::size_t count = 0;
    for (int i = 1; i < argc; ++i) {
        const std::filesystem::path path = argv[i];
        if (std::filesystem::is_directory(path)) {
            for (const auto &entry : std::filesystem::directory_iterator(path)) {
                if (entry.is_regular_file()) {
                    run(entry.path());
                    ++count;
                }
            }
        } else if (std::filesystem::exists(path)) {
            run(path);
            ++count;
        } else {
            std::fprintf(stderr, "missing %s\n", argv[i]);
            return 1;
        }
    }
    std::printf("replayed %zu inputs\n", count);
    return 0;
}
