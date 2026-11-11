// Valid files with mutated offsets, sizes, modes, widths, versions and tile counts.
#include "harness.h"

#include <cstdio>
#include <cstdlib>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t *data, std::size_t size) {
    const auto problem = fuzz::check_mutation(data, size);
    if (!problem.empty()) {
        std::fprintf(stderr, "%s\n", problem.c_str());
        std::abort();
    }
    return 0;
}
