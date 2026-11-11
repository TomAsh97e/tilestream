#pragma once

#include <tilestream/types.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace tilestream {

inline constexpr std::uint64_t max_bytes =
    std::numeric_limits<std::size_t>::max() < std::uint64_t{INT64_MAX}
        ? std::numeric_limits<std::size_t>::max()
        : std::uint64_t{INT64_MAX};
inline constexpr std::uint64_t workspace_alignment = 256;
inline constexpr std::uint64_t max_window_tiles = INT32_MAX;

struct WorkspaceSpec {
    std::uint64_t image_count = 0;
    std::uint64_t tile_count = 0;
    TileShape tile;
    // Query the actual primitive and take the maximum over sequential windows.
    std::uint64_t scan_temp_bytes = 0;
};

[[nodiscard]] Status image_span(const ImageSpec &image, std::uint64_t pitch_bytes,
                                std::uint64_t &bytes) noexcept;
[[nodiscard]] Status compress_bound(const ImageSpec &image, TileShape tile,
                                    std::uint64_t &bytes) noexcept;
[[nodiscard]] Status workspace_size(const WorkspaceSpec &spec, std::uint64_t &bytes) noexcept;

} // namespace tilestream
