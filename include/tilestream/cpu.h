#pragma once

#include <tilestream/api.h>

namespace tilestream::cpu {

// Buffers are caller-owned and must not overlap. FP32 is handled as exact bit words.
// Compression requires compress_bound capacity; failures leave outputs unchanged.
[[nodiscard]] Status compress(ConstImageView input, TileShape tile, MutableBytes output,
                              std::uint64_t &bytes_written) noexcept;
[[nodiscard]] Status decompress(ConstBytes input, ImageView output,
                                const ParseLimits &limits = {}) noexcept;

} // namespace tilestream::cpu
