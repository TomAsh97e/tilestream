#include "baselines.h"

#include <cuda.h>
#include <nvcomp.h>
#include <nvcomp/lz4.h>

namespace bench {

namespace {

using bench::ok;

bool ok(nvcompStatus_t status, const char *what) {
    if (status != nvcompSuccess) {
        std::fprintf(stderr, "%s failed with nvCOMP status %d\n", what, static_cast<int>(status));
    }
    return status == nvcompSuccess;
}

std::uint64_t align(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

template <class T>
bool upload_array(const std::vector<T> &host, cudaStream_t stream, cuda::DeviceBuffer &buffer) {
    return upload(host.data(), host.size() * sizeof(T), stream, buffer);
}

// The batched LZ4 API on fixed-size chunks of the raw pixels of every image. The chunks are linear
// ranges of the row-major image, not 2-D tiles.
struct Lz4 {
    nvcompBatchedLZ4CompressOpts_t compress = nvcompBatchedLZ4CompressDefaultOpts;
    nvcompBatchedLZ4DecompressOpts_t decompress = nvcompBatchedLZ4DecompressDefaultOpts;
    std::uint64_t chunk = 0;
    std::size_t count = 0;
    std::uint64_t total = 0;
    std::vector<std::uint64_t> sizes;   // Compressed size of every chunk.
    std::vector<std::uint64_t> offsets; // Aligned position of every chunk in the packed stream.
    std::uint64_t packed = 0;
    // Device arrays for compression into fixed slots and for decompression from the packed stream.
    cuda::DeviceBuffer raw_ptrs, raw_sizes, slot_ptrs, slot_sizes, slots;
    cuda::DeviceBuffer packed_ptrs, packed_sizes, actual_sizes, stream_data, statuses;
    cuda::DeviceBuffer compress_temp, decompress_temp;
};

bool installed(Result &r) {
    nvcompProperties_t properties{};
    if (!ok(nvcompGetProperties(&properties), "nvcompGetProperties")) {
        return false;
    }
    if (properties.version != NVCOMP_VER) {
        std::fprintf(stderr, "nvCOMP %u does not match the headers (%u)\n", properties.version,
                     static_cast<unsigned>(NVCOMP_VER));
        return false;
    }
    int runtime = 0;
    if (!ok(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion")) {
        return false;
    }
    if (properties.cudart_version / 1000 != static_cast<unsigned>(runtime) / 1000) {
        std::fprintf(stderr, "nvCOMP was built for CUDA %u, the runtime is %d\n",
                     properties.cudart_version, runtime);
        return false;
    }
    // Zero means that the device has no decompression engine.
    int engine = 0;
    CUdevice device = 0;
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&engine, CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK, device) !=
            CUDA_SUCCESS) {
        engine = -1;
    }
    r.method.emplace_back("nvcomp_version", std::to_string(NVCOMP_VER_MAJOR) + "." +
                                                std::to_string(NVCOMP_VER_MINOR) + "." +
                                                std::to_string(NVCOMP_VER_PATCH) + "." +
                                                std::to_string(NVCOMP_VER_BUILD));
    r.method.emplace_back("nvcomp_cudart", std::to_string(properties.cudart_version));
    r.method.emplace_back("decompression_engine_mask", std::to_string(engine));
    return true;
}

bool configure(const Options &o, const Workload &w, Lz4 &z, Result &r) {
    const auto type = w.images.front().spec.type;
    const auto sample = bench::sample_bytes(type);
    // The controlled default holds as many bytes as one TileStream tile.
    z.chunk = o.chunk != 0 ? o.chunk : std::uint64_t{o.tile.width} * o.tile.height * sample;
    if (o.lz4_type == "typed") {
        z.compress.data_type =
            type == ts::PixelType::uint16 ? NVCOMP_TYPE_USHORT : NVCOMP_TYPE_UINT;
    } else if (o.lz4_type != "char") {
        std::fprintf(stderr, "unknown LZ4 data type %s\n", o.lz4_type.c_str());
        return false;
    }
    z.compress.bitshuffle_mode =
        o.bitshuffle ? NVCOMP_BITSHUFFLE_MSB_FIRST : NVCOMP_BITSHUFFLE_NONE;
    z.decompress.data_type = z.compress.data_type;
    z.decompress.bitshuffle_mode = z.compress.bitshuffle_mode;
    if (o.backend == "cuda") {
        z.decompress.backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;
    } else if (o.backend == "hardware") {
        z.decompress.backend = NVCOMP_DECOMPRESS_BACKEND_HARDWARE;
    } else if (o.backend == "default") {
        z.decompress.backend = NVCOMP_DECOMPRESS_BACKEND_DEFAULT;
    } else {
        std::fprintf(stderr, "unknown backend %s\n", o.backend.c_str());
        return false;
    }
    if (z.chunk % sample != 0 || z.chunk > nvcompLZ4CompressionMaxAllowedChunkSize) {
        std::fprintf(stderr, "chunk of %llu bytes is not supported\n",
                     static_cast<unsigned long long>(z.chunk));
        return false;
    }
    r.method.emplace(r.method.begin(), "codec", "nvcomp_lz4");
    r.method.emplace_back("chunk_bytes", std::to_string(z.chunk));
    r.method.emplace_back("data_type", o.lz4_type);
    r.method.emplace_back("bitshuffle", o.bitshuffle ? "msb_first" : "none");
    r.method.emplace_back("backend", o.backend);
    return true;
}

// Compresses the pixels in t.pixels once, then packs the chunks into one aligned stream on the
// host, as an archive of the method would store them.
bool prepare(const Workload &w, const Target &t, cudaStream_t stream, Lz4 &z, Pinned &host,
             Result &r) {
    nvcompAlignmentRequirements_t in{};
    nvcompAlignmentRequirements_t out{};
    std::size_t slot = 0;
    if (!ok(nvcompBatchedLZ4CompressGetRequiredAlignments(z.compress, &in),
            "nvcompBatchedLZ4CompressGetRequiredAlignments") ||
        !ok(nvcompBatchedLZ4DecompressGetRequiredAlignments(z.decompress, &out),
            "nvcompBatchedLZ4DecompressGetRequiredAlignments") ||
        !ok(nvcompBatchedLZ4CompressGetMaxOutputChunkSize(z.chunk, z.compress, &slot),
            "nvcompBatchedLZ4CompressGetMaxOutputChunkSize")) {
        return false;
    }
    slot = align(slot, std::max(in.output, out.input));
    std::vector<const void *> raw;
    std::vector<std::size_t> raw_sizes;
    for (std::size_t i = 0; i < w.images.size(); ++i) {
        const auto size = w.images[i].bytes.size();
        for (std::uint64_t k = 0; k < size; k += z.chunk) {
            raw.push_back(static_cast<const std::uint8_t *>(t.pixels[i].data()) + k);
            raw_sizes.push_back(std::min<std::uint64_t>(z.chunk, size - k));
        }
        z.total += size;
    }
    z.count = raw.size();
    if (z.chunk % std::max(in.input, out.output) != 0 ||
        !ok(cuda::DeviceBuffer::allocate(z.count * slot, stream, z.slots), "allocate")) {
        std::fprintf(stderr, "chunks do not meet the nvCOMP alignment\n");
        return false;
    }
    std::vector<void *> slots(z.count);
    for (std::size_t k = 0; k < z.count; ++k) {
        slots[k] = static_cast<std::uint8_t *>(z.slots.data()) + k * slot;
    }
    std::size_t compress_temp = 0;
    std::size_t decompress_temp = 0;
    if (!upload_array(raw, stream, z.raw_ptrs) || !upload_array(raw_sizes, stream, z.raw_sizes) ||
        !upload_array(slots, stream, z.slot_ptrs) ||
        !ok(cuda::DeviceBuffer::allocate(z.count * sizeof(std::size_t), stream, z.slot_sizes),
            "allocate") ||
        !ok(cuda::DeviceBuffer::allocate(z.count * sizeof(nvcompStatus_t), stream, z.statuses),
            "allocate") ||
        !ok(nvcompBatchedLZ4CompressGetTempSize(z.count, z.chunk, z.compress, &compress_temp,
                                                z.total, stream),
            "nvcompBatchedLZ4CompressGetTempSize") ||
        !ok(nvcompBatchedLZ4DecompressGetTempSize(z.count, z.chunk, z.decompress, &decompress_temp,
                                                  z.total, stream),
            "nvcompBatchedLZ4DecompressGetTempSize") ||
        !ok(cuda::DeviceBuffer::allocate(compress_temp, stream, z.compress_temp), "allocate") ||
        !ok(cuda::DeviceBuffer::allocate(decompress_temp, stream, z.decompress_temp), "allocate")) {
        return false;
    }
    r.workspace = std::max<std::uint64_t>(r.workspace, std::max(compress_temp, decompress_temp));
    if (!ok(nvcompBatchedLZ4CompressAsync(
                static_cast<const void *const *>(z.raw_ptrs.data()),
                static_cast<const std::size_t *>(z.raw_sizes.data()), z.chunk, z.count,
                z.compress_temp.data(), z.compress_temp.size(),
                static_cast<void *const *>(z.slot_ptrs.data()),
                static_cast<std::size_t *>(z.slot_sizes.data()), z.compress,
                static_cast<nvcompStatus_t *>(z.statuses.data()), stream),
            "nvcompBatchedLZ4CompressAsync")) {
        return false;
    }
    std::vector<std::size_t> sizes(z.count);
    std::vector<nvcompStatus_t> statuses(z.count);
    if (!ok(cudaMemcpyAsync(sizes.data(), z.slot_sizes.data(), z.count * sizeof(std::size_t),
                            cudaMemcpyDeviceToHost, stream),
            "download") ||
        !ok(cudaMemcpyAsync(statuses.data(), z.statuses.data(), z.count * sizeof(nvcompStatus_t),
                            cudaMemcpyDeviceToHost, stream),
            "download") ||
        !ok(cuda::synchronize(stream), "synchronize")) {
        return false;
    }
    for (std::size_t k = 0; k < z.count; ++k) {
        if (!ok(statuses[k], "LZ4 chunk compression")) {
            return false;
        }
        z.sizes.push_back(sizes[k]);
        z.offsets.push_back(z.packed);
        z.packed = align(z.packed + sizes[k], out.input);
    }
    if (!host.allocate(z.packed) ||
        !ok(cuda::DeviceBuffer::allocate(z.packed, stream, z.stream_data), "allocate")) {
        return false;
    }
    for (std::size_t k = 0; k < z.count; ++k) {
        if (!ok(cudaMemcpyAsync(host.data() + z.offsets[k], slots[k], z.sizes[k],
                                cudaMemcpyDeviceToHost, stream),
                "download")) {
            return false;
        }
    }
    std::vector<const void *> packed(z.count);
    for (std::size_t k = 0; k < z.count; ++k) {
        packed[k] = static_cast<const std::uint8_t *>(z.stream_data.data()) + z.offsets[k];
    }
    if (!upload_array(packed, stream, z.packed_ptrs) ||
        !upload_array(std::vector<std::size_t>(z.sizes.begin(), z.sizes.end()), stream,
                      z.packed_sizes) ||
        !ok(cuda::DeviceBuffer::allocate(z.count * sizeof(std::size_t), stream, z.actual_sizes),
            "allocate") ||
        !ok(cudaMemcpyAsync(z.stream_data.data(), host.data(), z.packed, cudaMemcpyHostToDevice,
                            stream),
            "upload") ||
        !ok(cuda::synchronize(stream), "synchronize")) {
        return false;
    }
    r.payload = 0;
    for (const auto size : z.sizes) {
        r.payload += size;
    }
    // The batched API has no container; an archive needs the compressed size of every chunk.
    r.container = z.packed + 8 * z.count;
    r.method.emplace_back("chunks", std::to_string(z.count));
    return true;
}

bool decompress(const Lz4 &z, cudaStream_t stream) {
    return ok(
        nvcompBatchedLZ4DecompressAsync(static_cast<const void *const *>(z.packed_ptrs.data()),
                                        static_cast<const std::size_t *>(z.packed_sizes.data()),
                                        static_cast<const std::size_t *>(z.raw_sizes.data()),
                                        static_cast<std::size_t *>(z.actual_sizes.data()), z.count,
                                        z.decompress_temp.data(), z.decompress_temp.size(),
                                        static_cast<void *const *>(z.raw_ptrs.data()), z.decompress,
                                        static_cast<nvcompStatus_t *>(z.statuses.data()), stream),
        "nvcompBatchedLZ4DecompressAsync");
}

// Every chunk decompressed without error to its full size.
bool check(const Lz4 &z, cudaStream_t stream) {
    std::vector<nvcompStatus_t> statuses(z.count);
    std::vector<std::size_t> actual(z.count);
    std::vector<std::size_t> expected(z.count);
    if (!ok(cudaMemcpyAsync(statuses.data(), z.statuses.data(), z.count * sizeof(nvcompStatus_t),
                            cudaMemcpyDeviceToHost, stream),
            "download") ||
        !ok(cudaMemcpyAsync(actual.data(), z.actual_sizes.data(), z.count * sizeof(std::size_t),
                            cudaMemcpyDeviceToHost, stream),
            "download") ||
        !ok(cudaMemcpyAsync(expected.data(), z.raw_sizes.data(), z.count * sizeof(std::size_t),
                            cudaMemcpyDeviceToHost, stream),
            "download") ||
        !ok(cuda::synchronize(stream), "synchronize")) {
        return false;
    }
    for (std::size_t k = 0; k < z.count; ++k) {
        if (!ok(statuses[k], "LZ4 chunk decompression") || actual[k] != expected[k]) {
            return false;
        }
    }
    return true;
}

bool clear(const Target &t, cudaStream_t stream) {
    for (const auto &pixels : t.pixels) {
        if (!ok(cudaMemsetAsync(pixels.data(), 0xa5, pixels.size(), stream), "cudaMemsetAsync")) {
            return false;
        }
    }
    return true;
}

bool setup(const Options &o, const Workload &w, cudaStream_t stream, Target &t, Lz4 &z,
           Pinned &host, Result &r) {
    if (!installed(r) || !configure(o, w, z, r) || !prepare_target(w, stream, t, r)) {
        return false;
    }
    for (std::size_t i = 0; i < w.images.size(); ++i) {
        if (!ok(cudaMemcpyAsync(t.pixels[i].data(), w.images[i].bytes.data(),
                                w.images[i].bytes.size(), cudaMemcpyHostToDevice, stream),
                "upload")) {
            return false;
        }
    }
    return prepare(w, t, stream, z, host, r) && clear(t, stream);
}

} // namespace

bool lz4_calibrate(const Options &o, const Workload &w, cudaStream_t stream, Result &r) {
    Target t;
    Lz4 z;
    Pinned host;
    if (!setup(o, w, stream, t, z, host, r)) {
        return false;
    }
    const std::vector<Stage> stages{
        {"h2d", true,
         [&] {
             return ok(cudaMemcpyAsync(z.stream_data.data(), host.data(), z.packed,
                                       cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync");
         }},
        {"decompress", true, [&] { return decompress(z, stream); }},
        {"calibrate", true, [&] { return calibrate(t, stream); }},
    };
    if (!measure_stages(stages, o, stream, r) || !check(z, stream) ||
        !verify(w, t.outputs, t.results, stream)) {
        return false;
    }
    r.counter = "uncompressed decoded bytes";
    r.counted = w.uncompressed;
    r.transferred = z.packed;
    r.pinned = host.size();
    return true;
}

bool lz4_gpu(const Options &o, const Workload &w, bool compress, cudaStream_t stream, Result &r) {
    Target t;
    Lz4 z;
    Pinned host;
    if (!setup(o, w, stream, t, z, host, r)) {
        return false;
    }
    if (compress) {
        // The slots are overwritten with the same bytes, so the packed stream stays valid.
        for (std::size_t i = 0; i < w.images.size(); ++i) {
            if (!ok(cudaMemcpyAsync(t.pixels[i].data(), w.images[i].bytes.data(),
                                    w.images[i].bytes.size(), cudaMemcpyHostToDevice, stream),
                    "upload")) {
                return false;
            }
        }
    }
    const auto enqueue = [&] {
        if (!compress) {
            return decompress(z, stream);
        }
        return ok(nvcompBatchedLZ4CompressAsync(
                      static_cast<const void *const *>(z.raw_ptrs.data()),
                      static_cast<const std::size_t *>(z.raw_sizes.data()), z.chunk, z.count,
                      z.compress_temp.data(), z.compress_temp.size(),
                      static_cast<void *const *>(z.slot_ptrs.data()),
                      static_cast<std::size_t *>(z.slot_sizes.data()), z.compress,
                      static_cast<nvcompStatus_t *>(z.statuses.data()), stream),
                  "nvcompBatchedLZ4CompressAsync");
    };
    if (!measure_gpu(enqueue, o, stream, r)) {
        return false;
    }
    if (compress) {
        // The measured compression must reproduce the sizes of the setup and decode back.
        std::vector<std::size_t> sizes(z.count);
        if (!ok(cudaMemcpy(sizes.data(), z.slot_sizes.data(), z.count * sizeof(std::size_t),
                           cudaMemcpyDeviceToHost),
                "download") ||
            !std::equal(sizes.begin(), sizes.end(), z.sizes.begin()) || !clear(t, stream) ||
            !decompress(z, stream)) {
            std::fprintf(stderr, "LZ4 compression is not reproducible\n");
            return false;
        }
    }
    if (!check(z, stream)) {
        return false;
    }
    for (std::size_t i = 0; i < w.images.size(); ++i) {
        Bytes decoded(w.images[i].bytes.size());
        if (!ok(cudaMemcpy(decoded.data(), t.pixels[i].data(), decoded.size(),
                           cudaMemcpyDeviceToHost),
                "download") ||
            decoded != w.images[i].bytes) {
            std::fprintf(stderr, "LZ4 round trip of image %zu differs\n", i);
            return false;
        }
    }
    r.counter = compress ? "uncompressed input bytes" : "uncompressed decoded bytes";
    r.counted = w.uncompressed;
    r.output = compress ? r.payload : w.uncompressed;
    r.transferred = 0;
    return true;
}

} // namespace bench
