#include "baselines.h"

#include <tilestream/tiff.h>

#include <tiffio.h>
#include <zstd.h>

#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <thread>

namespace bench {

namespace {

// Persistent threads, so that an operation does not pay for thread creation.
class Workers {
  public:
    explicit Workers(unsigned count) : count_(count) {
        for (unsigned k = 1; k < count; ++k) {
            threads_.emplace_back([this, k] { loop(k); });
        }
    }
    ~Workers() {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            ++generation_;
        }
        wake_.notify_all();
        for (auto &thread : threads_) {
            thread.join();
        }
    }
    Workers(const Workers &) = delete;
    Workers &operator=(const Workers &) = delete;

    // Runs job(k) for every worker k; k = 0 runs on the calling thread.
    bool run(const std::function<bool(unsigned)> &job) {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            job_ = &job;
            pending_ = count_ - 1;
            failed_ = false;
            ++generation_;
        }
        wake_.notify_all();
        const bool passed = job(0);
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this] { return pending_ == 0; });
        return passed && !failed_;
    }

  private:
    void loop(unsigned k) {
        std::uint64_t seen = 0;
        for (;;) {
            const std::function<bool(unsigned)> *job = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [&] { return generation_ != seen; });
                seen = generation_;
                if (stop_) {
                    return;
                }
                job = job_;
            }
            const bool passed = (*job)(k);
            const std::lock_guard<std::mutex> lock(mutex_);
            failed_ = failed_ || !passed;
            if (--pending_ == 0) {
                done_.notify_one();
            }
        }
    }

    unsigned count_;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    const std::function<bool(unsigned)> *job_ = nullptr;
    std::uint64_t generation_ = 0;
    unsigned pending_ = 0;
    bool failed_ = false;
    bool stop_ = false;
};

float host_calibrate(float x) {
    const float difference = x - 100.0F;
    return difference * 1.25F;
}

// Offsets of the images in one contiguous host block.
std::vector<std::uint64_t> layout(const Workload &w, bool archives) {
    std::vector<std::uint64_t> offsets;
    std::uint64_t at = 0;
    for (std::size_t i = 0; i < w.images.size(); ++i) {
        offsets.push_back(at);
        at += archives ? w.files[i].size() : w.images[i].bytes.size();
    }
    offsets.push_back(at);
    return offsets;
}

bool copy_pixels(const Target &t, const Workload &w, const std::uint8_t *host,
                 const std::vector<std::uint64_t> &at, cudaStream_t stream) {
    for (std::size_t i = 0; i < w.images.size(); ++i) {
        if (!ok(cudaMemcpyAsync(t.pixels[i].data(), host + at[i], w.images[i].bytes.size(),
                                cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync")) {
            return false;
        }
    }
    return true;
}

// Library versions seen at run time must match the headers the adapters were built with.
bool zstd_installed(Result &r) {
    if (ZSTD_versionNumber() != ZSTD_VERSION_NUMBER) {
        std::fprintf(stderr, "libzstd %s does not match the headers (%s)\n", ZSTD_versionString(),
                     ZSTD_VERSION_STRING);
        return false;
    }
    r.method.emplace_back("zstd_version", ZSTD_versionString());
    return true;
}

bool tiff_installed(Result &r) {
    const std::string version = TIFFGetVersion();
    const std::string expected = "LIBTIFF, Version " + std::to_string(TIFFLIB_MAJOR_VERSION) + "." +
                                 std::to_string(TIFFLIB_MINOR_VERSION) + "." +
                                 std::to_string(TIFFLIB_MICRO_VERSION);
    if (version.rfind(expected + "\n", 0) != 0) {
        std::fprintf(stderr, "libtiff %s does not match the headers (%s)\n", version.c_str(),
                     expected.c_str());
        return false;
    }
    r.method.emplace_back("libtiff_version", expected.substr(17));
    return true;
}

std::uint64_t tile_bytes(const char *path) {
    TIFF *tiff = TIFFOpen(path, "r");
    if (tiff == nullptr) {
        return 0;
    }
    std::uint64_t *counts = nullptr;
    std::uint64_t sum = 0;
    if (TIFFGetField(tiff, TIFFTAG_TILEBYTECOUNTS, &counts) == 1) {
        for (std::uint32_t k = 0; k < TIFFNumberOfTiles(tiff); ++k) {
            sum += counts[k];
        }
    }
    TIFFClose(tiff);
    return sum;
}

bool read_file(const std::string &path, std::uint8_t *data, std::uint64_t size) {
    std::FILE *file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return false;
    }
    const bool read = std::fread(data, 1, size, file) == size && std::fgetc(file) == EOF;
    std::fclose(file);
    if (!read) {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
    }
    return read;
}

bool write_file(const std::string &path, const void *data, std::uint64_t size) {
    std::ofstream out(path, std::ios::binary);
    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    return static_cast<bool>(out);
}

} // namespace

bool prepare_target(const Workload &w, cudaStream_t stream, Target &t, Result &r) {
    const auto n = w.images.size();
    t.pixels.resize(n);
    t.outputs.resize(n);
    std::vector<cuda::BatchItem> items(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto &image = w.images[i];
        const auto &spec = image.spec;
        if (!ok(cuda::DeviceBuffer::allocate(image.bytes.size(), stream, t.pixels[i]),
                "allocate") ||
            !ok(cuda::DeviceBuffer::allocate(std::uint64_t{spec.width} * spec.height * 4, stream,
                                             t.outputs[i]),
                "allocate")) {
            return false;
        }
        items[i].input = {t.pixels[i].data(), spec, image.row(), t.pixels[i].size()};
        items[i].output = {t.outputs[i].data(),
                           {ts::PixelType::float32, spec.width, spec.height},
                           spec.width * 4ULL,
                           t.outputs[i].size()};
        calibration(items[i]);
        r.output += t.outputs[i].size();
    }
    return ok(cuda::prepare_batch({cuda::Operation::calibrate, items.data(), n, {}}, t.plan,
                                  stream),
              "prepare_batch") &&
           workspace_for(*t.plan, stream, t.workspace, r) &&
           ok(cuda::DeviceBuffer::allocate(n * sizeof(ts::DeviceResult), stream, t.results),
              "allocate");
}

bool calibrate(const Target &t, cudaStream_t stream) {
    return ok(cuda::calibrate_async(
                  *t.plan, t.workspace.bytes(),
                  {static_cast<ts::DeviceResult *>(t.results.data()), t.pixels.size()}, stream),
              "calibrate_async");
}

bool verify(const Workload &w, const std::vector<cuda::DeviceBuffer> &outputs,
            const cuda::DeviceBuffer &results, cudaStream_t stream) {
    std::vector<ts::DeviceResult> records;
    if (!check_results(results, w.images.size(), stream, records)) {
        return false;
    }
    for (std::size_t i = 0; i < w.images.size(); ++i) {
        const auto &image = w.images[i];
        const auto count = std::uint64_t{image.spec.width} * image.spec.height;
        std::vector<std::uint32_t> device(count);
        if (outputs[i].size() != count * 4 ||
            !ok(cudaMemcpy(device.data(), outputs[i].data(), count * 4, cudaMemcpyDeviceToHost),
                "download")) {
            return false;
        }
        for (std::uint64_t k = 0; k < count; ++k) {
            float x = 0;
            if (image.spec.type == ts::PixelType::uint16) {
                std::uint16_t v = 0;
                std::memcpy(&v, image.bytes.data() + 2 * k, 2);
                x = v;
            } else {
                std::memcpy(&x, image.bytes.data() + 4 * k, 4);
            }
            const float y = host_calibrate(x);
            std::uint32_t expected = 0;
            std::memcpy(&expected, &y, 4);
            const bool nan = std::isnan(y) && (device[k] & 0x7fffffffU) > 0x7f800000U;
            if (device[k] != expected && !nan) {
                std::fprintf(stderr, "calibrated output of image %zu differs at sample %llu\n", i,
                             static_cast<unsigned long long>(k));
                return false;
            }
        }
    }
    return true;
}

bool raw_calibrate(const Options &o, const Workload &w, cudaStream_t stream, Result &r) {
    const auto at = layout(w, false);
    Target t;
    Pinned host;
    if (!prepare_target(w, stream, t, r) || !host.allocate(at.back())) {
        return false;
    }
    for (std::size_t i = 0; i < w.images.size(); ++i) {
        std::memcpy(host.data() + at[i], w.images[i].bytes.data(), w.images[i].bytes.size());
    }
    const std::vector<Stage> stages{
        {"h2d", true, [&] { return copy_pixels(t, w, host.data(), at, stream); }},
        {"calibrate", true, [&] { return calibrate(t, stream); }},
    };
    if (!measure_stages(stages, o, stream, r) || !verify(w, t.outputs, t.results, stream)) {
        return false;
    }
    r.counter = "uncompressed decoded bytes";
    r.counted = w.uncompressed;
    r.transferred = w.uncompressed;
    r.payload = r.container = w.uncompressed;
    r.pinned = host.size();
    r.method = {{"codec", "raw"}};
    return true;
}

bool tsc_calibrate(const Options &o, const Workload &w, cudaStream_t stream, Result &r) {
    const auto n = w.images.size();
    const auto at = layout(w, true);
    Pinned host;
    std::vector<cuda::DeviceBuffer> sources(n);
    std::vector<cuda::DeviceBuffer> outputs(n);
    std::vector<cuda::BatchItem> items(n);
    if (!host.allocate(at.back())) {
        return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
        const auto &spec = w.images[i].spec;
        std::memcpy(host.data() + at[i], w.files[i].data(), w.files[i].size());
        if (!ok(cuda::DeviceBuffer::allocate(w.files[i].size(), stream, sources[i]), "allocate") ||
            !ok(cuda::DeviceBuffer::allocate(std::uint64_t{spec.width} * spec.height * 4, stream,
                                             outputs[i]),
                "allocate")) {
            return false;
        }
        items[i].compressed_input.bytes = {sources[i].data(), w.files[i].size()};
        items[i].compressed_input.metadata = &w.metadata[i];
        items[i].roi = {0, 0, spec.width, spec.height};
        items[i].output = {outputs[i].data(),
                           {ts::PixelType::float32, spec.width, spec.height},
                           spec.width * 4ULL,
                           outputs[i].size()};
        calibration(items[i]);
        r.output += outputs[i].size();
    }
    std::unique_ptr<cuda::BatchPlan> plan;
    cuda::DeviceBuffer workspace;
    cuda::DeviceBuffer results;
    if (!ok(cuda::prepare_batch({cuda::Operation::decode_calibrate, items.data(), n, {}}, plan,
                                stream),
            "prepare_batch") ||
        !workspace_for(*plan, stream, workspace, r) ||
        !ok(cuda::DeviceBuffer::allocate(n * sizeof(ts::DeviceResult), stream, results),
            "allocate")) {
        return false;
    }
    const std::vector<Stage> stages{
        {"h2d", true,
         [&] {
             for (std::size_t i = 0; i < n; ++i) {
                 if (!ok(cudaMemcpyAsync(sources[i].data(), host.data() + at[i], w.files[i].size(),
                                         cudaMemcpyHostToDevice, stream),
                         "cudaMemcpyAsync")) {
                     return false;
                 }
             }
             return true;
         }},
        {"decode_calibrate", true,
         [&] {
             return ok(cuda::decode_calibrate_async(
                           *plan, workspace.bytes(),
                           {static_cast<ts::DeviceResult *>(results.data()), n}, stream),
                       "decode_calibrate_async");
         }},
    };
    if (!measure_stages(stages, o, stream, r) || !verify(w, outputs, results, stream)) {
        return false;
    }
    r.counter = "uncompressed decoded bytes";
    r.counted = w.uncompressed;
    r.transferred = w.container;
    r.pinned = host.size();
    r.method = {{"codec", "tilestream"}};
    return true;
}

// Independent frames of a fixed raw size, so that threads can decompress them in parallel straight
// into the pinned staging buffer.
bool zstd_calibrate(const Options &o, const Workload &w, cudaStream_t stream, Result &r) {
    struct Frame {
        std::uint64_t raw = 0;
        std::uint64_t raw_size = 0;
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
    };
    const auto frame_bytes = o.chunk != 0 ? o.chunk : std::uint64_t{1} << 20;
    const int level = o.level != 0 ? o.level : ZSTD_CLEVEL_DEFAULT;
    if (!zstd_installed(r) || o.threads == 0) {
        return false;
    }
    const auto at = layout(w, false);
    std::vector<Frame> frames;
    Bytes compressed;
    std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)> cctx(ZSTD_createCCtx(), ZSTD_freeCCtx);
    for (std::size_t i = 0; i < w.images.size(); ++i) {
        const auto &bytes = w.images[i].bytes;
        for (std::uint64_t k = 0; k < bytes.size(); k += frame_bytes) {
            Frame f{at[i] + k, std::min<std::uint64_t>(frame_bytes, bytes.size() - k),
                    compressed.size(), 0};
            compressed.resize(f.offset + ZSTD_compressBound(f.raw_size));
            f.size = ZSTD_compressCCtx(cctx.get(), compressed.data() + f.offset,
                                       compressed.size() - f.offset, bytes.data() + k, f.raw_size,
                                       level);
            if (ZSTD_isError(f.size) != 0) {
                std::fprintf(stderr, "ZSTD_compressCCtx: %s\n", ZSTD_getErrorName(f.size));
                return false;
            }
            compressed.resize(f.offset + f.size);
            frames.push_back(f);
        }
    }
    Target t;
    Pinned staging;
    if (!prepare_target(w, stream, t, r) || !staging.allocate(at.back())) {
        return false;
    }
    std::vector<std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)>> contexts;
    for (unsigned k = 0; k < o.threads; ++k) {
        contexts.emplace_back(ZSTD_createDCtx(), ZSTD_freeDCtx);
    }
    Workers workers(o.threads);
    const std::function<bool(unsigned)> job = [&](unsigned k) {
        const auto begin = frames.size() * k / o.threads;
        const auto end = frames.size() * (k + 1) / o.threads;
        for (auto f = begin; f < end; ++f) {
            const auto &frame = frames[f];
            const auto size =
                ZSTD_decompressDCtx(contexts[k].get(), staging.data() + frame.raw, frame.raw_size,
                                    compressed.data() + frame.offset, frame.size);
            if (size != frame.raw_size) {
                return false;
            }
        }
        return true;
    };
    const std::vector<Stage> stages{
        {"decompress", false, [&] { return workers.run(job); }},
        {"h2d", true, [&] { return copy_pixels(t, w, staging.data(), at, stream); }},
        {"calibrate", true, [&] { return calibrate(t, stream); }},
    };
    if (!measure_stages(stages, o, stream, r) || !verify(w, t.outputs, t.results, stream)) {
        return false;
    }
    r.counter = "uncompressed decoded bytes";
    r.counted = w.uncompressed;
    r.transferred = w.uncompressed;
    r.payload = compressed.size();
    // A frame table with one offset per frame; the frames carry their own sizes.
    r.container = compressed.size() + 8 * frames.size();
    r.threads = o.threads;
    r.pinned = staging.size();
    r.method.emplace(r.method.begin(), "codec", "zstd");
    r.method.emplace_back("level", std::to_string(level));
    r.method.emplace_back("frame_bytes", std::to_string(frame_bytes));
    r.method.emplace_back("frames", std::to_string(frames.size()));
    return true;
}

bool file_to_result(const Options &o, const Workload &w, cudaStream_t stream, Result &r) {
    const auto n = w.images.size();
    const auto format = o.scenario.substr(0, o.scenario.find('_'));
    const bool tiff = format == "tiff";
    const bool tsc = format == "tsc";
    std::error_code error;
    std::filesystem::create_directories(o.directory, error);
    if (tiff && !tiff_installed(r)) {
        return false;
    }
    std::vector<std::string> paths(n);
    std::vector<std::uint64_t> sizes(n);
    ts::io::TiffWriteOptions options;
    options.compression = ts::io::TiffCompression::deflate;
    options.tile_width = o.tile.width;
    options.tile_height = o.tile.height;
    options.deflate_level = o.level != 0 ? static_cast<std::uint32_t>(o.level) : 6;
    // libtiff applies the horizontal predictor only to integer samples here.
    options.horizontal_predictor = w.images.front().spec.type == ts::PixelType::uint16;
    for (std::size_t i = 0; i < n; ++i) {
        const auto &image = w.images[i];
        const auto &spec = image.spec;
        paths[i] = o.directory + "/image" + std::to_string(i) + "-" +
                   (spec.type == ts::PixelType::uint16 ? "u16-" : "f32-") +
                   std::to_string(spec.width) + "x" + std::to_string(spec.height) + "-s" +
                   std::to_string(o.seed) + "." + (tiff ? "tif" : format);
        bool written = false;
        if (tiff) {
            written = ok(ts::io::write_tiff(
                             paths[i].c_str(),
                             {image.bytes.data(), spec, image.row(), image.bytes.size()}, options),
                         "write_tiff");
            r.payload += tile_bytes(paths[i].c_str());
        } else if (tsc) {
            written = write_file(paths[i], w.files[i].data(), w.files[i].size());
        } else {
            written = write_file(paths[i], image.bytes.data(), image.bytes.size());
        }
        if (!written) {
            std::fprintf(stderr, "cannot write %s\n", paths[i].c_str());
            return false;
        }
        sizes[i] = std::filesystem::file_size(paths[i]);
        r.container += sizes[i];
    }
    Target t;
    if (!prepare_target(w, stream, t, r)) {
        return false;
    }
    std::vector<std::uint64_t> at{0};
    for (const auto size : sizes) {
        at.push_back(at.back() + size);
    }
    Pinned host;
    std::vector<ts::io::TiffImage> decoded(n);
    std::vector<ts::ImageMetadata> metadata(n);
    std::vector<cuda::DeviceBuffer> sources(n);
    std::vector<cuda::BatchItem> items(n);
    std::unique_ptr<cuda::BatchPlan> plan;
    cuda::DeviceBuffer workspace;
    if (!tiff && !host.allocate(at.back())) {
        return false;
    }
    for (std::size_t i = 0; i < n && tsc; ++i) {
        if (!ok(cuda::DeviceBuffer::allocate(sizes[i], stream, sources[i]), "allocate")) {
            return false;
        }
        const auto &spec = w.images[i].spec;
        items[i].compressed_input.bytes = {sources[i].data(), sizes[i]};
        items[i].compressed_input.metadata = &metadata[i];
        items[i].roi = {0, 0, spec.width, spec.height};
        items[i].output = {t.outputs[i].data(),
                           {ts::PixelType::float32, spec.width, spec.height},
                           spec.width * 4ULL,
                           t.outputs[i].size()};
        calibration(items[i]);
    }
    std::vector<Stage> stages;
    if (tiff) {
        // libtiff decodes into memory owned by the adapter, so the transfer reads pageable memory.
        stages.push_back(
            {"read_decode", false, [&] {
                 for (std::size_t i = 0; i < n; ++i) {
                     if (!ok(ts::io::read_tiff(paths[i].c_str(), decoded[i]), "read_tiff") ||
                         decoded[i].pitch_bytes != w.images[i].row()) {
                         return false;
                     }
                 }
                 return true;
             }});
        stages.push_back(
            {"h2d", true, [&] {
                 for (std::size_t i = 0; i < n; ++i) {
                     if (!ok(cudaMemcpyAsync(t.pixels[i].data(), decoded[i].pixels.data(),
                                             w.images[i].bytes.size(), cudaMemcpyHostToDevice,
                                             stream),
                             "cudaMemcpyAsync")) {
                         return false;
                     }
                 }
                 return true;
             }});
    } else {
        stages.push_back({"read", false, [&] {
                              for (std::size_t i = 0; i < n; ++i) {
                                  if (!read_file(paths[i], host.data() + at[i], sizes[i])) {
                                      return false;
                                  }
                              }
                              return true;
                          }});
    }
    if (tsc) {
        // Parsing and planning belong to the file path; prepare_batch waits for its stream.
        stages.push_back(
            {"parse_prepare", false, [&] {
                 for (std::size_t i = 0; i < n; ++i) {
                     if (!ok(ts::parse_metadata({host.data() + at[i], sizes[i]}, {}, metadata[i]),
                             "parse_metadata")) {
                         return false;
                     }
                 }
                 return ok(cuda::prepare_batch(
                               {cuda::Operation::decode_calibrate, items.data(), n, {}}, plan,
                               stream),
                           "prepare_batch") &&
                        (workspace.data() != nullptr || workspace_for(*plan, stream, workspace, r));
             }});
        stages.push_back({"h2d", true, [&] {
                              for (std::size_t i = 0; i < n; ++i) {
                                  if (!ok(cudaMemcpyAsync(sources[i].data(), host.data() + at[i],
                                                          sizes[i], cudaMemcpyHostToDevice, stream),
                                          "cudaMemcpyAsync")) {
                                      return false;
                                  }
                              }
                              return true;
                          }});
        stages.push_back({"decode_calibrate", true, [&] {
                              return ok(cuda::decode_calibrate_async(
                                            *plan, workspace.bytes(),
                                            {static_cast<ts::DeviceResult *>(t.results.data()), n},
                                            stream),
                                        "decode_calibrate_async");
                          }});
    } else {
        if (!tiff) {
            stages.push_back(
                {"h2d", true, [&] { return copy_pixels(t, w, host.data(), at, stream); }});
        }
        stages.push_back({"calibrate", true, [&] { return calibrate(t, stream); }});
    }
    if (!measure_stages(stages, o, stream, r) || !verify(w, t.outputs, t.results, stream)) {
        return false;
    }
    r.counter = "uncompressed decoded bytes";
    r.counted = w.uncompressed;
    r.transferred = tsc ? r.container : w.uncompressed;
    if (!tiff) {
        r.payload = tsc ? w.payload : r.container;
        r.pinned = host.size();
    }
    r.method.emplace(r.method.begin(), "codec", tiff ? "tiff_deflate" : tsc ? "tilestream" : "raw");
    r.method.emplace_back("cache", "warm");
    r.method.emplace_back("directory", o.directory);
    if (tiff) {
        r.method.emplace_back("deflate_level", std::to_string(options.deflate_level));
        r.method.emplace_back("predictor", options.horizontal_predictor ? "horizontal" : "none");
        r.method.emplace_back("tile",
                              std::to_string(o.tile.width) + "x" + std::to_string(o.tile.height));
    }
    return true;
}

#if !TILESTREAM_BENCH_NVCOMP
bool lz4_calibrate(const Options &, const Workload &, cudaStream_t, Result &) {
    std::fprintf(stderr, "built without nvCOMP (TILESTREAM_ENABLE_NVCOMP)\n");
    return false;
}

bool lz4_gpu(const Options &, const Workload &, bool, cudaStream_t, Result &) {
    std::fprintf(stderr, "built without nvCOMP (TILESTREAM_ENABLE_NVCOMP)\n");
    return false;
}
#endif

} // namespace bench
