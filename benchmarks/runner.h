#pragma once

// Options, workloads and timing loops shared by the TileStream scenarios and the baselines.
#include "harness.h"

#include <tilestream/api.h>
#include <tilestream/cpu.h>
#include <tilestream/cuda.h>
#include <tilestream/sizing.h>

#include <cuda_runtime_api.h>

#include <functional>
#include <memory>
#include <utility>

namespace bench {

namespace ts = tilestream;
namespace cuda = tilestream::cuda;

using Bytes = std::vector<std::uint8_t>;

struct Options {
    std::string scenario;
    std::string input;
    std::string output;
    ts::PixelType type = ts::PixelType::uint16;
    std::uint32_t width = 4096;
    std::uint32_t height = 4096;
    std::uint64_t seed = 2026517;
    ts::TileShape tile{128, 32};
    std::uint32_t batch = 1;
    std::uint32_t roi = 0; // Side of a square region; 0 means the full image.
    std::uint64_t roi_seed = 1;
    int samples = 30;
    int warmup = 5;
    double min_sample_ms = 2;
    // Baselines only. Level 0 is the codec default, chunk 0 the default of the scenario.
    std::uint32_t threads = 1;
    int level = 0;
    std::uint64_t chunk = 0;
    std::string lz4_type = "char";
    bool bitshuffle = false;
    std::string backend = "cuda";
    std::string directory = "build/bench-data";
};

// Every calibrated scenario uses the same parameters.
inline void calibration(cuda::BatchItem &item) {
    item.calibration.offset.scalar = 100.0F;
    item.calibration.gain.scalar = 1.25F;
}

inline bool ok(ts::Status status, const char *what) {
    if (status != ts::Status::success) {
        std::fprintf(stderr, "%s failed with status %u\n", what, static_cast<unsigned>(status));
    }
    return status == ts::Status::success;
}

inline bool ok(cudaError_t error, const char *what) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(error));
    }
    return error == cudaSuccess;
}

// Images, their canonical archives and one region per image.
struct Workload {
    std::string source;
    std::vector<bench::Image> images;
    std::vector<Bytes> files;
    std::vector<ts::ImageMetadata> metadata;
    std::vector<ts::Roi> rois;
    std::uint64_t uncompressed = 0;
    std::uint64_t payload = 0;
    std::uint64_t container = 0;
    std::uint64_t region_bytes = 0;
    std::uint64_t region_payload = 0;
};

inline std::vector<std::uint64_t> intersecting(const ts::ImageMetadata &m, const ts::Roi &roi) {
    const std::uint64_t across = (m.image.width + m.tile.width - 1) / m.tile.width;
    std::vector<std::uint64_t> tiles;
    for (std::uint64_t row = roi.y / m.tile.height;
         row <= (std::uint64_t{roi.y} + roi.height - 1) / m.tile.height; ++row) {
        for (std::uint64_t column = roi.x / m.tile.width;
             column <= (std::uint64_t{roi.x} + roi.width - 1) / m.tile.width; ++column) {
            tiles.push_back(row * across + column);
        }
    }
    return tiles;
}

inline bool make_workload(const Options &o, Workload &w) {
    std::mt19937_64 rng(o.roi_seed);
    for (std::uint32_t i = 0; i < o.batch; ++i) {
        bench::Image image;
        if (o.input.empty()) {
            image = bench::detector_frame(o.type, o.width, o.height, o.seed + i);
            w.source = "synthetic_detector";
        } else {
            std::string error;
            if (!bench::load_npy(o.input, image, error)) {
                std::fprintf(stderr, "%s: %s\n", o.input.c_str(), error.c_str());
                return false;
            }
            // Repeating a file in a batch does not create new data.
            w.source = o.batch > 1 ? "file_repeated" : "file";
        }
        const auto spec = image.spec;
        std::uint64_t bound = 0;
        std::uint64_t written = 0;
        if (!ok(ts::compress_bound(spec, o.tile, bound), "compress_bound")) {
            return false;
        }
        Bytes file(bound);
        ts::ImageMetadata metadata;
        if (!ok(ts::cpu::compress({image.bytes.data(), spec, image.row(), image.bytes.size()},
                                  o.tile, {file.data(), bound}, written),
                "cpu::compress")) {
            return false;
        }
        file.resize(written);
        if (!ok(ts::parse_metadata({file.data(), file.size()}, {}, metadata), "parse_metadata")) {
            return false;
        }
        ts::Roi roi{0, 0, spec.width, spec.height};
        if (o.roi != 0) {
            roi.width = std::min(o.roi, spec.width);
            roi.height = std::min(o.roi, spec.height);
            roi.x = static_cast<std::uint32_t>(rng() % (spec.width - roi.width + 1));
            roi.y = static_cast<std::uint32_t>(rng() % (spec.height - roi.height + 1));
        }
        w.uncompressed += image.bytes.size();
        w.container += file.size();
        for (const auto &tile : metadata.tiles) {
            w.payload += tile.length;
        }
        w.region_bytes += std::uint64_t{roi.width} * roi.height * bench::sample_bytes(spec.type);
        for (const auto t : intersecting(metadata, roi)) {
            w.region_payload += metadata.tiles[t].length;
        }
        w.images.push_back(std::move(image));
        w.files.push_back(std::move(file));
        w.metadata.push_back(std::move(metadata));
        w.rois.push_back(roi);
    }
    return true;
}

struct Result {
    std::vector<bench::Series> series; // The first series is the primary timing.
    std::string counter;
    std::uint64_t counted = 0;
    std::uint64_t repeat = 1;
    std::uint64_t workspace = 0;
    std::uint64_t streams = 0;
    std::uint64_t transferred = 0;
    std::uint64_t output = 0;
    // Compressed size of the method; zero means the TileStream archive of the workload.
    std::uint64_t payload = 0;
    std::uint64_t container = 0;
    std::uint64_t threads = 1;
    std::uint64_t pinned = 0;
    std::vector<std::pair<std::string, std::string>> method;
};

class Events {
  public:
    ~Events() {
        for (auto *event : events_) {
            static_cast<void>(cudaEventDestroy(event));
        }
    }
    cudaEvent_t make() {
        cudaEvent_t event = nullptr;
        if (cudaEventCreate(&event) != cudaSuccess) {
            return nullptr;
        }
        events_.push_back(event);
        return event;
    }

  private:
    std::vector<cudaEvent_t> events_;
};

inline float elapsed(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0;
    return cudaEventElapsedTime(&ms, a, b) == cudaSuccess ? ms : -1.0F;
}

// Short operations run `repeat` times between two events, so that one sample lasts at least
// min_sample_ms; samples are per operation. The host clock stops after the stop event completed.
inline bool measure_gpu(const std::function<bool()> &enqueue, const Options &o, cudaStream_t stream,
                        Result &r) {
    Events events;
    auto *start = events.make();
    auto *stop = events.make();
    if (start == nullptr || stop == nullptr) {
        return false;
    }
    const auto once = [&](std::uint64_t repeat, double &device, double &host) {
        const auto begin = bench::now_ms();
        if (!ok(cudaEventRecord(start, stream), "cudaEventRecord")) {
            return false;
        }
        for (std::uint64_t i = 0; i < repeat; ++i) {
            if (!enqueue()) {
                return false;
            }
        }
        if (!ok(cudaEventRecord(stop, stream), "cudaEventRecord") ||
            !ok(cudaEventSynchronize(stop), "cudaEventSynchronize")) {
            return false;
        }
        host = (bench::now_ms() - begin) / static_cast<double>(repeat);
        device = elapsed(start, stop) / static_cast<double>(repeat);
        return device >= 0;
    };
    double device = 0;
    double host = 0;
    double fastest = 1e30;
    for (int i = 0; i < o.warmup; ++i) {
        if (!once(1, device, host)) {
            return false;
        }
        fastest = std::min(fastest, device);
    }
    r.repeat = static_cast<std::uint64_t>(
        std::clamp(std::ceil(o.min_sample_ms / std::max(fastest, 1e-4)), 1.0, 1000.0));
    bench::Series d{"device", "cuda_event", {}};
    bench::Series h{"host_end_to_end", "steady_clock", {}};
    for (int i = 0; i < o.samples; ++i) {
        if (!once(r.repeat, device, host)) {
            return false;
        }
        d.ms.push_back(device);
        h.ms.push_back(host);
    }
    r.series = {d, h};
    r.streams = 1;
    return true;
}

inline bool measure_cpu(const std::function<bool()> &run, const Options &o, Result &r) {
    double fastest = 1e30;
    for (int i = 0; i < o.warmup; ++i) {
        const auto begin = bench::now_ms();
        if (!run()) {
            return false;
        }
        fastest = std::min(fastest, bench::now_ms() - begin);
    }
    r.repeat = static_cast<std::uint64_t>(
        std::clamp(std::ceil(o.min_sample_ms / std::max(fastest, 1e-4)), 1.0, 1000.0));
    bench::Series h{"host", "steady_clock", {}};
    for (int i = 0; i < o.samples; ++i) {
        const auto begin = bench::now_ms();
        for (std::uint64_t k = 0; k < r.repeat; ++k) {
            if (!run()) {
                return false;
            }
        }
        h.ms.push_back((bench::now_ms() - begin) / static_cast<double>(r.repeat));
    }
    r.series = {h};
    return true;
}

// A host stage runs and is timed on the CPU; a device stage only enqueues work in the stream and is
// timed with events.
struct Stage {
    std::string name;
    bool device = true;
    std::function<bool()> run;
};

// Each operation runs the stages in order and waits for the stream, so a host stage never
// overwrites memory that a previous transfer still reads. device_total covers the operation only
// when every stage is device work.
inline bool measure_stages(const std::vector<Stage> &stages, const Options &o, cudaStream_t stream,
                           Result &r) {
    Events events;
    std::vector<cudaEvent_t> marks(2 * stages.size());
    for (auto &event : marks) {
        event = events.make();
        if (event == nullptr) {
            return false;
        }
    }
    const bool device =
        std::all_of(stages.begin(), stages.end(), [](const Stage &s) { return s.device; });
    std::vector<double> parts(stages.size());
    double host = 0;
    double span = 0;
    const auto once = [&](std::uint64_t repeat) {
        std::fill(parts.begin(), parts.end(), 0.0);
        host = 0;
        span = 0;
        for (std::uint64_t k = 0; k < repeat; ++k) {
            const auto begin = bench::now_ms();
            for (std::size_t s = 0; s < stages.size(); ++s) {
                if (!stages[s].device) {
                    const auto t = bench::now_ms();
                    if (!stages[s].run()) {
                        return false;
                    }
                    parts[s] += bench::now_ms() - t;
                } else if (!ok(cudaEventRecord(marks[2 * s], stream), "cudaEventRecord") ||
                           !stages[s].run() ||
                           !ok(cudaEventRecord(marks[2 * s + 1], stream), "cudaEventRecord")) {
                    return false;
                }
            }
            if (!ok(cuda::synchronize(stream), "synchronize")) {
                return false;
            }
            host += bench::now_ms() - begin;
            for (std::size_t s = 0; s < stages.size(); ++s) {
                if (stages[s].device) {
                    parts[s] += elapsed(marks[2 * s], marks[2 * s + 1]);
                }
            }
            if (device) {
                span += elapsed(marks.front(), marks.back());
            }
        }
        const auto n = static_cast<double>(repeat);
        host /= n;
        span /= n;
        for (auto &part : parts) {
            part /= n;
        }
        return true;
    };
    double fastest = 1e30;
    for (int i = 0; i < o.warmup; ++i) {
        if (!once(1)) {
            return false;
        }
        fastest = std::min(fastest, host);
    }
    r.repeat = static_cast<std::uint64_t>(
        std::clamp(std::ceil(o.min_sample_ms / std::max(fastest, 1e-4)), 1.0, 1000.0));
    r.series = {{"host_end_to_end", "steady_clock", {}}};
    if (device) {
        r.series.push_back({"device_total", "cuda_event", {}});
    }
    const auto first = r.series.size();
    for (const auto &stage : stages) {
        r.series.push_back({stage.name, stage.device ? "cuda_event" : "steady_clock", {}});
    }
    for (int i = 0; i < o.samples; ++i) {
        if (!once(r.repeat)) {
            return false;
        }
        r.series[0].ms.push_back(host);
        if (device) {
            r.series[1].ms.push_back(span);
        }
        for (std::size_t s = 0; s < stages.size(); ++s) {
            r.series[first + s].ms.push_back(parts[s]);
        }
    }
    r.streams = 1;
    return true;
}

// Page-locked host memory.
class Pinned {
  public:
    Pinned() = default;
    Pinned(const Pinned &) = delete;
    Pinned &operator=(const Pinned &) = delete;
    ~Pinned() {
        if (data_ != nullptr) {
            static_cast<void>(cudaFreeHost(data_));
        }
    }
    bool allocate(std::uint64_t size) {
        void *p = nullptr;
        if (!ok(cudaMallocHost(&p, std::max<std::uint64_t>(size, 1)), "cudaMallocHost")) {
            return false;
        }
        data_ = static_cast<std::uint8_t *>(p);
        size_ = size;
        return true;
    }
    std::uint8_t *data() const {
        return data_;
    }
    std::uint64_t size() const {
        return size_;
    }

  private:
    std::uint8_t *data_ = nullptr;
    std::uint64_t size_ = 0;
};

inline bool upload(const void *data, std::uint64_t size, cudaStream_t stream,
                   cuda::DeviceBuffer &buffer) {
    return ok(cuda::DeviceBuffer::allocate(size, stream, buffer), "allocate") &&
           ok(cudaMemcpyAsync(buffer.data(), data, size, cudaMemcpyHostToDevice, stream), "upload");
}

inline bool check_results(const cuda::DeviceBuffer &results, std::size_t count, cudaStream_t stream,
                          std::vector<ts::DeviceResult> &records) {
    records.resize(count);
    return ok(cudaMemcpyAsync(records.data(), results.data(), count * sizeof(ts::DeviceResult),
                              cudaMemcpyDeviceToHost, stream),
              "download") &&
           ok(cuda::synchronize(stream), "synchronize") &&
           std::all_of(records.begin(), records.end(),
                       [](const auto &record) { return ok(record.status, "device result"); });
}

inline bool workspace_for(const cuda::BatchPlan &plan, cudaStream_t stream,
                          cuda::DeviceBuffer &workspace, Result &r) {
    std::uint64_t size = 0;
    if (!ok(ts::workspace_size(plan.workspace_spec(), size), "workspace_size")) {
        return false;
    }
    r.workspace = std::max(r.workspace, size);
    return ok(cuda::DeviceBuffer::allocate(size, stream, workspace), "allocate");
}

} // namespace bench
