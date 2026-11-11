// Measures one benchmark case and writes every sample as JSON. benchmarks/run.py runs the cases,
// records the environment and aggregates the results.
#include "baselines.h"
#include "runner.h"

#include <tilestream/api.h>
#include <tilestream/cpu.h>
#include <tilestream/cuda.h>
#include <tilestream/info.h>
#include <tilestream/sizing.h>

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>

namespace ts = tilestream;
namespace cuda = tilestream::cuda;

namespace {

using namespace bench;

// Device-resident input pixels into device-resident archives; the output must equal the CPU files.
bool gpu_encode(const Options &o, const Workload &w, cudaStream_t stream, Result &r) {
    const auto n = w.images.size();
    std::vector<cuda::DeviceBuffer> inputs(n);
    std::vector<cuda::DeviceBuffer> outputs(n);
    std::vector<cuda::BatchItem> items(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto &image = w.images[i];
        std::uint64_t bound = 0;
        if (!upload(image.bytes.data(), image.bytes.size(), stream, inputs[i]) ||
            !ok(ts::compress_bound(image.spec, o.tile, bound), "compress_bound") ||
            !ok(cuda::DeviceBuffer::allocate(bound, stream, outputs[i]), "allocate")) {
            return false;
        }
        items[i].input = {inputs[i].data(), image.spec, image.row(), image.bytes.size()};
        items[i].compressed_output = outputs[i].bytes();
    }
    std::unique_ptr<cuda::BatchPlan> plan;
    cuda::DeviceBuffer workspace;
    cuda::DeviceBuffer results;
    if (!ok(cuda::prepare_batch({cuda::Operation::compress, items.data(), n, o.tile}, plan, stream),
            "prepare_batch") ||
        !workspace_for(*plan, stream, workspace, r) ||
        !ok(cuda::DeviceBuffer::allocate(n * sizeof(ts::DeviceResult), stream, results),
            "allocate")) {
        return false;
    }
    const auto enqueue = [&] {
        return ok(cuda::compress_async(*plan, workspace.bytes(),
                                       {static_cast<ts::DeviceResult *>(results.data()), n},
                                       stream),
                  "compress_async");
    };
    std::vector<ts::DeviceResult> records;
    if (!measure_gpu(enqueue, o, stream, r) || !check_results(results, n, stream, records)) {
        return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
        Bytes file(records[i].bytes_written);
        if (file.size() != w.files[i].size() ||
            !ok(cudaMemcpy(file.data(), outputs[i].data(), file.size(), cudaMemcpyDeviceToHost),
                "download") ||
            file != w.files[i]) {
            std::fprintf(stderr, "GPU archive %zu differs from the CPU archive\n", i);
            return false;
        }
    }
    r.counter = "uncompressed input bytes";
    r.counted = w.uncompressed;
    r.output = w.container;
    return true;
}

// Device-resident archives with parsed host metadata into device-resident pixels or FP32 output.
bool gpu_decode(const Options &o, const Workload &w, cuda::Operation operation, cudaStream_t stream,
                Result &r) {
    const auto n = w.images.size();
    const bool calibrated = operation == cuda::Operation::decode_calibrate;
    std::vector<cuda::DeviceBuffer> sources(n);
    std::vector<cuda::DeviceBuffer> outputs(n);
    std::vector<cuda::BatchItem> items(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto &roi = w.rois[i];
        const auto type = calibrated ? ts::PixelType::float32 : w.images[i].spec.type;
        const auto pitch = roi.width * bench::sample_bytes(type);
        if (!upload(w.files[i].data(), w.files[i].size(), stream, sources[i]) ||
            !ok(cuda::DeviceBuffer::allocate(pitch * roi.height, stream, outputs[i]), "allocate")) {
            return false;
        }
        items[i].compressed_input.bytes = {sources[i].data(), w.files[i].size()};
        items[i].compressed_input.metadata = &w.metadata[i];
        items[i].roi = roi;
        items[i].output = {
            outputs[i].data(), {type, roi.width, roi.height}, pitch, outputs[i].size()};
        calibration(items[i]);
        r.output += outputs[i].size();
    }
    std::unique_ptr<cuda::BatchPlan> plan;
    cuda::DeviceBuffer workspace;
    cuda::DeviceBuffer results;
    if (!ok(cuda::prepare_batch({operation, items.data(), n, {}}, plan, stream), "prepare_batch") ||
        !workspace_for(*plan, stream, workspace, r) ||
        !ok(cuda::DeviceBuffer::allocate(n * sizeof(ts::DeviceResult), stream, results),
            "allocate")) {
        return false;
    }
    const auto execute = operation == cuda::Operation::decompress   ? cuda::decompress_async
                         : operation == cuda::Operation::decode_roi ? cuda::decode_roi_async
                                                                    : cuda::decode_calibrate_async;
    const auto enqueue = [&] {
        return ok(execute(*plan, workspace.bytes(),
                          {static_cast<ts::DeviceResult *>(results.data()), n}, stream),
                  "decode");
    };
    std::vector<ts::DeviceResult> records;
    if (!measure_gpu(enqueue, o, stream, r) || !check_results(results, n, stream, records)) {
        return false;
    }
    if (!calibrated) {
        for (std::size_t i = 0; i < n; ++i) {
            const auto &image = w.images[i];
            const auto &roi = w.rois[i];
            const auto s = bench::sample_bytes(image.spec.type);
            Bytes decoded(outputs[i].size());
            if (!ok(cudaMemcpy(decoded.data(), outputs[i].data(), decoded.size(),
                               cudaMemcpyDeviceToHost),
                    "download")) {
                return false;
            }
            for (std::uint32_t y = 0; y < roi.height; ++y) {
                const auto *expected = image.bytes.data() + (roi.y + y) * image.row() + roi.x * s;
                if (std::memcmp(decoded.data() + y * roi.width * s, expected, roi.width * s) != 0) {
                    std::fprintf(stderr, "GPU decode of image %zu differs\n", i);
                    return false;
                }
            }
        }
    }
    r.counter = "uncompressed decoded bytes";
    r.counted = w.region_bytes;
    r.transferred = w.region_payload;
    return true;
}

// Archives in pinned host memory to calibrated FP32 output on the device: gather the intersecting
// tiles, prepare, copy them in one stream and decode in another. The stages overlap across images,
// so their sum is reported next to the total, never in place of it.
bool archive_roi(const Options &o, const Workload &w, cudaStream_t compute, Result &r) {
    const auto n = w.images.size();
    cudaStream_t copy = nullptr;
    if (!ok(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking), "cudaStreamCreate")) {
        return false;
    }
    const std::unique_ptr<CUstream_st, void (*)(cudaStream_t)> owned(
        copy, [](cudaStream_t s) { static_cast<void>(cudaStreamDestroy(s)); });
    std::vector<void *> pinned;
    const auto free_pinned = [&pinned] {
        for (auto *p : pinned) {
            static_cast<void>(cudaFreeHost(p));
        }
    };
    const auto pin = [&](std::uint64_t size) -> std::uint8_t * {
        void *p = nullptr;
        if (!ok(cudaMallocHost(&p, std::max<std::uint64_t>(size, 4)), "cudaMallocHost")) {
            return nullptr;
        }
        pinned.push_back(p);
        r.pinned += std::max<std::uint64_t>(size, 4);
        return static_cast<std::uint8_t *>(p);
    };
    std::vector<std::uint8_t *> archives(n);
    std::vector<std::uint8_t *> staging(n);
    std::vector<std::vector<std::uint64_t>> tiles(n);
    std::vector<std::vector<cuda::TileMapping>> mappings(n);
    std::vector<std::uint64_t> sizes(n);
    std::vector<cuda::DeviceBuffer> sources(n);
    std::vector<cuda::DeviceBuffer> outputs(n);
    std::vector<cuda::BatchItem> items(n);
    bool ready = true;
    for (std::size_t i = 0; i < n && ready; ++i) {
        const auto &roi = w.rois[i];
        tiles[i] = intersecting(w.metadata[i], roi);
        for (const auto t : tiles[i]) {
            sizes[i] += w.metadata[i].tiles[t].length;
        }
        archives[i] = pin(w.files[i].size());
        staging[i] = pin(sizes[i]);
        ready = archives[i] != nullptr && staging[i] != nullptr &&
                ok(cuda::DeviceBuffer::allocate(sizes[i], compute, sources[i]), "allocate") &&
                ok(cuda::DeviceBuffer::allocate(std::uint64_t{roi.width} * roi.height * 4, compute,
                                                outputs[i]),
                   "allocate");
        if (ready) {
            std::memcpy(archives[i], w.files[i].data(), w.files[i].size());
            items[i].roi = roi;
            items[i].output = {outputs[i].data(),
                               {ts::PixelType::float32, roi.width, roi.height},
                               roi.width * 4ULL,
                               outputs[i].size()};
            calibration(items[i]);
            r.output += outputs[i].size();
        }
    }
    Events events;
    std::vector<cudaEvent_t> marks(4 * n);
    for (auto &event : marks) {
        event = events.make();
        ready = ready && event != nullptr;
    }
    cuda::DeviceBuffer workspace;
    cuda::DeviceBuffer results;
    ready =
        ready && ok(cuda::DeviceBuffer::allocate(n * sizeof(ts::DeviceResult), compute, results),
                    "allocate");
    if (!ready) {
        static_cast<void>(cuda::synchronize(compute));
        free_pinned();
        return false;
    }
    auto *records = static_cast<ts::DeviceResult *>(results.data());

    bench::Series total{"host_end_to_end", "steady_clock", {}};
    bench::Series gather{"gather", "steady_clock", {}};
    bench::Series prepare{"prepare", "steady_clock", {}};
    bench::Series h2d{"h2d", "cuda_event_sum", {}};
    bench::Series decode{"decode_calibrate", "cuda_event_sum", {}};
    bench::Series device{"device_total", "cuda_event", {}};
    bench::Series stage_sum{"stage_sum", "derived", {}};
    bench::Series overlap{"device_overlap", "derived", {}};
    const auto once = [&](bool keep) {
        const auto t0 = bench::now_ms();
        for (std::size_t i = 0; i < n; ++i) {
            mappings[i].clear();
            std::uint64_t at = 0;
            for (const auto t : tiles[i]) {
                const auto &tile = w.metadata[i].tiles[t];
                std::memcpy(staging[i] + at, archives[i] + tile.offset, tile.length);
                mappings[i].push_back({t, at});
                at += tile.length;
            }
            auto &source = items[i].compressed_input;
            source.bytes = {sources[i].data(), sizes[i]};
            source.metadata = &w.metadata[i];
            source.mappings = mappings[i].data();
            source.mapping_count = mappings[i].size();
        }
        const auto t1 = bench::now_ms();
        std::vector<std::unique_ptr<cuda::BatchPlan>> plans(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (!ok(cuda::prepare_batch({cuda::Operation::decode_calibrate, &items[i], 1, {}},
                                        plans[i], compute),
                    "prepare_batch")) {
                return false;
            }
        }
        const auto t2 = bench::now_ms();
        if (workspace.data() == nullptr && !workspace_for(*plans[0], compute, workspace, r)) {
            return false;
        }
        for (std::size_t i = 0; i < n; ++i) {
            auto **e = &marks[4 * i];
            if (!ok(cudaEventRecord(e[0], copy), "cudaEventRecord") ||
                !ok(cudaMemcpyAsync(sources[i].data(), staging[i], sizes[i], cudaMemcpyHostToDevice,
                                    copy),
                    "cudaMemcpyAsync") ||
                !ok(cudaEventRecord(e[1], copy), "cudaEventRecord") ||
                !ok(cudaStreamWaitEvent(compute, e[1], 0), "cudaStreamWaitEvent") ||
                !ok(cudaEventRecord(e[2], compute), "cudaEventRecord") ||
                !ok(cuda::decode_calibrate_async(*plans[i], workspace.bytes(), {&records[i], 1},
                                                 compute),
                    "decode_calibrate_async") ||
                !ok(cudaEventRecord(e[3], compute), "cudaEventRecord")) {
                return false;
            }
        }
        if (!ok(cudaEventSynchronize(marks[4 * n - 1]), "cudaEventSynchronize")) {
            return false;
        }
        const auto t3 = bench::now_ms();
        if (!keep) {
            return true;
        }
        double copies = 0;
        double decodes = 0;
        for (std::size_t i = 0; i < n; ++i) {
            copies += elapsed(marks[4 * i], marks[4 * i + 1]);
            decodes += elapsed(marks[4 * i + 2], marks[4 * i + 3]);
        }
        const double span = elapsed(marks[0], marks[4 * n - 1]);
        total.ms.push_back(t3 - t0);
        gather.ms.push_back(t1 - t0);
        prepare.ms.push_back(t2 - t1);
        h2d.ms.push_back(copies);
        decode.ms.push_back(decodes);
        device.ms.push_back(span);
        stage_sum.ms.push_back((t1 - t0) + (t2 - t1) + copies + decodes);
        overlap.ms.push_back(copies + decodes - span);
        return true;
    };
    bool passed = true;
    for (int i = 0; i < o.warmup && passed; ++i) {
        passed = once(false);
    }
    for (int i = 0; i < o.samples && passed; ++i) {
        passed = once(true);
    }
    std::vector<ts::DeviceResult> checked;
    passed = passed && check_results(results, n, compute, checked);
    static_cast<void>(cuda::synchronize(copy));
    static_cast<void>(cuda::synchronize(compute));
    free_pinned();
    r.series = {total, gather, prepare, h2d, decode, device, stage_sum, overlap};
    r.streams = 2;
    r.counter = "uncompressed decoded bytes";
    r.counted = w.region_bytes;
    r.transferred = w.region_payload;
    return passed;
}

bool cpu_codec(const Options &o, const Workload &w, bool encode, Result &r) {
    std::vector<Bytes> archives(w.images.size());
    std::vector<Bytes> decoded(w.images.size());
    for (std::size_t i = 0; i < w.images.size(); ++i) {
        std::uint64_t bound = 0;
        if (!ok(ts::compress_bound(w.images[i].spec, o.tile, bound), "compress_bound")) {
            return false;
        }
        archives[i].resize(bound);
        decoded[i].resize(w.images[i].bytes.size());
    }
    const auto run = [&] {
        for (std::size_t i = 0; i < w.images.size(); ++i) {
            const auto &image = w.images[i];
            std::uint64_t written = 0;
            const auto status =
                encode ? ts::cpu::compress(
                             {image.bytes.data(), image.spec, image.row(), image.bytes.size()},
                             o.tile, {archives[i].data(), archives[i].size()}, written)
                       : ts::cpu::decompress(
                             {w.files[i].data(), w.files[i].size()},
                             {decoded[i].data(), image.spec, image.row(), decoded[i].size()});
            if (!ok(status, encode ? "cpu::compress" : "cpu::decompress")) {
                return false;
            }
        }
        return true;
    };
    if (!measure_cpu(run, o, r)) {
        return false;
    }
    for (std::size_t i = 0; i < w.images.size(); ++i) {
        const bool same =
            encode ? std::equal(w.files[i].begin(), w.files[i].end(), archives[i].begin())
                   : decoded[i] == w.images[i].bytes;
        if (!same) {
            std::fprintf(stderr, "CPU %s of image %zu differs\n", encode ? "encode" : "decode", i);
            return false;
        }
    }
    r.counter = encode ? "uncompressed input bytes" : "uncompressed decoded bytes";
    r.counted = w.uncompressed;
    r.output = encode ? w.container : w.uncompressed;
    return true;
}

void write_result(const Options &o, const Workload &w, const Result &r, std::uint64_t peak,
                  std::ostream &out) {
    bench::Json json;
    json.open('{').field("schema", "tilestream-benchmark-case/1").field("scenario", o.scenario);
    json.key("library").raw(ts::info_json());
    const auto &image = w.images.front().spec;
    json.key("dataset").open('{').field("source", w.source).field("path", o.input);
    json.field("type", image.type == ts::PixelType::uint16 ? "uint16" : "float32");
    json.field("width", std::uint64_t{image.width}).field("height", std::uint64_t{image.height});
    json.field("seed", o.seed).close('}');
    json.key("config").open('{').key("tile").open('[');
    json.value(std::uint64_t{o.tile.width}).value(std::uint64_t{o.tile.height}).close(']');
    json.field("batch", std::uint64_t{o.batch}).field("roi", std::uint64_t{o.roi});
    json.field("roi_seed", o.roi_seed).field("samples", static_cast<std::uint64_t>(o.samples));
    json.field("warmup", static_cast<std::uint64_t>(o.warmup))
        .field("min_sample_ms", o.min_sample_ms);
    json.field("repeat", r.repeat).field("cpu_threads", r.threads);
    json.field("streams", r.streams).close('}');
    if (!r.method.empty()) {
        json.key("method").open('{');
        for (const auto &[name, value] : r.method) {
            json.field(name, value);
        }
        json.close('}');
    }
    json.key("rois").open('[');
    for (const auto &roi : w.rois) {
        json.open('[').value(std::uint64_t{roi.x}).value(std::uint64_t{roi.y});
        json.value(std::uint64_t{roi.width}).value(std::uint64_t{roi.height}).close(']');
    }
    json.close(']');
    const auto payload = r.payload != 0 ? r.payload : w.payload;
    const auto container = r.container != 0 ? r.container : w.container;
    json.key("bytes").open('{').field("counter", r.counter).field("counted", r.counted);
    json.field("uncompressed", w.uncompressed).field("payload", payload);
    json.field("container", container);
    json.field("ratio_payload", static_cast<double>(w.uncompressed) / static_cast<double>(payload));
    json.field("ratio_container",
               static_cast<double>(w.uncompressed) / static_cast<double>(container));
    json.field("transferred", r.transferred).field("output", r.output).close('}');
    json.key("memory").open('{').field("workspace_bytes", r.workspace);
    json.field("peak_device_bytes", peak).field("pinned_host_bytes", r.pinned).close('}');
    json.key("throughput_gbps").open('{');
    // Throughput only for intervals that cover the whole operation, not for single stages.
    for (const auto &series : r.series) {
        if (series.name == "device" || series.name == "host" || series.name == "device_total" ||
            series.name == "host_end_to_end") {
            json.field(series.name, static_cast<double>(r.counted) / series.median() / 1e6);
        }
    }
    json.close('}');
    json.key("timings").open('[');
    for (const auto &series : r.series) {
        bench::write(json, series);
    }
    json.close(']').close('}');
    out << json.str() << '\n';
}

bool parse(int argc, char **argv, Options &o) {
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string name = argv[i];
        const std::string value = argv[i + 1];
        const auto number = [&] { return std::strtoull(value.c_str(), nullptr, 10); };
        if (name == "--scenario") {
            o.scenario = value;
        } else if (name == "--input") {
            o.input = value;
        } else if (name == "--output") {
            o.output = value;
        } else if (name == "--type") {
            o.type = value == "float32" ? ts::PixelType::float32 : ts::PixelType::uint16;
        } else if (name == "--width") {
            o.width = static_cast<std::uint32_t>(number());
        } else if (name == "--height") {
            o.height = static_cast<std::uint32_t>(number());
        } else if (name == "--seed") {
            o.seed = number();
        } else if (name == "--tile") {
            unsigned w = 0;
            unsigned h = 0;
            std::sscanf(value.c_str(), "%ux%u", &w, &h);
            o.tile = {static_cast<std::uint16_t>(w), static_cast<std::uint16_t>(h)};
        } else if (name == "--batch") {
            o.batch = static_cast<std::uint32_t>(number());
        } else if (name == "--roi") {
            o.roi = static_cast<std::uint32_t>(number());
        } else if (name == "--roi-seed") {
            o.roi_seed = number();
        } else if (name == "--samples") {
            o.samples = static_cast<int>(number());
        } else if (name == "--warmup") {
            o.warmup = static_cast<int>(number());
        } else if (name == "--min-sample-ms") {
            o.min_sample_ms = std::strtod(value.c_str(), nullptr);
        } else if (name == "--threads") {
            o.threads = static_cast<std::uint32_t>(number());
        } else if (name == "--level") {
            o.level = static_cast<int>(std::strtol(value.c_str(), nullptr, 10));
        } else if (name == "--chunk") {
            o.chunk = number();
        } else if (name == "--lz4-type") {
            o.lz4_type = value;
        } else if (name == "--bitshuffle") {
            o.bitshuffle = value == "1";
        } else if (name == "--backend") {
            o.backend = value;
        } else if (name == "--directory") {
            o.directory = value;
        } else {
            std::fprintf(stderr, "unknown option %s\n", name.c_str());
            return false;
        }
    }
    return argc % 2 == 1 && !o.scenario.empty() && o.batch > 0 && o.samples > 0 && o.warmup > 0 &&
           o.threads > 0;
}

} // namespace

int main(int argc, char **argv) {
    Options o;
    if (!parse(argc, argv, o)) {
        std::fprintf(stderr,
                     "usage: %s --scenario SCENARIO [--input file.npy | --type uint16|float32 "
                     "--width W --height H --seed S] [--tile WxH] [--batch N] [--roi SIDE] "
                     "[--roi-seed S] [--samples N] [--warmup N] [--min-sample-ms MS] "
                     "[--output file.json]\n"
                     "  TileStream: gpu_encode gpu_decode gpu_decode_roi gpu_decode_calibrate "
                     "archive_roi cpu_encode cpu_decode\n"
                     "  baselines: raw_calibrate tsc_calibrate zstd_calibrate lz4_calibrate "
                     "lz4_gpu_compress lz4_gpu_decompress raw_file tsc_file tiff_file\n"
                     "  [--threads N] [--level L] [--chunk BYTES] [--lz4-type char|typed] "
                     "[--bitshuffle 0|1] [--backend cuda|hardware|default] [--directory DIR]\n",
                     argv[0]);
        return 2;
    }
    // Only the ROI scenarios decode a region; everything else covers the whole image.
    if (o.scenario != "gpu_decode_roi" && o.scenario != "gpu_decode_calibrate" &&
        o.scenario != "archive_roi") {
        o.roi = 0;
    }
    Workload w;
    if (!make_workload(o, w)) {
        return 1;
    }
    Result r;
    std::uint64_t peak = 0;
    bool passed = false;
    if (o.scenario == "cpu_encode" || o.scenario == "cpu_decode") {
        passed = cpu_codec(o, w, o.scenario == "cpu_encode", r);
    } else {
        cudaStream_t stream = nullptr;
        cudaMemPool_t pool = nullptr;
        std::uint64_t zero = 0;
        if (!ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate") ||
            !ok(cudaDeviceGetDefaultMemPool(&pool, 0), "cudaDeviceGetDefaultMemPool") ||
            !ok(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero),
                "cudaMemPoolSetAttribute")) {
            return 1;
        }
        // Every library and harness allocation comes from the default stream-ordered pool.
        {
            if (o.scenario == "gpu_encode") {
                passed = gpu_encode(o, w, stream, r);
            } else if (o.scenario == "gpu_decode") {
                passed = gpu_decode(o, w, cuda::Operation::decompress, stream, r);
            } else if (o.scenario == "gpu_decode_roi") {
                passed = gpu_decode(o, w, cuda::Operation::decode_roi, stream, r);
            } else if (o.scenario == "gpu_decode_calibrate") {
                passed = gpu_decode(o, w, cuda::Operation::decode_calibrate, stream, r);
            } else if (o.scenario == "archive_roi") {
                passed = archive_roi(o, w, stream, r);
            } else if (o.scenario == "raw_calibrate") {
                passed = raw_calibrate(o, w, stream, r);
            } else if (o.scenario == "tsc_calibrate") {
                passed = tsc_calibrate(o, w, stream, r);
            } else if (o.scenario == "zstd_calibrate") {
                passed = zstd_calibrate(o, w, stream, r);
            } else if (o.scenario == "lz4_calibrate") {
                passed = lz4_calibrate(o, w, stream, r);
            } else if (o.scenario == "lz4_gpu_compress" || o.scenario == "lz4_gpu_decompress") {
                passed = lz4_gpu(o, w, o.scenario == "lz4_gpu_compress", stream, r);
            } else if (o.scenario == "raw_file" || o.scenario == "tsc_file" ||
                       o.scenario == "tiff_file") {
                passed = file_to_result(o, w, stream, r);
            } else {
                std::fprintf(stderr, "unknown scenario %s\n", o.scenario.c_str());
            }
            static_cast<void>(cuda::synchronize(stream));
        }
        std::uint64_t high = 0;
        passed = ok(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &high),
                    "cudaMemPoolGetAttribute") &&
                 passed;
        peak = high;
        static_cast<void>(cudaStreamDestroy(stream));
    }
    if (!passed) {
        return 1;
    }
    if (o.output.empty()) {
        write_result(o, w, r, peak, std::cout);
    } else {
        std::ofstream out(o.output);
        write_result(o, w, r, peak, out);
        if (!out) {
            std::fprintf(stderr, "cannot write %s\n", o.output.c_str());
            return 1;
        }
    }
    return 0;
}
