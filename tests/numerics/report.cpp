// Calibration error analysis for docs/numerical-analysis.md: the library calibration and the
// experimental variants against the FP64 reference, on synthetic stress cases and real fixtures.
//
// tilestream_numerics_report [--count N] [--output FILE] [--dump DIR] [--datasets DIR]
#include "numerics/support.h"
#include "numerics/variants.h"

#include <tilestream/api.h>
#include <tilestream/build_config.h>
#include <tilestream/cpu.h>
#include <tilestream/cuda.h>
#include <tilestream/sizing.h>

#include <cfenv>
#include <cinttypes>
#include <cstdio>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace ts = tilestream;
namespace cuda = tilestream::cuda;
using namespace numerics;

namespace {

constexpr std::uint64_t seed = 2026517;

void check(bool condition, const std::string &what) {
    if (!condition) {
        throw std::runtime_error(what);
    }
}

void check(cudaError_t error, const char *what) {
    check(error == cudaSuccess, std::string(what) + ": " + cudaGetErrorString(error));
}

void check(ts::Status status, const char *what) {
    check(status == ts::Status::success,
          std::string(what) + " failed with status " + std::to_string(static_cast<int>(status)));
}

// Reads a little-endian, C-ordered two-dimensional .npy array.
template <typename T>
std::vector<T> load_npy(const std::string &path, const char *descr, std::uint32_t &width,
                        std::uint32_t &height) {
    std::ifstream in(path, std::ios::binary);
    check(in.good(), "cannot open " + path);
    char magic[8] = {};
    unsigned char length[2] = {};
    in.read(magic, 8);
    in.read(reinterpret_cast<char *>(length), 2);
    check(in.good() && std::string(magic, 6) == "\x93NUMPY" && magic[6] == 1,
          path + " is not a version 1 .npy file");
    std::string header(static_cast<std::size_t>(length[0] | length[1] << 8), ' ');
    in.read(header.data(), static_cast<std::streamsize>(header.size()));
    const auto shape = header.find("'shape': (");
    check(header.find(std::string("'descr': '") + descr + "'") != std::string::npos &&
              header.find("'fortran_order': False") != std::string::npos &&
              shape != std::string::npos &&
              std::sscanf(header.c_str() + shape + 10, "%u, %u", &height, &width) == 2,
          path + " has an unexpected header");
    std::vector<T> data(std::uint64_t{width} * height);
    in.read(reinterpret_cast<char *>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(T)));
    check(in.good(), path + " is truncated");
    return data;
}

void save_npy(const std::string &path, const char *descr, const void *data, std::uint64_t count,
              std::uint64_t item) {
    std::string header = std::string("{'descr': '") + descr +
                         "', 'fortran_order': False, 'shape': (" + std::to_string(count) + ",), }";
    header.append(63 - (10 + header.size()) % 64, ' ');
    header += '\n';
    std::ofstream out(path, std::ios::binary);
    const unsigned char prefix[] = {0x93,
                                    'N',
                                    'U',
                                    'M',
                                    'P',
                                    'Y',
                                    1,
                                    0,
                                    static_cast<unsigned char>(header.size() & 0xff),
                                    static_cast<unsigned char>(header.size() >> 8)};
    out.write(reinterpret_cast<const char *>(prefix), sizeof(prefix));
    out << header;
    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(count * item));
    check(out.good(), "cannot write " + path);
}

// Normalization fixed before measuring: offset is the mean, gain 1 / (max - min).
template <typename T>
Case real_case(const std::string &directory, const char *name, const char *file,
               const char *descr) {
    Case result;
    result.name = name;
    result.integer = sizeof(T) == 2;
    const auto data = load_npy<T>(directory + "/" + file, descr, result.width, result.height);
    double low = data[0];
    double high = data[0];
    double sum = 0;
    for (const auto value : data) {
        low = std::min<double>(low, value);
        high = std::max<double>(high, value);
        sum += value;
    }
    const auto offset = static_cast<float>(sum / static_cast<double>(data.size()));
    const auto gain = static_cast<float>(1.0 / (high - low));
    for (const auto value : data) {
        result.add(value, offset, gain);
    }
    return result;
}

Case example_case() {
    struct Example {
        float x;
        float offset;
        float gain;
    };
    const Example examples[] = {
        {as_float(0x3f800001), 1.0F, 1e8F},
        {1234.5679F, 1234.5F, 1e-3F},
        {1001.0F, 1000.37F, 0.8F},
        {531.25F, 531.03117F, 1.0F / 840.0F},
        {3e38F, 2.9e38F, 2.0F},
        {FLT_MAX, -FLT_MAX, 0.25F},
        {0x1p-130F, 0.0F, 1.0F},
        {1000.0F, 0.0F, 0x1p-140F},
        {0x1p-100F, 0x1.8p-101F, 0x1p-30F},
    };
    Case result;
    result.name = "examples";
    result.width = static_cast<std::uint32_t>(std::size(examples));
    result.height = 1;
    for (const auto &e : examples) {
        result.add(e.x, e.offset, e.gain);
    }
    return result;
}

cuda::DeviceBuffer upload(const void *data, std::uint64_t bytes) {
    cuda::DeviceBuffer buffer;
    check(cuda::DeviceBuffer::allocate(bytes, nullptr, buffer), "allocate");
    check(cudaMemcpy(buffer.data(), data, bytes, cudaMemcpyHostToDevice), "upload");
    return buffer;
}

std::vector<float> download(const cuda::DeviceBuffer &buffer, std::uint64_t count) {
    std::vector<float> values(count);
    check(cudaMemcpy(values.data(), buffer.data(), count * 4, cudaMemcpyDeviceToHost), "download");
    return values;
}

// Library calibration with offset and gain maps, standalone or fused with decoding.
std::vector<float> library(const Case &c, bool fused) {
    const auto n = c.size();
    const std::uint64_t sample = c.integer ? 2 : 4;
    const ts::ImageSpec image{c.integer ? ts::PixelType::uint16 : ts::PixelType::float32, c.width,
                              c.height};
    const ts::ImageSpec region{ts::PixelType::float32, c.width, c.height};
    const void *pixels = c.integer ? static_cast<const void *>(c.samples.data()) : c.x.data();

    cuda::BatchItem item;
    cuda::DeviceBuffer input;
    std::vector<std::uint8_t> file;
    ts::ImageMetadata metadata;
    if (fused) {
        std::uint64_t bound = 0;
        std::uint64_t size = 0;
        check(ts::compress_bound(image, {}, bound), "compress_bound");
        file.resize(bound);
        check(ts::cpu::compress({pixels, image, c.width * sample, n * sample}, {},
                                {file.data(), bound}, size),
              "cpu::compress");
        check(ts::parse_metadata({file.data(), size}, {}, metadata), "parse_metadata");
        input = upload(file.data(), size);
        item.compressed_input.bytes = {input.data(), size};
        item.compressed_input.metadata = &metadata;
        item.roi = {0, 0, c.width, c.height};
    } else {
        input = upload(pixels, n * sample);
        item.input = {input.data(), image, c.width * sample, n * sample};
    }
    const auto offsets = upload(c.offset.data(), n * 4);
    const auto gains = upload(c.gain.data(), n * 4);
    cuda::DeviceBuffer output;
    check(cuda::DeviceBuffer::allocate(n * 4, nullptr, output), "allocate");
    item.output = {output.data(), region, c.width * 4ULL, n * 4};
    item.calibration.offset = {
        cuda::ParameterKind::map, 0.0F, {offsets.data(), region, c.width * 4ULL, n * 4}};
    item.calibration.gain = {
        cuda::ParameterKind::map, 0.0F, {gains.data(), region, c.width * 4ULL, n * 4}};

    const auto operation = fused ? cuda::Operation::decode_calibrate : cuda::Operation::calibrate;
    std::unique_ptr<cuda::BatchPlan> plan;
    check(cuda::prepare_batch({operation, &item, 1, {}}, plan, nullptr), "prepare_batch");
    std::uint64_t bytes = 0;
    check(ts::workspace_size(plan->workspace_spec(), bytes), "workspace_size");
    cuda::DeviceBuffer workspace;
    cuda::DeviceBuffer results;
    check(cuda::DeviceBuffer::allocate(bytes, nullptr, workspace), "allocate");
    check(cuda::DeviceBuffer::allocate(sizeof(ts::DeviceResult), nullptr, results), "allocate");
    const auto run = fused ? cuda::decode_calibrate_async : cuda::calibrate_async;
    check(run(*plan, workspace.bytes(), {static_cast<ts::DeviceResult *>(results.data()), 1},
              nullptr),
          "calibration");
    check(cuda::synchronize(nullptr), "synchronize");
    ts::DeviceResult record;
    check(cudaMemcpy(&record, results.data(), sizeof(record), cudaMemcpyDeviceToHost), "download");
    check(record.status, "device result");
    return download(output, n);
}

struct Outcome {
    const char *name;
    std::vector<float> values;
};

constexpr Variant experiments[] = {
    Variant::plain,           Variant::affine_fma, Variant::affine_separate, Variant::affine_plain,
    Variant::fast_intrinsics, Variant::fast_plain, Variant::fast_affine};
constexpr const char *experiment_names[] = {"plain",        "affine_fma",      "affine_separate",
                                            "affine_plain", "fast_intrinsics", "fast_plain",
                                            "fast_affine"};

// The affine form uses b = -(offset * gain), rounded once like a precomputed map.
std::vector<float> affine_offsets(const Case &c) {
    std::vector<float> b(c.size());
    for (std::uint64_t i = 0; i < c.size(); ++i) {
        b[i] = -(c.offset[i] * c.gain[i]);
    }
    return b;
}

std::vector<Outcome> run_all(const Case &c, const std::vector<float> &b) {
    std::vector<Outcome> outcomes;
    outcomes.push_back({"controlled", library(c, false)});
    outcomes.push_back({"fused", library(c, true)});
    const auto n = c.size();
    const auto x = upload(c.x.data(), n * 4);
    const auto offset = upload(c.offset.data(), n * 4);
    const auto gain = upload(c.gain.data(), n * 4);
    const auto affine = upload(b.data(), n * 4);
    cuda::DeviceBuffer output;
    check(cuda::DeviceBuffer::allocate(n * 4, nullptr, output), "allocate");
    const Inputs inputs{
        static_cast<const float *>(x.data()),    static_cast<const float *>(offset.data()),
        static_cast<const float *>(gain.data()), static_cast<const float *>(affine.data()),
        static_cast<float *>(output.data()),     n};
    for (std::size_t v = 0; v < std::size(experiments); ++v) {
        check(launch(experiments[v], inputs, nullptr), experiment_names[v]);
        check(cudaDeviceSynchronize(), experiment_names[v]);
        outcomes.push_back({experiment_names[v], download(output, n)});
    }
    return outcomes;
}

const std::vector<float> &find(const std::vector<Outcome> &outcomes, const char *name) {
    for (const auto &outcome : outcomes) {
        if (std::string(outcome.name) == name) {
            return outcome.values;
        }
    }
    throw std::runtime_error(std::string("no outcome ") + name);
}

class Json {
  public:
    explicit Json(std::FILE *out) : out_(out) {}

    void open(const char *key, char bracket) {
        item(key);
        std::fputc(bracket, out_);
        first_ = true;
    }
    void close(char bracket) {
        std::fputc(bracket, out_);
        first_ = false;
    }
    void value(const char *key, const std::string &text) {
        item(key);
        std::fprintf(out_, "\"%s\"", text.c_str());
    }
    void value(const char *key, std::uint64_t number) {
        item(key);
        std::fprintf(out_, "%" PRIu64, number);
    }
    void value(const char *key, double number) {
        item(key);
        if (std::isfinite(number)) {
            std::fprintf(out_, "%.17g", number);
        } else {
            std::fprintf(out_, "\"%s\"", std::isnan(number) ? "nan" : number > 0 ? "inf" : "-inf");
        }
    }
    void summary(const char *key, const Summary &s) {
        open(key, '{');
        value("count", s.count);
        value("max", s.max);
        value("median", s.median);
        value("p99", s.p99);
        close('}');
    }

  private:
    void item(const char *key) {
        if (!first_) {
            std::fputc(',', out_);
        }
        first_ = false;
        if (key != nullptr) {
            std::fprintf(out_, "\"%s\":", key);
        }
    }

    std::FILE *out_;
    bool first_ = true;
};

std::string hex(float value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", bits_of(value));
    return text;
}

std::uint64_t tally(std::uint64_t n, const std::function<bool(std::uint64_t)> &test) {
    std::uint64_t count = 0;
    for (std::uint64_t i = 0; i < n; ++i) {
        count += test(i) ? 1U : 0U;
    }
    return count;
}

void report_variant(Json &json, const Case &c, const std::vector<Kind> &kinds,
                    const std::vector<double> &refs, const std::vector<float> &controlled,
                    const Outcome &outcome) {
    const auto n = c.size();
    const auto &values = outcome.values;
    std::uint64_t shapes[5] = {};
    std::uint64_t nonfinite = 0;
    std::vector<double> absolute;
    std::vector<double> relative;
    std::vector<double> ulps;
    std::vector<double> normal;
    std::vector<double> subnormal;
    for (std::uint64_t i = 0; i < n; ++i) {
        ++shapes[static_cast<int>(shape(values[i]))];
        if (kinds[i] != Kind::zero && kinds[i] != Kind::subnormal && kinds[i] != Kind::normal) {
            continue;
        }
        if (!std::isfinite(values[i])) {
            ++nonfinite;
            continue;
        }
        const auto e = error(values[i], refs[i]);
        absolute.push_back(e.absolute);
        relative.push_back(e.relative);
        ulps.push_back(e.ulps);
        if (kinds[i] == Kind::normal) {
            normal.push_back(e.ulps);
        } else if (kinds[i] == Kind::subnormal) {
            subnormal.push_back(e.absolute);
        }
    }
    json.open(outcome.name, '{');
    json.open("shapes", '{');
    for (const auto s :
         {Shape::nan, Shape::infinite, Shape::zero, Shape::subnormal, Shape::normal}) {
        json.value(name(s), shapes[static_cast<int>(s)]);
    }
    json.close('}');
    json.value("differs_from_controlled",
               tally(n, [&](std::uint64_t i) { return !same(values[i], controlled[i]); }));
    json.value("violations", tally(n, [&](std::uint64_t i) {
                   return !violation(c.x[i], c.offset[i], c.gain[i], values[i]).empty();
               }));
    json.value("nonfinite_for_finite_reference", nonfinite);
    json.summary("absolute", summarize(std::move(absolute)));
    json.summary("relative", summarize(std::move(relative)));
    json.summary("ulps", summarize(std::move(ulps)));
    json.summary("ulps_normal", summarize(std::move(normal)));
    json.summary("absolute_subnormal", summarize(std::move(subnormal)));
    json.close('}');
}

void dump(const std::string &directory, const Case &c, const std::vector<float> &b,
          const std::vector<Outcome> &outcomes) {
    const auto base = directory + "/" + c.name;
    const auto n = c.size();
    if (c.integer) {
        save_npy(base + ".samples.npy", "<u2", c.samples.data(), n, 2);
    }
    save_npy(base + ".x.npy", "<f4", c.x.data(), n, 4);
    save_npy(base + ".offset.npy", "<f4", c.offset.data(), n, 4);
    save_npy(base + ".gain.npy", "<f4", c.gain.data(), n, 4);
    save_npy(base + ".b.npy", "<f4", b.data(), n, 4);
    for (const auto &outcome : outcomes) {
        save_npy(base + "." + outcome.name + ".npy", "<f4", outcome.values.data(), n, 4);
    }
}

// Every invariant the analysis relies on; a failure stops the report.
void verify(const Case &c, const std::vector<float> &b, const std::vector<Outcome> &outcomes,
            Json &json) {
    const auto n = c.size();
    const auto &controlled = find(outcomes, "controlled");
    const auto &fused = find(outcomes, "fused");
    const auto &fma = find(outcomes, "affine_fma");
    const auto &separate = find(outcomes, "affine_separate");
    const auto host = tally(n, [&](std::uint64_t i) {
        return !same(controlled[i], calibrate(c.x[i], c.offset[i], c.gain[i]));
    });
    const auto host_fma = tally(
        n, [&](std::uint64_t i) { return !same(fma[i], std::fmaf(c.gain[i], c.x[i], b[i])); });
    const auto host_separate = tally(n, [&](std::uint64_t i) {
        const float product = c.gain[i] * c.x[i];
        return !same(separate[i], product + b[i]);
    });
    const auto fused_bits =
        tally(n, [&](std::uint64_t i) { return bits_of(fused[i]) != bits_of(controlled[i]); });
    json.open("checks", '{');
    json.value("controlled_vs_host", host);
    json.value("fused_vs_controlled_bits", fused_bits);
    json.value("affine_fma_vs_host_fmaf", host_fma);
    json.value("affine_separate_vs_host", host_separate);
    json.close('}');
    check(host == 0 && fused_bits == 0 && host_fma == 0 && host_separate == 0,
          c.name + ": device and host disagree");
}

void report_case(Json &json, const Case &c, const std::string &dump_directory) {
    const auto n = c.size();
    const auto b = affine_offsets(c);
    const auto outcomes = run_all(c, b);
    std::vector<Kind> kinds(n);
    std::vector<double> refs(n);
    std::uint64_t counts[std::size(numerics::kinds)] = {};
    for (std::uint64_t i = 0; i < n; ++i) {
        kinds[i] = classify(c.x[i], c.offset[i], c.gain[i]);
        refs[i] = reference(c.x[i], c.offset[i], c.gain[i]);
        ++counts[static_cast<int>(kinds[i])];
    }
    json.open(nullptr, '{');
    json.value("name", c.name);
    json.value("integer", std::uint64_t{c.integer});
    json.value("width", std::uint64_t{c.width});
    json.value("height", std::uint64_t{c.height});
    json.value("samples", n);
    if (c.name.rfind("real_", 0) == 0) {
        json.value("offset", double{c.offset[0]});
        json.value("gain", double{c.gain[0]});
    }
    json.open("kinds", '{');
    for (const auto kind : numerics::kinds) {
        json.value(name(kind), counts[static_cast<int>(kind)]);
    }
    json.close('}');
    verify(c, b, outcomes, json);
    json.open("variants", '{');
    const auto &controlled = find(outcomes, "controlled");
    for (const auto &outcome : outcomes) {
        report_variant(json, c, kinds, refs, controlled, outcome);
    }
    json.close('}');
    json.close('}');
    if (!dump_directory.empty()) {
        dump(dump_directory, c, b, outcomes);
    }
}

void report_examples(Json &json) {
    const auto c = example_case();
    const auto b = affine_offsets(c);
    const auto outcomes = run_all(c, b);
    json.open("examples", '[');
    for (std::uint64_t i = 0; i < c.size(); ++i) {
        json.open(nullptr, '{');
        json.value("x", hex(c.x[i]));
        json.value("offset", hex(c.offset[i]));
        json.value("gain", hex(c.gain[i]));
        json.value("b", hex(b[i]));
        json.value("reference", reference(c.x[i], c.offset[i], c.gain[i]));
        json.open("results", '{');
        for (const auto &outcome : outcomes) {
            json.value(outcome.name, hex(outcome.values[i]));
        }
        json.close('}');
        json.close('}');
    }
    json.close(']');
}

void report_environment(Json &json) {
    int device = 0;
    int driver = 0;
    int runtime = 0;
    cudaDeviceProp properties{};
    check(cudaGetDevice(&device), "cudaGetDevice");
    check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    check(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
    check(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
    json.open("environment", '{');
    json.value("device", std::string(properties.name));
    json.value("compute_capability",
               std::to_string(properties.major) + "." + std::to_string(properties.minor));
    json.value("cuda_driver", std::uint64_t(driver));
    json.value("cuda_runtime", std::uint64_t(runtime));
    json.value("build_type", std::string(TILESTREAM_BUILD_TYPE));
    json.value("cxx_compiler", std::string(TILESTREAM_CXX_COMPILER));
    json.value("cuda_compiler", std::string(TILESTREAM_NUMERICS_CUDA_COMPILER));
    json.value("cxx_flags", std::string(TILESTREAM_NUMERICS_CXX_FLAGS));
    json.value("cuda_flags", std::string(TILESTREAM_NUMERICS_CUDA_FLAGS));
    json.value("fast_math_flags", std::string(TILESTREAM_NUMERICS_FAST_FLAGS));
    json.value("host_rounding",
               std::string(std::fegetround() == FE_TONEAREST ? "nearest" : "other"));
#if defined(__x86_64__)
    char csr[16];
    std::snprintf(csr, sizeof(csr), "0x%04x", _mm_getcsr());
    json.value("mxcsr", std::string(csr));
#endif
    json.close('}');
}

} // namespace

int main(int argc, char **argv) {
    std::uint64_t count = 1 << 17;
    std::string output;
    std::string dump_directory;
    std::string datasets = TILESTREAM_DATASETS_DIR;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string option = argv[i];
        if (option == "--count") {
            count = std::stoull(argv[i + 1]);
        } else if (option == "--output") {
            output = argv[i + 1];
        } else if (option == "--dump") {
            dump_directory = argv[i + 1];
        } else if (option == "--datasets") {
            datasets = argv[i + 1];
        } else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    if (count == 0 || count % case_width != 0) {
        std::fprintf(stderr, "--count must be a positive multiple of %u\n", case_width);
        return 2;
    }
    std::FILE *out = output.empty() ? stdout : std::fopen(output.c_str(), "w");
    if (out == nullptr) {
        std::fprintf(stderr, "cannot write %s\n", output.c_str());
        return 1;
    }
    try {
        auto cases = synthetic_cases(count, seed);
        cases.push_back(special_case());
        cases.push_back(
            real_case<std::uint16_t>(datasets, "real_ct_small", "ct_small_uint16.npy", "<u2"));
        cases.push_back(
            real_case<float>(datasets, "real_jacksboro_dem", "jacksboro_dem_float32.npy", "<f4"));
        Json json(out);
        json.open(nullptr, '{');
        json.value("tool", std::string("tilestream_numerics_report"));
        json.value("seed", seed);
        json.value("count", count);
        report_environment(json);
        json.open("cases", '[');
        for (const auto &c : cases) {
            report_case(json, c, dump_directory);
            std::fprintf(stderr, "%s: done\n", c.name.c_str());
        }
        json.close(']');
        report_examples(json);
        json.close('}');
        std::fputc('\n', out);
    } catch (const std::exception &failure) {
        std::fprintf(stderr, "error: %s\n", failure.what());
        return 1;
    }
    if (out != stdout) {
        std::fclose(out);
    }
    return 0;
}
