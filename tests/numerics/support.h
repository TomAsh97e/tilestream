#pragma once

// Calibration error analysis shared by the numerical tests and the report tool. The metrics,
// classes and bounds are those of docs/numerical-analysis.md and were fixed before measuring.
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace numerics {

static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559);
// Host float expressions must round to binary32 after every operation, as on the device.
static_assert(FLT_EVAL_METHOD == 0);

inline float as_float(std::uint32_t bits) {
    float value = 0.0F;
    std::memcpy(&value, &bits, 4);
    return value;
}

inline std::uint32_t bits_of(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    return bits;
}

// The library semantics on the host. These sources build with -ffp-contract=off.
inline float calibrate(float x, float offset, float gain) {
    const float difference = x - offset;
    return difference * gain;
}

// Higher-precision reference from the same binary32 inputs. It cannot overflow or underflow,
// but both operations may round, so it is not always the exact mathematical result.
inline double reference(float x, float offset, float gain) {
    const double difference = double{x} - double{offset};
    return difference * double{gain};
}

inline constexpr double min_normal = 0x1p-126;
inline constexpr double min_subnormal = 0x1p-149;
// Smallest magnitude that rounds to infinity in binary32.
inline constexpr double overflow_threshold = 0x1p128 - 0x1p103;
inline constexpr double unit_roundoff = 0x1p-24;

// Bounds of the controlled variant, with slack only for the error of the FP64 reference.
inline constexpr double normal_ulps = 1.5 + 0x1p-20;
inline constexpr double normal_relative = 0x1p-23 * (1 + 0x1p-20);
inline constexpr double subnormal_absolute = min_subnormal * (1 + 0x1p-20);
// Single rounding when the subtraction is exact (Sterbenz).
inline constexpr double exact_difference_ulps = 0.5 + 0x1p-20;

// Unit in the last place of binary32 at the magnitude of value.
inline double ulp(double value) {
    const double magnitude = std::fabs(value);
    if (magnitude < min_normal) {
        return min_subnormal;
    }
    int exponent = 0;
    static_cast<void>(std::frexp(magnitude, &exponent));
    return std::ldexp(1.0, exponent - 24);
}

struct Error {
    double absolute = 0;
    double relative = 0;
    double ulps = 0;
};

// Relative error divides by max(|reference|, FLT_MIN), so it stays finite near zero.
inline Error error(float result, double ref) {
    const double absolute = std::fabs(double{result} - ref);
    return {absolute, absolute / std::max(std::fabs(ref), min_normal), absolute / ulp(ref)};
}

// Expected class of a calibration, decided from the inputs and the FP64 reference.
enum class Kind {
    nan_input,
    invalid,
    infinite,
    intermediate_overflow,
    overflow,
    zero,
    subnormal,
    normal,
};

inline constexpr Kind kinds[] = {
    Kind::nan_input, Kind::invalid, Kind::infinite,  Kind::intermediate_overflow,
    Kind::overflow,  Kind::zero,    Kind::subnormal, Kind::normal};

inline const char *name(Kind kind) {
    switch (kind) {
    case Kind::nan_input:
        return "nan_input";
    case Kind::invalid:
        return "invalid";
    case Kind::infinite:
        return "infinite";
    case Kind::intermediate_overflow:
        return "intermediate_overflow";
    case Kind::overflow:
        return "overflow";
    case Kind::zero:
        return "zero";
    case Kind::subnormal:
        return "subnormal";
    default:
        return "normal";
    }
}

inline Kind classify(float x, float offset, float gain) {
    if (std::isnan(x) || std::isnan(offset) || std::isnan(gain)) {
        return Kind::nan_input;
    }
    const double ref = reference(x, offset, gain);
    if (std::isnan(ref)) {
        return Kind::invalid;
    }
    if (std::isinf(ref)) {
        return Kind::infinite;
    }
    // The threshold and every such difference are exact in FP64, so this is the binary32 test.
    if (std::fabs(double{x} - double{offset}) >= overflow_threshold) {
        return Kind::intermediate_overflow;
    }
    const double magnitude = std::fabs(ref);
    if (magnitude >= overflow_threshold) {
        return Kind::overflow;
    }
    if (magnitude == 0) {
        return Kind::zero;
    }
    return magnitude < min_normal ? Kind::subnormal : Kind::normal;
}

// Class of an actual binary32 result, for counting.
enum class Shape { nan, infinite, zero, subnormal, normal };

inline Shape shape(float value) {
    if (std::isnan(value)) {
        return Shape::nan;
    }
    if (std::isinf(value)) {
        return Shape::infinite;
    }
    if (value == 0) {
        return Shape::zero;
    }
    return std::fabs(value) < FLT_MIN ? Shape::subnormal : Shape::normal;
}

inline const char *name(Shape value) {
    switch (value) {
    case Shape::nan:
        return "nan";
    case Shape::infinite:
        return "infinite";
    case Shape::zero:
        return "zero";
    case Shape::subnormal:
        return "subnormal";
    default:
        return "normal";
    }
}

// NaN results match any NaN, since payloads are not part of the contract.
inline bool same(float a, float b) {
    return bits_of(a) == bits_of(b) || (std::isnan(a) && std::isnan(b));
}

// Empty when result satisfies the contract and the bounds of the controlled variant.
inline std::string violation(float x, float offset, float gain, float result) {
    const double ref = reference(x, offset, gain);
    const bool sign = std::signbit(result) == std::signbit(ref);
    switch (classify(x, offset, gain)) {
    case Kind::nan_input:
    case Kind::invalid:
        return std::isnan(result) ? "" : "expected NaN";
    case Kind::infinite:
        return std::isinf(result) && sign ? "" : "expected an infinity of the reference's sign";
    case Kind::intermediate_overflow: {
        // The rounded difference is infinite and the product follows IEEE rules from there.
        const float difference = x - offset;
        return same(result, difference * gain) ? "" : "expected the product of an infinity";
    }
    case Kind::overflow:
        if (std::isinf(result) && sign) {
            return "";
        }
        // The rounded difference may keep a product just above the threshold finite.
        if (std::fabs(result) == FLT_MAX && sign &&
            std::fabs(ref) < overflow_threshold * (1 + unit_roundoff)) {
            return "";
        }
        return "expected overflow";
    case Kind::zero:
        return result == 0 && sign ? "" : "expected a zero of the reference's sign";
    case Kind::subnormal: {
        if (!std::isfinite(result) || (result == 0 && !sign)) {
            return "expected a subnormal or zero result of the reference's sign";
        }
        return error(result, ref).absolute <= subnormal_absolute ? "" : "absolute error too large";
    }
    default: {
        if (std::isinf(result)) {
            // A product just below the threshold can round up after the rounded difference.
            return sign && std::fabs(ref) * (1 + unit_roundoff) >= overflow_threshold
                       ? ""
                       : "unexpected infinity";
        }
        const auto e = error(result, ref);
        if (std::isnan(result) || e.ulps > normal_ulps || e.relative > normal_relative) {
            return "error above 1.5 ulp";
        }
        return "";
    }
    }
}

// Inputs of one analysis case; integer cases keep the uint16 samples the device reads.
struct Case {
    std::string name;
    bool integer = false;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint16_t> samples;
    std::vector<float> x;
    std::vector<float> offset;
    std::vector<float> gain;

    std::uint64_t size() const {
        return x.size();
    }
    void add(float value, float o, float g) {
        x.push_back(value);
        offset.push_back(o);
        gain.push_back(g);
    }
    void add(std::uint16_t value, float o, float g) {
        samples.push_back(value);
        add(static_cast<float>(value), o, g);
    }
};

// Generators avoid std distributions, whose output differs between standard libraries.
class Random {
  public:
    explicit Random(std::uint64_t seed) : engine_(seed) {}

    std::uint64_t bits() {
        return engine_();
    }
    // Uniform in [low, high), rounded once to binary32.
    float uniform(double low, double high) {
        return static_cast<float>(low + (high - low) * static_cast<double>(bits() >> 11) * 0x1p-53);
    }
    // A normal binary32 value with an exponent in [low, high] and a random significand.
    float scaled(int low, int high, bool negative) {
        const auto span = static_cast<std::uint64_t>(high - low + 1);
        const auto exponent =
            static_cast<std::uint32_t>(low + 127 + static_cast<int>(bits() % span));
        const auto word = (negative ? 0x80000000U : 0U) | exponent << 23 |
                          static_cast<std::uint32_t>(bits() & 0x7fffffU);
        return as_float(word);
    }
    float scaled(int low, int high) {
        return scaled(low, high, (bits() & 1) != 0);
    }

  private:
    std::mt19937_64 engine_;
};

inline constexpr std::uint32_t case_width = 512;

inline Case start(const char *name, bool integer, std::uint64_t count) {
    Case result;
    result.name = name;
    result.integer = integer;
    result.width = case_width;
    result.height = static_cast<std::uint32_t>(count / case_width);
    return result;
}

// Words of every IEEE class: zeros, subnormals, normal extremes, infinities, NaNs and values
// near 1, 2^24 and 2^-24.
inline constexpr std::uint32_t special_words[] = {
    0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x00800000, 0x80800000,
    0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00001, 0x7f800001,
    0x3f800000, 0x3f800001, 0xbf800000, 0x4b800000, 0x33800000, 0x0c000000, 0x72000000};

// Synthetic cases of the analysis; count must be a multiple of case_width.
inline std::vector<Case> synthetic_cases(std::uint64_t count, std::uint64_t seed) {
    std::vector<Case> cases;
    Random rng(seed);

    auto detector = start("uint16_detector", true, count);
    for (std::uint64_t i = 0; i < count; ++i) {
        const auto value = static_cast<std::uint16_t>(rng.bits() & 0xffff);
        detector.add(value, rng.uniform(90, 110), rng.uniform(0.5, 2));
    }
    cases.push_back(std::move(detector));

    auto field = start("float32_field", false, count);
    for (std::uint64_t i = 0; i < count; ++i) {
        const auto x = static_cast<float>(i % case_width);
        const auto y = static_cast<float>(i / case_width);
        const float value =
            100.0F + 0.01F * x + 0.02F * y + 1e-4F * static_cast<float>(rng.bits() % 4096);
        field.add(value, rng.uniform(95, 105), rng.uniform(0.9, 1.1));
    }
    cases.push_back(std::move(field));

    // x a few ulps from the offset, across binades and with both signs of gain.
    auto cancellation = start("cancellation", false, count);
    for (std::uint64_t i = 0; i < count; ++i) {
        const float offset = rng.scaled(0, 20, false);
        const auto steps = static_cast<std::int64_t>(rng.bits() % 129) - 64;
        const float value = as_float(static_cast<std::uint32_t>(bits_of(offset) + steps));
        cancellation.add(value, offset,
                         rng.uniform(0.5, 2) * ((rng.bits() & 1) != 0 ? -1.0F : 1.0F));
    }
    cases.push_back(std::move(cancellation));

    auto range = start("dynamic_range", false, count);
    for (std::uint64_t i = 0; i < count; ++i) {
        range.add(rng.scaled(-126, 127), rng.scaled(-126, 127), rng.scaled(-40, 40));
    }
    cases.push_back(std::move(range));

    // Products from 2^-176 to 2^-99: normal, subnormal and underflowing to zero.
    auto underflow = start("underflow", false, count);
    for (std::uint64_t i = 0; i < count; ++i) {
        underflow.add(rng.scaled(-126, -100), rng.scaled(-126, -100), rng.scaled(-50, -1));
    }
    cases.push_back(std::move(underflow));

    // One sample in eight overflows in the subtraction, with a finite exact result.
    auto overflow = start("overflow", false, count);
    for (std::uint64_t i = 0; i < count; ++i) {
        if (i % 8 == 0) {
            overflow.add(rng.scaled(127, 127, false), rng.scaled(127, 127, true),
                         rng.scaled(-40, -2));
        } else {
            overflow.add(rng.scaled(100, 127), rng.scaled(100, 127), rng.scaled(-30, 30));
        }
    }
    cases.push_back(std::move(overflow));

    auto tiny = start("gain_near_zero", false, count);
    for (std::uint64_t i = 0; i < count; ++i) {
        float gain = 0;
        switch (rng.bits() % 4) {
        case 0:
            gain = (rng.bits() & 1) != 0 ? -0.0F : 0.0F;
            break;
        case 1:
            gain = as_float(static_cast<std::uint32_t>(rng.bits() & 0x807fffffU));
            break;
        case 2:
            gain = rng.scaled(-126, -100);
            break;
        default:
            gain = rng.scaled(-100, -90);
            break;
        }
        tiny.add(rng.uniform(0, 65535), rng.uniform(90, 110), gain);
    }
    cases.push_back(std::move(tiny));

    auto words = start("random_bits", false, count);
    for (std::uint64_t i = 0; i < count; ++i) {
        const auto value = rng.bits();
        words.add(as_float(static_cast<std::uint32_t>(value)),
                  as_float(static_cast<std::uint32_t>(value >> 32)),
                  as_float(static_cast<std::uint32_t>(rng.bits())));
    }
    cases.push_back(std::move(words));
    return cases;
}

// Every combination of three special words.
inline Case special_case() {
    constexpr auto n = static_cast<std::uint32_t>(std::size(special_words));
    auto result = start("special_grid", false, 0);
    result.width = n;
    result.height = n * n;
    for (std::uint32_t i = 0; i < n * n * n; ++i) {
        result.add(as_float(special_words[i % n]), as_float(special_words[i / n % n]),
                   as_float(special_words[i / n / n]));
    }
    return result;
}

// Contract cases with their exact expected words; any NaN matches an expected NaN.
struct Row {
    std::uint32_t x;
    std::uint32_t offset;
    std::uint32_t gain;
    std::uint32_t expected;
    const char *note;
};

inline constexpr Row contract_rows[] = {
    {0x7fc12345, 0x3f800000, 0x3f800000, 0x7fc00000, "quiet NaN sample with a payload"},
    {0x7f800001, 0x3f800000, 0x3f800000, 0x7fc00000, "signaling NaN sample"},
    {0x3f800000, 0xffc00001, 0x3f800000, 0x7fc00000, "NaN offset"},
    {0x3f800000, 0x00000000, 0x7fc00000, 0x7fc00000, "NaN gain"},
    {0x7f800000, 0x7f800000, 0x3f800000, 0x7fc00000, "inf - inf"},
    {0x7f800000, 0x3f800000, 0x00000000, 0x7fc00000, "inf * 0"},
    {0x3f800000, 0x3f800000, 0x7f800000, 0x7fc00000, "0 * inf"},
    {0x7f800000, 0x3f800000, 0x40000000, 0x7f800000, "(inf - 1) * 2"},
    {0x7f800000, 0x3f800000, 0xc0000000, 0xff800000, "(inf - 1) * -2"},
    {0xff800000, 0x7f800000, 0x3f800000, 0xff800000, "(-inf - inf) * 1"},
    {0x3f800000, 0x7f800000, 0x3f800000, 0xff800000, "(1 - inf) * 1"},
    {0x3f800000, 0xff800000, 0xbf800000, 0xff800000, "(1 + inf) * -1"},
    {0x40400000, 0x3f800000, 0x7f800000, 0x7f800000, "(3 - 1) * inf"},
    {0x3f800000, 0x40400000, 0x7f800000, 0xff800000, "(1 - 3) * inf"},
    {0x40a00000, 0x40a00000, 0x40400000, 0x00000000, "(5 - 5) * 3 is +0"},
    {0x40a00000, 0x40a00000, 0xc0400000, 0x80000000, "(5 - 5) * -3 is -0"},
    {0x80000000, 0x00000000, 0x3f800000, 0x80000000, "(-0 - +0) * 1 is -0"},
    {0x80000000, 0x80000000, 0x3f800000, 0x00000000, "(-0 - -0) * 1 is +0"},
    {0x00000000, 0x80000000, 0xbf800000, 0x80000000, "(+0 - -0) * -1 is -0"},
    {0x40400000, 0x3f800000, 0x00000000, 0x00000000, "(3 - 1) * +0"},
    {0x40400000, 0x3f800000, 0x80000000, 0x80000000, "(3 - 1) * -0"},
    {0x3f800000, 0x40400000, 0x00000000, 0x80000000, "(1 - 3) * +0"},
    {0x7b800000, 0x00000000, 0x44800000, 0x7f800000, "2^120 * 2^10 overflows"},
    {0x7b800000, 0x00000000, 0xc4800000, 0xff800000, "2^120 * -2^10 overflows"},
    {0x7f7fffff, 0x00000000, 0x3f800000, 0x7f7fffff, "FLT_MAX * 1"},
    {0x7f000000, 0x00000000, 0x3fffffff, 0x7f7fffff, "2^127 * (2 - 2^-23) is FLT_MAX"},
    {0x7f000000, 0x00000000, 0x40000000, 0x7f800000, "2^127 * 2 overflows"},
    {0x7f7fffff, 0x00000000, 0x3f800001, 0x7f800000, "FLT_MAX * (1 + 2^-23) overflows"},
    {0x7f7fffff, 0xff7fffff, 0x3e800000, 0x7f800000, "difference overflows, exact FLT_MAX/2"},
    {0xff7fffff, 0x7f7fffff, 0x3e800000, 0xff800000, "difference overflows, exact -FLT_MAX/2"},
    {0x7f7fffff, 0xff7fffff, 0x00000000, 0x7fc00000, "difference overflows, exact 0"},
    {0x00000001, 0x00000000, 0x3f800000, 0x00000001, "2^-149 is not flushed"},
    {0x00000003, 0x00000001, 0x3f800000, 0x00000002, "subnormal difference is exact"},
    {0x00800000, 0x00400000, 0x3f800000, 0x00400000, "2^-126 - 2^-127 is subnormal"},
    {0x3f800000, 0x00000000, 0x00000001, 0x00000001, "subnormal gain is not flushed"},
    {0x00000200, 0x00000000, 0x3a800000, 0x00000000, "2^-150 ties to even +0"},
    {0x80000200, 0x00000000, 0x3a800000, 0x80000000, "-2^-150 ties to even -0"},
    {0x00000003, 0x00000000, 0x3f000000, 0x00000002, "1.5 * 2^-149 ties to even 2^-148"},
    {0x00000200, 0x00000000, 0x3ac00000, 0x00000001, "0.75 * 2^-149 rounds to 2^-149"},
    {0x4b800000, 0x3f000000, 0x3f800000, 0x4b800000, "2^24 - 0.5 ties to even 2^24"},
    {0x477fff00, 0x3f000000, 0x3f800000, 0x477ffe80, "65535 - 0.5 is exact"},
    {0x3f800001, 0x3f800000, 0x4cbebc20, 0x413ebc20, "(1 + 2^-23 - 1) * 1e8 is exact"},
};

// Nearest-rank statistics: the q-quantile is the smallest value with at least q*n values at or
// below it.
struct Summary {
    std::uint64_t count = 0;
    double max = 0;
    double median = 0;
    double p99 = 0;
};

inline double rank(const std::vector<double> &sorted, double q) {
    const auto n = static_cast<double>(sorted.size());
    const auto index = static_cast<std::size_t>(std::ceil(q * n));
    return sorted[index == 0 ? 0 : index - 1];
}

inline Summary summarize(std::vector<double> values) {
    Summary result;
    result.count = values.size();
    if (values.empty()) {
        return result;
    }
    std::sort(values.begin(), values.end());
    result.max = values.back();
    result.median = rank(values, 0.5);
    result.p99 = rank(values, 0.99);
    return result;
}

} // namespace numerics
