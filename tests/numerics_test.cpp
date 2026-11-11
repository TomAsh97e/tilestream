// Calibration error suite on the host: the metric definitions, the floating-point environment
// and the host form of the calibration against the bounds fixed in docs/numerical-analysis.md.
#include "numerics/support.h"

#include <gtest/gtest.h>

#include <cfenv>
#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace {

using namespace numerics;

TEST(Numerics, HostEnvironmentRoundsToNearestWithoutFlushing) {
    EXPECT_EQ(std::fegetround(), FE_TONEAREST);
#if defined(__x86_64__)
    const auto csr = _mm_getcsr();
    RecordProperty("mxcsr", std::to_string(csr));
    EXPECT_EQ(csr & 0x6000U, 0U) << "SSE rounding is not to nearest";
    EXPECT_EQ(csr & 0x8000U, 0U) << "flush to zero is enabled";
    EXPECT_EQ(csr & 0x0040U, 0U) << "denormals are zero is enabled";
#endif
    RecordProperty("cxx_flags", TILESTREAM_NUMERICS_CXX_FLAGS);
    // A product of subnormal-producing operands must stay subnormal.
    volatile float tiny = 0x1p-140F;
    EXPECT_EQ(tiny * 0x1p-8F, 0x1p-148F);
}

TEST(Numerics, UlpAndErrorDefinitions) {
    EXPECT_EQ(ulp(1.0), 0x1p-23);
    EXPECT_EQ(ulp(1.9999), 0x1p-23);
    EXPECT_EQ(ulp(-2.0), 0x1p-22);
    EXPECT_EQ(ulp(0.0), 0x1p-149);
    EXPECT_EQ(ulp(0x1p-140), 0x1p-149);
    EXPECT_EQ(ulp(0x1p-126), 0x1p-149);
    EXPECT_EQ(ulp(0x1p-125), 0x1p-148);
    EXPECT_EQ(ulp(FLT_MAX), 0x1p104);

    const auto near = error(1.0F, 1.0 + 0x1p-24);
    EXPECT_EQ(near.absolute, 0x1p-24);
    EXPECT_EQ(near.ulps, 0.5);
    // Below FLT_MIN the relative error divides by FLT_MIN instead of the reference.
    const auto tiny = error(0.0F, 0x1p-140);
    EXPECT_EQ(tiny.relative, 0x1p-14);
    EXPECT_EQ(tiny.ulps, 512.0);
}

TEST(Numerics, ClassesAtTheirBoundaries) {
    EXPECT_EQ(classify(FLT_MAX, 0, 1), Kind::normal);
    EXPECT_EQ(classify(FLT_MAX, 0, as_float(0x3f800001)), Kind::overflow);
    EXPECT_EQ(classify(FLT_MAX, -FLT_MAX, 0.25F), Kind::intermediate_overflow);
    EXPECT_EQ(classify(0x1p127F, -0x1.fffffcp126F, 0.5F), Kind::normal);
    EXPECT_EQ(classify(0x1p-126F, 0, 1), Kind::normal);
    EXPECT_EQ(classify(0x1p-126F, 0x1p-127F, 1), Kind::subnormal);
    EXPECT_EQ(classify(1, 1, 5), Kind::zero);
    EXPECT_EQ(classify(1, 2, 0), Kind::zero);
    EXPECT_EQ(classify(INFINITY, 1, 1), Kind::infinite);
    EXPECT_EQ(classify(INFINITY, INFINITY, 1), Kind::invalid);
    EXPECT_EQ(classify(1, 0, NAN), Kind::nan_input);
}

TEST(Numerics, NearestRankStatistics) {
    const auto s = summarize({5, 1, 4, 2, 3});
    EXPECT_EQ(s.count, 5U);
    EXPECT_EQ(s.max, 5);
    EXPECT_EQ(s.median, 3);
    EXPECT_EQ(s.p99, 5);
    std::vector<double> hundred;
    for (int i = 1; i <= 100; ++i) {
        hundred.push_back(i);
    }
    EXPECT_EQ(summarize(hundred).median, 50);
    EXPECT_EQ(summarize(hundred).p99, 99);
    EXPECT_EQ(summarize({}).count, 0U);
}

TEST(Numerics, HostContractRows) {
    for (const auto &row : contract_rows) {
        const auto result = calibrate(as_float(row.x), as_float(row.offset), as_float(row.gain));
        EXPECT_TRUE(same(result, as_float(row.expected)))
            << row.note << ": " << std::hex << bits_of(result);
        EXPECT_EQ(violation(as_float(row.x), as_float(row.offset), as_float(row.gain), result), "")
            << row.note;
    }
}

TEST(Numerics, HostCalibrationWithinBounds) {
    auto cases = synthetic_cases(1 << 16, 2026);
    cases.push_back(special_case());
    for (const auto &c : cases) {
        std::uint64_t failures = 0;
        for (std::uint64_t i = 0; i < c.size(); ++i) {
            const auto result = calibrate(c.x[i], c.offset[i], c.gain[i]);
            const auto problem = violation(c.x[i], c.offset[i], c.gain[i], result);
            if (!problem.empty() && failures++ < 5) {
                ADD_FAILURE() << c.name << " sample " << i << ": " << problem << std::hex
                              << " x=" << bits_of(c.x[i]) << " offset=" << bits_of(c.offset[i])
                              << " gain=" << bits_of(c.gain[i]) << " result=" << bits_of(result);
            }
        }
        EXPECT_EQ(failures, 0U) << c.name;
    }
}

// Within a factor of two of the offset the subtraction is exact, so one rounding remains.
TEST(Numerics, ExactSubtractionNearTheOffset) {
    const auto cases = synthetic_cases(1 << 16, 2027);
    const auto &c = cases[2];
    ASSERT_EQ(c.name, "cancellation");
    double worst = 0;
    for (std::uint64_t i = 0; i < c.size(); ++i) {
        ASSERT_TRUE(c.x[i] >= c.offset[i] / 2 && c.x[i] <= c.offset[i] * 2);
        ASSERT_EQ(double{c.x[i] - c.offset[i]}, double{c.x[i]} - double{c.offset[i]});
        const auto result = calibrate(c.x[i], c.offset[i], c.gain[i]);
        worst = std::max(worst, error(result, reference(c.x[i], c.offset[i], c.gain[i])).ulps);
    }
    EXPECT_LE(worst, exact_difference_ulps);
}

} // namespace
