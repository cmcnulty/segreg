// piecewise.test.ts
import { findIntersection, piecewise, FittedSegment, fetchAndSaveChart, adjustSegmentsToSnapIntersections } from "../src/piecewise.js"; // note the .js extension for ESM
import seedrandom from "seedrandom";


describe("piecewise regression", () => {
  test("should produce expected segments", () => {
    // const t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
    // const v = [1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7];

    const t = [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70];
    const v = [
      4672.92, // 55
      5037.72, // 56
      5402.4, // 57
      5675.88, // 58
      5957.04, // 59
      6230.64, // 60
      6504.12, // 61
      6777.72, // 62
      7051.20, // 63
      7324.80, // 64
      7598.28, // 65 first point of second segment, last point of first segment
      7598.28, // 66
      7598.28, // 67
      7598.28, // 68
      7598.28, // 69
      7598.28 // 70
    ];

    const model = piecewise(t, v);

    // Expect two segments
    expect(model.segments.length).toBe(2);

    // Optionally, add more checks on the boundaries or coefficients
    expect(model.segments[0].start_t).toBe(55);
    expect(model.segments[0].end_t).toBe(65);
    expect(model.segments[1].start_t).toBe(66);
    expect(model.segments[1].end_t).toBe(70);
  });
});


describe("piecewise regression fake data", () => {
  test("should produce expected segments", () => {
    // populate t with 1 to 13
    const t = Array.from({ length: 16 }, (_, i) => i + 1);
    // populate v with 1 6 times and 7 7 times
    const v = [...Array.from({ length: 10 }, (_, i) => i + 1), ...Array.from({ length: 6 }, (_, i) => 11)];

    const model = piecewise(t, v);

    expect(model.segments.length).toBe(2);

    // Optionally, add more checks on the boundaries or coefficients
    expect(model.segments[0].start_t).toBe(1);
    expect(model.segments[0].end_t).toBe(11);
    expect(model.segments[1].start_t).toBe(12);
    expect(model.segments[1].end_t).toBe(16);
  });
});


describe("Single Line Test", () => {
  /**
   * Generate a standard normal random number using the Box–Muller transform.
   * We assume that rng() returns a uniform random number in (0,1).
   */
  function randomNormal(rng: seedrandom.prng): number {
    let u = 0, v = 0;
    // Avoid u or v being 0.
    do { u = rng(); } while (u === 0);
    do { v = rng(); } while (v === 0);
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  test("should yield a single segment with coefficients near the true values", () => {
    // Seed the random number generator.
    const rng = seedrandom("1");
    const intercept = -45.0;
    const slope = 0.7;

    // Generate t as an array of integers from 0 to 1999.
    const t = Array.from({ length: 2000 }, (_, i) => i);

    // Generate v as: intercept + slope*t + Gaussian noise.
    const v = t.map((tVal) => intercept + slope * tVal + randomNormal(rng));

    // Fit the piecewise regression.
    const model = piecewise(t, v);

    // Assert that a single segment is found.
    expect(model.segments.length).toBe(1);
    const seg = model.segments[0];

    // Assert the segment covers the whole domain.
    expect(seg.start_t).toBe(0);
    expect(seg.end_t).toBe(1999);

    // Assert that the regression coefficients are close to the original values.
    // Using toBeCloseTo with precision 0 means the values are equal when rounded to 0 decimals.
    expect(seg.coeffs.intercept).toBeCloseTo(intercept, 0);
    expect(seg.coeffs.slope).toBeCloseTo(slope, 0);
  });
});


// Test 1: Five Segments
describe("Test Five Segments", () => {
  test("should find five segments with proper breakpoints and slopes of 1", () => {
    // Generate data: t from 1900 to 1999 and v = t % 20.
    const t = Array.from({ length: 100 }, (_, i) => 1900 + i);
    const v = t.map((tVal) => tVal % 20);

    // Fit the piecewise regression.
    const model = piecewise(t, v);

    // Expect five segments.
    expect(model.segments.length).toBe(5);

    // Every segment should have a slope of approximately 1.
    model.segments.forEach((seg) => {
      expect(seg.coeffs.slope).toBeCloseTo(1.0);
    });

    // Verify the segments start at the correct times.
    expect(model.segments[0].start_t).toBe(1900);
    expect(model.segments[1].start_t).toBe(1920);
    expect(model.segments[2].start_t).toBe(1940);
    expect(model.segments[3].start_t).toBe(1960);
    expect(model.segments[4].start_t).toBe(1980);
  });
});

// Test 2: Messy t-values
describe("Test Messy t-values", () => {
  test("should handle uneven, out-of-order, float t-values and produce two constant segments", () => {
    // Generate data.
    const t = [1.0, 0.2, 0.5, 0.4, 2.3, 1.1];
    const v = [5, 0, 0, 0, 5, 5];

    // Fit the piecewise regression.
    const model = piecewise(t, v);

    // Expect two segments.
    expect(model.segments.length).toBe(2);
    const [seg1, seg2] = model.segments;

    // After preprocessing the t-values are sorted.
    // Expected: first segment covers t in [0.2, 1.0] with constant value 0,
    // and second covers [1.0, 2.3] with constant value 5.
    expect(seg1.start_t).toBe(0.2);
    expect(seg1.end_t).toBe(.5);

    // THESE TWO ARE FAILING:
    /*
    Expected: 0
    Received: -2.3381294964028774
    */
    expect(seg1.coeffs.intercept).toBeCloseTo(0);


    /*
      Expected: 0
      Received: 6.834532374100719
    */
    expect(seg1.coeffs.slope).toBeCloseTo(0);

    // expect(seg2.start_t).toBe(1.0);
    expect(seg2.end_t).toBe(2.3);
    expect(seg2.coeffs.intercept).toBeCloseTo(5);
    expect(seg2.coeffs.slope).toBeCloseTo(0);
  });
});

// Test 3: Non-Unique t-values
describe("Test Non-Unique t-values", () => {
  test("should assign all points with the same t to the same segment", () => {
    // Generate first part: t1 from 0 to 99.
    const t1 = Array.from({ length: 100 }, (_, i) => i);
    // For simplicity, use Math.random() to simulate Gaussian noise around mean 3.
    const v1 = Array.from({ length: 100 }, () => 3 + boxMullerRandom());

    // Generate second part: t2 from 99 to 198.
    const t2 = Array.from({ length: 100 }, (_, i) => 99 + i);
    const v2 = Array.from({ length: 100 }, () => 20 + boxMullerRandom());

    // Combine the two datasets.
    const t = [...t1, ...t2];
    const v = [...v1, ...v2];

    // Fit the piecewise regression.
    const model = piecewise(t, v);

    // Expect two segments.
    expect(model.segments.length).toBe(2);
    const [seg1, seg2] = model.segments;

    expect(seg1.end_t).toBe(seg2.start_t - 1);
  });
});

describe("Test single line with NaNs", () => {
  test("should handle NaN values correctly and exclude them from segment domains", () => {
    // Use a deterministic approach to match np.random.seed(1)
    // Create a simple pseudo-random number generator with a seed
    const seed = 1;
    function seededRandom(seed: number, i: number): number {
      const x = Math.sin(seed + i) * 10000;
      return x - Math.floor(x) - 0.5; // Range roughly -0.5 to 0.5
    }

    // Generate data with linear trend and noise
    const intercept = -45.0;
    const slope = 0.7;
    const t = Array.from({ length: 2000 }, (_, i) => i);
    const v = t.map((tVal, i) => intercept + slope * tVal + seededRandom(seed, i));

    // Introduce NaNs at specific positions
    [0, 24, 400, 401, 402, 1000, 1999].forEach(idx => {
      v[idx] = NaN;
    });

    // Fit the piecewise regression
    const model = piecewise(t, v);

    // A single segment should be found, encompassing the whole domain
    // (excluding the leading and trailing NaNs)
    expect(model.segments.length).toBe(1);

    const seg = model.segments[0];
    expect(seg.start_t).toBe(1);
    expect(seg.end_t).toBe(1998);

    // Coefficients should approximately match those used to generate the data
    expect(seg.coeffs.intercept).toBeCloseTo(intercept, 0);
    expect(seg.coeffs.slope).toBeCloseTo(slope, 0);
  });
});


describe("Test segment intersection", () => {
  test("should find intersection between two segments", async () => {
      const t_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
      const v_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10];

      const model = piecewise(t_data, v_data, 0.01);

      // Check for intersection between the first two segments (if there are at least two).
      if (model.segments.length >= 2) {

        const intersection = findIntersection(model.segments[0], model.segments[1], .15);
        const mergedSegments = adjustSegmentsToSnapIntersections(model.segments, .15);

        expect(mergedSegments.length).toBe(2);
        expect(mergedSegments[0].end_t).toBe(10);
        expect(mergedSegments[1].start_t).toBe(10);
        expect(intersection?.intersects).toBe(true);
        expect(intersection?.point.t).toBe(10);
        expect(intersection?.point.y).toBe(10);
      }
  });

  test("should find intersection even if they're a bit off", async () => {
    const segments = [
      {
        start_t: 55,
        end_t: 65,
        coeffs: { intercept: -10963.723636363637, slope: 286.11163636363636 }
      },
      {
        start_t: 66,
        end_t: 70,
        coeffs: { intercept: 7598.28, slope: 0 }
      }
    ] as FittedSegment[];
    const intersection = findIntersection(segments[0], segments[1]);

    expect(intersection?.intersects).toBe(true);
    expect(intersection?.point.t).toBeCloseTo(64.87678);
    expect(intersection?.point.y).toBeCloseTo(7598.28);
  });
});

/**
 * Helper function: Generate a standard normally distributed number using the Box–Muller transform.
 * (No seeding here; for tests relying on specific noise distributions consider seeding or fixed values.)
 */
function boxMullerRandom(): number {
  let u = 0,
      v = 0;
  // Avoid u or v being 0.
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
