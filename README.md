# piecewise

This repo accompanies [Piecewise regression: when one line simply isn't enough](https://www.datadoghq.com/blog/engineering/piecewise-regression/), a blog post about Datadog's approach to piecewise regression. The code included here is intended to be minimal and readable; this is not a Swiss Army knife to solve all variations of piecewise regression problems.

This is a TypeScript/JavaScript implementation of the piecewise regression algorithm.

## Installation & dependencies

```bash
npm install segreg
```

Or clone this repo and build from source:

```bash
git clone https://github.com/cmcnulty/segreg.git
cd segreg
npm install
npm run build
```

The package has no runtime dependencies. Development dependencies include TypeScript, Jest, and ts-jest for testing.

## Usage

Start by preparing your data as arrays of timestamps (independent variables) and values (dependent variables).

```typescript
import { piecewise } from 'segreg';

// Generate sample data
const t = Array.from({ length: 10 }, (_, i) => i);
const v = [
  ...Array.from({ length: 5 }, (_, i) => 2 * i),
  ...Array.from({ length: 5 }, (_, i) => 10 - i)
].map(val => val + (Math.random() - 0.5) * 2); // Add some noise
```

Now, you're ready to fit a piecewise linear regression.

```typescript
const model = piecewise(t, v);
```

`model` is a `FittedModel` object. You can inspect the fitted segments to see their domains and regression coefficients.

```typescript
console.log(model.segments.length); // Number of segments
console.log(model.segments[0]);
// FittedSegment {
//   start_t: 0,
//   end_t: 5,
//   coeffs: { intercept: -0.857, slope: 2.224 }
// }
```

You can access the coefficients and domain information from each segment:

```typescript
const segment = model.segments[0];
console.log(segment.start_t);          // Starting t value
console.log(segment.end_t);            // Ending t value
console.log(segment.coeffs.intercept); // y-intercept
console.log(segment.coeffs.slope);     // Slope
```

## Advanced Features

### Intersection Detection

The library includes utilities for detecting and snapping segment intersections:

```typescript
import { findIntersection, adjustSegmentsToSnapIntersections } from 'segreg';

const model = piecewise(t, v);

// Find intersection between consecutive segments
if (model.segments.length >= 2) {
  const intersection = findIntersection(
    model.segments[0],
    model.segments[1],
    0.05 // snap radius ratio (5% of bounding box diagonal)
  );

  if (intersection.intersects) {
    console.log(`Intersection at t=${intersection.point.t}, y=${intersection.point.y}`);
  }
}

// Adjust all segments to snap to their intersections
const adjustedSegments = adjustSegmentsToSnapIntersections(model.segments, 0.05);
```

### Custom Stopping Criteria

You can adjust the `min_stop_frac` parameter to control when the algorithm stops merging segments:

```typescript
// More aggressive merging (fewer segments)
const model1 = piecewise(t, v, 0.05);

// Less aggressive merging (more segments)
const model2 = piecewise(t, v, 0.01);
```
