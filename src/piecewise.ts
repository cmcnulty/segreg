/**
 * TypeScript version of a piecewise regression library.
 * This version more closely mirrors the Python implementation.
 */

/* --------------------------- Data Structures --------------------------- */

// Regression coefficients: y = intercept + slope * t
interface Coeffs {
  intercept: number;
  slope: number;
}

export class FittedSegment {
  start_t: number; // first t value to which this segment applies
  end_t: number; // last t value to which this segment applies
  coeffs: Coeffs;

  constructor(start_t: number, end_t: number, coeffs: Coeffs) {
    this.start_t = start_t;
    this.end_t = end_t;
    this.coeffs = coeffs;
  }

}

class FittedModel {
  segments: FittedSegment[];

  constructor(segments: FittedSegment[]) {
    this.segments = segments;
  }
}

class Segment {
  start_index: number;
  end_index: number;
  coeffs: Coeffs;
  error: number;
  cov_data: number[]; // [n, sum(t), sum(v), sum((t-mean)^2), sum((v-mean)^2), sum((t-mean)*(v-mean))]

  constructor(
    start_index: number,
    end_index: number,
    coeffs: Coeffs,
    error: number,
    cov_data: number[]
  ) {
    this.start_index = start_index;
    this.end_index = end_index;
    this.coeffs = coeffs;
    this.error = error;
    this.cov_data = cov_data;
  }
}

class Merge {
  cost: number;
  left_seg: Segment;
  right_seg: Segment;
  new_seg: Segment;

  constructor(cost: number, left_seg: Segment, right_seg: Segment, new_seg: Segment) {
    this.cost = cost;
    this.left_seg = left_seg;
    this.right_seg = right_seg;
    this.new_seg = new_seg;
  }
}

class SegmentTracker {
  segments: Segment[];
  // Maps to track segment relationships
  private segmentMap: Map<number, Segment> = new Map();
  private prevMap: Map<number, Segment | null> = new Map();
  private nextMap: Map<number, Segment | null> = new Map();

  constructor(segments: Segment[]) {
    // Sort segments by start_index
    this.segments = [...segments].sort((a, b) => a.start_index - b.start_index);

    // Initialize maps
    for (let i = 0; i < this.segments.length; i++) {
      const segment = this.segments[i];
      this.segmentMap.set(segment.start_index, segment);

      // Set prev and next relationships
      this.prevMap.set(segment.start_index, i > 0 ? this.segments[i - 1] : null);
      this.nextMap.set(segment.start_index, i < this.segments.length - 1 ? this.segments[i + 1] : null);
    }
  }

  contains(segment: Segment): boolean {
    return this.segments.includes(segment);
  }

  getPrev(segment: Segment): Segment | null {
    return this.prevMap.get(segment.start_index) || null;
  }

  getNext(segment: Segment): Segment | null {
    return this.nextMap.get(segment.start_index) || null;
  }

  getNeighbors(segment: Segment): Segment[] {
    const neighbors: Segment[] = [];
    const prev = this.getPrev(segment);
    const next = this.getNext(segment);
    if (prev) neighbors.push(prev);
    if (next) neighbors.push(next);
    return neighbors;
  }

  applyMerge(merge: Merge): void {
    const { left_seg, right_seg, new_seg } = merge;

    // Update segments array
    const leftIndex = this.segments.indexOf(left_seg);
    const rightIndex = this.segments.indexOf(right_seg);

    if (leftIndex === -1 || rightIndex === -1) {
      throw new Error("Segments not found for merge");
    }

    // Remove the original segments
    this.segments.splice(rightIndex, 1);
    this.segments.splice(leftIndex, 1);

    // Add the new segment
    this.segments.push(new_seg);
    this.segments.sort((a, b) => a.start_index - b.start_index);

    // Update mapping
    this.segmentMap.delete(right_seg.start_index);
    this.segmentMap.set(new_seg.start_index, new_seg);

    // Update relationships
    const newIndex = this.segments.indexOf(new_seg);
    this.prevMap.set(new_seg.start_index, newIndex > 0 ? this.segments[newIndex - 1] : null);
    this.nextMap.set(new_seg.start_index, newIndex < this.segments.length - 1 ? this.segments[newIndex + 1] : null);

    // Update next segment's prev reference
    if (newIndex < this.segments.length - 1) {
      const nextSeg = this.segments[newIndex + 1];
      this.prevMap.set(nextSeg.start_index, new_seg);
    }

    // Update prev segment's next reference
    if (newIndex > 0) {
      const prevSeg = this.segments[newIndex - 1];
      this.nextMap.set(prevSeg.start_index, new_seg);
    }
  }

  unapplyMerge(merge: Merge): void {
    const { left_seg, right_seg, new_seg } = merge;

    // Remove the merged segment
    const mergedIndex = this.segments.indexOf(new_seg);
    if (mergedIndex === -1) {
      throw new Error("Merged segment not found");
    }

    this.segments.splice(mergedIndex, 1);

    // Add back the original segments
    this.segments.push(left_seg, right_seg);
    this.segments.sort((a, b) => a.start_index - b.start_index);

    // Update mappings
    this.segmentMap.delete(new_seg.start_index);
    this.segmentMap.set(left_seg.start_index, left_seg);
    this.segmentMap.set(right_seg.start_index, right_seg);

    // Update relationships
    // For each segment, set its prev and next
    for (let i = 0; i < this.segments.length; i++) {
      const segment = this.segments[i];
      this.prevMap.set(segment.start_index, i > 0 ? this.segments[i - 1] : null);
      this.nextMap.set(segment.start_index, i < this.segments.length - 1 ? this.segments[i + 1] : null);
    }
  }
}

/* --------------------------- Helper Functions --------------------------- */


/**
 * Helper function to build segments from index ranges
 * This is equivalent to Python's _build_segments function
 */
function buildSegments(t: number[], v: number[], ranges: Array<[number, number]>): Segment[] {
  const segments: Segment[] = [];

  for (const [start, end] of ranges) {
    // Calculate n, sum(t), sum(v)
    const n = end - start;
    let sumT = 0, sumV = 0;

    for (let i = start; i < end; i++) {
      sumT += t[i];
      sumV += v[i];
    }

    // Calculate means
    const meanT = sumT / n;
    const meanV = sumV / n;

    // Calculate variances and covariance
    let ct = 0, cv = 0, ctv = 0;

    for (let i = start; i < end; i++) {
      const dt = t[i] - meanT;
      const dv = v[i] - meanV;
      ct += dt * dt;
      cv += dv * dv;
      ctv += dt * dv;
    }

    // Calculate coefficients
    let slope: number, intercept: number, error: number;

    if (ct > 0) {
      slope = ctv / ct;
      intercept = meanV - slope * meanT;
      error = n >= 3 ? cv - (ctv * ctv) / ct : 0;
    } else {
      slope = 0;
      intercept = meanV;
      error = n >= 3 ? cv : 0;
    }

    const covData = [n, sumT, sumV, ct, cv, ctv];
    segments.push(new Segment(start, end, { intercept, slope }, error, covData));
  }

  return segments;
}


function preprocess(t: number[], v: number[]): { t: number[]; v: number[] } {
  if (t.length !== v.length) {
    throw new Error("`t` and `v` must have the same length.");
  }

  // Filter out non-finite values
  const validIndices: number[] = [];
  for (let i = 0; i < t.length; i++) {
    if (isFinite(t[i]) && isFinite(v[i])) {
      validIndices.push(i);
    }
  }

  // Order by t-values
  const sortOrder = [...validIndices].sort((a, b) => t[a] - t[b]);
  const sortedT = sortOrder.map(i => t[i]);
  const sortedV = sortOrder.map(i => v[i]);

  return { t: sortedT, v: sortedV };
}

// Fix for the fitLine function - more closely match Python's complex error calculation
function fitLine(
  t: number[],
  v: number[],
  start: number,
  end: number,
  covData?: number[]
): { coeffs: Coeffs; error: number; covData: number[] } {
  let n = end - start;

  // Use provided covariance data if available
  if (covData && covData.length === 6) {
    const [n, sumT, sumV, ct, cv, ctv] = covData;
    const meanT = sumT / n;
    const meanV = sumV / n;

    let slope: number, intercept: number, error: number;

    // IMPORTANT: Match Python's complex error calculation logic
    // nonzero_error = n >= 3
    const nonzero_error = n >= 3;

    if (ct !== 0 && nonzero_error) {
      slope = ctv / ct;
      intercept = meanV - slope * meanT;
      error = cv - (ctv * ctv) / ct;
    } else if (nonzero_error) {
      // ct == 0 but n >= 3
      slope = 0;
      intercept = meanV;
      error = cv;
    } else {
      // n < 3
      slope = ct !== 0 ? ctv / ct : 0;
      intercept = meanV - (slope * meanT);
      error = 0;  // For n < 3, error is always 0
    }

    return {
      coeffs: { intercept, slope },
      error,
      covData: [...covData]
    };
  }

  // Calculate from scratch
  let sumT = 0, sumV = 0;
  for (let i = start; i < end; i++) {
    sumT += t[i];
    sumV += v[i];
  }

  const meanT = sumT / n;
  const meanV = sumV / n;

  let ct = 0, cv = 0, ctv = 0;
  for (let i = start; i < end; i++) {
    const dt = t[i] - meanT;
    const dv = v[i] - meanV;
    ct += dt * dt;
    cv += dv * dv;
    ctv += dt * dv;
  }

  // IMPORTANT: Match Python's complex error calculation logic
  let slope: number, intercept: number, error: number;
  const nonzero_error = n >= 3;

  if (ct !== 0 && nonzero_error) {
    slope = ctv / ct;
    intercept = meanV - slope * meanT;
    error = cv - (ctv * ctv) / ct;
  } else if (nonzero_error) {
    slope = 0;
    intercept = meanV;
    error = cv;
  } else {
    slope = ct !== 0 ? ctv / ct : 0;
    intercept = meanV - (slope * meanT);
    error = 0;  // For n < 3, error is always 0
  }

  return {
    coeffs: { intercept, slope },
    error,
    covData: [n, sumT, sumV, ct, cv, ctv]
  };
}


function mergeCovData(d1: number[], d2: number[]): number[] {
  const result = d1.map((val, idx) => val + d2[idx]);

  const n1 = d1[0];
  const n2 = d2[0];
  const n12 = n1 * n2;
  const n3 = result[0];

  if (n12 > 0 && n3 > 0) {
    const deltat = (d1[1] * n2 - d2[1] * n1) / n12;
    const deltav = (d1[2] * n2 - d2[2] * n1) / n12;

    result[3] += (deltat * deltat * n12) / n3;
    result[4] += (deltav * deltav * n12) / n3;
    result[5] += (deltat * deltav * n12) / n3;
  }

  return result;
}

function makeSegment(t: number[], v: number[], left_seg: Segment, right_seg: Segment): Segment {
  const start_index = left_seg.start_index;
  const end_index = right_seg.end_index;

  // Merge covariance data
  const cov_data = mergeCovData(left_seg.cov_data, right_seg.cov_data);

  // Calculate new coefficients and error
  const { coeffs, error } = fitLine(t, v, start_index, end_index, cov_data);

  return new Segment(start_index, end_index, coeffs, error, cov_data);
}

function makeMerge(t: number[], v: number[], left_seg: Segment, right_seg: Segment): Merge {
  const new_seg = makeSegment(t, v, left_seg, right_seg);
  const cost = new_seg.error - left_seg.error - right_seg.error;
  return new Merge(cost, left_seg, right_seg, new_seg);
}

/**
 * Create initial segments and merges, faithfully implementing Python's
 * _get_initial_segments_and_merges function.
 */
function getInitialSegmentsAndMerges(t: number[], v: number[]): { segments: Segment[]; merges: Merge[] } {
  // First, find unique t-values and their indices (equivalent to np.unique with return_index=True)
  const uniqueValues: number[] = [];
  const uniqueIndices: number[] = [];

  for (let i = 0; i < t.length; i++) {
    if (i === 0 || t[i] !== t[i-1]) {
      uniqueValues.push(t[i]);
      uniqueIndices.push(i);
    }
  }

  // Check if number of unique values is even
  const evenN = uniqueIndices.length % 2 === 0;

  // Create index ranges [start, end) for each unique t-value
  // equivalent to: index_ranges = np.c_[unique_t, np.r_[unique_t[1:], len(t)]]
  const indexRanges: Array<[number, number]> = [];
  for (let i = 0; i < uniqueIndices.length; i++) {
    const start = uniqueIndices[i];
    const end = i < uniqueIndices.length - 1 ? uniqueIndices[i+1] : t.length;
    indexRanges.push([start, end]);
  }

  // Calculate averages for each unique t-value
  const averages: number[] = [];
  for (let i = 0; i < indexRanges.length; i++) {
    const [start, end] = indexRanges[i];

    if (end - start === 1) {
      // Just a single value
      averages.push(v[start]);
    } else {
      // Average multiple values with the same t
      let sum = 0;
      for (let j = start; j < end; j++) {
        sum += v[j];
      }
      averages.push(sum / (end - start));
    }
  }

  // Pair every other t with left or right based on value proximity
  // This is a critical step in Python's algorithm!
  // equivalent to: pair_left = np.less(*np.abs(np.ediff1d(averages, to_end=np.inf if even_n else None)).reshape(-1, 2).T)
  const pairLeft: boolean[] = [];

  for (let i = 0; i < Math.floor(averages.length / 2); i++) {
    const oddIdx = i * 2 + 1;
    if (oddIdx >= averages.length) continue;

    // For each odd index, compare absolute difference with left and right
    const leftDiff = Math.abs(averages[oddIdx] - averages[oddIdx-1]);

    // If we're at the last odd index and n is even, use Infinity for rightDiff
    const rightDiff = (oddIdx === averages.length - 1 && evenN) ?
      Infinity :
      Math.abs(averages[oddIdx] - averages[oddIdx+1]);

    pairLeft.push(leftDiff < rightDiff);
  }

  // Modify index ranges based on pairing decisions
  // First, pair odd indices with left neighbors where appropriate
  for (let i = 0; i < pairLeft.length; i++) {
    if (pairLeft[i]) {
      // odd index i*2+1 pairs with left (i*2)
      // index_ranges[:-1:2, 1] = index_ranges[1::2, 1] where pair_left
      indexRanges[i*2][1] = indexRanges[i*2+1][1];
    }
  }

  // Next, pair odd indices with right neighbors where appropriate
  for (let i = 0; i < pairLeft.length; i++) {
    if (!pairLeft[i] && (i*2+2) < indexRanges.length) {
      // odd index i*2+1 pairs with right (i*2+2)
      // index_ranges[2::2, 0] = index_ranges[1:-1:2, 0] where ~pair_left
      indexRanges[i*2+2][0] = indexRanges[i*2+1][0];
    }
  }

  // Select segment ranges from even indices
  // segment_ranges = index_ranges[::2]
  const segmentRanges: Array<[number, number]> = [];
  for (let i = 0; i < indexRanges.length; i += 2) {
    segmentRanges.push(indexRanges[i]);
  }

  // Build segments from the ranges
  const segments: Segment[] = buildSegments(t, v, segmentRanges);

  // Create merge ranges between consecutive segments
  // merge_ranges = np.c_[segment_ranges[:-1,0], segment_ranges[1:,1]]
  const mergeRanges: Array<[number, number]> = [];
  for (let i = 0; i < segmentRanges.length - 1; i++) {
    mergeRanges.push([segmentRanges[i][0], segmentRanges[i+1][1]]);
  }

  // Build merged segments
  const mergeSegments = buildSegments(t, v, mergeRanges);

  // Create Merge objects
  const merges: Merge[] = [];
  for (let i = 0; i < mergeSegments.length; i++) {
    const newSeg = mergeSegments[i];
    const cost = newSeg.error - segments[i].error - segments[i+1].error;
    merges.push(new Merge(cost, segments[i], segments[i+1], newSeg));
  }

  return { segments, merges };
}


// MinHeap implementation for managing merges
class MinHeap<T> {
  private heap: T[] = [];
  private compare: (a: T, b: T) => number;

  constructor(compare: (a: T, b: T) => number) {
    this.compare = compare;
  }

  push(item: T): void {
    this.heap.push(item);
    this.siftUp(this.heap.length - 1);
  }

  pop(): T | undefined {
    if (this.heap.length === 0) return undefined;

    const result = this.heap[0];
    const last = this.heap.pop()!;

    if (this.heap.length > 0) {
      this.heap[0] = last;
      this.siftDown(0);
    }

    return result;
  }

  peek(): T | undefined {
    return this.heap[0];
  }

  get size(): number {
    return this.heap.length;
  }

  private siftUp(index: number): void {
    const item = this.heap[index];

    while (index > 0) {
      const parentIndex = Math.floor((index - 1) / 2);
      const parent = this.heap[parentIndex];

      if (this.compare(item, parent) >= 0) break;

      this.heap[index] = parent;
      index = parentIndex;
    }

    this.heap[index] = item;
  }

  private siftDown(index: number): void {
    const item = this.heap[index];
    const length = this.heap.length;

    while (index < length) {
      const leftIndex = 2 * index + 1;
      const rightIndex = leftIndex + 1;

      if (leftIndex >= length) break;

      const leftChild = this.heap[leftIndex];
      const rightChild = rightIndex < length ? this.heap[rightIndex] : null;
      const smallerChildIndex =
        rightChild !== null && this.compare(rightChild, leftChild) < 0
          ? rightIndex
          : leftIndex;
      const smallerChild = this.heap[smallerChildIndex];

      if (this.compare(item, smallerChild) <= 0) break;

      this.heap[index] = smallerChild;
      index = smallerChildIndex;
    }

    this.heap[index] = item;
  }
}

/**
 * Main piecewise regression function.
 */
export function piecewise(t: number[], v: number[], min_stop_frac: number = 0.03): FittedModel {
  // Preprocess the data
  const { t: sortedT, v: sortedV } = preprocess(t, v);

  // Initialize segments and merges
  const { segments: initSegments, merges: initMerges } = getInitialSegmentsAndMerges(sortedT, sortedV);
  const segTracker = new SegmentTracker(initSegments);

  // Create a min heap for merges - similar to heapify in Python
  const mergeHeap = new MinHeap<Merge>((a, b) => a.cost - b.cost);
  initMerges.forEach(merge => mergeHeap.push(merge));

  // Track costs and best state
  let cumCost = 0.0;
  let biggestCostIncrease = 0.0;
  let mergesSinceBest: Merge[] = [];

  // Main merge loop
  while (segTracker.segments.length > 1 && mergeHeap.size > 0) {
    // Get next valid merge
    let nextMerge: Merge | undefined;
    while (mergeHeap.size > 0) {
      const merge = mergeHeap.pop()!;
      if (segTracker.contains(merge.left_seg) && segTracker.contains(merge.right_seg)) {
        nextMerge = merge;
        break;
      }
    }

    if (!nextMerge) break;

    // Update cost tracking
    cumCost += nextMerge.cost;
    const costIncrease = nextMerge.cost;
    biggestCostIncrease = Math.max(biggestCostIncrease, costIncrease);

    // Determine if this is the new best state
    if (biggestCostIncrease < min_stop_frac * cumCost ||
        costIncrease === biggestCostIncrease) {
      mergesSinceBest = [nextMerge];
    } else {
      mergesSinceBest.push(nextMerge);
    }

    // Apply the merge
    segTracker.applyMerge(nextMerge);

    // Add new potential merges
    const neighbors = segTracker.getNeighbors(nextMerge.new_seg);
    for (const neighbor of neighbors) {
      const [left, right] = nextMerge.new_seg.start_index < neighbor.start_index
        ? [nextMerge.new_seg, neighbor]
        : [neighbor, nextMerge.new_seg];
      mergeHeap.push(makeMerge(sortedT, sortedV, left, right));
    }
  }

  // Handle the edge case
  if (biggestCostIncrease <= min_stop_frac * cumCost) {
    mergesSinceBest = [];
  }

  // Undo merges to get back to best state
  for (let i = mergesSinceBest.length - 1; i >= 0; i--) {
    segTracker.unapplyMerge(mergesSinceBest[i]);
  }

  // Create fitted segments
  const fittedSegments = segTracker.segments
    .sort((a, b) => a.start_index - b.start_index)
    .map(seg => {
      return new FittedSegment(
        sortedT[seg.start_index],
        sortedT[seg.end_index - 1],
        seg.coeffs
      );
    });

  return new FittedModel(fittedSegments);
}


interface IntersectionResult {
  intersects: boolean;
  point: { t: number; y: number } | null;
}

interface Point {
  t: number;
  y: number;
}

interface IntersectionResult {
  intersects: boolean;
  point: Point | null;
}

/**
 * Determines if two line segments intersect using a circular "snap zone" around
 * the theoretical intersection point.
 *
 * @param segment1 First line segment
 * @param segment2 Second line segment
 * @param snapRadiusRatio Snap radius as a percentage of the bounding box diagonal
 * @returns Object containing intersection status and point (if any)
 */
export function findIntersection(
  segment1: FittedSegment,
  segment2: FittedSegment,
  snapRadiusRatio: number = 0.05 // 5% of diagonal by default
): IntersectionResult {
  // Get the actual points for both segments
  const s1Start = {
    t: segment1.start_t,
    y: segment1.coeffs.slope * segment1.start_t + segment1.coeffs.intercept
  };
  const s1End = {
    t: segment1.end_t,
    y: segment1.coeffs.slope * segment1.end_t + segment1.coeffs.intercept
  };

  const s2Start = {
    t: segment2.start_t,
    y: segment2.coeffs.slope * segment2.start_t + segment2.coeffs.intercept
  };
  const s2End = {
    t: segment2.end_t,
    y: segment2.coeffs.slope * segment2.end_t + segment2.coeffs.intercept
  };

  // Calculate the bounding rectangle of all segment points
  const allPoints = [s1Start, s1End, s2Start, s2End];
  const boundingBox = calculateBoundingBox(allPoints);
  const diagonalLength = calculateDiagonal(boundingBox);

  // Calculate snap radius as percentage of diagonal
  const snapRadius = diagonalLength * snapRadiusRatio;

  // Handle parallel lines (identical slopes)
  if (segment1.coeffs.slope === segment2.coeffs.slope) {
    // If intercepts are also the same, lines are collinear
    if (segment1.coeffs.intercept === segment2.coeffs.intercept) {
      // Check if segments overlap
      if ((segment1.start_t <= segment2.end_t && segment1.end_t >= segment2.start_t) ||
          (segment2.start_t <= segment1.end_t && segment2.end_t >= segment1.start_t)) {

        // For collinear overlapping segments, use the midpoint of the overlap as intersection
        const overlapStart = Math.max(segment1.start_t, segment2.start_t);
        const overlapEnd = Math.min(segment1.end_t, segment2.end_t);
        const t = (overlapStart + overlapEnd) / 2;
        const y = segment1.coeffs.slope * t + segment1.coeffs.intercept;
        return { intersects: true, point: { t, y } };
      } else {
        // Check if endpoints are close enough
        const distS1StartS2End = distance(s1Start, s2End);
        const distS1EndS2Start = distance(s1End, s2Start);

        if (distS1StartS2End <= snapRadius) {
          const midpoint = {
            t: (s1Start.t + s2End.t) / 2,
            y: (s1Start.y + s2End.y) / 2
          };
          return { intersects: true, point: midpoint };
        } else if (distS1EndS2Start <= snapRadius) {
          const midpoint = {
            t: (s1End.t + s2Start.t) / 2,
            y: (s1End.y + s2Start.y) / 2
          };
          return { intersects: true, point: midpoint };
        }
      }
    }

    // Parallel lines that are not close enough
    return { intersects: false, point: null };
  }

  // Calculate theoretical intersection point of the lines (even if outside segments)
  const t = (segment2.coeffs.intercept - segment1.coeffs.intercept) /
            (segment1.coeffs.slope - segment2.coeffs.slope);
  const y = segment1.coeffs.slope * t + segment1.coeffs.intercept;
  const intersectionPoint = { t, y };

  // We removed the direct hit check to ensure consistent behavior for all intersections
  // The developer controls what's considered an intersection solely through the snap radius

  // Get endpoints of each segment
  const segment1Start = {
    t: segment1.start_t,
    y: segment1.coeffs.slope * segment1.start_t + segment1.coeffs.intercept
  };
  const segment1End = {
    t: segment1.end_t,
    y: segment1.coeffs.slope * segment1.end_t + segment1.coeffs.intercept
  };

  const segment2Start = {
    t: segment2.start_t,
    y: segment2.coeffs.slope * segment2.start_t + segment2.coeffs.intercept
  };
  const segment2End = {
    t: segment2.end_t,
    y: segment2.coeffs.slope * segment2.end_t + segment2.coeffs.intercept
  };

  // Find the closest endpoint of each segment to the intersection
  const dist1Start = distance(segment1Start, intersectionPoint);
  const dist1End = distance(segment1End, intersectionPoint);
  const dist2Start = distance(segment2Start, intersectionPoint);
  const dist2End = distance(segment2End, intersectionPoint);

  // Get the minimum distance for each segment
  const dist1 = Math.min(dist1Start, dist1End);
  const dist2 = Math.min(dist2Start, dist2End);

  // If both closest points are within the snap radius, consider it an intersection
  if (dist1 <= snapRadius && dist2 <= snapRadius) {
    return { intersects: true, point: intersectionPoint };
  }

  return { intersects: false, point: null };
}

// Helper functions
function calculateBoundingBox(points: Point[]) {
  let minT = Infinity;
  let maxT = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  for (const point of points) {
    minT = Math.min(minT, point.t);
    maxT = Math.max(maxT, point.t);
    minY = Math.min(minY, point.y);
    maxY = Math.max(maxY, point.y);
  }

  return { minT, maxT, minY, maxY };
}

function calculateDiagonal(boundingBox: { minT: number, maxT: number, minY: number, maxY: number }) {
  const width = boundingBox.maxT - boundingBox.minT;
  const height = boundingBox.maxY - boundingBox.minY;
  return Math.sqrt(width * width + height * height);
}

function distance(p1: Point, p2: Point) {
  const dx = p2.t - p1.t;
  const dy = p2.y - p1.y;
  return Math.sqrt((dx * dx) + (dy * dy));
}

/**
 * Adjusts consecutive segments to meet precisely at their intersection points
 * when they're within the specified tolerance.
 *
 * @param segments Array of fitted segments to adjust
 * @param snapRadiusRatio Snap radius as a percentage of the bounding box diagonal
 * @returns Array of adjusted segments
 */
export function adjustSegmentsToSnapIntersections(
  segments: FittedSegment[],
  toleranceFactor: number = 0.05
): FittedSegment[] {
  if (segments.length < 2) return segments; // Nothing to adjust if there's only one segment

  // Deep clone segments to avoid modifying the original array
  const adjustedSegments = segments.map(seg => ({
    ...seg,
    coeffs: {...seg.coeffs}
  })) as FittedSegment[];

  for (let i = 0; i < adjustedSegments.length - 1; i++) {
    const seg1 = adjustedSegments[i];
    const seg2 = adjustedSegments[i + 1];

    const intersection = findIntersection(seg1, seg2, toleranceFactor);

    if (intersection?.intersects) {
      const { t, y } = intersection.point;

      // Snap endpoints to intersection point
      seg1.end_t = t;
      seg2.start_t = t;

      // Recalculate intercepts for modified segments
      seg1.coeffs.intercept = y - seg1.coeffs.slope * t;
      seg2.coeffs.intercept = y - seg2.coeffs.slope * t;
    }
  }

  return adjustedSegments;
}
