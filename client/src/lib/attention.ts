export type Matrix = number[][];

export type FeatureMapName = "elu" | "relu" | "exp";

export type AttentionRun = {
  tokens: string[];
  q: Matrix;
  k: Matrix;
  v: Matrix;
  exactWeights: Matrix;
  exactOutput: Matrix;
  linearOutput: Matrix;
  phiQ: Matrix;
  phiK: Matrix;
  state: Matrix;
  denominator: number[];
  mae: number;
  l2: number;
  topExact: number;
  topLinear: number;
  operations: { exact: number; linear: number };
};

export type ExampleName = "sentence" | "retrieval" | "drift";

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

export function dot(a: number[], b: number[]) {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

export function matMul(a: Matrix, b: Matrix): Matrix {
  if (!a.length || !b.length || a[0].length !== b.length) return [];
  const result = Array.from({ length: a.length }, () =>
    Array.from({ length: b[0].length }, () => 0),
  );
  for (let i = 0; i < a.length; i += 1) {
    for (let k = 0; k < b.length; k += 1) {
      for (let j = 0; j < b[0].length; j += 1) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return result;
}

export function transpose(matrix: Matrix): Matrix {
  return matrix[0]?.map((_, column) => matrix.map((row) => row[column])) ?? [];
}

export function softmax(values: number[]) {
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0) || 1;
  return exps.map((value) => value / total);
}

export function rowSoftmax(matrix: Matrix) {
  return matrix.map(softmax);
}

export function featureMap(values: number[], name: FeatureMapName) {
  if (name === "relu") {
    return values.map((value) => Math.max(0, value) + 0.001);
  }
  if (name === "exp") {
    return values.map((value) => Math.exp(clamp(value, -5, 5)));
  }
  return values.map((value) => (value >= 0 ? value + 1 : Math.exp(value)));
}

export function l2Norm(values: number[]) {
  return Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
}

export function normalize(values: number[]) {
  const norm = l2Norm(values) || 1;
  return values.map((value) => value / norm);
}

function hashString(input: string) {
  return input.split("").reduce((hash, character) => {
    return (hash * 31 + character.charCodeAt(0)) >>> 0;
  }, 2166136261);
}

function seededValue(seed: number, index: number) {
  const x = Math.sin(seed * 0.00001 + index * 12.9898) * 43758.5453;
  return (x - Math.floor(x)) * 2 - 1;
}

const SENTENCE_TOKENS = ["The", "cat", "sat", "on", "the", "mat"];
const RETRIEVAL_TOKENS = [
  "<s>",
  "blue",
  "noise",
  "quiet",
  "signal",
  "noise",
  "needle:42",
  "noise",
  "soft",
  "noise",
  "query",
  "?",
];
const DRIFT_TOKENS = [
  "A",
  "model",
  "reads",
  "a",
  "long",
  "stream",
  "while",
  "topics",
  "shift",
  "then",
  "asks",
  "why",
];

export function tokensForExample(example: ExampleName, length: number) {
  const source =
    example === "retrieval"
      ? RETRIEVAL_TOKENS
      : example === "drift"
        ? DRIFT_TOKENS
        : SENTENCE_TOKENS;
  return Array.from({ length }, (_, index) => source[index % source.length]);
}

function genericVectors(tokens: string[], dimension: number, salt: number) {
  return tokens.map((token, row) => {
    const seed = hashString(`${token}:${salt}:${row}`);
    return Array.from({ length: dimension }, (_, column) => {
      const base = seededValue(seed, column + salt);
      return Number((base * 0.85 + Math.sin((row + 1) * (column + 2)) * 0.12).toFixed(5));
    });
  });
}

function retrievalVectors(tokens: string[], dimension: number) {
  const q = genericVectors(tokens, dimension, 17);
  const k = genericVectors(tokens, dimension, 43);
  const v = genericVectors(tokens, dimension, 79);
  const needleIndex = tokens.findIndex((token) => token.includes("needle"));
  const queryIndex = tokens.length - 2;
  const axis = Array.from({ length: dimension }, (_, index) => (index % 2 ? 0.92 : 1));
  q[queryIndex] = axis.map((value) => value * 2.1);
  k[needleIndex] = axis.map((value) => value * 2.1);
  // Distractors remain positively correlated under phi_relu, making the trade-off visible.
  tokens.forEach((token, index) => {
    if (index !== needleIndex && index !== queryIndex) {
      k[index] = axis.map((value, column) => value * (0.35 + (column % 3) * 0.07));
    }
  });
  v[needleIndex] = Array.from({ length: dimension }, (_, index) => (index === 0 ? 1 : 0));
  v[queryIndex] = Array.from({ length: dimension }, () => 0);
  return { q, k, v };
}

function driftVectors(tokens: string[], dimension: number) {
  const vectors = genericVectors(tokens, dimension, 101);
  const last = tokens.length - 1;
  const topicA = Array.from({ length: dimension }, (_, index) => (index % 2 ? -0.9 : 0.9));
  vectors[last] = topicA;
  return {
    q: vectors.map((row) => row.map((value) => value * 0.95)),
    k: vectors.map((row) => row.map((value) => value * 0.8)),
    v: genericVectors(tokens, dimension, 131),
  };
}

export function makeInputs(
  example: ExampleName,
  length: number,
  dimension: number,
): { tokens: string[]; q: Matrix; k: Matrix; v: Matrix } {
  const tokens = tokensForExample(example, length);
  if (example === "retrieval") {
    const vectors = retrievalVectors(tokens, dimension);
    return { tokens, ...vectors };
  }
  if (example === "drift") {
    const vectors = driftVectors(tokens, dimension);
    return { tokens, ...vectors };
  }
  return {
    tokens,
    q: genericVectors(tokens, dimension, 11),
    k: genericVectors(tokens, dimension, 23),
    v: genericVectors(tokens, dimension, 37),
  };
}

export function runAttention(
  example: ExampleName,
  length: number,
  dimension: number,
  mapName: FeatureMapName,
  useNormalization: boolean,
): AttentionRun {
  const { tokens, q, k, v } = makeInputs(example, length, dimension);
  const scale = Math.sqrt(dimension);
  const scores = q.map((query) => k.map((key) => dot(query, key) / scale));
  const exactWeights = rowSoftmax(scores);
  const exactOutput = matMul(exactWeights, v);

  const phiQ = q.map((row) => featureMap(row, mapName));
  const phiK = k.map((row) => featureMap(row, mapName));
  const state = matMul(transpose(phiK), v);
  const keyMass = phiK.reduce(
    (accumulator, row) => accumulator.map((value, index) => value + row[index]),
    Array.from({ length: dimension }, () => 0),
  );
  const linearOutput = phiQ.map((query) => {
    const numerator = Array.from({ length: dimension }, (_, valueIndex) =>
      query.reduce((sum, queryValue, featureIndex) => sum + queryValue * state[featureIndex][valueIndex], 0),
    );
    if (!useNormalization) return numerator.map((value) => value / Math.max(1, length));
    const denominator = Math.max(0.0001, dot(query, keyMass));
    return numerator.map((value) => value / denominator);
  });
  const denominator = phiQ.map((query) => dot(query, keyMass));

  const flatExact = exactOutput.flat();
  const flatLinear = linearOutput.flat();
  const differences = flatExact.map((value, index) => value - flatLinear[index]);
  const mae = differences.reduce((sum, value) => sum + Math.abs(value), 0) / differences.length;
  const l2 = l2Norm(differences);
  const focusRow = example === "retrieval" ? length - 2 : 0;
  const topExact = exactWeights[focusRow].indexOf(Math.max(...exactWeights[focusRow]));
  const linearScores = phiK.map((key) => dot(phiQ[focusRow], key));
  const topLinear = linearOutput[focusRow]
    ? linearScores.indexOf(Math.max(...linearScores))
    : 0;

  return {
    tokens,
    q,
    k,
    v,
    exactWeights,
    exactOutput,
    linearOutput,
    phiQ,
    phiK,
    state,
    denominator,
    mae,
    l2,
    topExact,
    topLinear,
    operations: {
      // Counts track the dominant loops in this implementation, not device runtime.
      exact: length * length * (2 * dimension + 3),
      linear: length * dimension * dimension * 2 + length * dimension * 6,
    },
  };
}

export function formatVector(vector: number[], precision = 2) {
  return `[${vector.map((value) => value.toFixed(precision)).join(", ")}]`;
}

export function formatCount(value: number) {
  return new Intl.NumberFormat("en-US").format(Math.round(value));
}

export function complexityPoints(dimension: number, mapName: FeatureMapName) {
  const lengths = [8, 16, 32, 64, 128, 256];
  const mapCost = mapName === "exp" ? 2 : 1;
  return lengths.map((length) => ({
    length,
    exact: length * length * (2 * dimension + 3),
    linear: length * dimension * dimension * 2 + length * dimension * (5 + mapCost),
  }));
}

export function matrixMinMax(matrix: Matrix) {
  const values = matrix.flat();
  return { min: Math.min(...values), max: Math.max(...values) };
}

export function normalizeForColor(value: number, min: number, max: number) {
  return (value - min) / Math.max(0.00001, max - min);
}

export function vectorBarValues(vector: number[]) {
  const max = Math.max(...vector.map((value) => Math.abs(value)), 0.0001);
  return vector.map((value) => ({ value, ratio: Math.abs(value) / max }));
}

export const EXAMPLE_COPY: Record<ExampleName, { label: string; description: string }> = {
  sentence: { label: "Sentence / toy language", description: "The cat sat on the mat." },
  retrieval: { label: "Needle in a haystack", description: "Retrieve one relevant token among distractors." },
  drift: { label: "Topic drift", description: "A sequence whose local context changes over time." },
};

export const MAP_COPY: Record<FeatureMapName, { label: string; description: string }> = {
  elu: { label: "φ₁ · ELU + 1", description: "Positive smooth feature map" },
  relu: { label: "φ₂ · ReLU + ε", description: "Sparse, positive feature map" },
  exp: { label: "φ₃ · clipped exp", description: "Sharper positive features" },
};
