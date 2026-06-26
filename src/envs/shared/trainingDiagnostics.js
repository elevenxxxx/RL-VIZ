const TRAINING_RECORD_TEMPLATE = {
  episode: 0,
  reward: 0,
  steps: 0,
  success: false,
  win: false,
  actorLoss: null,
  criticLoss: null,
  actorParamDelta: null,
  criticParamDelta: null,
  ratioMean: null,
  ratioMin: null,
  ratioMax: null,
  entropy: null,
  advantageMean: null,
  advantageStd: null,
  stateNonzeroCount: null,
  legalActions: null,
  totalActions: null,
  legalRate: null,
  invalidAttempts: null,
  validAttempts: null,
  invalidPerValid: null,
  repeatRate: null,
  repeatMoveRate: null,
  attackMovesRate: null,
  actionLoopCount: null,
  outcome: null,
  actionCounts: [],
  topActions: [],
  qDelta: null,
  epsilon: null,
  maxQ: null,
  visitedStates: null,
};

export function createTrainingRecord(values = {}) {
  return {
    ...TRAINING_RECORD_TEMPLATE,
    ...values,
  };
}

export function hasMetric(records, key) {
  return records.some((record) => record[key] !== null && record[key] !== undefined);
}

export function latestMetric(records, key) {
  for (let i = records.length - 1; i >= 0; i--) {
    const value = records[i][key];
    if (value !== null && value !== undefined) {
      return value;
    }
  }
  return null;
}

export function rollingAverage(records, key, windowSize = 20) {
  return records.map((_, index) => {
    const start = Math.max(0, index - windowSize + 1);
    let sum = 0;
    let count = 0;

    for (let i = start; i <= index; i++) {
      const value = records[i][key];
      if (typeof value === "number" && Number.isFinite(value)) {
        sum += value;
        count++;
      }
    }

    return count > 0 ? sum / count : null;
  });
}

export function recentMean(records, key, count = 20) {
  const slice = records.slice(-count);
  let sum = 0;
  let seen = 0;

  for (const record of slice) {
    const value = record[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      sum += value;
      seen++;
    }
  }

  return seen > 0 ? sum / seen : null;
}

export function recentRate(records, key, count = 20) {
  const slice = records.slice(-count);
  if (slice.length === 0) return null;

  let positives = 0;
  for (const record of slice) {
    if (record[key]) positives++;
  }

  return positives / slice.length;
}

export function formatMetric(value, digits = 3) {
  if (value === null || value === undefined) return "--";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value !== "number" || !Number.isFinite(value)) return String(value);
  return value.toFixed(digits);
}

export function summarizeTraining(records, windowSize = 20) {
  return {
    latestEpisode: latestMetric(records, "episode"),
    latestReward: latestMetric(records, "reward"),
    rewardMean: recentMean(records, "reward", windowSize),
    stepsMean: recentMean(records, "steps", windowSize),
    successRate: recentRate(records, "success", windowSize),
    winRate: recentRate(records, "win", windowSize),
  };
}
