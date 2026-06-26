export function createEpisodeMetrics(values = {}) {
  return {
    algorithm: values.algorithm ?? "unknown",
    episodeIndex: values.episodeIndex ?? 0,
    path: Array.isArray(values.path) ? values.path : [],
    steps: values.steps ?? 0,
    reward: values.reward ?? 0,
    success: Boolean(values.success),
    failure: !values.success,
    pathLength: values.pathLength ?? values.steps ?? 0,
    trace: Array.isArray(values.trace) ? values.trace : [],
    updateInfo: values.updateInfo ?? {},
    metadata: values.metadata ?? {},
  };
}

export function movingAverage(values, windowSize = 10) {
  return values.map((_, index) => {
    const start = Math.max(0, index - windowSize + 1);
    let sum = 0;
    let count = 0;
    for (let i = start; i <= index; i++) {
      const value = values[i];
      if (typeof value === "number" && Number.isFinite(value)) {
        sum += value;
        count += 1;
      }
    }
    return count > 0 ? sum / count : null;
  });
}

export function successRateSeries(history, windowSize = 10) {
  return history.map((_, index) => {
    const start = Math.max(0, index - windowSize + 1);
    const slice = history.slice(start, index + 1);
    if (slice.length === 0) return null;
    const successes = slice.reduce((sum, episode) => sum + (episode.success ? 1 : 0), 0);
    return successes / slice.length;
  });
}

export function comparisonSnapshot(histories = {}) {
  const entries = Object.entries(histories);
  return entries.map(([algorithm, history]) => {
    const last = history.at(-1) ?? null;
    return {
      algorithm,
      episodes: history.length,
      latestReward: last?.reward ?? 0,
      latestSteps: last?.steps ?? 0,
      successRate: successRateSeries(history, 10).at(-1) ?? 0,
    };
  });
}
