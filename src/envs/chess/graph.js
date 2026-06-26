import {
  formatMetric,
  hasMetric,
  latestMetric,
  rollingAverage,
  summarizeTraining,
} from "../shared/trainingDiagnostics.js";

let rewardChart = null;
let lossChart = null;
let diagnosticsChart = null;
let distributionChart = null;
let layoutReady = false;

function injectStyles() {
  if (document.getElementById("training-dashboard-styles")) {
    return;
  }

  const style = document.createElement("style");
  style.id = "training-dashboard-styles";
  style.textContent = `
    .training-dashboard {
      width: min(1200px, 96vw);
      display: grid;
      gap: 18px;
      margin: 0 auto;
      padding: 8px 0 28px;
      color: var(--ink-text, #2c3432);
      font-family: var(--font-ui, "PingFang SC", sans-serif);
    }
    .training-dashboard__header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-end;
      flex-wrap: wrap;
    }
    .training-dashboard__title {
      font-size: 18px;
      font-weight: 700;
      margin: 0;
      color: var(--ink-text, #2c3432);
      font-family: var(--font-heading, serif);
    }
    .training-dashboard__subtitle {
      margin: 4px 0 0;
      color: var(--ink-text-muted, #68716d);
      font-size: 13px;
    }
    .training-dashboard__cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
    }
    .training-dashboard__card {
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid rgba(60, 70, 60, 0.08);
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.06);
      padding: 13px 15px;
      backdrop-filter: blur(12px);
    }
    .training-dashboard__card-label {
      display: block;
      font-size: 12px;
      color: var(--ink-text-muted, #68716d);
      margin-bottom: 6px;
      letter-spacing: 0.04em;
    }
    .training-dashboard__card-value {
      display: block;
      font-size: 24px;
      font-weight: 700;
      color: var(--ink-text, #2c3432);
    }
    .training-dashboard__charts {
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
    }
    .training-dashboard__debug-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr);
      gap: 16px;
    }
    .training-dashboard__panel {
      border-radius: 24px;
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid rgba(60, 70, 60, 0.08);
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.06);
      padding: 14px;
      backdrop-filter: blur(12px);
    }
    .training-dashboard__panel-title {
      font-size: 18px;
      font-weight: 700;
      margin: 0 0 10px;
      color: var(--ink-text, #2c3432);
    }
    .training-dashboard__chart {
      width: 100%;
      height: 280px;
    }
    .training-dashboard__chart--compact {
      height: 240px;
    }
    .training-dashboard__list {
      display: grid;
      gap: 10px;
      margin: 0;
      padding: 0;
      list-style: none;
    }
    .training-dashboard__list-item {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(95, 158, 160, 0.08);
    }
    .training-dashboard__list-label {
      font-size: 13px;
      color: var(--ink-text, #2c3432);
      font-weight: 600;
    }
    .training-dashboard__list-value {
      font-size: 12px;
      color: var(--ink-text-muted, #68716d);
      text-align: right;
    }
    .training-dashboard__empty {
      margin: 0;
      color: var(--ink-text-muted, #68716d);
      font-size: 13px;
    }
    @media (max-width: 860px) {
      .training-dashboard__debug-grid {
        grid-template-columns: 1fr;
      }
    }
  `;
  document.head.appendChild(style);
}

const SERIES_COLORS = {
  Reward: "#5F9EA0",
  "Reward MA": "#5F9EA0",
  Steps: "#6F7FA8",
  "Steps MA": "#8C97B6",
  "Actor Loss": "#D2A24C",
  "Critic Loss": "#6F7FA8",
  "Actor Delta": "#5F9EA0",
  "Critic Delta": "#6E8B74",
  "Ratio Mean": "#5F9EA0",
  "Ratio Min": "#A0AFCA",
  "Ratio Max": "#C75B5B",
  Entropy: "#A66DDA",
  "Adv Mean": "#6E8B74",
  "Adv Std": "#D2A24C",
  "State Nonzero": "#8A9591",
  "Legal Actions": "#6E8B74",
  "Legal Rate": "#5F9EA0",
  "Invalid/Valid": "#C75B5B",
  "Repeat Move Rate": "#C75B5B",
  "Repeat State Rate": "#D28D4C",
  "Attack Move Rate": "#6E8B74",
  "Win Episodes": "#6E8B74",
  "Loss Episodes": "#C75B5B",
  "Truncated Episodes": "#A0AFCA",
  "Q Delta": "#C75B5B",
  Epsilon: "#C9A34A",
  "Max Q": "#5F9EA0",
  "Visited States": "#6F7FA8",
};

function findSeriesColor(name) {
  return Object.entries(SERIES_COLORS).find(([key]) => name.startsWith(key))?.[1] ?? "#c7ced0";
}

function styleSeries(series) {
  return series.map((item) => {
    const color = findSeriesColor(item.name);
    const isMovingAverage = item.name.includes("MA");
    return {
      ...item,
      symbol: "none",
      showSymbol: false,
      lineStyle: {
        width: isMovingAverage ? 3 : 2,
        type: isMovingAverage ? "dashed" : "solid",
        color,
      },
      itemStyle: { color },
      areaStyle: item.name === "Reward" || item.name === "Reward MA"
        ? { color: `${color}22` }
        : undefined,
      emphasis: {
        focus: "series",
      },
    };
  });
}

function ensureLayout() {
  const dom = document.getElementById("graph-container");
  if (!dom) return null;

  injectStyles();

  if (!layoutReady) {
    dom.innerHTML = `
      <div class="training-dashboard">
        <div class="training-dashboard__header">
          <div>
            <h2 class="training-dashboard__title" id="training-dashboard-title">RL Training Diagnostics</h2>
            <p class="training-dashboard__subtitle" id="training-dashboard-subtitle">Track whether training is healthy and why performance stalls.</p>
          </div>
        </div>
        <div class="training-dashboard__cards" id="training-dashboard-cards"></div>
        <div class="training-dashboard__charts">
          <div class="training-dashboard__panel">
            <h3 class="training-dashboard__panel-title" id="reward-panel-title">Reward</h3>
            <div class="training-dashboard__chart" id="reward-chart"></div>
          </div>
          <div class="training-dashboard__panel">
            <h3 class="training-dashboard__panel-title" id="loss-panel-title">Loss</h3>
            <div class="training-dashboard__chart" id="loss-chart"></div>
          </div>
          <div class="training-dashboard__panel">
            <h3 class="training-dashboard__panel-title" id="diagnostics-panel-title">Diagnostics</h3>
            <div class="training-dashboard__chart" id="diagnostics-chart"></div>
          </div>
          <div class="training-dashboard__debug-grid">
            <div class="training-dashboard__panel">
              <h3 class="training-dashboard__panel-title" id="action-frequency-panel-title">Top 5 Actions Frequency</h3>
              <div id="action-frequency-summary"></div>
            </div>
            <div class="training-dashboard__panel">
              <h3 class="training-dashboard__panel-title" id="distribution-panel-title">Win vs Loss Reward Distribution</h3>
              <div class="training-dashboard__chart training-dashboard__chart--compact" id="distribution-chart"></div>
            </div>
          </div>
        </div>
      </div>
    `;
    dom.style.width = "100%";
    dom.style.height = "auto";
    layoutReady = true;
  }

  const rewardDom = document.getElementById("reward-chart");
  const lossDom = document.getElementById("loss-chart");
  const diagnosticsDom = document.getElementById("diagnostics-chart");
  const distributionDom = document.getElementById("distribution-chart");

  if (!rewardChart && rewardDom) {
    rewardChart = echarts.init(rewardDom);
  }
  if (!lossChart && lossDom) {
    lossChart = echarts.init(lossDom);
  }
  if (!diagnosticsChart && diagnosticsDom) {
    diagnosticsChart = echarts.init(diagnosticsDom);
  }
  if (!distributionChart && distributionDom) {
    distributionChart = echarts.init(distributionDom);
  }

  if (!window.__trainingDashboardResizeBound) {
    window.__trainingDashboardResizeBound = true;
    window.addEventListener("resize", () => {
      rewardChart?.resize();
      lossChart?.resize();
      diagnosticsChart?.resize();
      distributionChart?.resize();
    });
  }

  return dom;
}

function buildCards(records, options) {
  const summary = summarizeTraining(records, options.movingAverageWindow);
  const cards = [
    { label: "Latest Episode", value: formatMetric(summary.latestEpisode, 0) },
    { label: `Reward MA(${options.movingAverageWindow})`, value: formatMetric(summary.rewardMean, 3) },
    { label: `Steps MA(${options.movingAverageWindow})`, value: formatMetric(summary.stepsMean, 2) },
    { label: `Success Rate(${options.movingAverageWindow})`, value: summary.successRate === null ? "--" : `${(summary.successRate * 100).toFixed(1)}%` },
    { label: `Win Rate(${options.movingAverageWindow})`, value: summary.winRate === null ? "--" : `${(summary.winRate * 100).toFixed(1)}%` },
  ];

  if (options.mode === "ppo") {
    cards.push(
      { label: "Actor Loss", value: formatMetric(latestMetric(records, "actorLoss"), 4) },
      { label: "Critic Loss", value: formatMetric(latestMetric(records, "criticLoss"), 4) },
      { label: "Actor Delta", value: formatMetric(latestMetric(records, "actorParamDelta"), 4) },
      { label: "Critic Delta", value: formatMetric(latestMetric(records, "criticParamDelta"), 4) },
      { label: "Ratio Mean", value: formatMetric(latestMetric(records, "ratioMean"), 4) },
      { label: "Entropy", value: formatMetric(latestMetric(records, "entropy"), 4) },
      { label: "Adv Std", value: formatMetric(latestMetric(records, "advantageStd"), 4) },
      { label: "State Nonzero", value: formatMetric(latestMetric(records, "stateNonzeroCount"), 1) },
      { label: "Legal Actions", value: formatMetric(latestMetric(records, "legalActions"), 1) },
      { label: "Legal Rate", value: latestMetric(records, "legalRate") === null ? "--" : `${(latestMetric(records, "legalRate") * 100).toFixed(1)}%` },
      { label: "Invalid / Valid", value: formatMetric(latestMetric(records, "invalidPerValid"), 3) },
      { label: "Repeat Move Rate", value: latestMetric(records, "repeatMoveRate") === null ? "--" : `${(latestMetric(records, "repeatMoveRate") * 100).toFixed(1)}%` },
    );
  } else if (options.mode === "qlearning") {
    cards.push(
      { label: "Q Delta", value: formatMetric(latestMetric(records, "qDelta"), 4) },
      { label: "Epsilon", value: formatMetric(latestMetric(records, "epsilon"), 4) },
      { label: "Max Q", value: formatMetric(latestMetric(records, "maxQ"), 4) },
      { label: "Visited States", value: formatMetric(latestMetric(records, "visitedStates"), 0) },
    );
  }

  return cards;
}

function renderCards(records, options) {
  const cardsEl = document.getElementById("training-dashboard-cards");
  if (!cardsEl) return;

  const cards = buildCards(records, options);
  const toneForLabel = (label) => {
    if (label.includes("Reward")) return "#5F9EA0";
    if (label.includes("Success") || label.includes("Win")) return "#6E8B74";
    if (label.includes("Loss") || label.includes("Invalid")) return "#C75B5B";
    if (label.includes("Episode")) return "#C9A34A";
    if (label.includes("Critic")) return "#6F7FA8";
    if (label.includes("Actor")) return "#D2A24C";
    if (label.includes("Entropy")) return "#A66DDA";
    if (label.includes("Repeat")) return "#C75B5B";
    return "#2C3432";
  };
  cardsEl.innerHTML = cards.map((card) => `
    <div class="training-dashboard__card">
      <span class="training-dashboard__card-label">${card.label}</span>
      <span class="training-dashboard__card-value" style="color:${toneForLabel(card.label)}">${card.value}</span>
    </div>
  `).join("");
}

function aggregateRecentTopActions(records, windowSize = 20, limit = 5) {
  const recent = records.slice(-windowSize);
  const aggregated = new Map();
  let totalCount = 0;

  for (const record of recent) {
    for (const entry of record.actionCounts ?? []) {
      const previous = aggregated.get(entry.actionId) ?? {
        actionId: entry.actionId,
        count: 0,
        label: entry.label ?? `action:${entry.actionId}`,
      };
      previous.count += entry.count ?? 0;
      aggregated.set(entry.actionId, previous);
      totalCount += entry.count ?? 0;
    }
  }

  return Array.from(aggregated.values())
    .sort((lhs, rhs) => rhs.count - lhs.count || lhs.actionId - rhs.actionId)
    .slice(0, limit)
    .map((entry) => ({
      ...entry,
      rate: totalCount > 0 ? entry.count / totalCount : 0,
    }));
}

function renderActionFrequencySummary(records, options) {
  const container = document.getElementById("action-frequency-summary");
  if (!container) return;

  const topActions = aggregateRecentTopActions(records, options.movingAverageWindow, 5);
  if (topActions.length === 0) {
    container.innerHTML = `<p class="training-dashboard__empty">暂无动作频率数据。</p>`;
    return;
  }

  container.innerHTML = `
    <ol class="training-dashboard__list">
      ${topActions.map((entry, index) => `
        <li class="training-dashboard__list-item">
          <span class="training-dashboard__list-label">#${index + 1} ${entry.label}</span>
          <span class="training-dashboard__list-value">${entry.count} moves · ${(entry.rate * 100).toFixed(1)}%</span>
        </li>
      `).join("")}
    </ol>
  `;
}

function buildRewardDistribution(records, binCount = 12) {
  const rewardsByOutcome = {
    win: records.filter((record) => record.outcome === "win").map((record) => record.reward),
    loss: records.filter((record) => record.outcome === "loss").map((record) => record.reward),
    truncated: records.filter((record) => record.outcome === "truncated").map((record) => record.reward),
  };
  const allRewards = Object.values(rewardsByOutcome).flat();

  if (allRewards.length === 0) {
    return null;
  }

  let minReward = Math.min(...allRewards);
  let maxReward = Math.max(...allRewards);
  if (minReward === maxReward) {
    minReward -= 0.5;
    maxReward += 0.5;
  }

  const width = (maxReward - minReward) / binCount;
  const bins = new Array(binCount).fill(null).map((_, index) => ({
    start: minReward + width * index,
    end: index === binCount - 1 ? maxReward : minReward + width * (index + 1),
    win: 0,
    loss: 0,
    truncated: 0,
  }));

  const bucketIndexFor = (reward) => {
    const rawIndex = Math.floor((reward - minReward) / width);
    return Math.max(0, Math.min(binCount - 1, rawIndex));
  };

  for (const [outcome, rewards] of Object.entries(rewardsByOutcome)) {
    for (const reward of rewards) {
      bins[bucketIndexFor(reward)][outcome] += 1;
    }
  }

  return {
    labels: bins.map((bin) => `${bin.start.toFixed(2)} to ${bin.end.toFixed(2)}`),
    winData: bins.map((bin) => bin.win),
    lossData: bins.map((bin) => bin.loss),
    truncatedData: bins.map((bin) => bin.truncated),
  };
}

function rewardDistributionOption(records) {
  const distribution = buildRewardDistribution(records);
  if (!distribution) {
    return {
      title: {
        text: "Reward Distribution",
        textStyle: {
          fontSize: 16,
          fontWeight: 700,
          color: "#2C3432",
          fontFamily: '"STSong", "Songti SC", "Noto Serif SC", serif',
        },
      },
      graphic: {
        type: "text",
        left: "center",
        top: "middle",
        style: {
          text: "No outcome reward data yet",
          fill: "#68716D",
          fontSize: 13,
        },
      },
    };
  }

  const series = styleSeries([
    { name: "Win Episodes", type: "bar", stack: "reward-distribution", data: distribution.winData },
    { name: "Loss Episodes", type: "bar", stack: "reward-distribution", data: distribution.lossData },
    { name: "Truncated Episodes", type: "bar", stack: "reward-distribution", data: distribution.truncatedData },
  ]).map((item) => ({
    ...item,
    barMaxWidth: 24,
    areaStyle: undefined,
  }));

  return {
    backgroundColor: "#ffffff",
    color: Object.values(SERIES_COLORS),
    animationDuration: 300,
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      backgroundColor: "rgba(255, 255, 255, 0.96)",
      borderColor: "rgba(60, 70, 60, 0.08)",
      borderWidth: 1,
      textStyle: { color: "#2C3432", fontSize: 13 },
    },
    legend: {
      top: 8,
      textStyle: { color: "#68716D", fontSize: 12 },
    },
    grid: {
      left: "7%",
      right: "4%",
      bottom: "16%",
      top: "20%",
      containLabel: true,
    },
    xAxis: {
      type: "category",
      data: distribution.labels,
      axisLabel: {
        color: "#68716D",
        fontSize: 11,
        rotate: 18,
      },
      axisLine: {
        lineStyle: { color: "#D6DBD8" },
      },
    },
    yAxis: {
      type: "value",
      name: "Episodes",
      axisLabel: { color: "#68716D", fontSize: 12 },
      axisLine: {
        show: true,
        lineStyle: { color: "#D6DBD8" },
      },
      splitLine: {
        lineStyle: { color: "#E8EBE8" },
      },
      nameTextStyle: { color: "#68716D", fontSize: 12 },
    },
    series,
  };
}

function buildRewardSeries(records, options) {
  const rewardMA = rollingAverage(records, "reward", options.movingAverageWindow);
  const stepsMA = rollingAverage(records, "steps", options.movingAverageWindow);
  return [
    {
      name: "Reward",
      type: "line",
      smooth: true,
      yAxisIndex: 0,
      data: records.map((record) => record.reward),
    },
    {
      name: `Reward MA(${options.movingAverageWindow})`,
      type: "line",
      smooth: true,
      yAxisIndex: 0,
      data: rewardMA,
    },
    {
      name: "Steps",
      type: "line",
      smooth: true,
      yAxisIndex: 1,
      data: records.map((record) => record.steps),
    },
    {
      name: `Steps MA(${options.movingAverageWindow})`,
      type: "line",
      smooth: true,
      yAxisIndex: 1,
      data: stepsMA,
    },
  ];
}

function buildLossSeries(records, options) {
  const series = [];

  if (options.mode === "ppo") {
    if (hasMetric(records, "actorLoss")) {
      series.push({ name: "Actor Loss", type: "line", smooth: true, yAxisIndex: 0, data: records.map((record) => record.actorLoss) });
    }
    if (hasMetric(records, "criticLoss")) {
      series.push({ name: "Critic Loss", type: "line", smooth: true, yAxisIndex: 0, data: records.map((record) => record.criticLoss) });
    }
    if (hasMetric(records, "actorParamDelta")) {
      series.push({ name: "Actor Delta", type: "line", smooth: true, yAxisIndex: 1, data: records.map((record) => record.actorParamDelta) });
    }
    if (hasMetric(records, "criticParamDelta")) {
      series.push({ name: "Critic Delta", type: "line", smooth: true, yAxisIndex: 1, data: records.map((record) => record.criticParamDelta) });
    }
  } else if (options.mode === "qlearning") {
    if (hasMetric(records, "qDelta")) {
      series.push({ name: "Q Delta", type: "line", smooth: true, yAxisIndex: 0, data: records.map((record) => record.qDelta) });
    }
    if (hasMetric(records, "maxQ")) {
      series.push({ name: "Max Q", type: "line", smooth: true, yAxisIndex: 1, data: records.map((record) => record.maxQ) });
    }
  }

  return series;
}

function buildDiagnosticsSeries(records, options) {
  const series = [];

  if (hasMetric(records, "success")) {
    series.push({ name: "Success Rate", type: "line", smooth: true, yAxisIndex: 0, data: records.map((record) => record.success === null || record.success === undefined ? null : Number(record.success)) });
  }
  if (hasMetric(records, "win")) {
    series.push({ name: "Win Rate", type: "line", smooth: true, yAxisIndex: 0, data: records.map((record) => record.win === null || record.win === undefined ? null : Number(record.win)) });
  }

  if (options.mode === "ppo") {
    if (hasMetric(records, "legalRate")) {
      series.push({ name: "Legal Rate", type: "line", smooth: true, yAxisIndex: 0, data: records.map((record) => record.legalRate) });
    }
    if (hasMetric(records, "ratioMean")) {
      series.push({ name: "Ratio Mean", type: "line", smooth: true, yAxisIndex: 1, data: records.map((record) => record.ratioMean) });
    }
    if (hasMetric(records, "ratioMin")) {
      series.push({ name: "Ratio Min", type: "line", smooth: true, yAxisIndex: 1, data: records.map((record) => record.ratioMin) });
    }
    if (hasMetric(records, "ratioMax")) {
      series.push({ name: "Ratio Max", type: "line", smooth: true, yAxisIndex: 1, data: records.map((record) => record.ratioMax) });
    }
    if (hasMetric(records, "entropy")) {
      series.push({ name: "Entropy", type: "line", smooth: true, yAxisIndex: 1, data: records.map((record) => record.entropy) });
    }
    if (hasMetric(records, "advantageMean")) {
      series.push({ name: "Adv Mean", type: "line", smooth: true, yAxisIndex: 1, data: records.map((record) => record.advantageMean) });
    }
    if (hasMetric(records, "advantageStd")) {
      series.push({ name: "Adv Std", type: "line", smooth: true, yAxisIndex: 1, data: records.map((record) => record.advantageStd) });
    }
    if (hasMetric(records, "stateNonzeroCount")) {
      series.push({ name: "State Nonzero", type: "line", smooth: true, yAxisIndex: 1, data: records.map((record) => record.stateNonzeroCount) });
    }
    if (hasMetric(records, "legalActions")) {
      series.push({ name: "Legal Actions", type: "line", smooth: true, yAxisIndex: 1, data: records.map((record) => record.legalActions) });
    }
    if (hasMetric(records, "invalidPerValid")) {
      series.push({ name: "Invalid/Valid", type: "line", smooth: true, yAxisIndex: 1, data: records.map((record) => record.invalidPerValid) });
    }
    if (hasMetric(records, "repeatRate")) {
      series.push({ name: "Repeat State Rate", type: "line", smooth: true, yAxisIndex: 0, data: records.map((record) => record.repeatRate) });
    }
    if (hasMetric(records, "repeatMoveRate")) {
      series.push({ name: "Repeat Move Rate", type: "line", smooth: true, yAxisIndex: 0, data: records.map((record) => record.repeatMoveRate) });
    }
    if (hasMetric(records, "attackMovesRate")) {
      series.push({ name: "Attack Move Rate", type: "line", smooth: true, yAxisIndex: 0, data: records.map((record) => record.attackMovesRate) });
    }
  } else if (options.mode === "qlearning") {
    if (hasMetric(records, "epsilon")) {
      series.push({ name: "Epsilon", type: "line", smooth: true, yAxisIndex: 1, data: records.map((record) => record.epsilon) });
    }
    if (hasMetric(records, "visitedStates")) {
      series.push({ name: "Visited States", type: "line", smooth: true, yAxisIndex: 1, data: records.map((record) => record.visitedStates) });
    }
  }

  return series;
}

function selectedLegendMap(series, mode, chartKey) {
  const selected = {};

  const isCore = (name) => {
    if (chartKey === "reward") {
      return name === "Reward"
        || name.startsWith("Reward MA")
        || name === "Steps"
        || name.startsWith("Steps MA");
    }

    if (chartKey === "loss") {
      if (mode === "ppo") {
        return name === "Actor Loss"
          || name === "Critic Loss"
          || name === "Actor Delta"
          || name === "Critic Delta";
      }
      return name === "Q Delta" || name === "Max Q";
    }

    if (chartKey === "diagnostics") {
      if (mode === "ppo") {
        return name === "Ratio Mean"
          || name === "Legal Rate"
          || name === "Entropy"
          || name === "Repeat Move Rate"
          || name === "State Nonzero"
          || name === "Invalid/Valid";
      }
      return name === "Success Rate" || name === "Win Rate" || name === "Epsilon" || name === "Visited States";
    }

    return false;
  };

  for (const item of series) {
    selected[item.name] = isCore(item.name);
  }

  return selected;
}

function baseChartOption(title, xAxisData, yAxisNames, series, mode, chartKey) {
  return {
    backgroundColor: "#ffffff",
    color: Object.values(SERIES_COLORS),
    animationDuration: 300,
    title: {
      text: title,
      textStyle: {
        fontSize: 18,
        fontWeight: 700,
        color: "#2C3432",
        fontFamily: '"STSong", "Songti SC", "Noto Serif SC", serif',
      },
    },
    tooltip: {
      trigger: "axis",
      backgroundColor: "rgba(255, 255, 255, 0.96)",
      borderColor: "rgba(60, 70, 60, 0.08)",
      borderWidth: 1,
      textStyle: {
        color: "#2C3432",
        fontSize: 14,
      },
      extraCssText: "box-shadow: 0 8px 24px rgba(0,0,0,0.08); border-radius: 14px;",
    },
    legend: {
      type: "scroll",
      top: 26,
      itemGap: 18,
      itemWidth: 14,
      itemHeight: 10,
      selected: selectedLegendMap(series, mode, chartKey),
      textStyle: {
        color: "#68716D",
        fontSize: 13,
      },
      pageTextStyle: {
        color: "#68716D",
        fontSize: 12,
      },
      pageIconColor: "#5F9EA0",
    },
    grid: {
      left: "5%",
      right: "5%",
      bottom: "12%",
      top: "26%",
      containLabel: true,
    },
    xAxis: {
      type: "category",
      name: "Episode",
      data: xAxisData,
      axisLine: {
        lineStyle: { color: "#D6DBD8" },
      },
      axisLabel: {
        color: "#68716D",
        fontSize: 12,
      },
      splitLine: {
        show: false,
      },
      nameTextStyle: {
        color: "#68716D",
        fontSize: 12,
      },
    },
    yAxis: yAxisNames.map((name, index) => ({
      type: "value",
      name,
      position: index === 0 ? "left" : "right",
      axisLine: {
        show: true,
        lineStyle: { color: "#D6DBD8" },
      },
      axisLabel: {
        color: "#68716D",
        fontSize: 12,
      },
      splitLine: {
        lineStyle: {
          color: "#E8EBE8",
        },
      },
      nameTextStyle: {
        color: "#68716D",
        fontSize: 12,
      },
    })),
    series: styleSeries(series),
  };
}

export function drawlineGraph(records, options = {}) {
  if (!records || records.length === 0) return;
  if (!ensureLayout()) return;

  const normalizedOptions = {
    title: options.title ?? "RL Training Diagnostics",
    subtitle: options.subtitle ?? "Use the dashboard to inspect reward trends, stability, and failure modes.",
    mode: options.mode ?? (hasMetric(records, "qDelta") ? "qlearning" : "ppo"),
    movingAverageWindow: options.movingAverageWindow ?? 20,
  };

  const episodes = records.map((record) => record.episode);
  document.getElementById("training-dashboard-title").textContent = normalizedOptions.title;
  document.getElementById("training-dashboard-subtitle").textContent = normalizedOptions.subtitle;
  document.getElementById("reward-panel-title").textContent = "Reward";
  document.getElementById("loss-panel-title").textContent = normalizedOptions.mode === "ppo" ? "Loss" : "Learning Signals";
  document.getElementById("diagnostics-panel-title").textContent = "Diagnostics";
  document.getElementById("action-frequency-panel-title").textContent = `Top 5 Actions Frequency (Last ${normalizedOptions.movingAverageWindow})`;
  document.getElementById("distribution-panel-title").textContent = "Win vs Loss Reward Distribution";

  renderCards(records, normalizedOptions);
  renderActionFrequencySummary(records, normalizedOptions);

  rewardChart?.setOption(baseChartOption(
    "Reward",
    episodes,
    ["Reward", "Steps"],
    buildRewardSeries(records, normalizedOptions),
    normalizedOptions.mode,
    "reward",
  ), true);

  lossChart?.setOption(baseChartOption(
    normalizedOptions.mode === "ppo" ? "Loss" : "Learning Signals",
    episodes,
    normalizedOptions.mode === "ppo" ? ["Loss", "Delta"] : ["Value", "Value"],
    buildLossSeries(records, normalizedOptions),
    normalizedOptions.mode,
    "loss",
  ), true);

  diagnosticsChart?.setOption(baseChartOption(
    "Diagnostics",
    episodes,
    normalizedOptions.mode === "ppo" ? ["Rate", "Diagnostics"] : ["Rate", "Exploration"],
    buildDiagnosticsSeries(records, normalizedOptions),
    normalizedOptions.mode,
    "diagnostics",
  ), true);

  distributionChart?.setOption(rewardDistributionOption(records), true);
}

export function initGraph() {
  ensureLayout();
}

export const renderTrainingDashboard = drawlineGraph;
