import { id2piece, num2rc } from "../chess/utils.js";

const VIEWER_MAX_EPISODES = 20;

function injectEpisodeViewerStyles() {
  if (document.getElementById("episode-viewer-styles")) return;

  const style = document.createElement("style");
  style.id = "episode-viewer-styles";
  style.textContent = `
    .episode-viewer {
      width: min(1180px, 100%);
      margin: 18px auto 0;
      display: grid;
      gap: 16px;
      color: var(--ink-text, #2C3432);
    }
    .episode-viewer__shell {
      border-radius: 28px;
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid rgba(60, 70, 60, 0.08);
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.06);
      backdrop-filter: blur(12px);
      padding: 18px;
      display: grid;
      gap: 14px;
    }
    .episode-viewer__header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      flex-wrap: wrap;
    }
    .episode-viewer__title {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
    }
    .episode-viewer__subtitle {
      margin: 6px 0 0;
      color: var(--ink-text-muted, #68716D);
      font-size: 13px;
      line-height: 1.6;
    }
    .episode-viewer__episode-select {
      min-width: 180px;
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid rgba(60, 70, 60, 0.12);
      background: rgba(255, 255, 255, 0.95);
      color: var(--ink-text, #2C3432);
    }
    .episode-viewer__controls {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    .episode-viewer__button {
      width: auto;
      min-width: 88px;
      padding: 10px 14px;
    }
    .episode-viewer__step-indicator {
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid rgba(60, 70, 60, 0.08);
      color: var(--ink-text-muted, #68716D);
      font-size: 13px;
    }
    .episode-viewer__summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
    }
    .episode-viewer__summary-card {
      padding: 12px 14px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.86);
      border: 1px solid rgba(60, 70, 60, 0.06);
    }
    .episode-viewer__summary-label {
      display: block;
      font-size: 12px;
      color: var(--ink-text-muted, #68716D);
      margin-bottom: 6px;
      letter-spacing: 0.04em;
    }
    .episode-viewer__summary-value {
      display: block;
      font-size: 20px;
      font-weight: 700;
      color: var(--ink-text, #2C3432);
    }
    .episode-viewer__grid {
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr);
      gap: 16px;
      align-items: stretch;
    }
    .episode-viewer__panel {
      border-radius: 24px;
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid rgba(60, 70, 60, 0.08);
      padding: 14px;
      display: grid;
      gap: 12px;
      min-height: 100%;
    }
    .episode-viewer__panel-title {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
    }
    .episode-viewer__empty {
      margin: 0;
      color: var(--ink-text-muted, #68716D);
      font-size: 13px;
    }
    .episode-viewer__rollout-stage {
      min-height: 420px;
      display: grid;
      place-items: center;
      border-radius: 20px;
      background: rgba(243, 244, 242, 0.85);
      border: 1px solid rgba(60, 70, 60, 0.08);
      overflow: hidden;
      padding: 14px;
    }
    .episode-viewer__meta {
      display: grid;
      gap: 8px;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    }
    .episode-viewer__meta-item {
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(95, 158, 160, 0.06);
      border: 1px solid rgba(95, 158, 160, 0.08);
    }
    .episode-viewer__meta-label {
      display: block;
      font-size: 12px;
      color: var(--ink-text-muted, #68716D);
      margin-bottom: 4px;
    }
    .episode-viewer__meta-value {
      display: block;
      font-size: 14px;
      color: var(--ink-text, #2C3432);
      line-height: 1.5;
      word-break: break-word;
    }
    .episode-viewer__charts {
      display: grid;
      gap: 16px;
    }
    .episode-viewer__chart {
      width: 100%;
      height: 220px;
    }
    .episode-viewer__chart--compact {
      height: 200px;
    }
    .episode-viewer__chess-board {
      display: grid;
      grid-template-columns: repeat(9, minmax(40px, 1fr));
      gap: 6px;
      width: min(100%, 520px);
    }
    .episode-viewer__chess-cell {
      aspect-ratio: 1 / 1;
      border-radius: 14px;
      border: 1px solid rgba(92, 79, 66, 0.18);
      background: linear-gradient(180deg, rgba(255,253,248,0.92) 0%, rgba(244,236,223,0.92) 100%);
      display: grid;
      place-items: center;
      font-size: 22px;
      color: #505755;
      position: relative;
    }
    .episode-viewer__chess-cell--red {
      color: #C75B5B;
    }
    .episode-viewer__chess-cell--from {
      box-shadow: inset 0 0 0 2px rgba(201, 163, 74, 0.7);
    }
    .episode-viewer__chess-cell--to {
      box-shadow: inset 0 0 0 2px rgba(95, 158, 160, 0.88);
      background: linear-gradient(180deg, rgba(240,249,249,0.98) 0%, rgba(226,241,241,0.96) 100%);
    }
    .episode-viewer__maze-board {
      display: grid;
      gap: 2px;
      width: min(100%, 520px);
      background: rgba(104, 113, 109, 0.08);
      padding: 8px;
      border-radius: 18px;
    }
    .episode-viewer__maze-cell {
      aspect-ratio: 1 / 1;
      border-radius: 4px;
      background: #F3F4F2;
      position: relative;
    }
    .episode-viewer__maze-cell--wall {
      background: #505755;
    }
    .episode-viewer__maze-cell--start {
      background: #6E8B74;
    }
    .episode-viewer__maze-cell--goal {
      background: #C75B5B;
    }
    .episode-viewer__maze-cell--path {
      background: rgba(201, 163, 74, 0.35);
    }
    .episode-viewer__maze-cell--agent {
      box-shadow: inset 0 0 0 2px rgba(95, 158, 160, 0.9);
      background: rgba(95, 158, 160, 0.28);
    }
    .episode-viewer__maze-agent-dot {
      position: absolute;
      inset: 22%;
      border-radius: 999px;
      background: #5F9EA0;
    }
    @media (max-width: 980px) {
      .episode-viewer__grid {
        grid-template-columns: 1fr;
      }
    }
  `;
  document.head.appendChild(style);
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined) return "--";
  if (typeof value !== "number" || !Number.isFinite(value)) return String(value);
  return value.toFixed(digits);
}

function formatPercent(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "--";
  return `${(value * 100).toFixed(1)}%`;
}

function ensureEcharts() {
  return typeof window !== "undefined" && window.echarts ? window.echarts : null;
}

function chartBaseOption() {
  return {
    backgroundColor: "#ffffff",
    animationDuration: 240,
    grid: {
      left: "8%",
      right: "6%",
      top: "14%",
      bottom: "12%",
      containLabel: true,
    },
    tooltip: {
      trigger: "axis",
      backgroundColor: "rgba(255,255,255,0.96)",
      borderColor: "rgba(60,70,60,0.08)",
      borderWidth: 1,
      textStyle: {
        color: "#2C3432",
        fontSize: 13,
      },
    },
    xAxis: {
      axisLine: {
        lineStyle: { color: "#D6DBD8" },
      },
      axisLabel: {
        color: "#68716D",
        fontSize: 12,
      },
      splitLine: {
        lineStyle: { color: "#E8EBE8" },
      },
    },
    yAxis: {
      axisLine: {
        lineStyle: { color: "#D6DBD8" },
      },
      axisLabel: {
        color: "#68716D",
        fontSize: 12,
      },
      splitLine: {
        lineStyle: { color: "#E8EBE8" },
      },
    },
  };
}

function buildPolicyOption(step) {
  const entries = Array.isArray(step?.policyTopActions) ? step.policyTopActions : [];
  if (entries.length === 0) {
    return {
      ...chartBaseOption(),
      graphic: {
        type: "text",
        left: "center",
        top: "middle",
        style: {
          text: "No policy data for this step",
          fill: "#68716D",
          fontSize: 13,
        },
      },
    };
  }

  return {
    ...chartBaseOption(),
    tooltip: {
      ...chartBaseOption().tooltip,
      formatter: (params) => {
        const row = Array.isArray(params) ? params[0] : params;
        const extra = entries[row.dataIndex];
        const prob = typeof row.value === "number" ? `${(row.value * 100).toFixed(2)}%` : "--";
        const entropy = formatNumber(step?.entropy, 4);
        return `${extra.label}<br/>Prob: ${prob}<br/>Entropy: ${entropy}`;
      },
    },
    xAxis: {
      ...chartBaseOption().xAxis,
      type: "value",
      max: 1,
      name: "Probability",
      nameTextStyle: { color: "#68716D", fontSize: 12 },
    },
    yAxis: {
      ...chartBaseOption().yAxis,
      type: "category",
      inverse: true,
      data: entries.map((entry) => entry.label),
    },
    series: [
      {
        type: "bar",
        data: entries.map((entry) => entry.probability ?? 0),
        itemStyle: {
          color: "#5F9EA0",
          borderRadius: [0, 8, 8, 0],
        },
        label: {
          show: true,
          position: "right",
          color: "#2C3432",
          formatter: ({ value }) => `${(value * 100).toFixed(1)}%`,
        },
      },
    ],
  };
}

function rewardColor(value) {
  if (value > 0) return "#6E8B74";
  if (value < 0) return "#C75B5B";
  return "#AAB4B1";
}

function buildRewardOption(step) {
  const breakdown = step?.rewardBreakdown ?? {};
  const entries = Object.entries(breakdown);
  if (entries.length === 0) {
    return {
      ...chartBaseOption(),
      graphic: {
        type: "text",
        left: "center",
        top: "middle",
        style: {
          text: "No reward breakdown for this step",
          fill: "#68716D",
          fontSize: 13,
        },
      },
    };
  }

  return {
    ...chartBaseOption(),
    xAxis: {
      ...chartBaseOption().xAxis,
      type: "value",
      name: "Reward",
      nameTextStyle: { color: "#68716D", fontSize: 12 },
    },
    yAxis: {
      ...chartBaseOption().yAxis,
      type: "category",
      inverse: true,
      data: entries.map(([key]) => key),
    },
    series: [
      {
        type: "bar",
        data: entries.map(([, value]) => value ?? 0),
        itemStyle: {
          color: (params) => rewardColor(params.value),
          borderRadius: [0, 8, 8, 0],
        },
        label: {
          show: true,
          position: "right",
          color: "#2C3432",
          formatter: ({ value }) => formatNumber(value, 3),
        },
      },
    ],
  };
}

function stateArrayToChessBoard(state) {
  const pieces = [];
  if (!Array.isArray(state)) return pieces;

  for (let id = 0; id < state.length; id++) {
    if (state[id] === 90) continue;
    const pieceInfo = id2piece(id);
    if (!pieceInfo) continue;
    const [side, pieceName, pieceNum] = pieceInfo;
    const [r, c] = num2rc(state[id]);
    pieces.push({
      id,
      side,
      pieceName,
      pieceNum,
      r,
      c,
    });
  }
  return pieces;
}

function renderChessRollout(container, step) {
  const actionInfo = step?.info?.move ?? {};
  const fromKey = actionInfo.from ? `${actionInfo.from.r},${actionInfo.from.c}` : null;
  const toKey = actionInfo.to ? `${actionInfo.to.r},${actionInfo.to.c}` : null;
  const pieceMap = new Map(
    stateArrayToChessBoard(step?.stateAfter ?? step?.stateBefore).map((piece) => [
      `${piece.r},${piece.c}`,
      piece,
    ])
  );

  const board = document.createElement("div");
  board.className = "episode-viewer__chess-board";

  for (let r = 0; r < 10; r++) {
    for (let c = 0; c < 9; c++) {
      const cell = document.createElement("div");
      const key = `${r},${c}`;
      const piece = pieceMap.get(key);
      cell.className = "episode-viewer__chess-cell";
      if (piece?.side === "red") {
        cell.classList.add("episode-viewer__chess-cell--red");
      }
      if (key === fromKey) {
        cell.classList.add("episode-viewer__chess-cell--from");
      }
      if (key === toKey) {
        cell.classList.add("episode-viewer__chess-cell--to");
      }
      cell.textContent = piece?.pieceName ?? "";
      cell.title = piece
        ? `${piece.side === "red" ? "红" : "黑"}${piece.pieceName}${piece.pieceNum > 0 ? piece.pieceNum : ""}`
        : `(${r + 1}, ${c + 1})`;
      board.appendChild(cell);
    }
  }

  container.innerHTML = "";
  container.appendChild(board);
}

function renderMazeRollout(container, step, trace) {
  const grid = trace?.metadata?.mazeGrid;
  if (!Array.isArray(grid) || grid.length === 0) {
    container.innerHTML = `<p class="episode-viewer__empty">No maze grid data.</p>`;
    return;
  }

  const rows = grid.length;
  const cols = grid[0]?.length ?? 0;
  const board = document.createElement("div");
  board.className = "episode-viewer__maze-board";
  board.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;

  const pathSet = new Set(
    Array.isArray(step?.info?.pathSoFar)
      ? step.info.pathSoFar.map((point) => `${point.x},${point.y}`)
      : []
  );
  const agentPos = step?.info?.positionAfter ?? null;
  const start = trace?.metadata?.start ?? null;
  const goal = trace?.metadata?.goal ?? null;

  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const cell = document.createElement("div");
      cell.className = "episode-viewer__maze-cell";
      if (grid[y][x]) {
        cell.classList.add("episode-viewer__maze-cell--wall");
      } else {
        if (pathSet.has(`${x},${y}`)) {
          cell.classList.add("episode-viewer__maze-cell--path");
        }
        if (start && start.x === x && start.y === y) {
          cell.classList.add("episode-viewer__maze-cell--start");
        }
        if (goal && goal.x === x && goal.y === y) {
          cell.classList.add("episode-viewer__maze-cell--goal");
        }
        if (agentPos && agentPos.x === x && agentPos.y === y) {
          cell.classList.add("episode-viewer__maze-cell--agent");
          const dot = document.createElement("div");
          dot.className = "episode-viewer__maze-agent-dot";
          cell.appendChild(dot);
        }
      }
      board.appendChild(cell);
    }
  }

  container.innerHTML = "";
  container.appendChild(board);
}

function renderRollout(container, trace, step) {
  if (!trace || !step) {
    container.innerHTML = `<p class="episode-viewer__empty">暂无 episode 轨迹数据。</p>`;
    return;
  }
  if (trace.envType === "maze") {
    renderMazeRollout(container, step, trace);
    return;
  }
  renderChessRollout(container, step, trace);
}

function renderSummary(summaryEl, trace, step) {
  const cards = [
    { label: "Episode", value: String((trace?.episode ?? 0) + 1), color: "#C9A34A" },
    { label: "Total Reward", value: formatNumber(trace?.totalReward, 3), color: "#5F9EA0" },
    { label: "Total Steps", value: formatNumber(trace?.totalSteps, 0), color: "#2C3432" },
    { label: "Step Reward", value: formatNumber(step?.reward, 3), color: rewardColor(step?.reward ?? 0) },
  ];

  summaryEl.innerHTML = cards.map((card) => `
    <div class="episode-viewer__summary-card">
      <span class="episode-viewer__summary-label">${card.label}</span>
      <span class="episode-viewer__summary-value" style="color:${card.color}">${card.value}</span>
    </div>
  `).join("");
}

function renderMeta(metaEl, trace, step) {
  const selected = step?.selectedAction ?? {};
  const items = [
    { label: "Selected Action", value: selected.label || step?.actionLabel || "--" },
    { label: "Selected Probability", value: selected.probability === null || selected.probability === undefined ? "--" : formatPercent(selected.probability) },
    { label: "Entropy", value: formatNumber(step?.entropy, 4) },
    { label: "Done", value: step?.done ? "Yes" : "No" },
    { label: "Episode Summary", value: trace?.summary?.outcome ?? trace?.summary?.successLabel ?? "--" },
    { label: "Extra Info", value: step?.info?.detail ?? "--" },
  ];

  metaEl.innerHTML = items.map((item) => `
    <div class="episode-viewer__meta-item">
      <span class="episode-viewer__meta-label">${item.label}</span>
      <span class="episode-viewer__meta-value">${item.value}</span>
    </div>
  `).join("");
}

export function createEpisodeViewer({
  containerId,
  title = "Single Episode RL Viewer",
  subtitle = "Replay one training episode with rollout, policy, and reward details.",
} = {}) {
  const root = document.getElementById(containerId);
  if (!root) {
    return {
      clear() {},
      pushEpisode() {},
      setEpisodes() {},
    };
  }

  injectEpisodeViewerStyles();
  root.classList.add("episode-viewer");
  root.innerHTML = `
    <div class="episode-viewer__shell">
      <div class="episode-viewer__header">
        <div>
          <h2 class="episode-viewer__title">${title}</h2>
          <p class="episode-viewer__subtitle">${subtitle}</p>
        </div>
        <select class="episode-viewer__episode-select" id="${containerId}-episode-select"></select>
      </div>
      <div class="episode-viewer__controls">
        <button class="episode-viewer__button" id="${containerId}-prev">上一步</button>
        <button class="episode-viewer__button" id="${containerId}-play">自动播放</button>
        <button class="episode-viewer__button" id="${containerId}-next">下一步</button>
        <span class="episode-viewer__step-indicator" id="${containerId}-step-indicator">Step -- / --</span>
      </div>
      <div class="episode-viewer__summary" id="${containerId}-summary"></div>
      <div class="episode-viewer__grid">
        <div class="episode-viewer__panel">
          <h3 class="episode-viewer__panel-title">Rollout Layer</h3>
          <div class="episode-viewer__rollout-stage" id="${containerId}-rollout"></div>
          <div class="episode-viewer__meta" id="${containerId}-meta"></div>
        </div>
        <div class="episode-viewer__charts">
          <div class="episode-viewer__panel">
            <h3 class="episode-viewer__panel-title">Policy Layer</h3>
            <div class="episode-viewer__chart episode-viewer__chart--compact" id="${containerId}-policy-chart"></div>
          </div>
          <div class="episode-viewer__panel">
            <h3 class="episode-viewer__panel-title">Reward Layer</h3>
            <div class="episode-viewer__chart episode-viewer__chart--compact" id="${containerId}-reward-chart"></div>
          </div>
        </div>
      </div>
    </div>
  `;

  const episodeSelect = document.getElementById(`${containerId}-episode-select`);
  const stepIndicator = document.getElementById(`${containerId}-step-indicator`);
  const rolloutEl = document.getElementById(`${containerId}-rollout`);
  const summaryEl = document.getElementById(`${containerId}-summary`);
  const metaEl = document.getElementById(`${containerId}-meta`);
  const echarts = ensureEcharts();
  const policyChart = echarts ? echarts.init(document.getElementById(`${containerId}-policy-chart`)) : null;
  const rewardChart = echarts ? echarts.init(document.getElementById(`${containerId}-reward-chart`)) : null;

  let episodes = [];
  let selectedEpisodeIndex = -1;
  let selectedStepIndex = 0;
  let autoplayTimer = null;

  function stopAutoplay() {
    if (autoplayTimer) {
      window.clearInterval(autoplayTimer);
      autoplayTimer = null;
    }
    const playButton = document.getElementById(`${containerId}-play`);
    if (playButton) {
      playButton.textContent = "自动播放";
    }
  }

  function currentTrace() {
    return episodes[selectedEpisodeIndex] ?? null;
  }

  function currentStep() {
    const trace = currentTrace();
    if (!trace || !Array.isArray(trace.steps) || trace.steps.length === 0) return null;
    return trace.steps[Math.max(0, Math.min(selectedStepIndex, trace.steps.length - 1))];
  }

  function syncEpisodeOptions() {
    episodeSelect.innerHTML = episodes.length === 0
      ? `<option value="-1">暂无 episode</option>`
      : episodes
        .map((trace, index) => {
          const label = `Episode ${trace.episode + 1} · reward ${formatNumber(trace.totalReward, 2)} · steps ${trace.totalSteps}`;
          return `<option value="${index}">${label}</option>`;
        })
        .join("");
    episodeSelect.value = String(selectedEpisodeIndex);
  }

  function renderCurrent() {
    const trace = currentTrace();
    const step = currentStep();
    if (!trace || !step) {
      rolloutEl.innerHTML = `<p class="episode-viewer__empty">暂无 episode 轨迹数据。</p>`;
      summaryEl.innerHTML = "";
      metaEl.innerHTML = "";
      stepIndicator.textContent = "Step -- / --";
      policyChart?.setOption(buildPolicyOption(null), true);
      rewardChart?.setOption(buildRewardOption(null), true);
      return;
    }

    stepIndicator.textContent = `Step ${selectedStepIndex + 1} / ${trace.steps.length}`;
    renderSummary(summaryEl, trace, step);
    renderMeta(metaEl, trace, step);
    renderRollout(rolloutEl, trace, step);
    policyChart?.setOption(buildPolicyOption(step), true);
    rewardChart?.setOption(buildRewardOption(step), true);
  }

  function selectEpisode(index) {
    if (episodes.length === 0) {
      selectedEpisodeIndex = -1;
      selectedStepIndex = 0;
      syncEpisodeOptions();
      renderCurrent();
      return;
    }
    selectedEpisodeIndex = Math.max(0, Math.min(index, episodes.length - 1));
    selectedStepIndex = 0;
    syncEpisodeOptions();
    renderCurrent();
  }

  function setStepIndex(nextIndex) {
    const trace = currentTrace();
    if (!trace) return;
    selectedStepIndex = Math.max(0, Math.min(nextIndex, trace.steps.length - 1));
    renderCurrent();
  }

  document.getElementById(`${containerId}-prev`).addEventListener("click", () => {
    stopAutoplay();
    setStepIndex(selectedStepIndex - 1);
  });
  document.getElementById(`${containerId}-next`).addEventListener("click", () => {
    stopAutoplay();
    setStepIndex(selectedStepIndex + 1);
  });
  document.getElementById(`${containerId}-play`).addEventListener("click", () => {
    const trace = currentTrace();
    if (!trace || trace.steps.length === 0) return;
    if (autoplayTimer) {
      stopAutoplay();
      return;
    }
    const playButton = document.getElementById(`${containerId}-play`);
    if (playButton) playButton.textContent = "暂停";
    autoplayTimer = window.setInterval(() => {
      if (!currentTrace()) {
        stopAutoplay();
        return;
      }
      if (selectedStepIndex >= currentTrace().steps.length - 1) {
        stopAutoplay();
        return;
      }
      setStepIndex(selectedStepIndex + 1);
    }, 800);
  });
  episodeSelect.addEventListener("change", (event) => {
    stopAutoplay();
    selectEpisode(Number(event.target.value));
  });

  if (!window.__episodeViewerResizeBound) {
    window.__episodeViewerResizeBound = true;
    window.addEventListener("resize", () => {
      policyChart?.resize();
      rewardChart?.resize();
    });
  }

  renderCurrent();

  return {
    clear() {
      episodes = [];
      selectedEpisodeIndex = -1;
      selectedStepIndex = 0;
      stopAutoplay();
      syncEpisodeOptions();
      renderCurrent();
    },
    setEpisodes(nextEpisodes = []) {
      episodes = nextEpisodes.slice(-VIEWER_MAX_EPISODES);
      selectedEpisodeIndex = episodes.length > 0 ? episodes.length - 1 : -1;
      selectedStepIndex = 0;
      stopAutoplay();
      syncEpisodeOptions();
      renderCurrent();
    },
    pushEpisode(trace) {
      if (!trace || !Array.isArray(trace.steps) || trace.steps.length === 0) return;
      episodes = [...episodes.slice(-(VIEWER_MAX_EPISODES - 1)), trace];
      selectedEpisodeIndex = episodes.length - 1;
      selectedStepIndex = 0;
      stopAutoplay();
      syncEpisodeOptions();
      renderCurrent();
    },
  };
}
