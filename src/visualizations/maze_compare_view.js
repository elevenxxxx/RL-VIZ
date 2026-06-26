import { movingAverage, successRateSeries } from "../metrics/episode_metrics.js";

const COMPARE_THEME = {
  text: "#f5e7a6",
  textMuted: "#d8cd93",
  reward: "#63d8d0",
  rewardSoft: "#93ebe4",
  success: "#7cd8bc",
  danger: "#ff9e7f",
  actor: "#f0d572",
  critic: "#93d3f2",
  tooltip: "rgba(6, 41, 46, 0.96)",
  border: "rgba(108, 214, 203, 0.16)",
  axis: "rgba(108, 214, 203, 0.24)",
  grid: "rgba(108, 214, 203, 0.12)",
};

function drawGrid(ctx, canvas, grid) {
  const rows = grid.length;
  const cols = grid[0].length;
  const cellSize = Math.min(canvas.width / cols, canvas.height / rows);
  const offsetX = (canvas.width - cols * cellSize) / 2;
  const offsetY = (canvas.height - rows * cellSize) / 2;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#F7F8F6";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      ctx.fillStyle = grid[y][x] === 1 ? "#505755" : "#F3F4F2";
      ctx.fillRect(offsetX + x * cellSize, offsetY + y * cellSize, cellSize - 1, cellSize - 1);
    }
  }

  return { cellSize, offsetX, offsetY };
}

export function drawMazeEpisode(canvas, grid, episode, options = {}) {
  if (!canvas || !Array.isArray(grid) || grid.length === 0) return;
  const ctx = canvas.getContext("2d");
  const layout = drawGrid(ctx, canvas, grid);
  const path = Array.isArray(episode?.path) ? episode.path : [];
  const start = options.start ?? { x: 1, y: 1 };
  const goal = options.goal ?? { x: grid[0].length - 2, y: grid.length - 2 };

  const project = (point) => ({
    x: layout.offsetX + point.x * layout.cellSize + layout.cellSize / 2,
    y: layout.offsetY + point.y * layout.cellSize + layout.cellSize / 2,
  });

  ctx.fillStyle = "#6E8B74";
  ctx.fillRect(
    layout.offsetX + start.x * layout.cellSize,
    layout.offsetY + start.y * layout.cellSize,
    layout.cellSize - 1,
    layout.cellSize - 1
  );

  ctx.fillStyle = "#C75B5B";
  ctx.fillRect(
    layout.offsetX + goal.x * layout.cellSize,
    layout.offsetY + goal.y * layout.cellSize,
    layout.cellSize - 1,
    layout.cellSize - 1
  );

  if (path.length > 1) {
    ctx.strokeStyle = options.strokeStyle ?? "#5F9EA0";
    ctx.lineWidth = Math.max(2, layout.cellSize * 0.18);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    const first = project(path[0]);
    ctx.moveTo(first.x, first.y);
    for (let index = 1; index < path.length; index++) {
      const point = project(path[index]);
      ctx.lineTo(point.x, point.y);
    }
    ctx.stroke();
  }

  if (path.length > 0) {
    const last = project(path[path.length - 1]);
    ctx.beginPath();
    ctx.fillStyle = options.agentColor ?? "#D2A24C";
    ctx.arc(last.x, last.y, Math.max(4, layout.cellSize * 0.22), 0, Math.PI * 2);
    ctx.fill();
  }
}

export function populateEpisodeSelect(selectEl, episodes = [], selectedIndex = 0) {
  if (!selectEl) return;
  if (!Array.isArray(episodes) || episodes.length === 0) {
    selectEl.innerHTML = `<option value="-1">暂无训练轮次</option>`;
    selectEl.value = "-1";
    return;
  }

  selectEl.innerHTML = episodes.map((episode, index) => `
    <option value="${index}">第 ${episode.episodeIndex + 1} 轮 · 奖励 ${episode.reward.toFixed(2)} · 步数 ${episode.steps}</option>
  `).join("");
  selectEl.value = String(Math.max(0, Math.min(selectedIndex, episodes.length - 1)));
}

export function renderEpisodeMeta(container, label, episode) {
  if (!container) return;
  if (!episode) {
    container.innerHTML = `<p class="compare-meta__empty">当前还没有可查看的训练轮次。</p>`;
    return;
  }
  container.innerHTML = `
    <div class="compare-meta__row"><span>${label}</span><strong>第 ${episode.episodeIndex + 1} 轮</strong></div>
    <div class="compare-meta__row"><span>奖励</span><strong>${episode.reward.toFixed(3)}</strong></div>
    <div class="compare-meta__row"><span>步数</span><strong>${episode.steps}</strong></div>
    <div class="compare-meta__row"><span>成功</span><strong>${episode.success ? "是" : "否"}</strong></div>
  `;
}

function buildChartBase(title) {
  return {
    backgroundColor: "transparent",
    animationDuration: 280,
    title: {
      text: title,
      left: 16,
      top: 12,
      textStyle: {
        color: COMPARE_THEME.text,
        fontSize: 16,
        fontWeight: 700,
      },
    },
    legend: {
      top: 14,
      right: 18,
      textStyle: {
        color: COMPARE_THEME.textMuted,
        fontSize: 12,
      },
    },
    tooltip: {
      trigger: "axis",
      backgroundColor: COMPARE_THEME.tooltip,
      borderColor: COMPARE_THEME.border,
      borderWidth: 1,
      textStyle: {
        color: COMPARE_THEME.text,
        fontSize: 13,
      },
    },
    grid: {
      left: "8%",
      right: "5%",
      bottom: "12%",
      top: "22%",
      containLabel: true,
    },
    xAxis: {
      type: "category",
      axisLine: {
        lineStyle: { color: COMPARE_THEME.axis },
      },
      axisLabel: {
        color: COMPARE_THEME.textMuted,
        fontSize: 12,
      },
    },
    yAxis: {
      type: "value",
      axisLine: {
        show: true,
        lineStyle: { color: COMPARE_THEME.axis },
      },
      axisLabel: {
        color: COMPARE_THEME.textMuted,
        fontSize: 12,
      },
      splitLine: {
        lineStyle: { color: COMPARE_THEME.grid },
      },
    },
  };
}

function historySeries(history = [], key) {
  return history.map((episode) => episode[key]);
}

export function renderComparisonCharts(charts, histories = {}) {
  const qHistory = histories.q_learning ?? [];
  const ppoHistory = histories.ppo ?? [];
  const maxEpisodes = Math.max(qHistory.length, ppoHistory.length);
  const xAxis = Array.from({ length: maxEpisodes }, (_, index) => `${index + 1}`);

  charts.stepsChart?.setOption({
    ...buildChartBase("步数对比"),
    xAxis: {
      ...buildChartBase("步数对比").xAxis,
      data: xAxis,
    },
    series: [
      {
        name: "Q-learning 步数",
        type: "line",
        smooth: true,
        symbol: "none",
        lineStyle: { width: 2, color: COMPARE_THEME.reward },
        data: historySeries(qHistory, "steps"),
      },
      {
        name: "PPO 步数",
        type: "line",
        smooth: true,
        symbol: "none",
        lineStyle: { width: 2, color: COMPARE_THEME.critic },
        data: historySeries(ppoHistory, "steps"),
      },
      {
        name: "Q-learning 步数 MA(10)",
        type: "line",
        smooth: true,
        symbol: "none",
        lineStyle: { width: 2, type: "dashed", color: COMPARE_THEME.rewardSoft },
        data: movingAverage(historySeries(qHistory, "steps"), 10),
      },
      {
        name: "PPO 步数 MA(10)",
        type: "line",
        smooth: true,
        symbol: "none",
        lineStyle: { width: 2, type: "dashed", color: "#c3e7fb" },
        data: movingAverage(historySeries(ppoHistory, "steps"), 10),
      },
    ],
  }, true);

  charts.successChart?.setOption({
    ...buildChartBase("成功率对比"),
    xAxis: {
      ...buildChartBase("成功率对比").xAxis,
      data: xAxis,
    },
    yAxis: {
      ...buildChartBase("成功率对比").yAxis,
      max: 1,
    },
    series: [
      {
        name: "Q-learning 成功率",
        type: "line",
        smooth: true,
        symbol: "none",
        lineStyle: { width: 2, color: COMPARE_THEME.success },
        data: successRateSeries(qHistory, 10),
      },
      {
        name: "PPO 成功率",
        type: "line",
        smooth: true,
        symbol: "none",
        lineStyle: { width: 2, color: COMPARE_THEME.actor },
        data: successRateSeries(ppoHistory, 10),
      },
    ],
  }, true);

  charts.rewardChart?.setOption({
    ...buildChartBase("奖励对比"),
    xAxis: {
      ...buildChartBase("奖励对比").xAxis,
      data: xAxis,
    },
    series: [
      {
        name: "Q-learning 奖励",
        type: "line",
        smooth: true,
        symbol: "none",
        lineStyle: { width: 2, color: COMPARE_THEME.reward },
        data: historySeries(qHistory, "reward"),
      },
      {
        name: "PPO 奖励",
        type: "line",
        smooth: true,
        symbol: "none",
        lineStyle: { width: 2, color: COMPARE_THEME.danger },
        data: historySeries(ppoHistory, "reward"),
      },
    ],
  }, true);
}
