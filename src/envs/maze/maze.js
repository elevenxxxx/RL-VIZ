const p5Module = await import("https://cdn.jsdelivr.net/npm/p5@1.9.0/+esm");
const p5 = p5Module.default;
const { drawlineGraph } = await import("../chess/graph.js");
const { createTrainingRecord } = await import("../shared/trainingDiagnostics.js");
const { createEpisodeTrace, createEpisodeStepRecord } = await import("../shared/episodeTrace.js");
const { createEpisodeViewer } = await import("../shared/episodeViewer.js");
const { setStepRender } = await import("./steps.js");

const MAZE_ACTION_LABELS = ["上", "下", "左", "右"];
const MAZE_EPISODE_TRACE_INTERVAL = 5;
const mazeEpisodeViewer = createEpisodeViewer({
  containerId: "episode-viewer-root-maze",
  title: "Single Episode RL Visualization",
  subtitle: "Replay one Maze Q-learning episode with path, policy scores, and reward decomposition.",
  maxEpisodes: 50,
});
// export const config = {
//   cols: 25,// 必须奇数
//   rows: 25,
//   cellSize: 30,

//   renderTrain: true,
//   renderSpeed: 10,
//   stepMode: false,
//   stop: false,

//   trainEpisodes: 300,
//   maxSteps: 400,
//   currentEpisode: 0,

//   alpha: 0.1,//学习率
//   gamma: 0.95,//折扣因子 对未来奖励的影响程度
//   epsilon: 0.1,//探索率
// };
const initConfig = {
  cols: 25,// 必须奇数
  rows: 25,
  cellSize: 35,

  renderTrain: true,
  renderSpeed: 10,
  stepMode: true,
  stop: false,

  trainEpisodes: 300,
  maxSteps: 400,
  currentEpisode: 0,

  EpRenderInterval: 20,

  alpha: 0.1,//学习率
  gamma: 0.95,//折扣因子 对未来奖励的影响程度
  epsilon: 0.1,//探索率

  now_train_step: 0,
  now_test_step: 0,

  draw_maze: true,
  draw_heatmap: true,
  draw_paths: true,
  draw_current_path: true,
  draw_policy: true,
  draw_agent: true,
  draw_goal: true,
  draw_tooltip: true,

  high_render: false,
}

export const config = new Proxy(structuredClone(initConfig), {
  set(target, key, value) {
    target[key] = value;

    // 触发UI更新
    if (key === "currentEpisode") {
      notify("currentEpisode", value);
    }
    if (key === "stepMode") {
      notify("stepMode", value);
    }

    return true;
  }
});
const init_Agentinfo = {
  Qagent_s: rc2stateId(1, 1),
  Qagent_a: -1,
  Qagent_r: 0,
  Qagent_qs: rc2stateId(1, 1),
};
export const Agentinfo = new Proxy(structuredClone(init_Agentinfo), {
  set(target, key, value) {
    target[key] = value;

    // 触发UI更新
    if (key === "Qagent_s" || key === "Qagent_a" || key === "Qagent_r" || key === "Qagent_qs") {
      notify(key, value);
    }

    return true;
  }
});

const watchers = {};

export function watch(key, fn) {
  if (!watchers[key]) watchers[key] = [];
  watchers[key].push(fn);
}

function notify(key, value) {
  if (watchers[key]) {
    watchers[key].forEach(fn => fn(value));
  }
}
let grid = [];
let agent = {};
let start;
let goal;
let Q = {};//存储Q值
let QTrace = {};//存储附加信息
let visitCount = {};//存储访问次数
let Qactive = {}

let records = [];//存储训练信息

let hoverCell = null;
let hoverStartTime = 0;
let hoverDelay = 400; // ms

let isTest = false

let episodePaths = [];
let currentPath = [];
let edgeWeight = {};
const HEATMAP_START = [224, 231, 255];
const HEATMAP_END = [91, 33, 182];
const TRAJECTORY_COLOR = [235, 96, 96];

function lerpRgb(start, end, t) {
  return [
    start[0] + (end[0] - start[0]) * t,
    start[1] + (end[1] - start[1]) * t,
    start[2] + (end[2] - start[2]) * t,
  ];
}

function drawMaze(p) {
  for (let y = 0; y < config.rows; y++) {
    for (let x = 0; x < config.cols; x++) {

      if (grid[y][x].wall) {
        p.fill(30, 40, 60);
      } else {
        p.fill(240);
      }

      p.stroke(200);
      p.rect(
        x * config.cellSize,
        y * config.cellSize,
        config.cellSize,
        config.cellSize,
        4
      );
    }
  }
}
function drawHeatmap(p) {
  let maxVisit = 1;

  for (let k in visitCount) {
    maxVisit = Math.max(maxVisit, visitCount[k]);
  }

  p.noStroke();

  for (let y = 0; y < config.rows; y++) {
    for (let x = 0; x < config.cols; x++) {

      const cell = grid[y][x];

      if (cell.wall) continue;

      const s = rc2stateId(x, y);
      const v = visitCount[s] || 0;

      if (v === 0) continue;

      const intensity = v / maxVisit;
      const [r, g, b] = lerpRgb(HEATMAP_START, HEATMAP_END, intensity);
      const alpha = 120 * intensity;

      p.fill(r, g, b, alpha);
      p.rect(
        x * config.cellSize,
        y * config.cellSize,
        config.cellSize,
        config.cellSize
      );
    }
  }
}
function drawTooltip(p) {
  if (!hoverCell) return;

  const now = performance.now();

  if (now - hoverStartTime < hoverDelay) return;

  const tooltip = document.getElementById("maze_tooltip");

  const [x, y] = hoverCell.split(",").map(Number);

  const s = rc2stateId(x, y);
  const q = getQ(s);
  const trace = getQTrace ? getQTrace(s) : null;

  const actions = ["↑", "↓", "←", "→"];

  let html = `<b>${s}</b><br/>`;
  html += `
    <div style="
        font-size: 20px;
        font-family: 'Times New Roman', serif;
    ">
        Q(s,a) += α[r + γ max_a' Q(s',a') − Q(s,a)]
    </div>
`;

  for (let i = 0; i < 4; i++) {
    html += `<div style="font-size:18px;">${i}${actions[i]}: ${q[i].toFixed(3)}</div>`;

    let Qcolor = "#5F9EA0";
    if (s in Qactive && Qactive[s] == i) {
      Qcolor = "#6E8B74";
    }

    if (trace && trace[i]) {
      html += `<div style="font-size:18px;color:${Qcolor}">
        ${trace[i].values}
      </div>`;
    }
  }

  tooltip.innerHTML = html;
  tooltip.style.left = p.mouseX + 130 + "px";
  tooltip.style.top = p.mouseY + 12 + "px";
  tooltip.style.display = "block";
}
function drawGoal(p) {
  p.fill(199, 91, 91);
  p.rect(
    goal.x * config.cellSize,
    goal.y * config.cellSize,
    config.cellSize,
    config.cellSize,
    6
  );
}

function drawAgent(p) {
  const cx = agent.x * config.cellSize + config.cellSize / 2;
  const cy = agent.y * config.cellSize + config.cellSize / 2;
  const size = config.cellSize * 0.8;

  let dir = agent.dir;

  p.fill(95, 158, 160);
  p.noStroke();

  // ❗初始状态 → 圆
  if (!dir) {
    p.circle(cx, cy, size);
    return;
  }

  p.beginShape();

  if (dir === "right") {
    p.vertex(cx + size / 2, cy);
    p.vertex(cx - size / 2, cy - size / 3);
    p.vertex(cx - size / 2, cy + size / 3);
  }

  else if (dir === "left") {
    p.vertex(cx - size / 2, cy);
    p.vertex(cx + size / 2, cy - size / 3);
    p.vertex(cx + size / 2, cy + size / 3);
  }

  else if (dir === "down") {
    p.vertex(cx, cy + size / 2);
    p.vertex(cx - size / 3, cy - size / 2);
    p.vertex(cx + size / 3, cy - size / 2);
  }

  else if (dir === "up") {
    p.vertex(cx, cy - size / 2);
    p.vertex(cx - size / 3, cy + size / 2);
    p.vertex(cx + size / 3, cy + size / 2);
  }

  // agent.dir = dir;

  p.endShape(p.CLOSE);
}
function drawPolicy(p) {
  p.textAlign(p.CENTER, p.CENTER);
  p.textSize(22);
  p.fill(104, 113, 109);

  for (let y = 0; y < config.rows; y++) {
    for (let x = 0; x < config.cols; x++) {
      if (grid[y][x].wall) continue;

      const s = rc2stateId(x, y);
      const q = getQ(s);

      const maxQ = Math.max(...q);
      const bestActions = [];

      for (let i = 0; i < 4; i++) {
        if (q[i] === maxQ) bestActions.push(i);
      }

      let symbol = "";

      if (bestActions.length === 4) {
        symbol = "·";
      } else {
        // 如果多个最大值，随机一个（或取第一个）
        const a = bestActions[0];

        if (a === 0) symbol = "↑";
        if (a === 1) symbol = "↓";
        if (a === 2) symbol = "←";
        if (a === 3) symbol = "→";
      }

      p.fill(104, 113, 109);
      p.text(
        symbol,
        x * config.cellSize + config.cellSize / 2,
        y * config.cellSize + config.cellSize / 2
      );
    }
  }
}
function drawPaths_nomi(p) {
  p.noFill(); for (let i = 0; i < episodePaths.length; i++) {
    const path = episodePaths[i];
    for (let j = 1; j < path.length; j++) {
      const a = path[j - 1];
      const b = path[j];

      const key = `${a.x},${a.y}->${b.x},${b.y}`;
      const w = edgeWeight[key] || 1;

      const t = j / path.length;
      const thickness = (1 - t) * 6 + w * 0.5;

      const alpha = Math.min(200, 30 + w * 20);

      p.stroke(TRAJECTORY_COLOR[0], TRAJECTORY_COLOR[1], TRAJECTORY_COLOR[2], alpha);
      p.strokeWeight(thickness);

      p.line(
        a.x * config.cellSize + config.cellSize / 2,
        a.y * config.cellSize + config.cellSize / 2,
        b.x * config.cellSize + config.cellSize / 2,
        b.y * config.cellSize + config.cellSize / 2
      );
    }
  }
  p.strokeWeight(1);
  p.stroke(232, 235, 232);
}
function drawPaths(p) {
  p.noFill();
  p.push();

  for (let i = 0; i < episodePaths.length; i++) {
    const path = episodePaths[i];
    const pathScale = 0.7 + i / episodePaths.length * 0.3;

    for (let j = 1; j < path.length; j++) {
      const a = path[j - 1];
      const b = path[j];

      const key = `${a.x},${a.y}->${b.x},${b.y}`;
      const w = edgeWeight[key] || 1;

      const t = j / path.length;
      const curveT = Math.pow(1 - t, 1.6);

      const baseThickness = Math.max(8, episodePaths.length * 0.6);
      const thickness = (baseThickness * curveT + w * 0.2) * pathScale;

      const alphaBase = 160;
      const alpha = Math.min(alphaBase, (100 + w * 5) * pathScale);

      p.stroke(TRAJECTORY_COLOR[0], TRAJECTORY_COLOR[1], TRAJECTORY_COLOR[2], alpha);
      p.strokeWeight(thickness);

      const ax = a.x * config.cellSize + config.cellSize / 2;
      const ay = a.y * config.cellSize + config.cellSize / 2;
      const bx = b.x * config.cellSize + config.cellSize / 2;
      const by = b.y * config.cellSize + config.cellSize / 2;

      const mi = 7;

      for (let s = 0; s < mi; s++) {
        const t0 = s / mi;
        const t1 = (s + 1) / mi;

        const curveLocal = Math.pow(1 - t0, 1.6);

        const x1 = p.lerp(ax, bx, t0);
        const y1 = p.lerp(ay, by, t0);
        const x2 = p.lerp(ax, bx, t1);
        const y2 = p.lerp(ay, by, t1);

        const localThickness = thickness * curveLocal;
        const localAlpha = alpha * (0.6 + 0.4 * curveLocal);

        p.stroke(TRAJECTORY_COLOR[0], TRAJECTORY_COLOR[1], TRAJECTORY_COLOR[2], localAlpha);
        p.strokeWeight(localThickness);

        p.line(x1, y1, x2, y2);
      }
    }
  }

  p.strokeWeight(1);
  p.stroke(232, 235, 232);
  p.pop();
}
function drawCurrentPath_nomi(p) {
  if (currentPath.length < 2) return;

  p.push();
  p.stroke(TRAJECTORY_COLOR[0], TRAJECTORY_COLOR[1], TRAJECTORY_COLOR[2], 180);
  p.strokeWeight(3);
  p.noFill();

  for (let i = 1; i < currentPath.length; i++) {
    const a = currentPath[i - 1];
    const b = currentPath[i];
    const key = `${a.x},${a.y}->${b.x},${b.y}`;
    const w = edgeWeight[key] || 1;

    p.line(
      a.x * config.cellSize + config.cellSize / 2,
      a.y * config.cellSize + config.cellSize / 2,
      b.x * config.cellSize + config.cellSize / 2,
      b.y * config.cellSize + config.cellSize / 2
    );
  }

  p.pop();
}
function drawCurrentPath(p) {
  if (currentPath.length < 2) return;

  p.push();

  for (let i = 1; i < currentPath.length; i++) {
    const a = currentPath[i - 1];
    const b = currentPath[i];

    const ax = a.x * config.cellSize + config.cellSize / 2;
    const ay = a.y * config.cellSize + config.cellSize / 2;
    const bx = b.x * config.cellSize + config.cellSize / 2;
    const by = b.y * config.cellSize + config.cellSize / 2;

    const mi = 7;

    for (let s = 0; s < mi; s++) {
      let t = s / mi;

      const curveT = Math.pow(1 - t, 1.8);

      const x = p.lerp(ax, bx, t);
      const y = p.lerp(ay, by, t);

      const base = 10;
      const w = base * curveT;

      const key = `${a.x},${a.y}->${b.x},${b.y}`;
      const ew = edgeWeight?.[key] || 1;

      p.stroke(TRAJECTORY_COLOR[0], TRAJECTORY_COLOR[1], TRAJECTORY_COLOR[2], 120 + ew * 10);
      p.strokeWeight(w);

      const nx = p.lerp(ax, bx, (s + 1) / mi);
      const ny = p.lerp(ay, by, (s + 1) / mi);

      p.line(x, y, nx, ny);
    }
  }

  p.pop();
}
function initQTable() {
  const el = document.getElementById("QTable");

  const actions = ["↑", "↓", "←", "→"];

  let html = `
  <tr>
    <th>a\\Q</th>
    <th id="q_sa">Q(s,a)</th>
    <th id="q_s2a2">Q(s',a')</th>
  </tr>
  `;

  for (let a = 0; a < 4; a++) {
    html += `
      <tr>
        <th>${a}(${actions[a]})</th>
        <td id="q_sa_${a}">0</td>
        <td id="q_s2a_${a}">0</td>
      </tr>
    `;
  }

  el.innerHTML = html;
}
function addLoops(g, cols, rows, rate = 0.3) {
  for (let y = 1; y < rows - 1; y++) {
    for (let x = 1; x < cols - 1; x++) {
      if (Math.random() < rate && g[y][x].wall) {
        g[y][x].wall = false;
      }
    }
  }
}
function createMazeDFS(cols, rows) {
  const g = [];

  for (let y = 0; y < rows; y++) {
    const row = [];
    for (let x = 0; x < cols; x++) {
      row.push({
        x,
        y,
        wall: true,
        visited: false,
      });
    }
    g.push(row);
  }

  const stack = [];

  const start = g[1][1];
  start.wall = false;
  start.visited = true;
  stack.push(start);

  const dirs = [
    [0, -2],
    [2, 0],
    [0, 2],
    [-2, 0],
  ];

  function neighbors(cell) {
    const res = [];

    for (let [dx, dy] of dirs) {
      const nx = cell.x + dx;
      const ny = cell.y + dy;

      if (
        nx > 0 &&
        ny > 0 &&
        nx < cols &&
        ny < rows &&
        !g[ny][nx].visited
      ) {
        res.push(g[ny][nx]);
      }
    }

    return res;
  }

  while (stack.length > 0) {
    const cur = stack[stack.length - 1];
    const nbs = neighbors(cur);

    if (nbs.length === 0) {
      stack.pop();
      continue;
    }

    const next = nbs[Math.floor(Math.random() * nbs.length)];

    const wx = (cur.x + next.x) / 2;
    const wy = (cur.y + next.y) / 2;

    g[wy][wx].wall = false;
    next.wall = false;
    next.visited = true;

    stack.push(next);
  }
  addLoops(g, cols, rows);
  return g;
}
function addEdge(a, b, weight = 1) {
  const key = `${a.x},${a.y}->${b.x},${b.y}`;
  let w = (edgeWeight[key] || 0) + weight;
  if (w > 0)
    edgeWeight[key] = w;
  else
    delete edgeWeight[key];
}


function rc2stateId(x, y) {
  return `${x},${y}`;
}
function stateId2rc(stateId) {
  //console.log("stateId", stateId);
  const [x, y] = stateId.split(",");
  return { x, y };
}
function getQTrace(s) {
  if (!QTrace[s]) {
    QTrace[s] = [
      null, null, null, null
    ];
  }
  return QTrace[s];
}
function getVisit(s) {
  if (!visitCount[s]) visitCount[s] = 0;
  return visitCount[s];
}
function getQ(s) {
  if (!Q[s]) Q[s] = [0, 0, 0, 0];
  return Q[s];
}

function getVisitedStateCount() {
  return Object.keys(visitCount).length;
}

function getGlobalMaxQ() {
  let maxQ = Number.NEGATIVE_INFINITY;

  for (const values of Object.values(Q)) {
    for (const value of values) {
      if (value > maxQ) {
        maxQ = value;
      }
    }
  }

  return Number.isFinite(maxQ) ? maxQ : 0;
}

function getMazePolicySnapshot(s, selectedAction) {
  const q = getQ(s);
  const maxQ = Math.max(...q);
  const bestActions = [];
  for (let i = 0; i < q.length; i++) {
    if (q[i] === maxQ) bestActions.push(i);
  }

  const exploreProb = config.epsilon / q.length;
  const exploitProb = bestActions.length > 0 ? (1 - config.epsilon) / bestActions.length : 0;
  const entries = q.map((value, actionId) => {
    const probability = exploreProb + (bestActions.includes(actionId) ? exploitProb : 0);
    return {
      actionId,
      label: `${MAZE_ACTION_LABELS[actionId]} · Q=${value.toFixed(3)}`,
      probability,
      qValue: value,
    };
  }).sort((lhs, rhs) => rhs.probability - lhs.probability || rhs.qValue - lhs.qValue);

  const entropy = entries.reduce((sum, entry) => {
    if (entry.probability <= 0) return sum;
    return sum - entry.probability * Math.log(entry.probability + 1e-8);
  }, 0);

  return {
    entropy,
    topActions: entries.slice(0, 5),
    selectedAction: {
      actionId: selectedAction,
      label: MAZE_ACTION_LABELS[selectedAction] ?? `a=${selectedAction}`,
      probability: entries.find((entry) => entry.actionId === selectedAction)?.probability ?? null,
    },
  };
}
//  ε-greedy
function chooseAction(s) {
  const q = getQ(s);

  if (!isTest && Math.random() < config.epsilon) {
    return Math.floor(Math.random() * 4);
  }

  return q.indexOf(Math.max(...q));
}

function updateQ(s, a, r, s2) {
  const q = getQ(s);
  const q2 = getQ(s2);

  const maxNext = Math.max(...q2);

  const oldQ = q[a];

  const tdTarget = r + config.gamma * maxNext;
  const tdError = tdTarget - oldQ;

  const newQ = oldQ + config.alpha * tdError;

  q[a] = newQ;

  const trace = getQTrace(s);

  trace[a] = {
    values: `
= ${oldQ.toFixed(3)} + ${config.alpha} × (${r.toFixed(3)} + ${config.gamma} × ${maxNext.toFixed(3)} - ${oldQ.toFixed(3)})
`,
    result: newQ.toFixed(3)
  };

  Qactive[s] = a;

  let qsrc = stateId2rc(s);
  let qs2rc = stateId2rc(s2);

  document.getElementById(`q_sa`).textContent = `Q(${qsrc.x}-${qsrc.y},a)`;
  document.getElementById(`q_s2a2`).textContent = `Q(${qs2rc.x}-${qs2rc.y},a')`;

  for (let i = 0; i < 4; i++) {
    let Qcolor1 = "#68716D";
    if (s in Qactive && Qactive[s] == i) {
      Qcolor1 = "#6E8B74";
    }

    document.getElementById(`q_sa_${i}`).textContent = q[i].toFixed(3);
    document.getElementById(`q_sa_${i}`).style.color = Qcolor1;

    const el2 = document.getElementById(`q_s2a_${i}`);
    el2.textContent = q2[i].toFixed(3);

    if (q2[i] === maxNext) {
      el2.style.color = "#eb5454";   // 高亮色
      el2.style.fontWeight = "bold";
    } else {
      el2.style.color = "#e6e6e6";
      el2.style.fontWeight = "normal";
    }
  }

  return {
    qDelta: Math.abs(newQ - oldQ),
    newQ,
    maxNext,
  };
}

const sketch = (p) => {
  p.setup = () => {
    const canvas = p.createCanvas(
      config.cols * config.cellSize,
      config.rows * config.cellSize
    );
    canvas.parent("game");

    fresh();
  };

  p.draw = () => {
    p.background(243, 244, 242);

    if (config.draw_maze) drawMaze(p);
    if (config.draw_heatmap) drawHeatmap(p);

    if (config.draw_paths) {
      if (config.high_render) drawPaths(p);
      else drawPaths_nomi(p);
    }
    if (config.draw_current_path) {
      if (config.high_render) drawCurrentPath(p);
      else drawCurrentPath_nomi(p);
    }
    if (config.draw_policy) drawPolicy(p);
    if (config.draw_agent) drawAgent(p);
    if (config.draw_goal) drawGoal(p);

    if (config.draw_tooltip) drawTooltip(p);
  };

  const tooltip = document.getElementById("maze_tooltip");


  p.mouseMoved = () => {
    const x = Math.floor(p.mouseX / config.cellSize);
    const y = Math.floor(p.mouseY / config.cellSize);

    if (
      x < 0 || y < 0 ||
      x >= config.cols ||
      y >= config.rows
    ) {
      hoverCell = null;
      return;
    }

    const newCell = `${x},${y}`;

    // ====== 关键：如果换格子，重置计时 ======
    if (hoverCell !== newCell) {
      hoverCell = newCell;
      hoverStartTime = performance.now();
      document.getElementById("maze_tooltip").style.display = "none";
    }
  };
  return p;
};
const stepRenderV = {
  a: -1,
  s2: -1,
  reward: -1,
  d: 0,
}
function ResetAgentinfo() {
  const new_Agentinfo = structuredClone(init_Agentinfo);
  Object.keys(new_Agentinfo).forEach(k => {
    Agentinfo[k] = new_Agentinfo[k];
  });
  config.now_train_step = 0;
  config.now_test_step = 0;
}
export async function stepRender() {
  const RunStepFunctions = {
    0: RunStep0,
    1: RunStep1,
    2: RunStep2
  };
  if (config.stepMode) {
    RunStepFunctions[config.now_train_step]();
    await setStepRender("train", config.now_train_step);
    config.now_train_step++
  }
  else {
    config.isTest = true;
    RunStepFunctions[config.now_test_step]();
    config.isTest = false;
    await setStepRender("test", config.now_test_step);
    config.now_test_step++
  }

  if (config.now_train_step > 2) {
    //ResetAgentinfo();
    config.now_train_step = 0;
  }
  if (config.now_test_step > 1) {
    //ResetAgentinfo();
    config.now_test_step = 0;
  }

  function RunStep0() {
    console.log("RunStep0", Agentinfo);
    const s = rc2stateId(agent.x, agent.y);
    Agentinfo.Qagent_s = s;
    Agentinfo.Qagent_a = chooseAction(s);
  }
  function RunStep1() {
    console.log("RunStep1", Agentinfo);
    const res = step(Agentinfo.Qagent_a);
    Agentinfo.Qagent_qs = res.s2;
    Agentinfo.Qagent_r = res.reward;
    if (res.done) {
      alert("到达终点");
    }

  }
  function RunStep2() {
    console.log("RunStep2", Agentinfo);
    updateQ(Agentinfo.Qagent_s, Agentinfo.Qagent_a, Agentinfo.Qagent_r, Agentinfo.Qagent_qs);
  }
}

export function SingleStep() {
  const s = rc2stateId(agent.x, agent.y);
  const a = chooseAction(s);
  const { s2, reward, done: d } = step(a);

  Agentinfo.Qagent_s = s;
  Agentinfo.Qagent_a = a;
  Agentinfo.Qagent_r = reward;
  Agentinfo.Qagent_qs = s2;

  if (config.stepMode) {
    updateQ(s, a, reward, s2);
  }

  if (d) {
    alert("end");
  }
}
function step(action) {
  if (
    action === undefined ||
    action === null ||
    typeof action !== "number" ||
    action < 0 ||
    action >= 4
  ) {
    console.warn("invalid action:", action);
    action = Math.floor(Math.random() * 4);
  }

  const dirs = [
    [0, -1],
    [0, 1],
    [-1, 0],
    [1, 0],
  ];

  const [dx, dy] = dirs[action];
  const stateBefore = rc2stateId(agent.x, agent.y);
  const positionBefore = { x: agent.x, y: agent.y };
  switch (action) {
    case 0:
      agent.dir = "up";
      break;
    case 1:
      agent.dir = "down";
      break;
    case 2:
      agent.dir = "left";
      break;
    case 3:
      agent.dir = "right";
      break;
    default:
      agent.dir = null;
  }
  const nx = agent.x + dx;
  const ny = agent.y + dy;

  let reward = -0.01;
  let done = false;
  const rewardBreakdown = {
    stepReward: -0.01,
    wallPenalty: 0,
    backtrackPenalty: 0,
    goalReward: 0,
  };

  let nextX = agent.x;
  let nextY = agent.y;

  // ====== 撞墙检测 ======
  const hitWall =
    nx < 0 ||
    ny < 0 ||
    nx >= config.cols ||
    ny >= config.rows ||
    grid[ny][nx].wall;

  if (hitWall) {
    reward = -0.1;
    rewardBreakdown.stepReward = 0;
    rewardBreakdown.wallPenalty = -0.1;
  } else {
    nextX = nx;
    nextY = ny;
  }

  const isBacktrack =
    nextX === agent.prevX && nextY === agent.prevY;

  if (isBacktrack && !(nextX === agent.x && nextY === agent.y)) {
    reward -= 0.05;
    rewardBreakdown.backtrackPenalty = -0.05;
  }

  agent.prevX = agent.x;
  agent.prevY = agent.y;

  agent.tx = nextX;
  agent.ty = nextY;

  agent.x = agent.tx;
  agent.y = agent.ty;

  // ====== 到达终点 ======
  if (agent.x === goal.x && agent.y === goal.y) {
    reward = 20;
    rewardBreakdown.stepReward = 0;
    rewardBreakdown.wallPenalty = 0;
    rewardBreakdown.backtrackPenalty = 0;
    rewardBreakdown.goalReward = 20;
    done = true;
  }

  const s = stateBefore;
  const s2 = rc2stateId(agent.tx, agent.ty);


  currentPath.push({
    x: agent.x,
    y: agent.y
  });
  if (!isTest) {
    visitCount[s2] = (visitCount[s2] || 0) + 1;
  }
  return {
    s,
    s2,
    reward,
    done,
    rewardBreakdown,
    hitWall,
    isBacktrack,
    positionBefore,
    positionAfter: { x: agent.x, y: agent.y },
  };
}

async function trainEpisode(episodeId = 0) {
  reset();

  let s = rc2stateId(agent.x, agent.y);
  let done = false;
  let stepCount = 0;
  let totalReward = 0;
  let qDeltaSum = 0;
  const episodeSteps = [];

  while (!done && stepCount < config.maxSteps && !config.stop) {

    const a = chooseAction(s);
    const policySnapshot = getMazePolicySnapshot(s, a);
    const { s2, reward, done: d, rewardBreakdown, hitWall, isBacktrack, positionBefore, positionAfter } = step(a);
    totalReward += reward;
    const updateInfo = updateQ(s, a, reward, s2);
    qDeltaSum += updateInfo.qDelta;

    Agentinfo.Qagent_s = s;
    Agentinfo.Qagent_a = a;
    Agentinfo.Qagent_r = reward;
    Agentinfo.Qagent_qs = s2;

    episodeSteps.push(createEpisodeStepRecord({
      index: stepCount,
      stateBefore: { stateId: s, position: positionBefore },
      stateAfter: { stateId: s2, position: positionAfter },
      actionId: a,
      actionLabel: MAZE_ACTION_LABELS[a] ?? `a=${a}`,
      selectedAction: policySnapshot.selectedAction,
      reward,
      rewardBreakdown,
      policyTopActions: policySnapshot.topActions,
      entropy: policySnapshot.entropy,
      done: d,
      info: {
        detail: d ? "goal reached" : hitWall ? "hit wall" : isBacktrack ? "backtrack" : "move",
        positionBefore,
        positionAfter,
        pathSoFar: currentPath.map((point) => ({ x: point.x, y: point.y })),
      },
    }));

    s = s2;
    done = d;

    stepCount++;

    if (config.renderTrain) {
      await new Promise(r => setTimeout(r, config.renderSpeed));// 延时10ms，使动画更清晰
    }
  }
  if (done || stepCount >= config.maxSteps) {
    if (!isTest) {
      episodePaths.push([...currentPath]);
      for (let i = 1; i < currentPath.length; i++) {
        addEdge(currentPath[i - 1], currentPath[i]);
      }
      // 只保留最近20条
      if (episodePaths.length > 20) {
        for (let i = 1; i < episodePaths[0].length; i++) {
          addEdge(episodePaths[0][i - 1], episodePaths[0][i], -1);
        }
        episodePaths.shift(); // 删除最旧的一条
      }
    }
    currentPath = [];
  }

  let record = createTrainingRecord({
    episode: episodeId,
    steps: stepCount,
    reward: totalReward,
    success: done,
    win: done,
    qDelta: stepCount > 0 ? qDeltaSum / stepCount : 0,
    epsilon: config.epsilon,
    maxQ: getGlobalMaxQ(),
    visitedStates: getVisitedStateCount(),
  });
  console.log(record);
  if (!config.stop) {
    records.push(record);
    const shouldStoreEpisodeTrace =
      (episodeId === 0) ||
      ((episodeId + 1) % MAZE_EPISODE_TRACE_INTERVAL === 0) ||
      (episodeId + 1 === config.trainEpisodes);
    if (shouldStoreEpisodeTrace) {
      mazeEpisodeViewer.pushEpisode(createEpisodeTrace({
        envType: "maze",
        episode: episodeId,
        totalReward,
        totalSteps: episodeSteps.length,
        summary: {
          outcome: done ? "success" : "truncated",
          successLabel: done ? "Reached Goal" : "Stopped / Max Steps",
        },
        metadata: {
          mazeGrid: grid.map((row) => row.map((cell) => cell.wall ? 1 : 0)),
          start: { ...start },
          goal: { ...goal },
        },
        steps: episodeSteps,
      }));
    }
    drawlineGraph(records, {
      title: "Maze Q-learning Diagnostics",
      subtitle: "Track reward, path length, exploration, and whether Q-values are still changing.",
      mode: "qlearning",
      movingAverageWindow: 20,
    });
  }
}
export async function stepTrainEpisode() {
  config.currentEpisode++;
  await trainEpisode(config.currentEpisode);
}

export async function train() {
  for (config.currentEpisode = 0; config.currentEpisode < config.trainEpisodes; config.currentEpisode++) {
    if (config.stop) break;
    await trainEpisode(config.currentEpisode);
    if (config.currentEpisode === config.trainEpisodes - 1) {
      await new Promise(r => setTimeout(r, config.renderSpeed));
    } else if (!config.renderTrain && config.currentEpisode % config.EpRenderInterval === 0) {
      await new Promise(r => setTimeout(r, config.renderSpeed));
    }
  }
  drawlineGraph(records, {
    title: "Maze Q-learning Diagnostics",
    subtitle: "Track reward, path length, exploration, and whether Q-values are still changing.",
    mode: "qlearning",
    movingAverageWindow: 20,
  });
  alert("训练完成");
  reset();
}
export async function test() {
  reset();

  let s = rc2stateId(agent.x, agent.y);
  let done = false;
  let stepCount = 0;
  isTest = true

  while (!done && stepCount < config.maxSteps) {

    const a = chooseAction(s);
    const { s2, reward, done: d } = step(a);

    s = s2;
    done = d;

    stepCount++;

    await new Promise(r => setTimeout(r, config.renderSpeed));// 延时10ms，使动画更清晰
  }
  isTest = false
  alert("测试完成,总步数：" + stepCount + ",是否到达终点：" + done);
  reset();
}
//刷新整个界面
export function fresh() {
  //重置环境
  grid = createMazeDFS(config.cols, config.rows);
  start = { x: 1, y: 1 };
  goal = { x: config.cols - 2, y: config.rows - 2 };
  //重置智能体
  resetAgent();
  //重置可视化
  resetVis();

  episodePaths = [];
  currentPath = [];
  edgeWeight = {}
}
//重置智能体
export function resetAgent() {
  agent = {
    x: start.x,
    y: start.y,
    tx: start.x,
    ty: start.y,
    prevX: start.x,
    prevY: start.y,
    dir: null,
  };
  Q = {}
  visitCount = {}
  QTrace = {}
  Qactive = {}
  records = []
  config.stop = false;
  config.currentEpisode = 0;

  isTest = false

  episodePaths = [];
  currentPath = [];
  edgeWeight = {};
  mazeEpisodeViewer.clear();

  ResetAgentinfo();

  initQTable();

  document.getElementById("cfgQValue").value = formatQTable(Q);
  document.getElementById("cfgQValue").style.display = "none";
}


export function reset() {
  agent = {
    x: start.x,
    y: start.y,
    tx: start.x,
    ty: start.y,
    prevX: start.x,
    prevY: start.y,
    dir: null,
  };

  ResetAgentinfo();

  config.stop = false;
  resetVis();
  currentPath = [];
}
//重置可视化
export function resetVis() {
  hoverCell = null;
  hoverStartTime = 0;
  isTest = false
  Qactive = {}
  config.now_train_step = 0;
  config.now_test_step = 0;
}
function formatQTable(Q) {
  const cols = ["0", "1", "2", "3"];

  // 收集所有 state key
  const states = Object.keys(Q)
    .map(k => k.split(",").map(Number))
    .sort((a, b) => a[0] - b[0] || a[1] - b[1]);

  let out = "";

  // 表头
  out += "  Q   " + cols.map(c => c.padEnd(8)).join("") + "\n";

  for (let [x, y] of states) {
    const key = `${x},${y}`;
    const q = Q[key] || [0, 0, 0, 0];

    let row = `${key}  `;

    for (let i = 0; i < 4; i++) {
      row += q[i].toFixed(2).padEnd(8);
    }

    out += row + "\n";
  }

  return out;
}
export function debug() {
  console.log("Q:", Q);
  //document.getElementById("cfgQValue").value = JSON.stringify(Q, null, 2);
  document.getElementById("cfgQValue").value = formatQTable(Q);
  document.getElementById("cfgQValue").style.display = "block";
}

export function stopTrain() {
  config.stop = true;
}
export function restore() {
  const new_config = structuredClone(initConfig);

  Object.keys(new_config).forEach(k => {
    config[k] = new_config[k];
  });

  document.getElementById("cfgQValue").style.display = "none";
  alert("已恢复初始配置");
}
export function applyConfig() {
  const get = (id) => document.getElementById(id);

  get("cfgRenderTrain").onchange = (e) => {
    config.renderTrain = e.target.checked;
  }

  get("cfgRenderSpeed").oninput = (e) =>
    (config.renderSpeed = +e.target.value);

  get("cfgStepMode").onchange = (e) =>
    (config.stepMode = e.target.checked);

  get("cfgHighRender").onchange = (e) =>
    (config.high_render = e.target.checked);

  get("cfgShowPath").onchange = (e) => {
    config.draw_current_path = e.target.checked
    config.draw_paths = e.target.checked
  }

  get("cfgShowMarkers").onchange = (e) => {
    config.draw_policy = e.target.checked
  }

  get("cfgShowHeatmap").onchange = (e) =>
    (config.draw_heatmap = e.target.checked);

  get("cfgEpisodes").oninput = (e) =>
    (config.trainEpisodes = +e.target.value);

  get("cfgEpRenderInterval").oninput = (e) =>
    (config.EpRenderInterval = +e.target.value);

  get("cfgMaxSteps").oninput = (e) =>
    (config.maxSteps = +e.target.value);

  get("cfgCols").oninput = (e) =>
    (config.cols = +e.target.value);

  get("cfgRows").oninput = (e) =>
    (config.rows = +e.target.value);

  get("cfgAlpha").oninput = (e) =>
    (config.alpha = +e.target.value);

  get("cfgGamma").oninput = (e) =>
    (config.gamma = +e.target.value);

  get("cfgEpsilon").oninput = (e) =>
    (config.epsilon = +e.target.value);
}

let P = new p5(sketch);
