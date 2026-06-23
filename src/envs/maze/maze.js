import p5 from "p5";

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
  stepMode: false,
  stop: false,

  trainEpisodes: 300,
  maxSteps: 400,
  currentEpisode: 0,

  EpRenderInterval: 20,

  alpha: 0.1,//学习率
  gamma: 0.95,//折扣因子 对未来奖励的影响程度
  epsilon: 0.1,//探索率

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
let agent;
let start;
let goal;
let Q = {};//存储Q值
let QTrace = {};//存储附加信息
let visitCount = {};//存储访问次数

let records = [];//存储训练信息

let hoverCell = null;
let hoverStartTime = 0;
let hoverDelay = 400; // ms

let isTest = false

let episodePaths = [];
let currentPath = [];
let edgeWeight = {};
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

      // ====== 用 alpha 而不是纯蓝覆盖 ======
      const alpha = 120 * intensity;

      p.fill(0, 0, 123, alpha); // 透明蓝
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

  for (let i = 0; i < 4; i++) {
    html += `<div style="font-size:18px;">${actions[i]}: ${q[i].toFixed(3)}</div>`;

    if (trace && trace[i]) {
      html += `<div style="font-size:18px;color:#0ff">
        ${trace[i].values}
      </div>`;
    }
  }

  tooltip.innerHTML = html;
  tooltip.style.left = p.mouseX + 12 + "px";
  tooltip.style.top = p.mouseY + 12 + "px";
  tooltip.style.display = "block";
}
function drawGoal(p) {
  p.fill(220, 50, 50);
  p.rect(
    goal.x * config.cellSize,
    goal.y * config.cellSize,
    config.cellSize,
    config.cellSize,
    6
  );
}

function drawAgent(p) {
  agent.x = p.lerp(agent.x, agent.tx, 0.2);
  agent.y = p.lerp(agent.y, agent.ty, 0.2);

  p.fill(50, 150, 255);
  p.circle(
    agent.x * config.cellSize + config.cellSize / 2,
    agent.y * config.cellSize + config.cellSize / 2,
    config.cellSize * 0.6
  );
}
function drawPolicy(p) {
  p.textAlign(p.CENTER, p.CENTER);
  p.textSize(18);
  p.fill(30);

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

      p.fill(0);
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

      // ====== 1. 方向衰减（越靠前越粗） ======
      const t = j / path.length; // 0 → 1
      const thickness = (1 - t) * 6 + w * 0.5;

      // ====== 2. 颜色衰减 ======
      const alpha = Math.min(200, 30 + w * 20);

      p.stroke(255, 0, 0, alpha);
      p.strokeWeight(thickness);

      // ====== 3. 画线 ======
      p.line(
        a.x * config.cellSize + config.cellSize / 2,
        a.y * config.cellSize + config.cellSize / 2,
        b.x * config.cellSize + config.cellSize / 2,
        b.y * config.cellSize + config.cellSize / 2
      );
    }
  }
  // reset 
  p.strokeWeight(1);
  p.stroke(200);
}
function drawPaths(p) {
  p.noFill();
  p.push();

  for (let i = 0; i < episodePaths.length; i++) {
    const path = episodePaths[i];

    // ⭐ 历史路径：整体再弱一点（关键）
    const pathScale = 0.7 + i / episodePaths.length * 0.3;
    // 0.6 ~ 1.0（越新的越明显）

    for (let j = 1; j < path.length; j++) {
      const a = path[j - 1];
      const b = path[j];

      const key = `${a.x},${a.y}->${b.x},${b.y}`;
      const w = edgeWeight[key] || 1;

      // =========================
      // 1. 方向衰减（核心结构）
      // =========================
      const t = j / path.length;

      // ⭐ 非线性（神经束风格）
      const curveT = Math.pow(1 - t, 1.6);

      // ⭐ 历史路径更细（重点）
      const baseThickness = Math.max(8, episodePaths.length * 0.6);
      const thickness = (baseThickness * curveT + w * 0.2) * pathScale;

      // =========================
      // 2. 颜色（更淡）
      // =========================
      const alphaBase = 160;
      const alpha = Math.min(alphaBase, (100 + w * 5) * pathScale);

      p.stroke(255, 0, 0, alpha);
      p.strokeWeight(thickness);

      // =========================
      // 3. 画“束状线”（关键升级）
      // =========================
      const ax = a.x * config.cellSize + config.cellSize / 2;
      const ay = a.y * config.cellSize + config.cellSize / 2;
      const bx = b.x * config.cellSize + config.cellSize / 2;
      const by = b.y * config.cellSize + config.cellSize / 2;

      const mi = 7; // ⭐ 比 currentPath 少（更轻）

      for (let s = 0; s < mi; s++) {
        const t0 = s / mi;
        const t1 = (s + 1) / mi;

        const curveLocal = Math.pow(1 - t0, 1.6);

        const x1 = p.lerp(ax, bx, t0);
        const y1 = p.lerp(ay, by, t0);
        const x2 = p.lerp(ax, bx, t1);
        const y2 = p.lerp(ay, by, t1);

        const localThickness = thickness * curveLocal; // ⭐ 更弱
        const localAlpha = alpha * (0.6 + 0.4 * curveLocal);

        p.stroke(255, 0, 0, localAlpha);
        p.strokeWeight(localThickness);

        p.line(x1, y1, x2, y2);
      }
    }
  }

  // reset state
  p.strokeWeight(1);
  p.stroke(200);
  p.pop();
}
function drawCurrentPath_nomi(p) {
  if (currentPath.length < 2) return;

  p.push();
  p.stroke(255, 0, 0, 180);
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

    const mi = 7; // 束的“密度”

    for (let s = 0; s < mi; s++) {
      let t = s / mi;

      // 非线性粗细（关键）
      const curveT = Math.pow(1 - t, 1.8);

      const x = p.lerp(ax, bx, t);
      const y = p.lerp(ay, by, t);

      const base = 10; // 最大粗细
      const w = base * curveT;

      // edge weight 可叠加
      const key = `${a.x},${a.y}->${b.x},${b.y}`;
      const ew = edgeWeight?.[key] || 1;

      p.stroke(255, 0, 0, 120 + ew * 10);
      p.strokeWeight(w);

      // 画短线段（形成“束”）
      const nx = p.lerp(ax, bx, (s + 1) / mi);
      const ny = p.lerp(ay, by, (s + 1) / mi);

      p.line(x, y, nx, ny);
    }
  }

  p.pop();
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
//  ε-greedy
function chooseAction(s) {
  const q = getQ(s);

  if (Math.random() < config.epsilon) {
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
    formula: `
Q(s,a) = Q(s,a) + \\alpha [r + \\gamma \\max Q(s',a') - Q(s,a)]
`,
    values: `
= ${oldQ.toFixed(3)} + ${config.alpha} × (${r.toFixed(3)} + ${config.gamma} × ${maxNext.toFixed(3)} - ${oldQ.toFixed(3)})
`,
    result: newQ.toFixed(3)
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
    p.background(245);//灰度值

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

export function step(action) {
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

  const nx = agent.x + dx;
  const ny = agent.y + dy;

  let reward = -0.01;
  let done = false;

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
  } else {
    nextX = nx;
    nextY = ny;
  }

  const isBacktrack =
    nextX === agent.prevX && nextY === agent.prevY;

  if (isBacktrack && !(nextX === agent.x && nextY === agent.y)) {
    reward -= 0.05;
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
    done = true;
  }

  const s = rc2stateId(agent.x, agent.y);
  const s2 = rc2stateId(agent.tx, agent.ty);


  currentPath.push({
    x: agent.x,
    y: agent.y
  });
  if (!isTest) {
    visitCount[s2] = (visitCount[s2] || 0) + 1;
  }

  if (config.stepMode) {
    updateQ(s, a, reward, s2);
  }

  return { s, s2, reward, done };
}

async function trainEpisode(episodeId = 0) {
  reset();

  let s = rc2stateId(agent.x, agent.y);
  let done = false;
  let stepCount = 0;
  let totalReward = 0;

  while (!done && stepCount < config.maxSteps && !config.stop) {

    const a = chooseAction(s);
    const { s2, reward, done: d } = step(a);
    totalReward += reward;
    updateQ(s, a, reward, s2);

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

  let record = {
    episode: episodeId,
    steps: stepCount,
    reward: totalReward,
    win: done,
  };
  console.log(record);
  if (!config.stop)
    records.push(record);
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
  alert("训练完成");
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
  };
  Q = {}
  visitCount = {}
  QTrace = {}
  records = []
  config.stop = false;
  config.currentEpisode = 0;

  isTest = false

  episodePaths = [];
  currentPath = [];
  edgeWeight = {};
}


export function reset() {
  agent = {
    x: start.x,
    y: start.y,
    tx: start.x,
    ty: start.y,
    prevX: start.x,
    prevY: start.y,
  };
  config.stop = false;
  resetVis();
  currentPath = [];
}
//重置可视化
export function resetVis() {
  hoverCell = null;
  hoverStartTime = 0;
  isTest = false

}
export function debug() {
  console.log(Q);
}

export function stopTrain() {
  config.stop = true;
}
export function restore() {
  const new_config = structuredClone(initConfig);

  Object.keys(new_config).forEach(k => {
    config[k] = new_config[k];
  });
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

