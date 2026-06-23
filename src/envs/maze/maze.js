import p5 from "p5";

export const config = {
  cols: 21,// 必须奇数
  rows: 21,
  cellSize: 28,

  renderTrain: true,
  renderSpeed: 10,
  stepMode: false,
  stop: false,

  trainEpisodes: 150,
  maxSteps: 400,
  currentEpisode: 0,

  alpha: 0.1,//学习率
  gamma: 0.95,//折扣因子 对未来奖励的影响程度
  epsilon: 0.1,//探索率
};

let grid = [];
let agent;
let start;
let goal;
let Q = {};
let records = []

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
  p.textSize(14);
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



function rc2stateId(x, y) {
  return `${x},${y}`;
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

  q[a] =
    q[a] +
    config.alpha * (r + config.gamma * maxNext - q[a]);
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
    p.background(245);
    drawMaze(p);
    drawPolicy(p);
    drawAgent(p);
    drawGoal(p);
  };
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
      await new Promise(r => setTimeout(r, 10));// 延时10ms，使动画更清晰
    }
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
  }
  alert("训练完成");
}
export async function test() {
  reset();

  let s = rc2stateId(agent.x, agent.y);
  let done = false;
  let stepCount = 0;

  while (!done && stepCount < config.maxSteps) {

    const a = chooseAction(s);
    const { s2, reward, done: d } = step(a);

    s = s2;
    done = d;

    stepCount++;

    await new Promise(r => setTimeout(r, 10));// 延时10ms，使动画更清晰
  }
  alert("测试完成,总步数：" + stepCount + ",是否到达终点：" + done);
}
//刷新整个界面
export function fresh() {
  //重置环境
  grid = createMazeDFS(config.cols, config.rows);
  start = { x: 1, y: 1 };
  goal = { x: config.cols - 2, y: config.rows - 2 };
  //重置智能体
  resetAgent()
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
  config.stop = false;
  config.currentEpisode = 0;
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
}

export function debug() {
  console.log(Q);
}

export function stopTrain() {
  config.stop = true;
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

  get("cfgEpisodes").oninput = (e) =>
    (config.trainEpisodes = +e.target.value);

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

new p5(sketch);

