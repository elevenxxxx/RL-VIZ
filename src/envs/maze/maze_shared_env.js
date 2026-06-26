const DEFAULT_COMPARE_CONFIG = {
  cols: 21,
  rows: 21,
  maxSteps: 300,
  loopRate: 0.3,
};

const ACTION_DELTAS = [
  { dx: 0, dy: -1, label: "Up" },
  { dx: 0, dy: 1, label: "Down" },
  { dx: -1, dy: 0, label: "Left" },
  { dx: 1, dy: 0, label: "Right" },
];

function ensureOdd(value, fallback) {
  const safe = Number.isFinite(value) ? Math.max(5, Math.floor(value)) : fallback;
  return safe % 2 === 1 ? safe : safe + 1;
}

function cloneGrid(grid) {
  return grid.map((row) => row.slice());
}

function addLoops(grid, cols, rows, rate = 0.3) {
  for (let y = 1; y < rows - 1; y++) {
    for (let x = 1; x < cols - 1; x++) {
      if (Math.random() < rate && grid[y][x] === 1) {
        grid[y][x] = 0;
      }
    }
  }
}

function createMazeDFS(cols, rows, loopRate = 0.3) {
  const grid = Array.from({ length: rows }, (_, y) =>
    Array.from({ length: cols }, (_, x) => ({
      x,
      y,
      wall: true,
      visited: false,
    }))
  );

  const stack = [];
  const start = grid[1][1];
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
    for (const [dx, dy] of dirs) {
      const nx = cell.x + dx;
      const ny = cell.y + dy;
      if (nx > 0 && ny > 0 && nx < cols && ny < rows && !grid[ny][nx].visited) {
        res.push(grid[ny][nx]);
      }
    }
    return res;
  }

  while (stack.length > 0) {
    const current = stack[stack.length - 1];
    const nbs = neighbors(current);

    if (nbs.length === 0) {
      stack.pop();
      continue;
    }

    const next = nbs[Math.floor(Math.random() * nbs.length)];
    const wx = (current.x + next.x) / 2;
    const wy = (current.y + next.y) / 2;

    grid[wy][wx].wall = false;
    next.wall = false;
    next.visited = true;
    stack.push(next);
  }

  const binaryGrid = grid.map((row) => row.map((cell) => (cell.wall ? 1 : 0)));
  addLoops(binaryGrid, cols, rows, loopRate);
  return binaryGrid;
}

export function createSharedMazeMap(options = {}) {
  const cols = ensureOdd(options.cols, DEFAULT_COMPARE_CONFIG.cols);
  const rows = ensureOdd(options.rows, DEFAULT_COMPARE_CONFIG.rows);
  const loopRate = Number.isFinite(options.loopRate) ? options.loopRate : DEFAULT_COMPARE_CONFIG.loopRate;
  return createMazeDFS(cols, rows, loopRate);
}

export class SharedMazeEnv {
  constructor(options = {}) {
    this.cols = ensureOdd(options.cols, DEFAULT_COMPARE_CONFIG.cols);
    this.rows = ensureOdd(options.rows, DEFAULT_COMPARE_CONFIG.rows);
    this.maxSteps = Number.isFinite(options.maxSteps) ? options.maxSteps : DEFAULT_COMPARE_CONFIG.maxSteps;
    this.loopRate = Number.isFinite(options.loopRate) ? options.loopRate : DEFAULT_COMPARE_CONFIG.loopRate;
    this.grid = Array.isArray(options.grid)
      ? cloneGrid(options.grid)
      : createSharedMazeMap({
        cols: this.cols,
        rows: this.rows,
        loopRate: this.loopRate,
      });
    this.start = options.start ? { ...options.start } : { x: 1, y: 1 };
    this.goal = options.goal ? { ...options.goal } : { x: this.cols - 2, y: this.rows - 2 };
    this.featureSize = this.rows * this.cols + 8;
    this.reset();
  }

  clone() {
    return new SharedMazeEnv({
      cols: this.cols,
      rows: this.rows,
      maxSteps: this.maxSteps,
      loopRate: this.loopRate,
      grid: this.serializeGrid(),
      start: this.start,
      goal: this.goal,
    });
  }

  serializeGrid() {
    return cloneGrid(this.grid);
  }

  reset() {
    this.agent = {
      x: this.start.x,
      y: this.start.y,
      prevX: this.start.x,
      prevY: this.start.y,
    };
    this.stepCount = 0;
    this.path = [{ x: this.agent.x, y: this.agent.y }];
    return this.getState();
  }

  getState() {
    return {
      x: this.agent.x,
      y: this.agent.y,
      prevX: this.agent.prevX,
      prevY: this.agent.prevY,
      stepCount: this.stepCount,
    };
  }

  getStateKey(state = this.getState()) {
    return `${state.x},${state.y}`;
  }

  getAvailableActions() {
    return [0, 1, 2, 3];
  }

  isWall(x, y) {
    return x < 0 || y < 0 || x >= this.cols || y >= this.rows || this.grid[y][x] === 1;
  }

  toFeatureVector(state = this.getState()) {
    const vector = new Float32Array(this.featureSize);
    const index = state.y * this.cols + state.x;
    vector[index] = 1;

    const base = this.rows * this.cols;
    vector[base] = state.x / Math.max(1, this.cols - 1);
    vector[base + 1] = state.y / Math.max(1, this.rows - 1);
    vector[base + 2] = (this.goal.x - state.x) / Math.max(1, this.cols - 1);
    vector[base + 3] = (this.goal.y - state.y) / Math.max(1, this.rows - 1);
    vector[base + 4] = this.isWall(state.x, state.y - 1) ? 1 : 0;
    vector[base + 5] = this.isWall(state.x, state.y + 1) ? 1 : 0;
    vector[base + 6] = this.isWall(state.x - 1, state.y) ? 1 : 0;
    vector[base + 7] = this.isWall(state.x + 1, state.y) ? 1 : 0;
    return Array.from(vector);
  }

  step(action) {
    const safeAction = Number.isInteger(action) && action >= 0 && action < ACTION_DELTAS.length ? action : 0;
    const stateBefore = this.getState();
    const { dx, dy } = ACTION_DELTAS[safeAction];
    const nx = this.agent.x + dx;
    const ny = this.agent.y + dy;

    let reward = -0.01;
    let done = false;
    let success = false;
    let truncated = false;
    const rewardBreakdown = {
      stepReward: -0.01,
      wallPenalty: 0,
      backtrackPenalty: 0,
      goalReward: 0,
    };

    let nextX = this.agent.x;
    let nextY = this.agent.y;
    const hitWall = this.isWall(nx, ny);
    if (hitWall) {
      reward = -0.1;
      rewardBreakdown.stepReward = 0;
      rewardBreakdown.wallPenalty = -0.1;
    } else {
      nextX = nx;
      nextY = ny;
    }

    const isBacktrack = nextX === this.agent.prevX && nextY === this.agent.prevY;
    if (isBacktrack && !(nextX === this.agent.x && nextY === this.agent.y)) {
      reward -= 0.05;
      rewardBreakdown.backtrackPenalty = -0.05;
    }

    this.agent.prevX = this.agent.x;
    this.agent.prevY = this.agent.y;
    this.agent.x = nextX;
    this.agent.y = nextY;
    this.stepCount += 1;

    if (this.agent.x === this.goal.x && this.agent.y === this.goal.y) {
      reward = 20;
      rewardBreakdown.stepReward = 0;
      rewardBreakdown.wallPenalty = 0;
      rewardBreakdown.backtrackPenalty = 0;
      rewardBreakdown.goalReward = 20;
      success = true;
      done = true;
    }

    if (!done && this.stepCount >= this.maxSteps) {
      done = true;
      truncated = true;
    }

    this.path.push({ x: this.agent.x, y: this.agent.y });

    return {
      state: this.getState(),
      reward,
      done,
      success,
      truncated,
      info: {
        action: safeAction,
        actionLabel: ACTION_DELTAS[safeAction].label,
        stateBefore,
        stateAfter: this.getState(),
        rewardBreakdown,
        hitWall,
        isBacktrack,
        pathSoFar: this.path.map((point) => ({ ...point })),
      },
    };
  }
}

export function getMazeActionLabels() {
  return ACTION_DELTAS.map((item) => item.label);
}
