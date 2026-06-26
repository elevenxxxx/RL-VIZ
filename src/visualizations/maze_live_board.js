function clonePath(path = []) {
  return path.map((point) => ({ x: point.x, y: point.y }));
}

function edgeKey(a, b) {
  return `${a.x},${a.y}->${b.x},${b.y}`;
}

export class MazeLiveBoard {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.grid = [];
    this.start = options.start ?? { x: 1, y: 1 };
    this.goal = options.goal ?? { x: 19, y: 19 };
    this.agent = { x: this.start.x, y: this.start.y, dir: null };
    this.currentPath = [];
    this.episodePaths = [];
    this.edgeWeight = {};
    this.maxEpisodePaths = 20;
    this.cellSize = 20;
    this.offsetX = 0;
    this.offsetY = 0;
  }

  setGrid(grid, options = {}) {
    this.grid = grid.map((row) => row.slice());
    this.start = options.start ?? this.start;
    this.goal = options.goal ?? this.goal;
    this.agent = { x: this.start.x, y: this.start.y, dir: null };
    this.currentPath = [];
    this.episodePaths = [];
    this.edgeWeight = {};
    this.render();
  }

  setEpisodePaths(paths = []) {
    this.episodePaths = paths.slice(-this.maxEpisodePaths).map((path) => clonePath(path));
    this.edgeWeight = {};
    for (const path of this.episodePaths) {
      this.addEpisodeEdges(path, 1);
    }
    this.render();
  }

  clearHistory() {
    this.currentPath = [];
    this.episodePaths = [];
    this.edgeWeight = {};
    this.agent = { x: this.start.x, y: this.start.y, dir: null };
    this.render();
  }

  beginEpisode(path = []) {
    this.currentPath = clonePath(path.length > 0 ? path : [{ x: this.start.x, y: this.start.y }]);
    const last = this.currentPath.at(-1) ?? this.start;
    this.agent = { x: last.x, y: last.y, dir: null };
    this.render();
  }

  updateCurrentPath(path = []) {
    this.currentPath = clonePath(path);
    const last = this.currentPath.at(-1);
    if (last) {
      const prev = this.currentPath.at(-2) ?? last;
      this.agent = {
        x: last.x,
        y: last.y,
        dir: this.inferDirection(prev, last),
      };
    }
    this.render();
  }

  commitEpisode(path = []) {
    const safePath = clonePath(path);
    if (safePath.length > 1) {
      this.episodePaths.push(safePath);
      this.addEpisodeEdges(safePath, 1);
      if (this.episodePaths.length > this.maxEpisodePaths) {
        const removed = this.episodePaths.shift();
        this.addEpisodeEdges(removed, -1);
      }
    }
    this.currentPath = [];
    this.render();
  }

  showEpisode(path = []) {
    this.currentPath = clonePath(path);
    const last = this.currentPath.at(-1);
    if (last) {
      this.agent = { x: last.x, y: last.y, dir: null };
    }
    this.render();
  }

  inferDirection(a, b) {
    if (!a || !b) return null;
    if (b.x > a.x) return "right";
    if (b.x < a.x) return "left";
    if (b.y > a.y) return "down";
    if (b.y < a.y) return "up";
    return null;
  }

  addEpisodeEdges(path = [], weight = 1) {
    for (let i = 1; i < path.length; i++) {
      const key = edgeKey(path[i - 1], path[i]);
      const next = (this.edgeWeight[key] || 0) + weight;
      if (next > 0) {
        this.edgeWeight[key] = next;
      } else {
        delete this.edgeWeight[key];
      }
    }
  }

  computeLayout() {
    if (!this.grid.length || !this.grid[0].length) return;
    const rows = this.grid.length;
    const cols = this.grid[0].length;
    this.cellSize = Math.min(
      Math.floor(this.canvas.width / cols),
      Math.floor(this.canvas.height / rows)
    );
    this.offsetX = Math.floor((this.canvas.width - cols * this.cellSize) / 2);
    this.offsetY = Math.floor((this.canvas.height - rows * this.cellSize) / 2);
  }

  toCanvasPoint(point) {
    return {
      x: this.offsetX + point.x * this.cellSize + this.cellSize / 2,
      y: this.offsetY + point.y * this.cellSize + this.cellSize / 2,
    };
  }

  drawMaze() {
    const ctx = this.ctx;
    const rows = this.grid.length;
    const cols = this.grid[0].length;

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        ctx.fillStyle = this.grid[y][x] === 1 ? "#505755" : "#F3F4F2";
        ctx.fillRect(
          this.offsetX + x * this.cellSize,
          this.offsetY + y * this.cellSize,
          this.cellSize - 1,
          this.cellSize - 1
        );
      }
    }
  }

  drawGoal() {
    const ctx = this.ctx;
    ctx.fillStyle = "#C75B5B";
    ctx.fillRect(
      this.offsetX + this.goal.x * this.cellSize,
      this.offsetY + this.goal.y * this.cellSize,
      this.cellSize - 1,
      this.cellSize - 1
    );

    ctx.fillStyle = "#6E8B74";
    ctx.fillRect(
      this.offsetX + this.start.x * this.cellSize,
      this.offsetY + this.start.y * this.cellSize,
      this.cellSize - 1,
      this.cellSize - 1
    );
  }

  drawPaths() {
    const ctx = this.ctx;
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    for (let i = 0; i < this.episodePaths.length; i++) {
      const path = this.episodePaths[i];
      const pathScale = 0.7 + (i / Math.max(1, this.episodePaths.length)) * 0.3;

      for (let j = 1; j < path.length; j++) {
        const a = path[j - 1];
        const b = path[j];
        const w = this.edgeWeight[edgeKey(a, b)] || 1;
        const t = j / path.length;
        const curveT = Math.pow(1 - t, 1.6);
        const baseThickness = Math.max(8, this.episodePaths.length * 0.6);
        const thickness = (baseThickness * curveT + w * 0.2) * pathScale;
        const alphaBase = 160;
        const alpha = Math.min(alphaBase, (100 + w * 5) * pathScale);

        const ax = this.toCanvasPoint(a).x;
        const ay = this.toCanvasPoint(a).y;
        const bx = this.toCanvasPoint(b).x;
        const by = this.toCanvasPoint(b).y;

        const segments = 7;
        for (let s = 0; s < segments; s++) {
          const t0 = s / segments;
          const t1 = (s + 1) / segments;
          const curveLocal = Math.pow(1 - t0, 1.6);
          const x1 = ax + (bx - ax) * t0;
          const y1 = ay + (by - ay) * t0;
          const x2 = ax + (bx - ax) * t1;
          const y2 = ay + (by - ay) * t1;
          const localThickness = thickness * curveLocal;
          const localAlpha = alpha * (0.6 + 0.4 * curveLocal);

          ctx.strokeStyle = `rgba(95, 158, 160, ${localAlpha / 255})`;
          ctx.lineWidth = localThickness;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        }
      }
    }

    ctx.restore();
  }

  drawCurrentPath() {
    if (this.currentPath.length < 2) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    for (let i = 1; i < this.currentPath.length; i++) {
      const a = this.currentPath[i - 1];
      const b = this.currentPath[i];
      const ax = this.toCanvasPoint(a).x;
      const ay = this.toCanvasPoint(a).y;
      const bx = this.toCanvasPoint(b).x;
      const by = this.toCanvasPoint(b).y;
      const segments = 7;

      for (let s = 0; s < segments; s++) {
        const t0 = s / segments;
        const t1 = (s + 1) / segments;
        const curveT = Math.pow(1 - t0, 1.8);
        const x1 = ax + (bx - ax) * t0;
        const y1 = ay + (by - ay) * t0;
        const x2 = ax + (bx - ax) * t1;
        const y2 = ay + (by - ay) * t1;
        const thickness = 10 * curveT;
        const ew = this.edgeWeight[edgeKey(a, b)] || 1;

        ctx.strokeStyle = `rgba(201, 163, 74, ${(120 + ew * 10) / 255})`;
        ctx.lineWidth = thickness;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }

    ctx.restore();
  }

  drawAgent() {
    const ctx = this.ctx;
    const center = this.toCanvasPoint(this.agent);
    const size = this.cellSize * 0.76;
    ctx.save();
    ctx.fillStyle = "#5F9EA0";
    ctx.shadowColor = "rgba(95, 158, 160, 0.22)";
    ctx.shadowBlur = 10;
    ctx.beginPath();

    if (!this.agent.dir) {
      ctx.arc(center.x, center.y, size * 0.42, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      return;
    }

    if (this.agent.dir === "right") {
      ctx.moveTo(center.x + size / 2, center.y);
      ctx.lineTo(center.x - size / 2, center.y - size / 3);
      ctx.lineTo(center.x - size / 2, center.y + size / 3);
    } else if (this.agent.dir === "left") {
      ctx.moveTo(center.x - size / 2, center.y);
      ctx.lineTo(center.x + size / 2, center.y - size / 3);
      ctx.lineTo(center.x + size / 2, center.y + size / 3);
    } else if (this.agent.dir === "down") {
      ctx.moveTo(center.x, center.y + size / 2);
      ctx.lineTo(center.x - size / 3, center.y - size / 2);
      ctx.lineTo(center.x + size / 3, center.y - size / 2);
    } else {
      ctx.moveTo(center.x, center.y - size / 2);
      ctx.lineTo(center.x - size / 3, center.y + size / 2);
      ctx.lineTo(center.x + size / 3, center.y + size / 2);
    }

    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  render() {
    if (!this.canvas || !this.grid.length) return;
    this.computeLayout();
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.fillStyle = "#F3F4F2";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.drawMaze();
    this.drawPaths();
    this.drawCurrentPath();
    this.drawGoal();
    this.drawAgent();
  }
}
