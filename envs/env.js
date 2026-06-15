import { EnemyAi } from "./enemy_ai.js";
// 棋子
class Piece {
    constructor(r, c, t, p) {
        this.r = r;
        this.c = c;
        this.t = t;
        this.p = p;
    }
}

// Board-棋盘
class Board {
    constructor() {
        this.pieces = [];
        this.init();
    }

    init() {
        this.pieces = [
            new Piece(0, 0, "车", "black"),
            new Piece(0, 1, "马", "black"),
            new Piece(0, 2, "相", "black"),
            new Piece(0, 3, "仕", "black"),
            new Piece(0, 4, "将", "black"),
            new Piece(0, 5, "仕", "black"),
            new Piece(0, 6, "相", "black"),
            new Piece(0, 7, "马", "black"),
            new Piece(0, 8, "车", "black"),
            new Piece(2, 1, "炮", "black"),
            new Piece(2, 7, "炮", "black"),
            new Piece(3, 0, "卒", "black"),
            new Piece(3, 2, "卒", "black"),
            new Piece(3, 4, "卒", "black"),
            new Piece(3, 6, "卒", "black"),
            new Piece(3, 8, "卒", "black"),

            new Piece(9, 0, "车", "red"),
            new Piece(9, 1, "马", "red"),
            new Piece(9, 2, "象", "red"),
            new Piece(9, 3, "士", "red"),
            new Piece(9, 4, "帅", "red"),
            new Piece(9, 5, "士", "red"),
            new Piece(9, 6, "象", "red"),
            new Piece(9, 7, "马", "red"),
            new Piece(9, 8, "车", "red"),
            new Piece(7, 1, "炮", "red"),
            new Piece(7, 7, "炮", "red"),
            new Piece(6, 0, "兵", "red"),
            new Piece(6, 2, "兵", "red"),
            new Piece(6, 4, "兵", "red"),
            new Piece(6, 6, "兵", "red"),
            new Piece(6, 8, "兵", "red"),
        ];
    }

    get(r, c) {
        return this.pieces.find(p => p.r === r && p.c === c);
    }

    remove(r, c) {
        this.pieces = this.pieces.filter(p => !(p.r === r && p.c === c));
    }

    move(piece, r, c) {
        const legal = this.getLegalMoves(piece);
        //  console.log(`${piece.p} ${piece.t} 可选位置:`, legal);
        if (!legal.some(m => m.r === r && m.c === c)) return false;

        const target = this.get(r, c);
        if (target && target.p === piece.p) return false;
        if (target) {
            //console.log(`${target.t}被吃掉拉`);
            this.remove(r, c);
        }

        piece.r = r;
        piece.c = c;
        return true;
    }

    // =======================
    // 合法走法
    // =======================
    getLegalMoves(p) {
        switch (p.t) {
            case "车": return this.rook(p);
            case "马": return this.knight(p);
            case "炮": return this.cannon(p);
            case "兵":
            case "卒": return this.pawn(p);
            case "象":
            case "相": return this.elephant(p);
            case "士":
            case "仕": return this.advisor(p);
            case "帅":
            case "将": return this.king(p);
            default: return [];
        }
    }

    //车
    rook(p) {
        let res = [];
        const dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]];

        for (let [dr, dc] of dirs) {
            let r = p.r, c = p.c;
            while (true) {
                r += dr; c += dc;
                if (r < 0 || r > 9 || c < 0 || c > 8) break;

                const t = this.get(r, c);
                if (!t) {
                    res.push({ r, c });
                } else {
                    if (t.p !== p.p) res.push({ r, c });
                    break;
                }
            }
        }
        return res;
    }

    //马
    knight(p) {
        const steps = [
            [2, 1], [2, -1], [-2, 1], [-2, -1],
            [1, 2], [1, -2], [-1, 2], [-1, -2]
        ];

        let res = [];

        for (let [dr, dc] of steps) {
            const r = p.r + dr;
            const c = p.c + dc;

            if (r < 0 || r > 9 || c < 0 || c > 8) continue;

            // 🧠 正确蹩马腿位置
            let legR = p.r;
            let legC = p.c;

            if (Math.abs(dr) === 2) {
                legR += dr / 2;
            } else {
                legC += dc / 2;
            }

            if (this.get(legR, legC)) continue;

            const t = this.get(r, c);
            if (!t || t.p !== p.p) res.push({ r, c });
        }

        return res;
    }
    //炮
    cannon(p) {
        let res = [];
        const dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]];

        for (let [dr, dc] of dirs) {
            let r = p.r, c = p.c;
            let jumped = false;

            while (true) {
                r += dr; c += dc;
                if (r < 0 || r > 9 || c < 0 || c > 8) break;

                const t = this.get(r, c);

                if (!jumped) {
                    if (!t) {
                        res.push({ r, c });
                    } else {
                        jumped = true;
                    }
                } else {
                    if (t) {
                        if (t.p !== p.p) res.push({ r, c });
                        break;
                    }
                }
            }
        }
        return res;
    }

    //兵 或卒
    pawn(p) {
        let res = [];
        const dir = p.p === "red" ? -1 : 1;

        const forward = this.get(p.r + dir, p.c);
        //前方没有棋子，直接走
        if (!forward && p.r + dir > 0) res.push({ r: p.r + dir, c: p.c });
        //前方有棋子，只能吃掉对方的棋子
        if (forward && forward.p !== p.p) res.push({ r: p.r + dir, c: p.c });

        if ((p.p === "red" && p.r <= 4) || (p.p === "black" && p.r >= 5)) {
            for (let dc of [-1, 1]) {
                const r = p.r, c = p.c + dc;
                if (c < 0 || c > 8) continue;

                const t = this.get(r, c);
                if (!t || t.p !== p.p) res.push({ r, c });
            }
        }
        res = res.filter(i => i.r >= 0 && i.r <= 9 && i.c >= 0 && i.c <= 8);
        return res;
    }
    //象 相
    elephant(p) {
        let res = [];
        const dirs = [
            [2, 2], [2, -2], [-2, 2], [-2, -2]
        ];

        for (let [dr, dc] of dirs) {
            const r = p.r + dr;
            const c = p.c + dc;

            if (r < 0 || r > 9 || c < 0 || c > 8) continue;

            // 🧠 不能过河
            if (p.p === "red" && r < 5) continue;
            if (p.p === "black" && r > 4) continue;

            // 🧠 象眼
            const eyeR = p.r + dr / 2;
            const eyeC = p.c + dc / 2;

            if (this.get(eyeR, eyeC)) continue;

            const t = this.get(r, c);
            if (!t || t.p !== p.p) res.push({ r, c });
        }

        return res;
    }
    //士 仕
    advisor(p) {
        let res = [];

        const dirs = [
            [1, 1], [1, -1], [-1, 1], [-1, -1]
        ];

        //console.log(`selected:${p.r},${p.c}`)
        for (let [dr, dc] of dirs) {
            const r = p.r + dr;
            const c = p.c + dc;

            // 🧠 九宫
            if (c < 3 || c > 5) continue;

            if (p.p === "red" && (r < 7 || r > 9)) continue;
            if (p.p === "black" && (r < 0 || r > 2)) continue;

            const t = this.get(r, c);
            if (!t || t.p !== p.p) res.push({ r, c });
        }

        return res;
    }
    //将 帅
    king(p) {
        let res = [];

        const dirs = [
            [1, 0], [-1, 0], [0, 1], [0, -1]
        ];

        for (let [dr, dc] of dirs) {
            const r = p.r + dr;
            const c = p.c + dc;

            // 九宫
            if (c < 3 || c > 5) continue;

            if (p.p === "red" && (r < 7 || r > 9)) continue;
            if (p.p === "black" && (r < 0 || r > 2)) continue;

            const t = this.get(r, c);
            if (!t || t.p !== p.p) res.push({ r, c });
        }

        // 🧠 飞将规则（非常重要）
        const enemyKing = this.pieces.find(x =>
            (x.t === "将" || x.t === "帅") && x.p !== p.p
        );

        //console.log(`enemyKing:${enemyKing}`)
        if (enemyKing && enemyKing.c === p.c) {
            let blocked = false;

            //console.log(`检测:${Math.min(p.r, enemyKing.r)}到${Math.max(p.r, enemyKing.r)}`)
            for (let r = Math.min(p.r, enemyKing.r) + 1;
                r < Math.max(p.r, enemyKing.r); r++) {
                if (this.get(r, p.c)) {
                    blocked = true;
                    break;
                }
            }

            if (!blocked) {
                res.push({ r: enemyKing.r, c: enemyKing.c });
            }
        }

        return res;
    }

}


// Game
export class Game {
    constructor() {
        this.board = new Board();
        this.selected = null;
        this.turn = "red";

        this.canvas = document.getElementById("board");
        this.ctx = this.canvas.getContext("2d");

        this.SIZE = 60;//棋盘格子大小，单位像素
        this.initEvents();
        this.render();
        this.enemy_ai = new EnemyAi();
    }

    initEvents() {
        //我方步进
        this.canvas.addEventListener("click", (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const c = Math.round(x / this.SIZE) - 1;
            const r = Math.round(y / this.SIZE) - 1;

            if (r < 0 || r > 9 || c < 0 || c > 8) return;

            const p = this.board.get(r, c);
            //点击选子
            if (!this.selected) {
                if (p && p.p === "red") {
                    this.selected = p;
                    this.render();
                }
                return;
            }
            //再次点击取消
            if (this.selected === p) {
                this.selected = null;
                this.render();
                return;
            }
            //走子
            const ok = this.board.move(this.selected, r, c);
            // console.log(this.selected);

            this.selected = null;
            this.render();

            if (ok)
                this.afterMove();
        });
    }

    afterMove() {

        const win = this.checkWin();

        if (win) {
            alert(win + " win!");

            this.reset();
            return;
        }

        this.turn = this.turn === "red" ? "black" : "red";
        //敌方走子
        if (this.turn === "black")
            setTimeout(() => this.step(), 300);
    }

    // 胜负判断
    checkWin() {
        const redKing = this.board.pieces.find(p => p.p === "red" && p.t === "帅");
        const blackKing = this.board.pieces.find(p => p.p === "black" && p.t === "将");

        if (!redKing) return "black";
        if (!blackKing) return "red";

        // 检查当前玩家是否还有合法走法
        for (let p of this.board.pieces) {
            if (p.p !== this.turn) continue;

            const moves = this.board.getLegalMoves(p);
            // console.log("moves:", moves);

            if (moves && moves.length > 0) {
                return null; // 还有棋可走
            }
        }

        // 没有任何合法走法 → 当前玩家失败
        return this.turn === "red" ? "black" : "red";
    }
    //敌方AI步进
    step(hander = "black") {
        let m_piece = [];
        m_piece = this.board.pieces.filter(p => p.p === hander);

        let moves = []
        let p = null
        let m = null
        if (hander == "black") {

            //接入敌方AI
            //black_state=this.board.pieces;
            //action=enemy_ai.step(black_state);

            //随机选择一个棋子执行一个动作
            //直到有路可走为止
            while (moves.length === 0) {
                p = m_piece[Math.floor(Math.random() * m_piece.length)];
                moves = this.board.getLegalMoves(p);
                m = moves[Math.floor(Math.random() * moves.length)];//{r,c}
            }
        }else if(hander == "red"){
            //我方AI
        }

        const ok = this.board.move(p, m.r, m.c);
        this.render();
        if (ok)
            this.afterMove();

    }

    reset() {
        this.board.init();
        this.selected = null;
        this.turn = "red";
        this.render();
    }

    //画棋盘
    drawBoard() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, 600, 670);

        ctx.strokeStyle = "#333";
        for (let r = 0; r < 10; r++) {
            ctx.beginPath();
            ctx.moveTo(60, 60 + r * 60);
            ctx.lineTo(540, 60 + r * 60);
            ctx.stroke();
        }

        for (let c = 0; c < 9; c++) {
            if (c == 0 || c == 8) {
                ctx.beginPath();
                ctx.moveTo(60 + c * 60, 60);
                ctx.lineTo(60 + c * 60, 600);
                ctx.stroke();
            } else {
                ctx.beginPath();
                ctx.moveTo(60 + c * 60, 60);
                ctx.lineTo(60 + c * 60, 60 * 5);
                ctx.stroke();

                ctx.beginPath();
                ctx.moveTo(60 + c * 60, 60 * 6);
                ctx.lineTo(60 + c * 60, 600);
                ctx.stroke();
            }

        }
        ctx.beginPath();
        ctx.moveTo(60 * 4, 60);
        ctx.lineTo(60 * 6, 60 * 3);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(60 * 6, 60);
        ctx.lineTo(60 * 4, 60 * 3);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(60 * 4, 60 * 10);
        ctx.lineTo(60 * 6, 60 * 8);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(60 * 6, 60 * 10);
        ctx.lineTo(60 * 4, 60 * 8);
        ctx.stroke();

    }

    //画棋子
    drawPieces() {
        const ctx = this.ctx;
        //console.log("还剩下", this.board.pieces.length, "个棋子");

        for (let p of this.board.pieces) {
            const x = (p.c + 1) * 60;
            const y = (p.r + 1) * 60;

            ctx.beginPath();
            ctx.arc(x, y, 22, 0, Math.PI * 2);
            ctx.fillStyle = "#fff";
            ctx.fill();
            ctx.stroke();

            ctx.fillStyle = p.p === "red" ? "red" : "black";
            ctx.font = "20px KaiTi";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(p.t, x, y);
            if (this.selected && this.selected === p) {
                // console.log("this.selected:",this.selected);
                // console.log("p:",p);
                ctx.save();

                ctx.beginPath();
                ctx.arc(x, y, 28, 0, Math.PI * 2);
                ctx.strokeStyle = "#00AEEF";
                ctx.lineWidth = 4;
                ctx.stroke();

                ctx.restore();
            }
        }
    }
    //画文字
    drawText() {
        const ctx = this.ctx;

        ctx.save();

        ctx.fillStyle = "#8B5A2B"; // 深棕色更像木棋盘
        ctx.font = "30px KaiTi";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";

        // 河界 y 位置（中间）
        const y = 5.5 * 60;

        // 四个字 x 分布（中间两列）
        ctx.fillText("楚", 2.5 * 60, y);
        ctx.fillText("河", 3.5 * 60, y);
        ctx.fillText("汉", 6.5 * 60, y);
        ctx.fillText("界", 7.5 * 60, y);

        ctx.restore();
    }

    //画标记
    drawCornerMark(x, y, size = 10) {
        const ctx = this.ctx

        ctx.beginPath();
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 2;

        // 左上角
        ctx.moveTo(x - size, y - size);
        ctx.lineTo(x - size, y - size / 2);
        ctx.moveTo(x - size, y - size);
        ctx.lineTo(x - size / 2, y - size);

        // 右上角
        ctx.moveTo(x + size, y - size);
        ctx.lineTo(x + size, y - size / 2);
        ctx.moveTo(x + size, y - size);
        ctx.lineTo(x + size / 2, y - size);

        // 左下角
        ctx.moveTo(x - size, y + size);
        ctx.lineTo(x - size, y + size / 2);
        ctx.moveTo(x - size, y + size);
        ctx.lineTo(x - size / 2, y + size);

        // 右下角
        ctx.moveTo(x + size, y + size);
        ctx.lineTo(x + size, y + size / 2);
        ctx.moveTo(x + size, y + size);
        ctx.lineTo(x + size / 2, y + size);

        ctx.stroke();
    };
    drawMark() {
        const ctx = this.ctx;

        ctx.save();
        ctx.fillStyle = "#333";

        const drawDot = (x, y) => {
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fill();
        };

        // 炮位（两边各2个）
        const cannonRows = [2, 7];
        const cannonCols = [1, 7];

        for (let r of cannonRows) {
            for (let c of cannonCols) {
                this.drawCornerMark((c + 1) * 60, (r + 1) * 60);
            }
        }

        // 兵位（五列）
        const pawnRows = [3, 6];
        const pawnCols = [0, 2, 4, 6, 8];

        for (let r of pawnRows) {
            for (let c of pawnCols) {
                this.drawCornerMark((c + 1) * 60, (r + 1) * 60);
            }
        }

        ctx.restore();
    }

    render() {
        const ctx = this.ctx;

        ctx.clearRect(0, 0, this.width, this.height);

        this.drawBoard();
        this.drawMark();
        this.drawText();

        this.drawPieces();

    }
}
