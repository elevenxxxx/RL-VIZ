export class EnemyAi {
    constructor(depth = 2) {
        this.depth = depth;
        this.pieceValue = {
            "将": 100000,
            "帅": 100000,
            "车": 900,
            "马": 450,
            "炮": 450,
            "象": 200,
            "相": 200,
            "仕": 200,
            "士": 200,
            "卒": 100,
            "兵": 100,
        };
    }

    step(board, side = "black") {
        return this.chooseMove(board, side);
    }

    chooseMove(board, side = "black") {
        const moves = this.orderMoves(board.getAllLegalMoves(side), board);
        if (moves.length === 0) return null;

        let bestMove = moves[0];
        let bestScore = -Infinity;
        let alpha = -Infinity;
        let beta = Infinity;

        for (let move of moves) {
            const nextBoard = this.applyMove(board, move);
            const score = this.alphaBeta(
                nextBoard,
                this.depth - 1,
                alpha,
                beta,
                this.opponent(side)
            );

            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }

            alpha = Math.max(alpha, bestScore);
        }

        return bestMove;
    }

    alphaBeta(board, depth, alpha, beta, side) {
        if (depth === 0 || this.isTerminal(board)) {
            return this.evaluate(board);
        }

        const moves = this.orderMoves(board.getAllLegalMoves(side), board);
        if (moves.length === 0) {
            return this.evaluate(board);
        }

        if (side === "black") {
            let value = -Infinity;

            for (let move of moves) {
                const nextBoard = this.applyMove(board, move);
                value = Math.max(
                    value,
                    this.alphaBeta(nextBoard, depth - 1, alpha, beta, "red")
                );
                alpha = Math.max(alpha, value);
                if (alpha >= beta) break;
            }

            return value;
        }

        let value = Infinity;

        for (let move of moves) {
            const nextBoard = this.applyMove(board, move);
            value = Math.min(
                value,
                this.alphaBeta(nextBoard, depth - 1, alpha, beta, "black")
            );
            beta = Math.min(beta, value);
            if (alpha >= beta) break;
        }

        return value;
    }

    applyMove(board, move) {
        const nextBoard = board.clone();
        const piece = nextBoard.getPbyId(move.pieceId);
        nextBoard.move(piece, move.toR, move.toC);
        return nextBoard;
    }

    isTerminal(board) {
        const redKing = board.pieces.find(p => p.p === "red" && p.t === "帅");
        const blackKing = board.pieces.find(p => p.p === "black" && p.t === "将");
        return !redKing || !blackKing;
    }

    evaluate(board) {
        const redKing = board.pieces.find(p => p.p === "red" && p.t === "帅");
        const blackKing = board.pieces.find(p => p.p === "black" && p.t === "将");

        if (!redKing) return 1000000;
        if (!blackKing) return -1000000;

        let score = 0;

        for (let piece of board.pieces) {
            const base = this.pieceValue[piece.t] || 0;
            let bonus = 0;

            if (piece.t === "卒") {
                bonus += piece.r * 8;
            } else if (piece.t === "兵") {
                bonus += (9 - piece.r) * 8;
            }

            if (piece.p === "black") {
                score += base + bonus;
            } else {
                score -= base + bonus;
            }
        }

        const blackMobility = board.getAllLegalMoves("black", false).length;
        const redMobility = board.getAllLegalMoves("red", false).length;
        score += (blackMobility - redMobility) * 2;

        if (board.isInCheck("red")) score += 50;
        if (board.isInCheck("black")) score -= 50;

        return score;
    }

    orderMoves(moves, board) {
        return [...moves].sort((a, b) => this.moveScore(board, b) - this.moveScore(board, a));
    }

    moveScore(board, move) {
        const target = board.get(move.toR, move.toC);
        if (!target) return 0;

        const attacker = board.getPbyId(move.pieceId);
        const gain = this.pieceValue[target.t] || 0;
        const risk = attacker ? (this.pieceValue[attacker.t] || 0) : 0;

        return gain * 10 - risk;
    }

    opponent(side) {
        return side === "black" ? "red" : "black";
    }
}
