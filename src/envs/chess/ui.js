const { Game } = await import("./env.js");
const { piece2id, rc2num, num2rc, encode_action } = await import("./utils.js");
const { Agent } = await import("./agent.js");
const { drawlineGraph } = await import("./graph.js");
const { createTrainingRecord } = await import("../shared/trainingDiagnostics.js");
const { createEpisodeViewer } = await import("../shared/episodeViewer.js");

const env = new Game();
let agent = new Agent(env);
let trainingActive = false;
const episodeViewer = createEpisodeViewer({
    containerId: "episode-viewer-root",
    title: "Single Episode RL Visualization",
    subtitle: "Replay one Chinese Chess PPO episode with rollout, policy top actions, and reward breakdown.",
});

function formatNumber(value, digits = 3) {
    if (value === null || value === undefined) return "--";
    if (typeof value !== "number" || !Number.isFinite(value)) return String(value);
    return value.toFixed(digits);
}

function formatElapsed(ms = 0) {
    const totalSeconds = Math.max(0, Math.floor(ms / 1000));
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function setTrainButtonDisabled(disabled) {
    const trainBtn = document.getElementById("trainBtn");
    if (!trainBtn) return;
    trainBtn.disabled = disabled;
    trainBtn.textContent = disabled ? "训练中..." : "训练模型";
}

function updateTrainingStatus({
    state = "Idle",
    currentEpisode = 0,
    totalEpisodes = 0,
    reward = null,
    actorLoss = null,
    criticLoss = null,
    actorDelta = null,
    ratioMean = null,
    elapsedMs = 0,
    etaMs = null,
    notice = "",
    errorMessage = "",
} = {}) {
    const progress = totalEpisodes > 0
        ? Math.min(100, (currentEpisode / totalEpisodes) * 100)
        : 0;

    document.getElementById("statusState").textContent = state;
    document.getElementById("statusEpisode").textContent = `${currentEpisode} / ${totalEpisodes}`;
    document.getElementById("statusProgress").textContent = `${progress.toFixed(1)}%`;
    document.getElementById("statusProgressBar").style.width = `${progress}%`;
    document.getElementById("statusReward").textContent = formatNumber(reward, 3);
    document.getElementById("statusActorLoss").textContent = formatNumber(actorLoss, 4);
    document.getElementById("statusCriticLoss").textContent = formatNumber(criticLoss, 4);
    document.getElementById("statusActorDelta").textContent = formatNumber(actorDelta, 4);
    document.getElementById("statusRatioMean").textContent = formatNumber(ratioMean, 4);
    document.getElementById("statusElapsed").textContent = formatElapsed(elapsedMs);
    document.getElementById("statusEta").textContent = etaMs === null ? "--:--" : formatElapsed(etaMs);
    document.getElementById("trainingNotice").textContent = errorMessage ? `${notice} ${errorMessage}`.trim() : notice;
}

function resetTrainingStatus() {
    updateTrainingStatus({
        state: "Idle",
        currentEpisode: 0,
        totalEpisodes: 0,
        reward: null,
        actorLoss: null,
        criticLoss: null,
        actorDelta: null,
        ratioMean: null,
        elapsedMs: 0,
        etaMs: null,
        notice: "当前未训练。",
    });
}

export function reset() {
    env.reset();
}

export async function train() {
    if (trainingActive) return;

    trainingActive = true;
    const totalEpisodes = 500;
    const startedAt = performance.now();

    setTrainButtonDisabled(true);
    episodeViewer.clear();
    updateTrainingStatus({
        state: "Training",
        currentEpisode: 0,
        totalEpisodes,
        elapsedMs: 0,
        etaMs: null,
        notice: "训练中，请稍等，图表和指标会逐步刷新。",
    });

    try {
        const history = await agent.train({
            episodes: totalEpisodes,
            chartUpdateInterval: 5,
            onProgress: (record, progressInfo) => {
                const elapsedMs = performance.now() - startedAt;
                const completedEpisodes = progressInfo.episode + 1;
                const etaMs = completedEpisodes > 0 && reportedTotalOrFallback(progressInfo.totalEpisodes, totalEpisodes) > completedEpisodes
                    ? (elapsedMs / completedEpisodes) * (reportedTotalOrFallback(progressInfo.totalEpisodes, totalEpisodes) - completedEpisodes)
                    : 0;

                updateTrainingStatus({
                    state: "Training",
                    currentEpisode: completedEpisodes,
                    totalEpisodes: reportedTotalOrFallback(progressInfo.totalEpisodes, totalEpisodes),
                    reward: record.reward,
                    actorLoss: record.actorLoss,
                    criticLoss: record.criticLoss,
                    actorDelta: record.actorParamDelta,
                    ratioMean: record.ratioMean,
                    elapsedMs,
                    etaMs,
                    notice: "训练中，请稍等，图表和指标会逐步刷新。",
                });
            },
            onEpisode: ({ episodeTrace }) => {
                if (episodeTrace) {
                    episodeViewer.pushEpisode(episodeTrace);
                }
            },
        });

        const lastRecord = history?.at(-1) ?? null;
        updateTrainingStatus({
            state: "Finished",
            currentEpisode: totalEpisodes,
            totalEpisodes,
            reward: lastRecord?.reward ?? null,
            actorLoss: lastRecord?.actorLoss ?? null,
            criticLoss: lastRecord?.criticLoss ?? null,
            actorDelta: lastRecord?.actorParamDelta ?? null,
            ratioMean: lastRecord?.ratioMean ?? null,
            elapsedMs: performance.now() - startedAt,
            etaMs: 0,
            notice: "训练完成，状态面板和图表已刷新。",
        });
    } catch (error) {
        console.error(error);
        updateTrainingStatus({
            state: "Error",
            currentEpisode: 0,
            totalEpisodes,
            elapsedMs: performance.now() - startedAt,
            etaMs: null,
            notice: "训练出错。",
            errorMessage: String(error?.message ?? error),
        });
    } finally {
        trainingActive = false;
        setTrainButtonDisabled(false);
    }
}
export async function test() {
    await agent.test();
}
export async function load() {

}
export async function save() {

}
export function drawGraph() {
    let history = [];
    for (let i = 0; i < 800; i++) {
        const outcome = i % 7 === 0 ? "loss" : i % 11 === 0 ? "truncated" : "win";
        const reward = outcome === "win"
            ? 6 + Math.random() * 4
            : outcome === "loss"
                ? -8 + Math.random() * 4
                : -1 + Math.random() * 3;
        const actionCounts = [
            { actionId: 12, count: 5 + Math.floor(Math.random() * 5), label: "车1:+9" },
            { actionId: 31, count: 3 + Math.floor(Math.random() * 4), label: "马1:+17" },
            { actionId: 48, count: 2 + Math.floor(Math.random() * 4), label: "炮1:-9" },
        ];
        history.push(createTrainingRecord({
            episode: i,
            reward,
            steps: 40 + Math.floor(Math.random() * 60),
            success: outcome === "win",
            win: outcome === "win",
            outcome,
            actorLoss: Math.random() * 3,
            criticLoss: Math.random() * 8,
            actorParamDelta: Math.random() * 0.3,
            criticParamDelta: Math.random() * 0.4,
            ratioMean: 0.8 + Math.random() * 0.4,
            ratioMin: Math.random() * 0.6,
            ratioMax: 1 + Math.random() * 0.8,
            entropy: 0.3 + Math.random() * 1.2,
            advantageMean: (Math.random() - 0.5) * 0.3,
            advantageStd: 0.8 + Math.random() * 0.4,
            stateNonzeroCount: 20 + Math.random() * 30,
            legalActions: 25 + Math.random() * 20,
            totalActions: 188,
            legalRate: 0.12 + Math.random() * 0.18,
            invalidAttempts: Math.floor(Math.random() * 3),
            validAttempts: 10 + Math.floor(Math.random() * 15),
            invalidPerValid: Math.random() * 0.2,
            repeatRate: Math.random() * 0.3,
            repeatMoveRate: Math.random() * 0.35,
            attackMovesRate: 0.15 + Math.random() * 0.3,
            actionLoopCount: Math.floor(Math.random() * 4),
            actionCounts,
            topActions: actionCounts,
        }));
    }
    drawlineGraph(history, {
        title: "Chinese Chess PPO Diagnostics",
        subtitle: "Sample diagnostics data for dashboard layout testing.",
        mode: "ppo",
        movingAverageWindow: 20,
    });
}

resetTrainingStatus();

function reportedTotalOrFallback(reportedTotal, fallbackTotal) {
    return typeof reportedTotal === "number" && reportedTotal > 0 ? reportedTotal : fallbackTotal;
}
// reset();
// // 挂到全局
// window.reset = reset;
// window.train = train;
// window.test = test;
// window.load = load;
// window.save = save;
// window.drawGraph = drawGraph;
