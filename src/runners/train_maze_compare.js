import { getTrainingRegistration } from "../envs/registry.js";
import { createEpisodeMetrics } from "../metrics/episode_metrics.js";
import { createEpisodeStepRecord, createEpisodeTrace } from "../envs/shared/episodeTrace.js";
import { createSharedMazeMap, getMazeActionLabels } from "../envs/maze/maze_shared_env.js";

const ACTION_LABELS = getMazeActionLabels();

function toDisplayLabel(label) {
  return label
    .replace("Up", "上")
    .replace("Down", "下")
    .replace("Left", "左")
    .replace("Right", "右");
}

function sanitizePolicyTopActions(entries = []) {
  return entries.map((entry) => ({
    actionId: entry.actionId,
    label: entry.label ? toDisplayLabel(entry.label) : toDisplayLabel(ACTION_LABELS[entry.actionId] ?? `a=${entry.actionId}`),
    probability: entry.probability ?? 0,
  }));
}

async function rolloutEpisode(env, agent, episodeIndex, options = {}) {
  const deterministic = Boolean(options.deterministic);
  const onStep = typeof options.onStep === "function" ? options.onStep : null;
  let state = env.reset();
  let totalReward = 0;
  let done = false;
  let success = false;
  let truncated = false;
  const trace = [];
  const trajectory = [];
  const path = [{ x: state.x, y: state.y }];

  while (!done) {
    const decision = await agent.selectAction(state, { deterministic });
    const transition = env.step(decision.action);
    totalReward += transition.reward;
    success = transition.success;
    truncated = transition.truncated;

    const stepRecord = createEpisodeStepRecord({
      index: trace.length,
      stateBefore: { ...transition.info.stateBefore },
      stateAfter: { ...transition.info.stateAfter },
      actionId: decision.action,
      actionLabel: toDisplayLabel(decision.actionLabel),
      selectedAction: decision.selectedAction
        ? {
          ...decision.selectedAction,
          label: toDisplayLabel(decision.selectedAction.label),
        }
        : null,
      reward: transition.reward,
      rewardBreakdown: transition.info.rewardBreakdown,
      policyTopActions: sanitizePolicyTopActions(decision.policyTopActions),
      entropy: decision.entropy ?? null,
      done: transition.done,
      info: {
        detail: transition.success ? "goal reached" : transition.truncated ? "max steps reached" : transition.info.hitWall ? "hit wall" : transition.info.isBacktrack ? "backtrack" : "move",
        pathSoFar: transition.info.pathSoFar,
        hitWall: transition.info.hitWall,
        isBacktrack: transition.info.isBacktrack,
      },
    });

    trace.push(stepRecord);
    path.push({ x: transition.state.x, y: transition.state.y });
    trajectory.push({
      state,
      nextState: transition.state,
      action: decision.action,
      reward: transition.reward,
      done: transition.done,
      logProb: decision.logProb,
      value: decision.value,
      stateVector: decision.stateVector ?? env.toFeatureVector(state),
    });

    if (typeof agent.observe === "function") {
      agent.observe({
        state,
        nextState: transition.state,
        action: decision.action,
        reward: transition.reward,
        done: transition.done,
      });
    }

    if (onStep) {
      await onStep(stepRecord, {
        episodeIndex,
        path: path.map((point) => ({ ...point })),
        done: transition.done,
        success: transition.success,
        truncated: transition.truncated,
      });
    }

    state = transition.state;
    done = transition.done;
  }

  const updateInfo = typeof agent.finishEpisode === "function"
    ? await agent.finishEpisode(trajectory, { episodeIndex, env })
    : {};

  const episodeTrace = createEpisodeTrace({
    envType: "maze",
    episode: episodeIndex,
    totalReward,
    totalSteps: trace.length,
    summary: {
      outcome: success ? "success" : truncated ? "truncated" : "failure",
      successLabel: success ? "Reached Goal" : truncated ? "Max Steps" : "Failed",
    },
    metadata: {
      mazeGrid: env.serializeGrid(),
      start: { ...env.start },
      goal: { ...env.goal },
    },
    steps: trace,
  });

  return createEpisodeMetrics({
    algorithm: agent.constructor.name,
    episodeIndex,
    path,
    steps: trace.length,
    reward: totalReward,
    success,
    pathLength: path.length,
    trace,
    updateInfo: {
      ...updateInfo,
      truncated,
    },
    metadata: {
      episodeTrace,
    },
  });
}

export function makeSharedMazeConfig(options = {}) {
  return {
    cols: Number.isFinite(options.cols) ? options.cols : 21,
    rows: Number.isFinite(options.rows) ? options.rows : 21,
    maxSteps: Number.isFinite(options.maxSteps) ? options.maxSteps : 300,
    loopRate: Number.isFinite(options.loopRate) ? options.loopRate : 0.3,
  };
}

export function createSharedMazeSnapshot(options = {}) {
  const envConfig = makeSharedMazeConfig(options);
  const grid = createSharedMazeMap(envConfig);
  return {
    envConfig,
    grid,
  };
}

export async function runMazeTraining({
  mode = "q_learning",
  episodes = 80,
  envConfig = {},
  mapSnapshot = null,
  agentConfig = {},
  onEpisode = null,
  onStep = null,
} = {}) {
  const registration = getTrainingRegistration("maze", mode);
  if (!registration) {
    throw new Error(`Unsupported maze mode: ${mode}`);
  }

  const sharedConfig = makeSharedMazeConfig(envConfig);
  const grid = Array.isArray(mapSnapshot?.grid)
    ? mapSnapshot.grid
    : createSharedMazeMap(sharedConfig);

  const env = registration.createEnvironment({
    ...sharedConfig,
    grid,
  });
  const agent = registration.createAgent(env, agentConfig);
  const history = [];

  for (let episodeIndex = 0; episodeIndex < episodes; episodeIndex++) {
    const episode = await rolloutEpisode(env, agent, episodeIndex, {
      deterministic: false,
      onStep: typeof onStep === "function"
        ? async (stepRecord, stepContext) => {
          await onStep(stepRecord, {
            mode,
            episodeIndex,
            episodes,
            history,
            grid,
            ...stepContext,
          });
        }
        : null,
    });
    history.push(episode);
    if (typeof onEpisode === "function") {
      await onEpisode(episode, {
        mode,
        episodeIndex,
        episodes,
        history,
        grid,
      });
    }
    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  return {
    mode,
    history,
    mapSnapshot: {
      envConfig: sharedConfig,
      grid,
    },
  };
}

export async function runMazeComparison({
  episodes = 80,
  envConfig = {},
  qLearningConfig = {},
  ppoConfig = {},
  onUpdate = null,
  onStep = null,
} = {}) {
  const mapSnapshot = createSharedMazeSnapshot(envConfig);
  const histories = {
    q_learning: [],
    ppo: [],
  };

  const qResult = await runMazeTraining({
    mode: "q_learning",
    episodes,
    envConfig: mapSnapshot.envConfig,
    mapSnapshot,
    agentConfig: qLearningConfig,
    onEpisode: async (episode, context) => {
      histories.q_learning = context.history;
      if (onUpdate) {
        await onUpdate({
          histories,
          mapSnapshot,
          latest: {
            algorithm: "q_learning",
            episode,
          },
        });
      }
    },
    onStep: async (stepRecord, stepContext) => {
      if (onStep) {
        await onStep(stepRecord, {
          algorithm: "q_learning",
          histories,
          mapSnapshot,
          ...stepContext,
        });
      }
    },
  });

  const ppoResult = await runMazeTraining({
    mode: "ppo",
    episodes,
    envConfig: mapSnapshot.envConfig,
    mapSnapshot,
    agentConfig: ppoConfig,
    onEpisode: async (episode, context) => {
      histories.ppo = context.history;
      if (onUpdate) {
        await onUpdate({
          histories,
          mapSnapshot,
          latest: {
            algorithm: "ppo",
            episode,
          },
        });
      }
    },
    onStep: async (stepRecord, stepContext) => {
      if (onStep) {
        await onStep(stepRecord, {
          algorithm: "ppo",
          histories,
          mapSnapshot,
          ...stepContext,
        });
      }
    },
  });

  return {
    mapSnapshot,
    results: {
      q_learning: qResult,
      ppo: ppoResult,
    },
    histories,
  };
}
