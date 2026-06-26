const noop = () => {};

globalThis.window = { addEventListener() {} };
globalThis.alert = noop;
globalThis.echarts = {
  init: () => ({ setOption: noop, resize: noop })
};

const ctx = {
  clearRect: noop,
  beginPath: noop,
  moveTo: noop,
  lineTo: noop,
  stroke: noop,
  arc: noop,
  fill: noop,
  fillText: noop,
  save: noop,
  restore: noop,
};

const canvas = {
  getContext: () => ctx,
  addEventListener: noop,
  getBoundingClientRect: () => ({ left: 0, top: 0 }),
};

globalThis.document = {
  getElementById(id) {
    if (id === "board") return canvas;
    if (id === "graph-container") return {};
    return null;
  }
};

const tf = await import("@tensorflow/tfjs");
const { initMap, encode_state } = await import("../src/envs/chess/utils.js");
const { Game } = await import("../src/envs/chess/env.js");
const { Agent } = await import("../src/envs/chess/agent.js");
const originalConsole = { ...console };
console.log = noop;
console.warn = noop;
console.info = noop;
console.error = noop;

function countNonZero(values) {
  let count = 0;
  for (const value of values) {
    if (value !== 0) count++;
  }
  return count;
}

function flatten(values) {
  return Array.isArray(values) ? values.flat(Infinity) : [values];
}

function snapshotActor(agent) {
  return {
    fc1Kernel: agent.actor.fc1.getWeights()[0].arraySync(),
    fc1Bias: agent.actor.fc1.getWeights()[1].arraySync(),
    meanKernel: agent.actor.mean.getWeights()[0].arraySync(),
    meanBias: agent.actor.mean.getWeights()[1].arraySync(),
    logStd: agent.actor.logStd.arraySync(),
  };
}

function ensureNetworksBuilt(agent) {
  if (agent.actor.fc1.getWeights().length > 0 && agent.critic.fc1.getWeights().length > 0) {
    return;
  }

  tf.tidy(() => {
    const sampleState = tf.tensor2d([encode_state(initMap)], [1, 1260]);
    agent.actor.forward(sampleState);
    agent.critic.value(sampleState);
  });
}

function diffStats(before, after) {
  const lhs = flatten(before);
  const rhs = flatten(after);
  let l1 = 0;
  let l2 = 0;
  let maxAbs = 0;

  for (let i = 0; i < lhs.length; i++) {
    const diff = rhs[i] - lhs[i];
    const abs = Math.abs(diff);
    l1 += abs;
    l2 += diff * diff;
    if (abs > maxAbs) maxAbs = abs;
  }

  return {
    l1,
    l2: Math.sqrt(l2),
    maxAbs,
  };
}

function aggregateActorDiff(before, after) {
  const keys = Object.keys(before);
  return keys.reduce((acc, key) => {
    const diff = diffStats(before[key], after[key]);
    acc.l1 += diff.l1;
    acc.l2 += diff.l2;
    acc.maxAbs = Math.max(acc.maxAbs, diff.maxAbs);
    return acc;
  }, { l1: 0, l2: 0, maxAbs: 0 });
}

async function collectSingleUpdateDiff() {
  const env = new Game();
  const agent = new Agent(env);
  agent.env.render_mode = "none";
  ensureNetworksBuilt(agent);

  let rollout = await agent.RunGame();
  if (!rollout) {
    throw new Error("RunGame returned null during single-update validation.");
  }
  if (rollout.length > 100) {
    rollout = rollout.slice(0, 100);
  }

  const before = snapshotActor(agent);
  const metrics = await agent.updateModel(rollout);
  const after = snapshotActor(agent);

  return {
    metrics,
    diff: aggregateActorDiff(before, after),
    bufferLength: rollout.length,
  };
}

async function collectTrainingHistory(episodes = 50) {
  const env = new Game();
  const agent = new Agent(env);
  agent.env.render_mode = "none";
  ensureNetworksBuilt(agent);

  const history = [];
  const startWeights = snapshotActor(agent);
  const beforeEval = await agent.evaluate(20);

  for (let episode = 0; episode < episodes; episode++) {
    env.reset();
    let rollout = await agent.RunGame();
    if (!rollout) {
      throw new Error(`RunGame returned null at episode ${episode}.`);
    }
    if (rollout.length > 100) {
      rollout = rollout.slice(0, 100);
    }

    let metrics = {};
    for (let step = 0; step < 5; step++) {
      metrics = await agent.updateModel(rollout);
    }

    history.push({
      episode,
      reward: rollout.reduce((sum, item) => sum + item.reward, 0),
      lossActor: metrics.lossActor,
      lossCritic: metrics.lossCritic,
      advMean: metrics.advMean,
      bufferLen: rollout.length,
    });
  }

  const endWeights = snapshotActor(agent);
  const afterEval = await agent.evaluate(20);
  const rewards = history.map(item => item.reward);
  const first100 = rewards.slice(0, 100);
  const last100 = rewards.slice(-100);
  const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;

  return {
    history,
    actorDiff: aggregateActorDiff(startWeights, endWeights),
    beforeEval,
    afterEval,
    first100Mean: first100.length > 0 ? mean(first100) : null,
    last100Mean: last100.length > 0 ? mean(last100) : null,
  };
}

const encoded = encode_state(initMap);
const nonZeroCount = countNonZero(encoded);
if (nonZeroCount <= 0) {
  throw new Error("encode_state(initMap) still produced no non-zero features.");
}

const singleUpdate = await collectSingleUpdateDiff();
if (singleUpdate.diff.maxAbs === 0) {
  throw new Error("Actor parameters did not change after updateModel().");
}

const trainingRun = await collectTrainingHistory(200);
if (trainingRun.actorDiff.maxAbs === 0) {
  throw new Error("Actor parameters did not change across training history.");
}

originalConsole.log(JSON.stringify({
  encodeState: {
    nonZeroCount,
    featureSize: encoded.length,
  },
  singleUpdate,
  trainingRun: {
    episodes: trainingRun.history.length,
    actorDiff: trainingRun.actorDiff,
    beforeEval: trainingRun.beforeEval,
    afterEval: trainingRun.afterEval,
    first100Mean: trainingRun.first100Mean,
    last100Mean: trainingRun.last100Mean,
    rewardHead: trainingRun.history.slice(0, 5).map(item => item.reward),
    rewardTail: trainingRun.history.slice(-5).map(item => item.reward),
    historyTail: trainingRun.history.slice(-5),
  },
}, null, 2));
