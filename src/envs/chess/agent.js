import * as tf from "@tensorflow/tfjs";
const { encode_state, decode_action, id2piece, num2rc } = await import("./utils.js");
const { drawlineGraph } = await import("./graph.js");
const { createTrainingRecord } = await import("../shared/trainingDiagnostics.js");
const { createEpisodeTrace, createEpisodeStepRecord } = await import("../shared/episodeTrace.js");

function flattenWeights(values) {
  return Array.isArray(values) ? values.flat(Infinity) : [values];
}

function captureTensorValues(tensor) {
  return Array.from(tensor.dataSync());
}

function captureActorParams(actor) {
  return {
    fc1Kernel: actor.fc1.getWeights()[0].arraySync(),
    fc1Bias: actor.fc1.getWeights()[1].arraySync(),
    logitsKernel: actor.logits.getWeights()[0].arraySync(),
    logitsBias: actor.logits.getWeights()[1].arraySync(),
  };
}

function captureCriticParams(critic) {
  return {
    fc1Kernel: critic.fc1.getWeights()[0].arraySync(),
    fc1Bias: critic.fc1.getWeights()[1].arraySync(),
    fc2Kernel: critic.fc2.getWeights()[0].arraySync(),
    fc2Bias: critic.fc2.getWeights()[1].arraySync(),
  };
}

function computeParamDelta(before, after) {
  let total = 0;
  for (const key of Object.keys(before)) {
    const lhs = flattenWeights(before[key]);
    const rhs = flattenWeights(after[key]);
    for (let i = 0; i < lhs.length; i++) {
      const diff = rhs[i] - lhs[i];
      total += diff * diff;
    }
  }
  return Math.sqrt(total);
}

function countNonZeroFeatures(encodedState) {
  let total = 0;
  for (const value of encodedState) {
    if (value !== 0) total++;
  }
  return total;
}

function createLegalMaskArray(legalActionIds, totalActions) {
  const mask = new Array(totalActions).fill(0);
  for (const actionId of legalActionIds) {
    mask[actionId] = 1;
  }
  return mask;
}

function meanOf(values) {
  if (!values || values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function formatSignedOffset(offset) {
  return offset > 0 ? `+${offset}` : `${offset}`;
}

function formatActionDebugLabel(actionId) {
  const [pieceId, offset] = decode_action(actionId);
  const pieceInfo = id2piece(pieceId);
  if (!pieceInfo) {
    return `action:${actionId}`;
  }

  const [, pieceName, pieceNum] = pieceInfo;
  const pieceLabel = pieceNum > 0 ? `${pieceName}${pieceNum}` : pieceName;
  return `${pieceLabel}:${formatSignedOffset(offset)}`;
}

function formatChessCoordinate(num) {
  if (typeof num !== "number" || num < 0 || num > 89) return "--";
  const [r, c] = num2rc(num);
  return `(${r + 1},${c + 1})`;
}

function describeChessAction(stateBefore, stateAfter, actionId) {
  const [pieceId] = decode_action(actionId);
  const pieceInfo = id2piece(pieceId);
  const fallbackLabel = formatActionDebugLabel(actionId);
  if (!pieceInfo || !Array.isArray(stateBefore) || pieceId < 0 || pieceId >= stateBefore.length) {
    return {
      label: fallbackLabel,
      move: null,
      detail: fallbackLabel,
    };
  }

  const [, pieceName, pieceNum] = pieceInfo;
  const pieceLabel = pieceNum > 0 ? `${pieceName}${pieceNum}` : pieceName;
  const fromNum = stateBefore[pieceId];
  const toNum = Array.isArray(stateAfter) && pieceId < stateAfter.length ? stateAfter[pieceId] : fromNum;
  const fromCoord = typeof fromNum === "number" && fromNum !== 90 ? num2rc(fromNum) : null;
  const toCoord = typeof toNum === "number" && toNum !== 90 ? num2rc(toNum) : fromCoord;
  const label = `${pieceLabel} ${formatChessCoordinate(fromNum)} → ${formatChessCoordinate(toNum)}`;

  return {
    label,
    move: fromCoord && toCoord
      ? {
        pieceId,
        pieceLabel,
        from: { r: fromCoord[0], c: fromCoord[1] },
        to: { r: toCoord[0], c: toCoord[1] },
      }
      : null,
    detail: `${pieceLabel} from ${formatChessCoordinate(fromNum)} to ${formatChessCoordinate(toNum)}`,
  };
}

function summarizeActionFrequency(actionCounts, totalMoves, topK = 5) {
  const entries = Array.from(actionCounts.entries())
    .sort((lhs, rhs) => rhs[1] - lhs[1] || lhs[0] - rhs[0]);

  return {
    actionCounts: entries.map(([actionId, count]) => ({
      actionId,
      count,
      rate: totalMoves > 0 ? count / totalMoves : 0,
      label: formatActionDebugLabel(actionId),
    })),
    topActions: entries.slice(0, topK).map(([actionId, count]) => ({
      actionId,
      count,
      rate: totalMoves > 0 ? count / totalMoves : 0,
      label: formatActionDebugLabel(actionId),
    })),
  };
}

function summarizeActionSpaceStats(legalActionCounts, totalActions, invalidAttempts, validAttempts, extra = {}) {
  const legalActions = meanOf(legalActionCounts);
  return {
    legalActions,
    totalActions,
    legalRate: totalActions > 0 ? legalActions / totalActions : 0,
    invalidAttempts,
    validAttempts,
    invalidPerValid: validAttempts > 0 ? invalidAttempts / validAttempts : invalidAttempts,
    repeatRate: extra.repeatRate ?? 0,
    actionLoopCount: extra.actionLoopCount ?? 0,
    attackMovesRate: extra.attackMovesRate ?? 0,
    repeatMoveRate: extra.repeatMoveRate ?? 0,
    actionCounts: extra.actionCounts ?? [],
    topActions: extra.topActions ?? [],
  };
}
class ActorNet {
  constructor(stateDim, hiddenDim = 256, actionDim = 188) {
    this.fc1 = tf.layers.dense({
      units: hiddenDim,
      activation: "relu",
      inputShape: [stateDim],
      kernelInitializer: 'glorotNormal',  // 明确指定初始化器
      biasInitializer: 'randomNormal'
    });

    this.logits = tf.layers.dense({
      units: actionDim,
      kernelInitializer: 'glorotNormal',  // 明确指定初始化器
      biasInitializer: 'randomNormal'
    });
    this.actionDim = actionDim;
  }

  forward(state) {
    return tf.tidy(() => {
      const x = this.fc1.apply(state);
      return this.logits.apply(x);
    });
  }

  maskLogits(logits, legalMask) {
    const invalidFill = tf.fill(logits.shape, -1e9);
    return tf.where(legalMask.cast("bool"), logits, invalidFill);
  }

  logProbFromAction(actionTensor, logits, legalMask) {
    return tf.tidy(() => {
      const maskedLogits = this.maskLogits(logits, legalMask);
      const logProbs = tf.logSoftmax(maskedLogits);
      const oneHot = tf.oneHot(actionTensor.toInt(), this.actionDim);
      return tf.sum(tf.mul(logProbs, oneHot), -1);
    });
  }

  sampleAction(state, legalMask) {
    return tf.tidy(() => {
      const logits = this.forward(state);
      const maskedLogits = this.maskLogits(logits, legalMask);
      const action = tf.multinomial(maskedLogits, 1).reshape([-1]);
      const logProb = this.logProbFromAction(action, logits, legalMask);
      return { action, logProb };
    });
  }

  sampleDetermineAction(state, legalMask) {
    return tf.tidy(() => {
      const logits = this.forward(state);
      const maskedLogits = this.maskLogits(logits, legalMask);
      return tf.argMax(maskedLogits, 1);
    })
  }
}
class CriticNet {
  constructor(stateDim, hiddenDim = 256) {
    this.model = tf.sequential();
    this.fc1 = tf.layers.dense({
      units: hiddenDim,
      activation: "relu",
      inputShape: [stateDim],
      kernelInitializer: 'glorotNormal',  // 明确指定初始化器
      biasInitializer: 'randomNormal'
    })
    this.model.add(this.fc1);
    this.fc2 = tf.layers.dense({
      units: 1,
      kernelInitializer: 'glorotNormal',
      biasInitializer: 'randomNormal'
    })
    this.model.add(this.fc2);
  }

  value(state) {
    //return this.model.predict(state);
    return this.model.apply(state)
  }
}
export class Agent {
  constructor(env, stateDim = 1260, actionDim = 188) {
    this.env = env;

    this.actor = new ActorNet(stateDim, 256, actionDim);
    this.critic = new CriticNet(stateDim);

    this.actionDim = actionDim;
    this.gamma = 0.99;
    this.lam = 0.95;
    this.clip = 0.2;
    this.actor_lr = 0.01;
    this.critic_lr = 0.01;

    this.actorOpt = tf.train.adam(this.actor_lr);
    //this.actorOpt = this.createClippedOptimizer(3e-4);
    this.criticOpt = tf.train.adam(this.critic_lr);
    //this.criticOpt = this.createClippedOptimizer(2e-3);
    this.entropyCoef = 0.02;//熵系数 反应探索的重视程度
  }
  // 裁剪优化度
  createClippedOptimizer(learningRate, maxGradNorm = 1.0) {
    const baseOpt = tf.train.adam(learningRate, 0.9, 0.999, 1e-8);

    return {
      minimize: (lossFn, variables) => {
        const { value, grads } = tf.variableGrads(lossFn, variables);// 计算梯度
        const clippedGrads = this.clipGradients(grads, maxGradNorm);// 裁剪梯度（按全局范数）
        baseOpt.applyGradients(clippedGrads);// 应用裁剪后的梯度
        return value;
      },
      applyGradients: (grads) => {
        const clippedGrads = this.clipGradients(grads, maxGradNorm);
        baseOpt.applyGradients(clippedGrads);
      }
    };
  }
  clipGradients(grads, maxNorm) {
    const eps = tf.scalar(1e-8);

    const filtered = [];
    const cleanGrads = {};

    for (const [name, grad] of Object.entries(grads)) {
      if (grad) {
        const g = tf.where(
          tf.isFinite(grad),
          grad,
          tf.zerosLike(grad)
        );

        cleanGrads[name] = g;
        filtered.push(g.square().sum());
      }
    }
    const totalNorm = tf.addN(filtered).sqrt();
    const scale = tf.clipByValue(
      maxNorm / totalNorm.add(eps),
      0,
      1
    );

    const clipped = {};
    for (const [name, g] of Object.entries(cleanGrads)) {
      clipped[name] = g.mul(scale);
    }

    totalNorm.dispose();
    scale.dispose();

    return clipped;
  }
  createLegalMaskTensor(legalActionIds) {
    return tf.tensor2d(
      [createLegalMaskArray(legalActionIds, this.actionDim)],
      [1, this.actionDim]
    );
  }

  logTopActionProbabilities(episodeNumber, state, legalActionIds, topK = 5) {
    const encodedState = encode_state(state);
    const legalMask = this.createLegalMaskTensor(legalActionIds);

    const topEntries = tf.tidy(() => {
      const stateTensor = tf.tensor2d([encodedState], [1, 1260]);
      const logits = this.actor.forward(stateTensor);
      const maskedLogits = this.actor.maskLogits(logits, legalMask);
      const probs = tf.softmax(maskedLogits).dataSync();

      return legalActionIds
        .map((actionId) => ({
          actionId,
          probability: probs[actionId],
          action: decode_action(actionId),
        }))
        .sort((lhs, rhs) => rhs.probability - lhs.probability)
        .slice(0, topK);
    });

    tf.dispose([legalMask]);

    console.log(`Episode ${episodeNumber}`);
    console.log("Top5动作概率");
    topEntries.forEach((entry, index) => {
      console.log(
        `${index + 1}. actionId=${entry.actionId}, prob=${entry.probability.toFixed(6)}, action=${JSON.stringify(entry.action)}`
      );
    });
  }

  capturePolicySnapshot(stateTensor, legalMaskTensor, legalActionIds, topK = 5) {
    return tf.tidy(() => {
      const logits = this.actor.forward(stateTensor);
      const maskedLogits = this.actor.maskLogits(logits, legalMaskTensor);
      const probs = Array.from(tf.softmax(maskedLogits).dataSync());
      const entropy = tf.mean(
        tf.neg(tf.sum(tf.mul(tf.softmax(maskedLogits), tf.log(tf.softmax(maskedLogits).add(1e-8))), -1))
      ).dataSync()[0];

      const probabilityByAction = {};
      const topActions = legalActionIds
        .map((actionId) => {
          const probability = probs[actionId] ?? 0;
          probabilityByAction[actionId] = probability;
          return {
            actionId,
            label: formatActionDebugLabel(actionId),
            probability,
          };
        })
        .sort((lhs, rhs) => rhs.probability - lhs.probability)
        .slice(0, topK);

      return {
        entropy,
        probabilityByAction,
        topActions,
      };
    });
  }

  async RunGame() {
    let state = this.env.reset();
    let done = false;
    const buffer = [];
    buffer.viewerSteps = [];
    let invalidAttempts = 0;
    let validAttempts = 0;
    let repeatedStates = 0;
    let actionLoopCount = 0;
    let attackMoves = 0;
    let repeatMoves = 0;
    const actionCounts = new Map();
    const legalActionCounts = [];
    let st = 0;
    while (true) {
      st++;
      let success = false;
      let nextState, reward, terminated, truncated;
      const legalActionIds = this.env.getLegalActionIds("red");
      if (legalActionIds.length === 0) {
        console.error("No legal actions available for red.");
        return null;
      }
      const legalMaskArray = createLegalMaskArray(legalActionIds, this.actionDim);
      const legalMaskTensor = tf.tensor2d([legalMaskArray], [1, this.actionDim]);

      legalActionCounts.push(legalActionIds.length);

      let actionIndex, logProbValue;
      let policySnapshot = {
        entropy: null,
        probabilityByAction: {},
        topActions: [],
      };
      let trial = 0;
      while (!success) {
        trial++;
        if (trial > 200) {
          console.error("Too many trials!")
          tf.dispose([legalMaskTensor]);
          return null;
        }

        let encoded_state = encode_state(state);
        const s = tf.tensor2d([encoded_state], [1, 1260]);
        policySnapshot = this.capturePolicySnapshot(s, legalMaskTensor, legalActionIds);
        const { action, logProb } = this.actor.sampleAction(s, legalMaskTensor);
        const sampledActionIndex = action.dataSync()[0];
        const candidates = [sampledActionIndex, ...legalActionIds.filter((candidate) => candidate !== sampledActionIndex)];
        let executedActionIndex = null;

        for (const candidate of candidates) {
          const res = await this.env.step(candidate);
          success = res[0];
          nextState = res[1];
          reward = res[2];
          terminated = res[3];
          truncated = res[4];
          if (success) {
            executedActionIndex = candidate;
            break;
          }
        }

        if (success) {
          actionIndex = executedActionIndex;
          validAttempts++;
          if (executedActionIndex === sampledActionIndex) {
            logProbValue = logProb.dataSync()[0];
          } else {
            logProbValue = tf.tidy(() => {
              const chosenAction = tf.tensor1d([executedActionIndex], "int32");
              const logits = this.actor.forward(s);
              return this.actor.logProbFromAction(chosenAction, logits, legalMaskTensor).dataSync()[0];
            });
            console.warn("[RunGame Fallback Action]", {
              sampledActionIndex,
              executedActionIndex,
              decodedSampledAction: decode_action(sampledActionIndex),
              decodedExecutedAction: decode_action(executedActionIndex),
              legalCount: legalActionIds.length,
              stateHash: this.env.stateToHash(state),
            });
          }
        } else {
          invalidAttempts++;
          console.error("[RunGame Exhausted Legal Candidates]", {
            sampledActionIndex,
            decodedSampledAction: decode_action(sampledActionIndex),
            legalCount: legalActionIds.length,
            stateHash: this.env.stateToHash(state),
          });
          success = true;
          actionIndex = sampledActionIndex;
          nextState = state.slice();
          reward = -1;
          terminated = false;
          truncated = true;
          logProbValue = logProb.dataSync()[0];
          this.env.lastTransitionMeta = {
            previousAction: this.env.previousActionIndex,
            stateHash: this.env.stateToHash(nextState),
            repeatStateOccurrence: 1,
            repeatedState: false,
            actionLoopTriggered: false,
            actionLoopPenalty: 0,
            tacticalSafetyPenalty: 0,
            terminated: false,
            truncated: true,
            doneType: "truncated",
            bootstrapState: nextState.slice(),
          };
        }
        tf.dispose([s, action, logProb]);
      }
      tf.dispose([legalMaskTensor]);

      done = terminated || truncated;
      const transitionMeta = this.env.lastTransitionMeta ?? {};
      const rawNextState = Array.isArray(transitionMeta.viewerState)
        ? transitionMeta.viewerState.slice()
        : (Array.isArray(nextState) ? nextState.slice() : []);
      const rewardRaw = typeof reward === "number" ? reward : 0;
      const actionDetail = describeChessAction(state, rawNextState, actionIndex);
      const loopPenalty =
        Number(transitionMeta.repeatedActionPenalty || 0) +
        Number(transitionMeta.actionLoopPenalty || 0) +
        Number(transitionMeta.repeatStatePenalty || 0);
      buffer.push({
        state: encode_state(state),
        action: actionIndex,
        reward: rewardRaw / 5,
        logProb: logProbValue,
        done: done ? 1 : 0,
        terminated: terminated ? 1 : 0,
        truncated: truncated ? 1 : 0,
        legalMask: legalMaskArray,
        legalActions: legalActionIds.length,
        totalActions: this.actionDim,
        previousAction: transitionMeta.previousAction ?? null,
        stateHash: transitionMeta.stateHash ?? null,
        repeatedState: transitionMeta.repeatedState ?? false,
        actionLoopTriggered: transitionMeta.actionLoopTriggered ?? false,
        tacticalSafetyPenalty: transitionMeta.tacticalSafetyPenalty ?? 0,
        attackPressureReward: transitionMeta.attackPressureReward ?? 0,
      });
      buffer.viewerSteps.push(createEpisodeStepRecord({
        index: buffer.viewerSteps.length,
        stateBefore: Array.isArray(state) ? state.slice() : [],
        stateAfter: rawNextState,
        actionId: actionIndex,
        actionLabel: actionDetail.label,
        selectedAction: {
          actionId: actionIndex,
          label: actionDetail.label,
          probability: policySnapshot.probabilityByAction[actionIndex] ?? null,
        },
        reward: rewardRaw,
        rewardBreakdown: {
          eatReward: Number(transitionMeta.eatReward || 0),
          materialReward: Number(transitionMeta.materialReward || 0),
          attackReward: Number(transitionMeta.attackPressureReward || 0),
          loopPenalty,
          safetyPenalty: Number(transitionMeta.tacticalSafetyPenalty || 0),
        },
        policyTopActions: policySnapshot.topActions,
        entropy: Number.isFinite(policySnapshot.entropy) ? policySnapshot.entropy : null,
        done,
        info: {
          detail: transitionMeta.doneType ?? "ongoing",
          legalActions: legalActionIds.length,
          move: actionDetail.move,
        },
      }));
      actionCounts.set(actionIndex, (actionCounts.get(actionIndex) ?? 0) + 1);
      if (transitionMeta.repeatedState) {
        repeatedStates++;
      }
      if (transitionMeta.actionLoopTriggered) {
        actionLoopCount++;
      }
      if ((transitionMeta.attackPressureReward ?? 0) > 0) {
        attackMoves++;
      }
      if ((transitionMeta.repeatedState ?? false) || (transitionMeta.actionLoopTriggered ?? false)) {
        repeatMoves++;
      }
      if (truncated && !terminated && Array.isArray(transitionMeta.bootstrapState)) {
        buffer.bootstrapState = transitionMeta.bootstrapState.slice();
      }
      if (buffer.length != this.env.episode) {
        console.error(`buffer length ${buffer.length} != episode ${this.env.episode}`)
      }
      state = nextState;

      if (done) break;
    }

    const actionFrequency = summarizeActionFrequency(actionCounts, buffer.length);
    buffer.stats = summarizeActionSpaceStats(
      legalActionCounts,
      this.actionDim,
      invalidAttempts,
      validAttempts,
      {
        repeatRate: buffer.length > 0 ? repeatedStates / buffer.length : 0,
        actionLoopCount,
        attackMovesRate: buffer.length > 0 ? attackMoves / buffer.length : 0,
        repeatMoveRate: buffer.length > 0 ? repeatMoves / buffer.length : 0,
        actionCounts: actionFrequency.actionCounts,
        topActions: actionFrequency.topActions,
      }
    );
    return buffer;
  }
  async PlayGame(interval = 1000) {
    let state = this.env.reset();
    let done = false;

    let st = 0;
    while (true) {
      st++;
      //console.log("step", st);
      let success = false;
      let nextState, reward, terminated, truncated;

      let trial = 0;
      while (!success) {
        trial++;
        if (trial > 500) {
          console.error("Too many trials!")
          return null;
        }
        console.info("step:", st, "第", trial, "次尝试");

        const legalActionIds = this.env.getLegalActionIds("red");
        const legalMask = this.createLegalMaskTensor(legalActionIds);
        const { actionValue } = tf.tidy(() => {
          let encoded_state = encode_state(state);
          const s = tf.tensor2d([encoded_state], [1, 1260]);
          const actionTensor = this.actor.sampleDetermineAction(s, legalMask);
          return { actionValue: actionTensor.dataSync()[0] };
        });
        tf.dispose([legalMask]);
        const candidates = [actionValue, ...legalActionIds.filter((candidate) => candidate !== actionValue)];
        for (const candidate of candidates) {
          [success, nextState, reward, terminated, truncated] = await this.env.step(candidate, 1000);
          if (success) break;
        }
      }

      state = nextState;
      done = terminated || truncated;

      await new Promise(resolve => setTimeout(resolve, interval));//1s

      if (done) break;
    }

  }
  async evaluate(games = 20) {
    this.env.render_mode = "none";
    const rewards = [];
    let wins = 0;
    let losses = 0;
    let truncations = 0;

    for (let game = 0; game < games; game++) {
      let state = this.env.reset();
      let done = false;
      let totalReward = 0;

      while (!done) {
        let success = false;
        let nextState, reward, terminated, truncated;
        let trial = 0;

        while (!success) {
          trial++;
          if (trial > 500) {
            throw new Error(`Too many evaluation trials in game ${game}`);
          }

          const legalActionIds = this.env.getLegalActionIds("red");
          const legalMask = this.createLegalMaskTensor(legalActionIds);
          const { actionValue } = tf.tidy(() => {
            const encoded_state = encode_state(state);
            const s = tf.tensor2d([encoded_state], [1, 1260]);
            const actionTensor = this.actor.sampleDetermineAction(s, legalMask);
            return { actionValue: actionTensor.dataSync()[0] };
          });
          tf.dispose([legalMask]);
          const candidates = [actionValue, ...legalActionIds.filter((candidate) => candidate !== actionValue)];
          for (const candidate of candidates) {
            [success, nextState, reward, terminated, truncated] = await this.env.step(candidate, 0);
            if (success) break;
          }
        }

        totalReward += reward / 5;
        state = nextState;
        done = terminated || truncated;

        if (done) {
          if (terminated && reward === 100) wins++;
          else if (terminated && reward === -100) losses++;
          else if (truncated) truncations++;
        }
      }

      rewards.push(totalReward);
    }

    return {
      games,
      meanReward: rewards.reduce((sum, reward) => sum + reward, 0) / rewards.length,
      wins,
      losses,
      truncations,
      rewards,
    };
  }
  computeGAE(buffer) {
    //  const values = buffer.map(b =>
    //   this.critic.value(tf.tensor([b.state])).dataSync()[0]
    // );
    console.log('buffer length:', buffer.length);
    const states = tf.tensor(buffer.map(b => b.state));
    const values = this.critic.value(states).reshape([-1]).dataSync();
    if (values.some(a => isNaN(a))) {
      console.log("values存在NaN值");
    }
    if (values.every(a => a === 0)) {
      console.log("values全0");
    }
    const advantages = new Array(buffer.length);
    let adv = 0;

    for (let i = buffer.length - 1; i >= 0; i--) {
      const reward = buffer[i].reward;
      const done = buffer[i].done;

      const v = values[i];
      const vNext = (done || i === buffer.length - 1)
        ? 0
        : values[i + 1];
      //TD差分公式
      const delta =
        reward +
        this.gamma * vNext -
        v;

      adv = delta + this.gamma * this.lam * (1 - done) * adv;

      advantages[i] = adv;

      // if (i % 100 === 0 || i === buffer.length - 1 || i === 0) {
      //   console.log(`i=${i}, reward=${reward}, done=${done}, v=${v.toFixed(4)}, vNext=${vNext.toFixed(4)}, delta=${delta.toFixed(4)}, adv=${adv.toFixed(4)}`);
      // }
    }
    // console.log('advantages raw (first 5):', advantages.slice(0, 5));
    // console.log('advantages raw (last 5):', advantages.slice(-5));
    const mean = advantages.reduce((a, b) => a + b, 0) / advantages.length;

    const std = Math.sqrt(
      advantages.reduce((a, b) => a + (b - mean) ** 2, 0) /
      advantages.length
    );
    console.log(`advantages mean=${mean.toFixed(6)}, std=${std.toFixed(6)}`);
    console.log('advantages max:', Math.max(...advantages));
    console.log('advantages min:', Math.min(...advantages));
    if (advantages.every(a => a === 0)) {
      console.log("advantages全0");
    }
    // 归一化
    const normalized = advantages.map(a => (a - mean) / (std + 1e-8));
    console.log('normalized mean:', normalized.reduce((a, b) => a + b, 0) / normalized.length);
    console.log('normalized std:', Math.sqrt(normalized.reduce((a, b) => a + b * b, 0) / normalized.length));

    return { advantages: normalized, old_value: values };
  }

  async updateModel(buffer) {
    //let { advantages, old_value } = this.computeGAE(buffer);
    console.log('buffer length:', buffer.length);
    const states = tf.tensor(buffer.map(b => b.state));
    //console.log('states shape:', states.shape);//[buffer.length, 1260]
    const old_V = this.critic.value(states).reshape([-1])
    const old_values = old_V.dataSync();
    const old_V_detached = tf.tensor(old_values, old_V.shape);
    let bootstrapValue = 0;
    if (Array.isArray(buffer.bootstrapState) && buffer.length > 0 && buffer[buffer.length - 1].truncated) {
      bootstrapValue = tf.tidy(() => {
        const bootstrapTensor = tf.tensor2d([encode_state(buffer.bootstrapState)], [1, 1260]);
        return this.critic.value(bootstrapTensor).dataSync()[0];
      });
    }
    if (old_values.some(a => isNaN(a))) {
      console.log("values存在NaN值");
    }
    if (old_values.every(a => a === 0)) {
      console.log("values全0");
    }
    const advantages = new Array(buffer.length);
    let x = 0;

    for (let i = buffer.length - 1; i >= 0; i--) {
      const reward = buffer[i].reward;
      const terminated = buffer[i].terminated === 1;
      const truncated = buffer[i].truncated === 1;
      const isLast = i === buffer.length - 1;

      const v = old_values[i];
      const vNext = isLast
        ? (terminated ? 0 : truncated ? bootstrapValue : 0)
        : old_values[i + 1];
      //TD差分公式
      const delta =
        reward +
        this.gamma * vNext -
        v;

      const continuation = isLast ? 0 : (terminated ? 0 : 1);
      x = delta + this.gamma * this.lam * continuation * x;

      advantages[i] = x;
    }
    const adv_mean = advantages.reduce((a, b) => a + b, 0) / advantages.length;
    const adv_std = Math.sqrt(
      advantages.reduce((a, b) => a + (b - adv_mean) ** 2, 0) /
      advantages.length
    );
    // console.log(`advantages mean=${adv_mean.toFixed(6)}, adv_std=${adv_std.toFixed(6)}`);
    // console.log('advantages max:', Math.max(...advantages));
    // console.log('advantages min:', Math.min(...advantages));
    if (advantages.every(a => a === 0)) {
      console.log("advantages全0");
    }
    // 归一化
    //const normalized = advantages.map(a => (a - adv_mean) / (adv_std + 1e-8));
    // console.log('normalized mean:', normalized.reduce((a, b) => a + b, 0) / normalized.length);
    // console.log('normalized std:', Math.sqrt(normalized.reduce((a, b) => a + b * b, 0) / normalized.length));
    // console.log('Advantages stats:', {
    //   length: advantages.length,
    //   sum: advantages.reduce((a, b) => a + b, 0),
    //   mean: advantages.reduce((a, b) => a + b, 0) / advantages.length,
    //   max: Math.max(...advantages),
    //   min: Math.min(...advantages),
    //   allZero: advantages.every(v => v === 0)
    // });

    const normalizedAdvantages = advantages.map(a => (a - adv_mean) / (adv_std + 1e-8));

    const actions = tf.tensor1d(buffer.map(b => b.action), "int32");
    const legalMasks = tf.tensor2d(buffer.map(b => b.legalMask), [buffer.length, this.actionDim]);
    if (actions.dataSync().some(a => isNaN(a))) {
      console.log("actions存在NaN值");
    }
    if (actions.dataSync().every(a => a === 0)) {
      console.log("actions全0");
    }
    const oldLogProbs = tf.tensor(buffer.map(b => b.logProb));
    //console.log('oldLogProbs shape:', oldLogProbs.shape);//[buffer.length]
    const rawAdv = tf.tensor(advantages);
    const adv = tf.tensor(normalizedAdvantages);
    // const advTensor = tf.tensor1d(advantages);
    // const ad_mean = advTensor.mean();
    // const adv = advTensor.sub(ad_mean).div(advTensor.squaredDifference(ad_mean).mean().sqrt().add(1e-8));
    let metrics = {};
    console.log("开始优化Actor")
    // ===== Actor update =====
    //console.log('Actor fc1 trainable:', this.actor.fc1.trainable);
    // tf.tidy(() => {
    //   const { mean, std } = this.actor.forward(states);
    //   // console.log('mean:', mean.dataSync());
    //   // console.log('std:', std.dataSync());

    //   const u = tf.randomNormal(mean.shape, mean.dataSync()[0], std.dataSync()[0]);
    //   const z = u.sub(mean).div(std);
    //   const logProbNormal = tf.scalar(-0.5).mul(z.square())
    //     .sub(tf.scalar(0.5).mul(tf.log(tf.scalar(2 * Math.PI))))
    //     .sub(tf.log(std));
    //   const correction = tf.log(tf.scalar(1).sub(tf.tanh(u).square()).add(tf.scalar(1e-6)));
    //   const logProb = logProbNormal.sub(correction).sum(-1);
    //   // console.log('logProb:', logProb.dataSync());
    //   // console.log('oldLogProbs:', oldLogProbs.dataSync());
    //   // const ratio = tf.exp(tf.sub(logProb, oldLogProbs));
    //   // // console.log('ratio:', ratio.dataSync());
    //   // console.log('adv:', adv.dataSync());
    //   // const surr1 = tf.mul(ratio, adv);
    //   // //console.log('surr1:', surr1);
    //   // const surr2 = tf.mul(
    //   //   tf.clipByValue(ratio, 1 - this.clip, 1 + this.clip),
    //   //   adv
    //   // );
    //   // //console.log('surr2:', surr2);

    //   // const loss = tf.neg(tf.mean(tf.minimum(surr1, surr2)));
    //   const kl = tf.mean(tf.sub(oldLogProbs, logProb));
    //   //metrics.lossActor = loss.dataSync()[0];
    //   metrics.kl = kl.dataSync()[0];

    // });
    const ratioStats = tf.tidy(() => {
      const logits = this.actor.forward(states);
      const logProb = this.actor.logProbFromAction(actions, logits, legalMasks);
      const ratio = tf.exp(tf.sub(logProb, oldLogProbs));
      return {
        min: ratio.min().dataSync()[0],
        max: ratio.max().dataSync()[0],
        mean: ratio.mean().dataSync()[0],
      };
    });

    const actorBefore = captureActorParams(this.actor);
    let a_loss = this.actorOpt.minimize(() => {
      const logits = this.actor.forward(states);
      const maskedLogits = this.actor.maskLogits(logits, legalMasks);
      const logProb = this.actor.logProbFromAction(actions, logits, legalMasks);
      const ratio = tf.exp(tf.sub(logProb, oldLogProbs));
      const surr1 = tf.mul(ratio, adv);
      const surr2 = tf.mul(
        tf.clipByValue(ratio, 1 - this.clip, 1 + this.clip),
        adv
      );
      const policyLoss = tf.neg(tf.mean(tf.minimum(surr1, surr2)));
      const probs = tf.softmax(maskedLogits);
      const entropy = tf.mean(
        tf.neg(tf.sum(tf.mul(probs, tf.log(probs.add(1e-8))), -1))
      );
      metrics.entropy = entropy.dataSync()[0];
      const loss = tf.sub(policyLoss, tf.mul(this.entropyCoef, entropy));

      return loss;
    }, true);
    console.log('Actor loss:', a_loss.dataSync()[0]);
    metrics.lossActor = a_loss.dataSync()[0];
    metrics.actorParamDelta = computeParamDelta(actorBefore, captureActorParams(this.actor));

    // tf.tidy(() => {
    //   const logits = this.actor.forward(states);
    //   console.log('logits:', logits);
    //   const logProbsAll = tf.logSoftmax(logits);
    //   console.log('logProbsAll:', logProbsAll.dataSync());
    //   const actionsInt = actions.toInt();
    //   console.log("actionsInt shape:", actionsInt.shape);
    //   const oneHot = tf.oneHot(actionsInt, logits.shape[1]);
    //   console.log('oneHot shape:', oneHot.shape);
    //   const logProb = tf.sum(tf.mul(logProbsAll, oneHot), -1);
    //   console.log('oldLogProbs:', oldLogProbs.dataSync());
    //   const ratio = tf.exp(
    //     tf.clipByValue(
    //       tf.sub(logProb, oldLogProbs),
    //       -10,
    //       10
    //     )
    //   );
    //   const ratioMax = ratio.max().dataSync()[0];
    //   console.log('Max ratio:', ratioMax);
    //   const logProbs = tf.logSoftmax(logits);
    //   const probs = tf.exp(logProbs);
    //   // const probs = tf.softmax(logits);
    //   console.log('probs shape:', probs.shape);
    //   console.log('probs:', probs.dataSync());
    //   const entropy = tf.mean(tf.neg(tf.sum(tf.mul(probs, tf.log(probs.add(1e-8))), -1)));
    //   console.log('entropy:', entropy.dataSync()[0]);
    //   const kl = tf.mean(tf.sub(oldLogProbs, logProb));
    //   const surr1 = tf.mul(ratio, adv);
    //   console.log('surr1:', surr1.dataSync());
    //   const surr2 = tf.mul(tf.clipByValue(ratio, 1 - this.clip, 1 + this.clip), adv);
    //   console.log('surr2:', surr2.dataSync());
    //   const loss = tf.neg(tf.mean(tf.minimum(surr1, surr2)));
    //   console.log('loss:', loss.dataSync()[0]);
    //   const loss2 = tf.sub(loss, tf.mul(this.entropyCoef, entropy));
    //   console.log('loss2:', loss2.dataSync()[0]);

    //   metrics.lossActor = loss2.dataSync()[0];
    //   metrics.kl = kl.dataSync()[0];
    //   metrics.entropy = entropy.dataSync()[0];
    // });
    // this.actorOpt.minimize(() => {
    //   // 只做计算，不 dataSync
    //   const logits = this.actor.forward(states);
    //   const logProbsAll = tf.logSoftmax(logits);
    //   const actionsInt = actions.toInt();
    //   const oneHot = tf.oneHot(actionsInt, logits.shape[1]);
    //   const logProb = tf.sum(tf.mul(logProbsAll, oneHot), -1);
    //   const ratio = tf.exp(
    //     tf.clipByValue(
    //       tf.sub(logProb, oldLogProbs),
    //       -10,
    //       10
    //     )
    //   );
    //   const surr1 = tf.mul(ratio, adv);
    //   const surr2 = tf.mul(tf.clipByValue(ratio, 1 - this.clip, 1 + this.clip), adv);
    //   const loss = tf.neg(tf.mean(tf.minimum(surr1, surr2)));
    //   //const probs = tf.softmax(logits);
    //   const logProbs = tf.logSoftmax(logits);
    //   const probs = tf.exp(logProbs);
    //   const entropy = tf.mean(tf.neg(tf.sum(tf.mul(probs, tf.log(probs.add(1e-8))), -1)));
    //   const loss2 = tf.sub(loss, tf.mul(this.entropyCoef, entropy));
    //   return loss2;
    // });

    const weights_fc1 = this.actor.fc1.getWeights()[0];
    const weights_logits = this.actor.logits.getWeights()[0];
    console.log('Actor fc1 Weight mean before/after:', weights_fc1.mean().dataSync()[0]);
    console.log('Actor logits Weight mean before/after:', weights_logits.mean().dataSync()[0]);

    console.log("开始优化Critic")
    // ===== Critic update =====
    // console.log("critic fc1 trainable:", this.critic.fc1.trainable);
    // console.log("critic fc2 trainable:", this.critic.fc2.trainable);
    const returns = tf.add(rawAdv, old_V_detached);
    //console.log('old_value:', old_V.dataSync());
    const targets = returns.clone();

    // console.log('Critic values:', this.critic.value(states).reshape([-1]).dataSync());
    // console.log('Critic returns:', targets.dataSync());

    // 更新 Critic
    const criticBefore = captureCriticParams(this.critic);
    let closs = this.criticOpt.minimize(() => {
      const values = this.critic.value(states).reshape([-1]);
      return tf.losses.meanSquaredError(targets, values);
    }, true);
    console.log('Critic loss:', closs.dataSync()[0]);
    metrics.lossCritic = closs.dataSync()[0];
    metrics.criticParamDelta = computeParamDelta(criticBefore, captureCriticParams(this.critic));
    // tf.tidy(() => {
    //   const values = this.critic.value(states).reshape([-1]);
    //   console.log('Critic values:', values.dataSync());
    //   console.log('Critic adv:', adv.dataSync());
    //   const returns = tf.add(adv, values);
    //   console.log('Critic returns:', returns.dataSync());
    //   let loss = tf.losses.meanSquaredError(returns, values);
    //   metrics.lossCritic = loss.dataSync()[0];
    // });
    // this.criticOpt.minimize(() => {
    //   const values = this.critic.value(states).reshape([-1]);
    //   const returns = tf.add(adv, values);
    //   //console.log("returns", returns);
    //   const delta = 1.0;
    //   const errors = tf.sub(returns, values);
    //   const absErrors = tf.abs(errors);
    //   // Huber Loss 的等效公式（无需条件判断）
    //   // L = 0.5 * error^2  if |error| <= delta
    //   // L = delta * |error| - 0.5 * delta^2  if |error| > delta
    //   // 等价于：L = 0.5 * min(|error|, delta)^2 + delta * max(|error| - delta, 0)

    //   const clippedAbs = tf.minimum(absErrors, delta);
    //   const quadraticPart = tf.mul(0.5, tf.square(clippedAbs));
    //   const linearPart = tf.mul(delta, tf.maximum(tf.sub(absErrors, delta), 0));
    //   const huberLoss = tf.add(quadraticPart, linearPart);

    //   return tf.mean(huberLoss);
    //   //  let loss = tf.losses.meanSquaredError(returns, values);
    //   // console.log("loss", loss);
    //   // metrics.lossCritic = loss.dataSync()[0];
    //   // return loss;
    // });

    // this.critic.model.layers.forEach((layer, index) => {
    //   const weights = layer.getWeights();
    //   if (weights.length > 0) {
    //     let totalSum = 0;
    //     let totalCount = 0;
    //     weights.forEach((weight) => {
    //       const data = weight.dataSync();
    //       totalSum += data.reduce((a, b) => a + b, 0);
    //       totalCount += data.length;
    //     });
    //     const totalMean = totalSum / totalCount;
    //     console.log(`Critic Layer ${layer.name} 总均值: ${totalMean.toFixed(6)}`);
    //   }
    // });
    const c_w_fc1 = this.critic.fc1.getWeights()[0];
    const c_w_fc2 = this.critic.fc2.getWeights()[0];
    console.log('Critic fc1 Weight mean before/after:', c_w_fc1.mean().dataSync()[0]);
    console.log('Critic fc2 Weight mean before/after:', c_w_fc2.mean().dataSync()[0]);

    const result = {
      lossActor: metrics.lossActor,
      lossCritic: metrics.lossCritic,
      actorParamDelta: metrics.actorParamDelta,
      criticParamDelta: metrics.criticParamDelta,
      ratioMin: ratioStats.min,
      ratioMax: ratioStats.max,
      ratioMean: ratioStats.mean,
      kl: metrics.kl,
      entropy: metrics.entropy,
      advMean: adv.mean().dataSync()[0],
      advStd: Math.sqrt(adv.square().mean().dataSync()[0]),
    };
    tf.dispose([states, actions, legalMasks, oldLogProbs, rawAdv, adv, targets, returns, a_loss, closs]);
    return result;
  }

  async train(options = {}) {
    const episodes = typeof options === "number" ? options : (options.episodes ?? 500);
    const onProgress = typeof options === "object" ? options.onProgress : null;
    const onEpisode = typeof options === "object" ? options.onEpisode : null;
    const chartUpdateInterval = typeof options === "object" ? (options.chartUpdateInterval ?? 1) : 1;
    const episodeTraceInterval = typeof options === "object" ? (options.episodeTraceInterval ?? 1) : 1;
    const logEpisodes = new Set([1, 100, 200, 300, 400]);
    this.env.render_mode = "none"
    let history = [];
    for (let i = 0; i < episodes; i++) {
      this.env.reset();
      let buffer = await this.RunGame();
      if (buffer == null) {
        console.error("RunGame return null, episode:", i);
        console.log(history);
        throw new Error(`RunGame returned null at episode ${i + 1}`);
      }
      //console.log(`Episode ${i} ended. reward: ${buffer.reduce((a, b) => a + b.reward, 0)}`);
      // for (let j = 0; j < 3; j++) {
      //   const randomIndex = Math.floor(Math.random() * buffer.length);
      //   console.log(`buffer${randomIndex}:`, buffer[randomIndex]);
      // }
      if (buffer.length > 100) {
        console.error("buffer长度大于100:", buffer.length);
        buffer = buffer.slice(0, 100); // 改为取前100个样本
      }
      let metrics = {}
      for (let j = 0; j < 5; j++) {
        metrics = await this.updateModel(buffer);
      }
      const episodeReward = buffer.reduce((sum, transition) => sum + transition.reward, 0);
      const stateNonzeroCount = buffer.length > 0
        ? buffer.reduce((sum, transition) => sum + countNonZeroFeatures(transition.state), 0) / buffer.length
        : 0;
      const finalTransition = buffer.at(-1) ?? null;
      const didWin = finalTransition?.terminated === 1 && finalTransition.reward > 0;
      const didLose = finalTransition?.terminated === 1 && finalTransition.reward < 0;
      const outcome = didWin ? "win" : didLose ? "loss" : "truncated";
      const actionSpaceStats = buffer.stats ?? summarizeActionSpaceStats([], this.actionDim, 0, buffer.length);
      const record = createTrainingRecord({
        episode: i,
        reward: episodeReward,
        steps: buffer.length,
        success: didWin,
        win: didWin,
        outcome,
        actorLoss: metrics.lossActor,
        criticLoss: metrics.lossCritic,
        actorParamDelta: metrics.actorParamDelta,
        criticParamDelta: metrics.criticParamDelta,
        ratioMean: metrics.ratioMean,
        ratioMin: metrics.ratioMin,
        ratioMax: metrics.ratioMax,
        entropy: metrics.entropy,
        advantageMean: metrics.advMean,
        advantageStd: metrics.advStd,
        stateNonzeroCount,
        legalActions: actionSpaceStats.legalActions,
        totalActions: actionSpaceStats.totalActions,
        legalRate: actionSpaceStats.legalRate,
        invalidAttempts: actionSpaceStats.invalidAttempts,
        validAttempts: actionSpaceStats.validAttempts,
        invalidPerValid: actionSpaceStats.invalidPerValid,
        repeatRate: actionSpaceStats.repeatRate,
        actionLoopCount: actionSpaceStats.actionLoopCount,
        attackMovesRate: actionSpaceStats.attackMovesRate,
        repeatMoveRate: actionSpaceStats.repeatMoveRate,
        actionCounts: actionSpaceStats.actionCounts,
        topActions: actionSpaceStats.topActions,
      });
      console.log(record);
      history.push(record);

      if ((i + 1) % 50 === 0) {
        const recent = history.slice(-50);
        const avgReward = recent.reduce((sum, item) => sum + item.reward, 0) / recent.length;
        const avgRepeatRate = recent.reduce((sum, item) => sum + (item.repeatRate ?? 0), 0) / recent.length;
        const totalActionLoops = recent.reduce((sum, item) => sum + (item.actionLoopCount ?? 0), 0);
        const avgAttackMovesRate = recent.reduce((sum, item) => sum + (item.attackMovesRate ?? 0), 0) / recent.length;
        const avgRepeatMoveRate = recent.reduce((sum, item) => sum + (item.repeatMoveRate ?? 0), 0) / recent.length;
        const avgEntropy = recent.reduce((sum, item) => sum + (item.entropy ?? 0), 0) / recent.length;
        const topActionSummary = (record.topActions ?? [])
          .slice(0, 5)
          .map((entry) => `${entry.label} ${(entry.rate * 100).toFixed(1)}%`)
          .join(", ");
        console.log(
          `[PPO Debug] episode=${i + 1} repeatRate=${avgRepeatRate.toFixed(3)} ` +
          `repeatMoveRate=${avgRepeatMoveRate.toFixed(3)} attackMovesRate=${avgAttackMovesRate.toFixed(3)} ` +
          `invalidLoopCount=${totalActionLoops} avgReward=${avgReward.toFixed(3)} entropy=${avgEntropy.toFixed(3)}`
        );
        console.log(`[PPO Debug] episode=${i + 1} top5Actions=${topActionSummary || "none"}`);
      }

      const progressInfo = {
        episode: i,
        totalEpisodes: episodes,
        metrics,
        history,
      };
      if (onProgress) {
        await onProgress(record, progressInfo);
      }

      const shouldEmitEpisodeTrace =
        (i === 0) ||
        ((i + 1) % episodeTraceInterval === 0) ||
        (i === episodes - 1);

      if (onEpisode && shouldEmitEpisodeTrace) {
        const episodeTrace = createEpisodeTrace({
          envType: "chess",
          episode: i,
          totalReward: buffer.viewerSteps.reduce((sum, step) => sum + (step.reward ?? 0), 0),
          totalSteps: buffer.viewerSteps.length,
          summary: {
            outcome,
            successLabel: didWin ? "Red Win" : outcome === "loss" ? "Black Win" : "Truncated",
          },
          metadata: {},
          steps: buffer.viewerSteps,
        });
        await onEpisode({
          ...progressInfo,
          record,
          episodeTrace,
        });
      }

      if ((i + 1) % chartUpdateInterval === 0 || i === episodes - 1) {
        drawlineGraph(history, {
          title: "Chinese Chess PPO Diagnostics",
          subtitle: "Inspect reward, losses, ratio stability, and whether policy parameters are actually moving.",
          mode: "ppo",
          movingAverageWindow: 20,
        });
      }

      const episodeNumber = i + 1;
      if (logEpisodes.has(episodeNumber)) {
        const initialState = this.env.reset();
        const initialLegalActionIds = this.env.getLegalActionIds("red");
        this.logTopActionProbabilities(episodeNumber, initialState, initialLegalActionIds);
      }

      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    drawlineGraph(history, {
      title: "Chinese Chess PPO Diagnostics",
      subtitle: "Inspect reward, losses, ratio stability, and whether policy parameters are actually moving.",
      mode: "ppo",
      movingAverageWindow: 20,
    });
    console.log(history);
    return history;
  }
  async test() {
    this.env.render_mode = "render"
    await this.PlayGame(1000);
  }
}
