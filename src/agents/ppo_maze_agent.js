import * as tf from "@tensorflow/tfjs";
import { getMazeActionLabels } from "../envs/maze/maze_shared_env.js";

const ACTION_LABELS = getMazeActionLabels();

function sampleFromProbabilities(probabilities) {
  const rand = Math.random();
  let cumulative = 0;
  for (let i = 0; i < probabilities.length; i++) {
    cumulative += probabilities[i];
    if (rand <= cumulative) return i;
  }
  return probabilities.length - 1;
}

function topActionsFromProbabilities(probabilities) {
  return probabilities
    .map((probability, actionId) => ({
      actionId,
      label: ACTION_LABELS[actionId],
      probability,
    }))
    .sort((lhs, rhs) => rhs.probability - lhs.probability)
    .slice(0, 4);
}

function entropyFromProbabilities(probabilities) {
  return probabilities.reduce((sum, probability) => {
    if (probability <= 0) return sum;
    return sum - probability * Math.log(probability + 1e-8);
  }, 0);
}

function normalize(values) {
  if (values.length === 0) return [];
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  const std = Math.sqrt(variance) + 1e-8;
  return values.map((value) => (value - mean) / std);
}

export class PPOMazeAgent {
  constructor(env, options = {}) {
    this.env = env;
    this.stateDim = env.featureSize;
    this.actionDim = 4;
    this.gamma = Number.isFinite(options.gamma) ? options.gamma : 0.99;
    this.lam = Number.isFinite(options.lam) ? options.lam : 0.95;
    this.clip = Number.isFinite(options.clip) ? options.clip : 0.2;
    this.actorLr = Number.isFinite(options.actorLr) ? options.actorLr : 3e-4;
    this.criticLr = Number.isFinite(options.criticLr) ? options.criticLr : 1e-3;
    this.entropyCoef = Number.isFinite(options.entropyCoef) ? options.entropyCoef : 0.01;
    this.updateEpochs = Number.isFinite(options.updateEpochs) ? options.updateEpochs : 4;
    this.hiddenDim = Number.isFinite(options.hiddenDim) ? options.hiddenDim : 64;

    this.actor = tf.sequential({
      layers: [
        tf.layers.dense({
          units: this.hiddenDim,
          activation: "relu",
          inputShape: [this.stateDim],
          kernelInitializer: "glorotUniform",
        }),
        tf.layers.dense({
          units: this.actionDim,
          kernelInitializer: "glorotUniform",
        }),
      ],
    });

    this.critic = tf.sequential({
      layers: [
        tf.layers.dense({
          units: this.hiddenDim,
          activation: "relu",
          inputShape: [this.stateDim],
          kernelInitializer: "glorotUniform",
        }),
        tf.layers.dense({
          units: 1,
          kernelInitializer: "glorotUniform",
        }),
      ],
    });

    this.actorOpt = tf.train.adam(this.actorLr);
    this.criticOpt = tf.train.adam(this.criticLr);
  }

  encodeState(state) {
    return this.env.toFeatureVector(state);
  }

  evaluateState(stateVector) {
    return tf.tidy(() => {
      const stateTensor = tf.tensor2d([stateVector], [1, this.stateDim]);
      const logits = this.actor.predict(stateTensor);
      const probs = Array.from(tf.softmax(logits).dataSync());
      const value = this.critic.predict(stateTensor).dataSync()[0];
      return { probs, value };
    });
  }

  async selectAction(state, options = {}) {
    const deterministic = Boolean(options.deterministic);
    const stateVector = this.encodeState(state);
    const { probs, value } = this.evaluateState(stateVector);
    const action = deterministic
      ? probs.indexOf(Math.max(...probs))
      : sampleFromProbabilities(probs);
    const probability = probs[action] ?? 0;
    return {
      action,
      actionLabel: ACTION_LABELS[action],
      selectedAction: {
        actionId: action,
        label: ACTION_LABELS[action],
        probability,
      },
      policyTopActions: topActionsFromProbabilities(probs),
      entropy: entropyFromProbabilities(probs),
      logProb: Math.log(probability + 1e-8),
      value,
      stateVector,
    };
  }

  computeAdvantages(trajectory) {
    const rewards = trajectory.map((step) => step.reward);
    const values = trajectory.map((step) => step.value ?? 0);
    const advantages = new Array(trajectory.length).fill(0);
    let gae = 0;
    for (let index = trajectory.length - 1; index >= 0; index--) {
      const nextValue = index === trajectory.length - 1 ? 0 : values[index + 1];
      const delta = rewards[index] + this.gamma * nextValue * (trajectory[index].done ? 0 : 1) - values[index];
      gae = delta + this.gamma * this.lam * (trajectory[index].done ? 0 : 1) * gae;
      advantages[index] = gae;
    }
    const returns = advantages.map((advantage, index) => advantage + values[index]);
    return {
      advantages,
      normalizedAdvantages: normalize(advantages),
      returns,
    };
  }

  async finishEpisode(trajectory) {
    if (!Array.isArray(trajectory) || trajectory.length === 0) {
      return {
        lossActor: null,
        lossCritic: null,
        entropy: null,
      };
    }

    const { normalizedAdvantages, returns } = this.computeAdvantages(trajectory);
    const states = tf.tensor2d(trajectory.map((step) => step.stateVector), [trajectory.length, this.stateDim]);
    const actions = tf.tensor1d(trajectory.map((step) => step.action), "int32");
    const oldLogProbs = tf.tensor1d(trajectory.map((step) => step.logProb ?? 0));
    const advantagesTensor = tf.tensor1d(normalizedAdvantages);
    const returnsTensor = tf.tensor1d(returns);

    let actorLossValue = 0;
    let criticLossValue = 0;
    let entropyValue = 0;

    for (let epoch = 0; epoch < this.updateEpochs; epoch++) {
      const actorLoss = this.actorOpt.minimize(() => {
        const logits = this.actor.apply(states);
        const logProbs = tf.logSoftmax(logits);
        const oneHot = tf.oneHot(actions, this.actionDim);
        const selectedLogProbs = tf.sum(tf.mul(logProbs, oneHot), -1);
        const ratio = tf.exp(tf.sub(selectedLogProbs, oldLogProbs));
        const surrogate1 = tf.mul(ratio, advantagesTensor);
        const surrogate2 = tf.mul(
          tf.clipByValue(ratio, 1 - this.clip, 1 + this.clip),
          advantagesTensor
        );
        const policyLoss = tf.neg(tf.mean(tf.minimum(surrogate1, surrogate2)));
        const probs = tf.softmax(logits);
        const entropy = tf.mean(tf.neg(tf.sum(tf.mul(probs, tf.log(probs.add(1e-8))), -1)));
        entropyValue = entropy.dataSync()[0];
        return tf.sub(policyLoss, tf.mul(this.entropyCoef, entropy));
      }, true);

      const criticLoss = this.criticOpt.minimize(() => {
        const values = this.critic.apply(states).reshape([-1]);
        return tf.losses.meanSquaredError(returnsTensor, values);
      }, true);

      actorLossValue = actorLoss.dataSync()[0];
      criticLossValue = criticLoss.dataSync()[0];
      actorLoss.dispose();
      criticLoss.dispose();
    }

    tf.dispose([states, actions, oldLogProbs, advantagesTensor, returnsTensor]);

    return {
      lossActor: actorLossValue,
      lossCritic: criticLossValue,
      entropy: entropyValue,
    };
  }
}
