import { getMazeActionLabels } from "../envs/maze/maze_shared_env.js";

const ACTION_LABELS = getMazeActionLabels();

function argmax(values) {
  let bestIndex = 0;
  let bestValue = values[0];
  for (let i = 1; i < values.length; i++) {
    if (values[i] > bestValue) {
      bestValue = values[i];
      bestIndex = i;
    }
  }
  return bestIndex;
}

function entropyOf(entries) {
  return entries.reduce((sum, entry) => {
    if (!entry.probability || entry.probability <= 0) return sum;
    return sum - entry.probability * Math.log(entry.probability + 1e-8);
  }, 0);
}

export class QLearningMazeAgent {
  constructor(env, options = {}) {
    this.env = env;
    this.alpha = Number.isFinite(options.alpha) ? options.alpha : 0.1;
    this.gamma = Number.isFinite(options.gamma) ? options.gamma : 0.95;
    this.epsilon = Number.isFinite(options.epsilon) ? options.epsilon : 0.12;
    this.epsilonDecay = Number.isFinite(options.epsilonDecay) ? options.epsilonDecay : 0.995;
    this.epsilonMin = Number.isFinite(options.epsilonMin) ? options.epsilonMin : 0.02;
    this.qTable = new Map();
  }

  getQValues(stateKey) {
    if (!this.qTable.has(stateKey)) {
      this.qTable.set(stateKey, [0, 0, 0, 0]);
    }
    return this.qTable.get(stateKey);
  }

  buildPolicySnapshot(qValues, selectedAction, deterministic = false) {
    const bestValue = Math.max(...qValues);
    const bestActions = qValues
      .map((value, index) => ({ value, index }))
      .filter((entry) => entry.value === bestValue)
      .map((entry) => entry.index);

    const epsilon = deterministic ? 0 : this.epsilon;
    const exploreProb = epsilon / ACTION_LABELS.length;
    const exploitProb = bestActions.length > 0 ? (1 - epsilon) / bestActions.length : 0;

    const topActions = qValues.map((value, actionId) => ({
      actionId,
      label: `${ACTION_LABELS[actionId]} · Q=${value.toFixed(3)}`,
      probability: exploreProb + (bestActions.includes(actionId) ? exploitProb : 0),
    })).sort((lhs, rhs) => rhs.probability - lhs.probability || rhs.actionId - lhs.actionId);

    return {
      topActions,
      entropy: entropyOf(topActions),
      selectedAction: {
        actionId: selectedAction,
        label: ACTION_LABELS[selectedAction],
        probability: topActions.find((entry) => entry.actionId === selectedAction)?.probability ?? null,
      },
    };
  }

  async selectAction(state, options = {}) {
    const deterministic = Boolean(options.deterministic);
    const stateKey = this.env.getStateKey(state);
    const qValues = this.getQValues(stateKey);

    let action;
    if (deterministic || Math.random() >= this.epsilon) {
      action = argmax(qValues);
    } else {
      action = Math.floor(Math.random() * ACTION_LABELS.length);
    }

    const policySnapshot = this.buildPolicySnapshot(qValues, action, deterministic);
    return {
      action,
      actionLabel: ACTION_LABELS[action],
      policyTopActions: policySnapshot.topActions.slice(0, 4),
      selectedAction: policySnapshot.selectedAction,
      entropy: policySnapshot.entropy,
      logProb: null,
      value: Math.max(...qValues),
      stateVector: null,
    };
  }

  observe(transition) {
    const stateKey = this.env.getStateKey(transition.state);
    const nextStateKey = this.env.getStateKey(transition.nextState);
    const qValues = this.getQValues(stateKey);
    const nextQValues = this.getQValues(nextStateKey);
    const oldQ = qValues[transition.action];
    const maxNext = transition.done ? 0 : Math.max(...nextQValues);
    const tdTarget = transition.reward + this.gamma * maxNext;
    const tdError = tdTarget - oldQ;
    qValues[transition.action] = oldQ + this.alpha * tdError;
    return {
      qDelta: Math.abs(qValues[transition.action] - oldQ),
      maxNext,
    };
  }

  async finishEpisode() {
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
    let maxQ = 0;
    for (const values of this.qTable.values()) {
      maxQ = Math.max(maxQ, ...values);
    }
    return {
      epsilon: this.epsilon,
      maxQ,
      tableSize: this.qTable.size,
    };
  }
}
