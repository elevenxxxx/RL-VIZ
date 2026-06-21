import * as tf from "@tensorflow/tfjs";
import { encode_state } from "./utils.js";
class ActorNet {
  constructor(stateDim, hiddenDim = 128, actionDim = 1) {
    this.fc1 = tf.layers.dense({
      units: hiddenDim,
      activation: "relu",
      inputShape: [stateDim],
    });

    this.mean = tf.layers.dense({
      units: actionDim,
      activation: "tanh",
    });

    this.logStd = tf.variable(tf.zeros([actionDim]));
  }

  forward(state) {
    return tf.tidy(() => {
      const x = this.fc1.apply(state);
      const mean = this.mean.apply(x);

      //const logStd = this.logStd.clipByValue(-20, 2);
      const logStd = tf.clipByValue(this.logStd, -20, 2);
      const std = tf.exp(logStd);

      return { mean, std };
    });
  }

  sampleAction(state) {
    return tf.tidy(() => {
      const { mean, std } = this.forward(state);

      const eps = tf.randomNormal(mean.shape);
      const u = tf.add(mean, tf.mul(std, eps));

      // log_prob（简化版）
      const logProb = tf.sum(
        tf.sub(
          tf.log(tf.div(1, tf.mul(std, tf.sqrt(2 * Math.PI)))),
          tf.div(tf.square(tf.sub(u, mean)), tf.mul(2, tf.square(std)))
        )
      );

      return { u, logProb };
    });
  }
}
class CriticNet {
  constructor(stateDim, hiddenDim = 128) {
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({
      units: hiddenDim,
      activation: "relu",
      inputShape: [stateDim],
    }));
    this.model.add(tf.layers.dense({ units: 1 }));
  }

  value(state) {
    return this.model.predict(state);
  }
}
export class Agent {
  constructor(env, stateDim = 1024, actionDim = 1) {
    this.env = env;

    this.actor = new ActorNet(stateDim);
    this.critic = new CriticNet(stateDim);

    this.gamma = 0.99;
    this.lam = 0.95;
    this.clip = 0.2;
    this.actionScale = 187;

    this.actorOpt = tf.train.adam(3e-4);
    this.criticOpt = tf.train.adam(1e-3);
  }

  // 连续 → 离散动作映射
  decodeAction(u) {
    const v = u.dataSync()[0];
    const a = Math.round(((v + 1) / 2) * this.actionScale);
    return Math.max(0, Math.min(187, a));
  }

  async collectEpisode() {
    let state = this.env.reset();
    let done = false;
    const buffer = [];

    while (true) {

      let success = false;
      let nextState, reward, terminated, truncated;

      let actionIndex, logProbValue;

      while (!success) {

        const s = tf.tensor(encode_state(state));

        const { u, logProb } = this.actor.sampleAction(s);

        actionIndex = this.decodeAction(u);
        const action = this.env.ModifyAction(actionIndex);

        ({ success, nextState, reward, terminated, truncated } =
          this.env.step(action));

        logProbValue = logProb.dataSync()[0];
        tf.dispose([s, u, logProb]);
      }

      buffer.push({
        state: encode_state(state),
        action: actionIndex,
        reward: reward,
        logProb: logProbValue,
        done: done ? 1 : 0,
      });

      state = nextState;
      done = terminated || truncated;
      if (done) break;
    }

    return buffer;
  }

  computeGAE(buffer) {
    //  const values = buffer.map(b =>
    //   this.critic.value(tf.tensor([b.state])).dataSync()[0]
    // );
    const states = tf.tensor(buffer.map(b => b.state));
    const values = this.critic.value(states).reshape([-1]).dataSync();

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

      adv =delta +this.gamma * this.lam * (1 - done) * adv;

      advantages[i] = adv;
    }

    const mean = advantages.reduce((a, b) => a + b, 0) / advantages.length;

    const std = Math.sqrt(
      advantages.reduce((a, b) => a + (b - mean) ** 2, 0) /
      advantages.length
    );

    return advantages.map(a => (a - mean) / (std + 1e-8));
  }

  async trainStep(buffer) {
    const advantages = this.computeGAE(buffer);

    const states = tf.tensor(buffer.map(b => b.state));
    const actions = tf.stack(buffer.map(b => b.action));
    const oldLogProbs = tf.stack(buffer.map(b => b.logProb));
    const adv = tf.tensor(advantages);

    // ===== Actor update =====
    this.actorOpt.minimize(() => {
      const { mean, std } = this.actor.forward(states);

      const logProb = tf.sum(
        tf.sub(
          tf.log(tf.div(1, tf.mul(std, tf.sqrt(2 * Math.PI)))),
          tf.div(tf.square(tf.sub(actions, mean)), tf.mul(2, tf.square(std)))
        ),
        -1
      );

      const ratio = tf.exp(tf.sub(logProb, oldLogProbs));

      const surr1 = tf.mul(ratio, adv);
      const surr2 = tf.mul(
        tf.clipByValue(ratio, 1 - this.clip, 1 + this.clip),
        adv
      );

      const loss = tf.neg(tf.mean(tf.minimum(surr1, surr2)));

      return loss;
    });

    // ===== Critic update =====
    this.criticOpt.minimize(() => {
      const values = this.critic.value(states).reshape([-1]);
      const returns = tf.add(adv, values);

      return tf.losses.meanSquaredError(returns, values);
    });

    tf.dispose([states, actions, oldLogProbs, adv]);
  }

  async train(episodes = 500) {
    for (let i = 0; i < episodes; i++) {
      const buffer = await this.collectEpisode();
      await this.trainStep(buffer);

      console.log(`episode ${i}, reward ${buffer.reduce((a, b) => a + b.reward, 0)}`);
    }
  }
}