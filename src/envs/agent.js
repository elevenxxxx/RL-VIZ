import * as tf from "@tensorflow/tfjs";

export class ActorNet {
  constructor(stateDim=32, hiddenDim = 128, actionDim = 1) {
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

      const logStd = this.logStd.clipByValue(-20, 2);
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
export class CriticNet {
  constructor(stateDim=32, hiddenDim = 128) {
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
  constructor(env, stateDim=32, actionDim = 1) {
    this.env = env;

    this.actor = new ActorNet(stateDim);
    this.critic = new CriticNet(stateDim);

    this.gamma = 0.99;
    this.lam = 0.95;
    this.clip = 0.2;
    this.actionScale = 379;

    this.actorOpt = tf.train.adam(3e-4);
    this.criticOpt = tf.train.adam(1e-3);
  }

  // 连续 → 离散动作映射
  decodeAction(u) {
    const v = u.dataSync()[0];
    const a = Math.round(((v + 1) / 2) * this.actionScale);
    return Math.max(0, Math.min(379, a));
  }

  async collectEpisode() {
    let state = this.env.reset();
    let done = false;

    const buffer = [];

    while (!done) {
      const s = tf.tensor([state]);

      const { u, logProb } = this.actor.sampleAction(s);
      const actionIndex = this.decodeAction(u);

      const { nextState, reward, terminated, truncated } = this.env.step(actionIndex);

      buffer.push({
        state,
        action: u,
        reward,
        logProb,
        nextState,
        done: d ? 1 : 0,
      });

      state = nextState;
      done = terminated || truncated;

      tf.dispose([s, u]);
    }

    return buffer;
  }

  computeGAE(buffer) {
    const values = buffer.map(b =>
      this.critic.value(tf.tensor([b.state])).dataSync()[0]
    );

    const nextValues = values.slice(1).concat([0]);

    let adv = 0;
    const advantages = [];

    for (let i = buffer.length - 1; i >= 0; i--) {
      const delta =
        buffer[i].reward +
        this.gamma * nextValues[i] * (1 - buffer[i].done) -
        values[i];

      adv = delta + this.gamma * this.lam * adv;
      advantages[i] = adv;
    }

    const mean = advantages.reduce((a, b) => a + b) / advantages.length;
    const std = Math.sqrt(
      advantages.map(a => (a - mean) ** 2).reduce((a, b) => a + b) /
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