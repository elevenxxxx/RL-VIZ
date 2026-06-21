import * as tf from "@tensorflow/tfjs";
import { encode_state } from "./utils.js";
class ActorNet {
  constructor(stateDim, hiddenDim = 256, actionDim = 188) {
    this.fc1 = tf.layers.dense({
      units: hiddenDim,
      activation: "relu",
      inputShape: [stateDim],
    });

    // this.mean = tf.layers.dense({
    //   units: actionDim,
    //   activation: "tanh",
    // });

    // this.logStd = tf.variable(tf.zeros([actionDim]));

    // 输出 logits categorical policy
    this.logits = tf.layers.dense({
      units: actionDim,
    });
  }

  forward(state) {
    return tf.tidy(() => {
      const x = this.fc1.apply(state);
      // const mean = this.mean.apply(x);

      // //const logStd = this.logStd.clipByValue(-20, 2);
      // const logStd = tf.clipByValue(this.logStd, -20, 2);
      // const std = tf.exp(logStd);

      // return { mean, std };
      const logits = this.logits.apply(x);
      return logits;
    });
  }

  sampleAction_Gaussian(state) {
    return tf.tidy(() => {
      const { mean, std } = this.forward(state);

      const eps = tf.randomNormal(mean.shape);
      const u_raw = tf.add(mean, tf.mul(std, eps));
      const u = tf.tanh(u_raw);
      // log_prob（简化版）
      // const logProb = tf.sum(
      //   tf.sub(
      //     tf.log(tf.div(1, tf.mul(std, tf.sqrt(2 * Math.PI)))),
      //     tf.div(tf.square(tf.sub(u, mean)), tf.mul(2, tf.square(std)))
      //   )
      // );
      //Jacobian修正
      const logProb = tf.log(u_raw) - tf.log(1 - tf.square(u));
      return { u, logProb };
    });
  }
  //Categorical policy
  sampleAction(state) {
    return tf.tidy(() => {
      const logits = this.forward(state);

      const probs = tf.softmax(logits);
      const dist = tf.randomUniform(probs.shape);

      // Gumbel-Max trick（更稳定）
      const gumbel = tf.neg(tf.log(tf.neg(tf.log(dist))));
      const y = tf.add(tf.log(probs), gumbel);

      const u = tf.argMax(y, -1);

      // logProb = log π(a|s)
      const logProb = tf.log(tf.gather(probs, u, 1).add(1e-8));

      return { u, logProb };
    });
  }
}
class CriticNet {
  constructor(stateDim, hiddenDim = 256) {
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
  constructor(env, stateDim = 1260, actionDim = 1) {
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

  // 连续 → 离散动作映射 Gaussian policy
  decodeAction(u) {
    const v = u.dataSync()[0];
    const a = Math.round(((v + 1) / 2) * this.actionScale);
    return Math.max(0, Math.min(187, a));
  }

  async RunGame() {
    let state = this.env.reset();
    let done = false;
    const buffer = [];
    let st = 0;
    while (true) {
      st++;
      //console.log("step", st);
      let success = false;
      let nextState, reward, terminated, truncated;

      let actionIndex, logProbValue;
      let trial = 0;
      while (!success) {
        trial++;
        if (trial > 500) {
          console.error("Too many trials!")
          return null;
        }
        console.info("step:", st, "第", trial, "次尝试");
        let encoded_state = encode_state(state);
        //console.log("encoded_state", encoded_state);//Float32Array(1260)
        const s = tf.tensor2d([encoded_state], [1, 1260]);//(1,1260)

        const { u, logProb } = this.actor.sampleAction(s);//应该是0~187
        //console.log("u", u.dataSync()[0]);

        // actionIndex = this.decodeAction(u);
        // console.log("actionIndex", actionIndex);
        actionIndex = this.env.ModifyAction(u.dataSync()[0]);
        if (actionIndex < 0) {
          console.error("internal error:ModifyAction return -1")
          return null;
        }

        //console.log("modified action", actionIndex);

        [success, nextState, reward, terminated, truncated] = this.env.step(actionIndex);

        if (success) {
          logProbValue = logProb.dataSync()[0];
        }
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

      adv = delta + this.gamma * this.lam * (1 - done) * adv;

      advantages[i] = adv;
    }

    const mean = advantages.reduce((a, b) => a + b, 0) / advantages.length;

    const std = Math.sqrt(
      advantages.reduce((a, b) => a + (b - mean) ** 2, 0) /
      advantages.length
    );

    return advantages.map(a => (a - mean) / (std + 1e-8));
  }

  async updateModel(buffer) {
    const advantages = this.computeGAE(buffer);
    //console.log("advantages", advantages);

    const states = tf.tensor(buffer.map(b => b.state));
    const actions = tf.stack(buffer.map(b => b.action));
    const oldLogProbs = tf.stack(buffer.map(b => b.logProb));
    const adv = tf.tensor(advantages);

    let metrics = {};
    console.log("开始优化Actor")
    // ===== Actor update =====
    this.actorOpt.minimize(() => {
      // const { mean, std } = this.actor.forward(states);

      // const logProb = tf.sum(
      //   tf.sub(
      //     tf.log(tf.div(1, tf.mul(std, tf.sqrt(2 * Math.PI)))),
      //     tf.div(tf.square(tf.sub(actions, mean)), tf.mul(2, tf.square(std)))
      //   ),
      //   -1
      // );
      const logits = this.actor.forward(states);
      // console.log("logits", logits);
      const logProbsAll = tf.logSoftmax(logits);
      //console.log("logProbsAll", logProbsAll);
      // 取对应 action 的 logProb
      //console.log('actions dtype:', actions.dtype);//float32
      const actionsInt = actions.toInt();

      const idx = tf.stack([
        tf.range(0, actionsInt.shape[0], 1, "int32"),
        actionsInt
      ], 1);
      // console.log("idx", idx);
      const oneHot = tf.oneHot(actionsInt, logits.shape[1]);// one-hot mask
      //const logProb = tf.gatherND(logProbsAll, idx);
      const logProb = tf.sum(tf.mul(logProbsAll, oneHot), -1);
      //console.log("logProb", logProb);
      const ratio = tf.exp(tf.sub(logProb, oldLogProbs));

      const surr1 = tf.mul(ratio, adv);
      const surr2 = tf.mul(
        tf.clipByValue(ratio, 1 - this.clip, 1 + this.clip),
        adv
      );

      const loss = tf.neg(tf.mean(tf.minimum(surr1, surr2)));
      // console.log("loss", loss);

      const probs = tf.softmax(logits);
      const entropy = tf.mean(
        tf.neg(tf.sum(tf.mul(probs, tf.log(probs.add(1e-8))), -1))
      );
      const kl = tf.mean(tf.sub(oldLogProbs, logProb));
      metrics.lossActor = loss.dataSync()[0];
      metrics.kl = kl.dataSync()[0];
      metrics.entropy = entropy.dataSync()[0];
      return loss;
    });
    console.log("开始优化Critic")
    // ===== Critic update =====
    this.criticOpt.minimize(() => {
      const values = this.critic.value(states).reshape([-1]);
      //console.log("values", values);
      const returns = tf.add(adv, values);
      //console.log("returns", returns);
      let loss = tf.losses.meanSquaredError(returns, values);
      // console.log("loss", loss);
      metrics.lossCritic = loss.dataSync()[0];
      return loss;
    });
    const result = {
      lossActor: metrics.lossActor,
      lossCritic: metrics.lossCritic,
      kl: metrics.kl,
      entropy: metrics.entropy,
      advMean: adv.mean().dataSync()[0]
    };
    tf.dispose([states, actions, oldLogProbs, adv]);
    return result;
  }

  async train(episodes = 500) {
    this.env.render_mode = "none"
    for (let i = 0; i < episodes; i++) {
      const buffer = await this.RunGame();
      if (buffer == null) {
        console.error("RunGame return null, episode:", i);
        return;
      }
      //console.log(`Episode ${i} ended. reward: ${buffer.reduce((a, b) => a + b.reward, 0)}`);
      // for (let j = 0; j < 3; j++) {
      //   const randomIndex = Math.floor(Math.random() * buffer.length);
      //   console.log(`buffer${randomIndex}:`, buffer[randomIndex]);
      // }
      const metrics = await this.updateModel(buffer);
      console.log({
        episode: i,
        reward: buffer.reduce((a, b) => a + b.reward, 0),
        lossActor: metrics.lossActor,//合法范围波动才对
        lossCritic: metrics.lossCritic,//应该逐渐下降才对
        kl: metrics.kl,//应该适中
        entropy: metrics.entropy,//衡量探索度，应该先高后底
        advMean: metrics.advMean,//应该接近0
      });
    }
  }
}