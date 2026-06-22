import * as tf from "@tensorflow/tfjs";
import { encode_state } from "./utils.js";
import { drawlineGraph } from "./graph.js";
class ActorNet {
  constructor(stateDim, hiddenDim = 256, actionDim = 1) {
    this.fc1 = tf.layers.dense({
      units: hiddenDim,
      activation: "relu",
      inputShape: [stateDim],
      kernelInitializer: 'glorotNormal',  // 明确指定初始化器
      biasInitializer: 'zeros'
    });

    this.mean = tf.layers.dense({
      units: actionDim,
      activation: "tanh",
      kernelInitializer: 'glorotNormal',  // 明确指定初始化器
      biasInitializer: 'zeros'
    });

    this.logStd = tf.variable(tf.zeros([actionDim]));

    // 输出 logits categorical policy
    // this.logits = tf.layers.dense({
    //   units: actionDim,
    // });
    // this.model = tf.sequential();
    // this.model.add(this.fc1);
    // this.model.add(this.logits);
    this.actionScale = 187;
  }

  forward(state) {
    return tf.tidy(() => {
      // console.log('输入 state shape:', state.shape);  // 应该是 [1, 1260]
      const x = this.fc1.apply(state);
      const mean = this.mean.apply(x);
      // console.log('fc1 输出 shape:', x.shape);  // 应该是 [1, 256]

      const logStd = tf.clipByValue(this.logStd, -20, 2);
      const std = tf.exp(logStd);

      return { mean, std };
      //const logits = this.logits.apply(x);
      // console.log('logits 输出 shape:', logits.shape);  // 应该是 [1, 188]
      //return logits;
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
      // const logits = this.forward(state);
      // if (logits.dataSync().some(val => isNaN(val))) {
      //   console.log("logits存在NaN值");
      // }
      // if (logits.dataSync().every(val => val === 0)) {
      //   console.log("logits为全0");
      // }
      const { mean, std } = this.forward(state);
      // console.log("mean", mean.dataSync()[0]);
      // console.log("std", std.dataSync()[0]);

      const u = tf.randomNormal(mean.shape, mean.dataSync()[0], std.dataSync()[0]);
      //console.log("u", u.dataSync()[0]);
      const z = u.sub(mean).div(std);
      //console.log("z", z.dataSync());
      const logProbNormal = tf.scalar(-0.5).mul(z.square())
        .sub(tf.scalar(0.5).mul(tf.log(tf.scalar(2 * Math.PI))))
        .sub(tf.log(std));
      //console.log("logProbNormal", logProbNormal.dataSync());

      const tanh_u = tf.tanh(u);
      //console.log("tanh_u", tanh_u.dataSync());
      const correction = tf.log(tf.scalar(1).sub(tanh_u.square()).add(tf.scalar(1e-6)));
      //console.log("correction", correction.dataSync());
      // 最终log概率（在最后一维求和）
      const logProb = logProbNormal.sub(correction).sum(-1);
      //console.log("logProb", logProb.dataSync());
      const a = tf.mul(tf.abs(tanh_u), this.actionScale);
      //console.log("a", a.dataSync());
      // 返回动作和对数概率
      // console.log("logProb", logProb.dataSync()[0]);
      // console.log("a", a.dataSync()[0]);
      return { u_continuous: a, logProb: logProb };
      // const probs = tf.softmax(logits);
      // const dist = tf.randomUniform(probs.shape);
      // const dist = tf.clipByValue(
      //   tf.randomUniform(probs.shape),
      //   1e-7,
      //   1 - 1e-7
      // );//裁剪 避免趋近于0或者1

      // Gumbel-Max trick（更稳定）
      // const gumbel = tf.neg(tf.log(tf.neg(tf.log(dist))));
      //const y = tf.add(tf.log(probs), gumbel);

      //const u = tf.argMax(y, -1);
      // const u = tf.multinomial(logits, 1).squeeze();  // 移除尺寸为1的维度
      // logProb = log π(a|s)
      // const logProbs = tf.logSoftmax(logits);  // 直接计算 log 概率
      // console.log("logProbs", logProbs.dataSync());
      //const logProb = tf.gather(logProbs, u, 1);  // 取对应动作的 log 概率
      // const logProb = tf.clipByValue(logProb2, -20, 0); // log概率通常在 -20 ~ 0
      // const logProb = tf.log(tf.gather(probs, u, 1).add(1e-8));
      //  console.log("logProb", logProb2.dataSync(), "logProbClipped", logProb.dataSync());
      //return { u, logProb };
    });
  }
  sampleDetermineAction(state) {
    tf.tidy(() => {
      const { mean, std } = this.forward(state);
      const tanh_u = tf.tanh(mean);//使用mean作为确定性策略
      const a = tf.mul(tf.abs(tanh_u), this.actionScale);
      return a;
    })
  }
}
class CriticNet {
  constructor(stateDim, hiddenDim = 256) {
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({
      units: hiddenDim,
      activation: "relu",
      inputShape: [stateDim],
      kernelInitializer: 'glorotNormal',  // 明确指定初始化器
      biasInitializer: 'zeros'
    }));
    this.model.add(tf.layers.dense({
      units: 1,
      kernelInitializer: 'glorotNormal',
      biasInitializer: 'zeros'
    }));
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
    this.actor_lr = 3e-4;
    this.critic_lr = 2e-3;

    this.actorOpt = tf.train.adam(this.actor_lr);
    //this.actorOpt = this.createClippedOptimizer(3e-4);
    this.criticOpt = tf.train.adam(this.critic_lr);
    //this.criticOpt = this.createClippedOptimizer(2e-3);
    this.entropyCoef = 0.01;//熵系数 反应探索的重视程度
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
  // 连续 → 离散动作映射 Gaussian policy
  decodeAction(u) {
    if (u < 0 || u > 187) {
      console.error("internal error:u out of range")
    }
    let v;
    if (u instanceof tf.Tensor) {
      v = u.dataSync()[0];  // 获取标量值
    } else {
      v = u;
    }
    return Math.max(0, Math.min(187, Math.round(v)));
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

      let actionIndex, logProbValue, action;
      let trial = 0;
      while (!success) {
        trial++;
        if (trial > 200) {
          console.error("Too many trials!")
          return null;
        }
        // if (trial > 180) {
        //   console.info("step:", st, "第", trial, "次尝试");
        // }

        let encoded_state = encode_state(state);
        //console.log("encoded_state", encoded_state);//Float32Array(1260)
        const s = tf.tensor2d([encoded_state], [1, 1260]);//(1,1260)

        const { u_continuous, logProb } = this.actor.sampleAction(s);//应该是0~187
        action = u_continuous.dataSync()[0];
        //console.log("action", action);
        let a_discrete = this.decodeAction(action);
        actionIndex = this.env.ModifyAction(a_discrete);
        //console.log("actionIndex", actionIndex);
        if (actionIndex < 0) {
          console.error("internal error:ModifyAction return -1")
          return null;
        }

        //console.log("modified action", actionIndex);
        const res = await this.env.step(actionIndex);
        success = res[0];
        nextState = res[1];
        reward = res[2];
        terminated = res[3];
        truncated = res[4];

        if (success) {
          logProbValue = logProb.dataSync()[0];
        }
        tf.dispose([s, u_continuous, logProb]);
      }

      done = terminated || truncated;
      buffer.push({
        state: encode_state(state),
        action: actionIndex,
        reward: reward / 10,
        logProb: logProbValue,
        done: done ? 1 : 0,
      });
      //console.log(`[${buffer.length},${this.env.episode}]`);
      if (buffer.length != this.env.episode) {
        console.error(`buffer length ${buffer.length} != episode ${this.env.episode}`)
      }
      state = nextState;

      if (done) break;
    }

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

      let actionIndex;
      let trial = 0;
      while (!success) {
        trial++;
        if (trial > 500) {
          console.error("Too many trials!")
          return null;
        }
        console.info("step:", st, "第", trial, "次尝试");

        // 使用 tf.tidy 自动清理内存并确保梯度不被追踪
        // const { actionIndex } = tf.tidy(() => {
        //   let encoded_state = encode_state(state);
        //   const s = tf.tensor2d([encoded_state], [1, 1260]);
        //   const { u, logProb } = this.actor.sampleAction(s);
        //   const actionIndex = this.env.ModifyAction(u.dataSync()[0]);
        //   return { actionIndex };
        // });
        const { actionIndex } = tf.tidy(() => {
          let encoded_state = encode_state(state);
          const s = tf.tensor2d([encoded_state], [1, 1260]);
          //const { u_continuous, logProb } = this.actor.sampleAction(s);
          const u_continous = this.actor.sampleDetermineAction(s);
          let u = this.decodeAction(u_continous.dataSync()[0]);
          const actionIndex = this.env.ModifyAction(u);
          return { actionIndex };
        });
        console.log("actionIndex", actionIndex);
        if (actionIndex < 0) {
          console.error("internal error:ModifyAction return -1")
          return null;
        }

        [success, nextState, reward, terminated, truncated] = await this.env.step(actionIndex, 1000);
      }

      state = nextState;
      done = terminated || truncated;

      await new Promise(resolve => setTimeout(resolve, interval));//1s

      if (done) break;
    }

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

      if (i % 100 === 0 || i === buffer.length - 1 || i === 0) {
        console.log(`i=${i}, reward=${reward}, done=${done}, v=${v.toFixed(4)}, vNext=${vNext.toFixed(4)}, delta=${delta.toFixed(4)}, adv=${adv.toFixed(4)}`);
      }
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

    return normalized;
  }

  async updateModel(buffer) {
    const advantages = this.computeGAE(buffer);
    //console.log("advantages", advantages);
    console.log('Advantages stats:', {
      length: advantages.length,
      sum: advantages.reduce((a, b) => a + b, 0),
      mean: advantages.reduce((a, b) => a + b, 0) / advantages.length,
      max: Math.max(...advantages),
      min: Math.min(...advantages),
      allZero: advantages.every(v => v === 0)
    });
    const states = tf.tensor(buffer.map(b => b.state));
    //console.log('states shape:', states.shape);//[buffer.length, 1260]
    const actions = tf.tensor(buffer.map(b => b.action));
    //console.log('actions shape:', actions.shape);//[buffer.length]
    if (actions.dataSync().some(a => isNaN(a))) {
      console.log("actions存在NaN值");
    }
    if (actions.dataSync().every(a => a === 0)) {
      console.log("actions全0");
    }
    const oldLogProbs = tf.tensor(buffer.map(b => b.logProb));
    //console.log('oldLogProbs shape:', oldLogProbs.shape);//[buffer.length]
    const adv = tf.tensor(advantages);
    let metrics = {};
    console.log("开始优化Actor")
    // ===== Actor update =====
    tf.tidy(() => {
      const { mean, std } = this.actor.forward(states);
      console.log('mean:', mean.dataSync());
      console.log('std:', std.dataSync());
      const logProb = tf.sum(
        tf.sub(
          tf.log(tf.div(1, tf.mul(std, tf.sqrt(2 * Math.PI)))),
          tf.div(tf.square(tf.sub(actions, mean)), tf.mul(2, tf.square(std)))
        ),
        -1
      );
      console.log('logProb:', logProb.dataSync());
      console.log('oldLogProbs:', oldLogProbs.dataSync());
      const ratio = tf.exp(tf.sub(logProb, oldLogProbs));
      console.log('ratio:', ratio.dataSync());
      console.log('adv:', adv.dataSync());
      const surr1 = tf.mul(ratio, adv);
      //console.log('surr1:', surr1);
      const surr2 = tf.mul(
        tf.clipByValue(ratio, 1 - this.clip, 1 + this.clip),
        adv
      );
      //console.log('surr2:', surr2);

      const loss = tf.neg(tf.mean(tf.minimum(surr1, surr2)));
      const kl = tf.mean(tf.sub(oldLogProbs, logProb));
      metrics.lossActor = loss.dataSync()[0];
      metrics.kl = kl.dataSync()[0];

    });
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

    const weights = this.actor.fc1.getWeights()[0];
    const norm = weights.norm().dataSync()[0];
    console.log('Weight norm before/after:', norm);
    if (weights.dataSync().some(v => isNaN(v)))
      console.log("The actor weights has NaN values");

    console.log("开始优化Critic")
    // ===== Critic update =====
    tf.tidy(() => {
      const values = this.critic.value(states).reshape([-1]);
      console.log('Critic values:', values.dataSync());
      //console.log('Critic adv:', adv.dataSync());
      const returns = tf.add(adv, values);
      console.log('Critic returns:', returns.dataSync());
      let loss = tf.losses.meanSquaredError(returns, values);
      //console.log('Critic loss:', loss.dataSync()[0]);
      metrics.lossCritic = loss.dataSync()[0];
    });
    this.criticOpt.minimize(() => {
      const values = this.critic.value(states).reshape([-1]);
      const returns = tf.add(adv, values);
      return tf.losses.meanSquaredError(returns, values);
      //console.log("returns", returns);
      // const delta = 1.0;
      // const errors = tf.sub(returns, values);
      // const absErrors = tf.abs(errors);
      // // Huber Loss 的等效公式（无需条件判断）
      // // L = 0.5 * error^2  if |error| <= delta
      // // L = delta * |error| - 0.5 * delta^2  if |error| > delta
      // // 等价于：L = 0.5 * min(|error|, delta)^2 + delta * max(|error| - delta, 0)

      // const clippedAbs = tf.minimum(absErrors, delta);
      // const quadraticPart = tf.mul(0.5, tf.square(clippedAbs));
      // const linearPart = tf.mul(delta, tf.maximum(tf.sub(absErrors, delta), 0));
      // const huberLoss = tf.add(quadraticPart, linearPart);

      // return tf.mean(huberLoss);
    });
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

    this.critic.model.layers.forEach((layer, index) => {
      const weights = layer.getWeights();
      if (weights.length > 0) {
        let totalSum = 0;
        let totalCount = 0;
        weights.forEach((weight) => {
          const data = weight.dataSync();
          totalSum += data.reduce((a, b) => a + b, 0);
          totalCount += data.length;
        });
        const totalMean = totalSum / totalCount;
        console.log(`Critic Layer ${layer.name} 总均值: ${totalMean.toFixed(6)}`);
      }
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
    let history = [];
    for (let i = 0; i < episodes; i++) {
      this.env.reset();
      let buffer = await this.RunGame();
      if (buffer == null) {
        console.error("RunGame return null, episode:", i);
        console.log(history);
        return;
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
      const metrics = await this.updateModel(buffer);
      // let record =
      // {
      //   episode: i,
      //   reward: buffer.reduce((a, b) => a + b.reward, 0),
      //   lossActor: metrics.lossActor,//合法范围波动才对
      //   lossCritic: metrics.lossCritic,//应该逐渐下降才对
      //   kl: metrics.kl,//应该适中
      //   entropy: metrics.entropy,//衡量探索度，应该先高后底
      //   advMean: metrics.advMean,//应该接近0
      //   buffer_len: buffer.length,
      // };
      let record =
      {
        episode: i,
        reward: buffer.reduce((a, b) => a + b.reward, 0),
        lossActor: metrics.lossActor,//合法范围波动才对
        lossCritic: metrics.lossCritic,//应该逐渐下降才对
        kl: metrics.kl,//应该适中
        advMean: metrics.advMean,//应该接近0
        buffer_len: buffer.length,
      };
      console.log(record);
      history.push(record);
    }
    drawlineGraph(history);//绘图
    alert("训练完成,开始评估");
    console.log(history);
    await this.test();
  }
  async test() {
    this.env.render_mode = "render"
    await this.PlayGame(1000);
  }
}