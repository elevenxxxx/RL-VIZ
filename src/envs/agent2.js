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
        this.actionScale = 187;
    }

    forward(state) {
        return tf.tidy(() => {
            // console.log('输入 state shape:', state.shape);  // 应该是 [1, 1260]
            const x = this.fc1.apply(state);
            const mean = this.mean.apply(x);
            // console.log('fc1 输出 shape:', x.shape);  // 应该是 [1, 256]

            const log_std = this.logStd.broadcastTo(mean.shape);
            const std = tf.exp(tf.clipByValue(log_std, -20, 2));

            return { mean, std };
            //const logits = this.logits.apply(x);
            // console.log('logits 输出 shape:', logits.shape);  // 应该是 [1, 188]
            //return logits;
        });
    }
    //Categorical policy
    sampleAction(state) {
        return tf.tidy(() => {
            const { mean, std } = this.forward(state);
            const u = tf.randomNormal(mean.shape, mean.dataSync()[0], std.dataSync()[0]);
            const z = u.sub(mean).div(std);
            const logProbNormal = tf.scalar(-0.5).mul(z.square())
                .sub(tf.scalar(0.5).mul(tf.log(tf.scalar(2 * Math.PI))))
                .sub(tf.log(std));
            const tanh_u = tf.tanh(u);
            const correction = tf.log(tf.scalar(1).sub(tanh_u.square()).add(tf.scalar(1e-6)));
            const logProb = logProbNormal.sub(correction).sum(-1);
            const a = tf.mul(tf.abs(tanh_u), this.actionScale);
            return { u_continuous: a, logProb: logProb };
        });
    }
    sampleDetermineAction(state) {
        return tf.tidy(() => {
            const { mean, std } = this.forward(state);
            //console.log("mean", mean.dataSync()[0]);
            //console.log("std", std.dataSync()[0]);
            const u = tf.randomNormal(mean.shape, mean.dataSync()[0], std.dataSync()[0]);
            const tanh_u = tf.tanh(u);//使用mean作为确定性策略
            //console.log("tanh_u", tanh_u.dataSync()[0]);
            const a = tf.mul(tf.abs(tanh_u), this.actionScale);
            //console.log("a", a.dataSync()[0]);
            return a;
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
            biasInitializer: 'zeros'
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
        this.actor_lr = 0.01;
        this.critic_lr = 0.01;

        this.actorOpt = tf.train.adam(this.actor_lr);
        this.criticOpt = tf.train.adam(this.critic_lr);
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
                let encoded_state = encode_state(state);
                const s = tf.tensor2d([encoded_state], [1, 1260]);//(1,1260)

                const { u_continuous, logProb } = this.actor.sampleAction(s);//应该是0~187
                action = u_continuous.dataSync()[0];
                let a_discrete = this.decodeAction(action);
                actionIndex = this.env.ModifyAction(a_discrete);

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
                reward: reward / 5,
                logProb: logProbValue,
                done: done ? 1 : 0,
            });
            if (buffer.length != this.env.episode) {
                console.error(`buffer length ${buffer.length} != episode ${this.env.episode}`)
            }
            state = nextState;

            if (done) break;
        }

        return buffer;
    }

    computeGAE(buffer) {
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
        // 归一化
        const normalized = advantages.map(a => (a - mean) / (std + 1e-8));
        return { advantages: normalized, old_value: values };
    }

    async updateModel(buffer) {
        let { advantages, old_value } = this.computeGAE(buffer);
        const states = tf.tensor(buffer.map(b => b.state));
        const actions = tf.tensor(buffer.map(b => b.action));
        const oldLogProbs = tf.tensor(buffer.map(b => b.logProb));
        const adv = tf.tensor(advantages);
        const old_V = tf.tensor(old_value);
        let metrics = {};
        // ===== Actor update =====
        tf.tidy(() => {
            const { mean, std } = this.actor.forward(states);
            const u = tf.randomNormal(mean.shape, mean.dataSync()[0], std.dataSync()[0]);
            const z = u.sub(mean).div(std);
            const logProbNormal = tf.scalar(-0.5).mul(z.square())
                .sub(tf.scalar(0.5).mul(tf.log(tf.scalar(2 * Math.PI))))
                .sub(tf.log(std));
            const correction = tf.log(tf.scalar(1).sub(tf.tanh(u).square()).add(tf.scalar(1e-6)));
            const logProb = logProbNormal.sub(correction).sum(-1);
            const kl = tf.mean(tf.sub(oldLogProbs, logProb));
            metrics.kl = kl.dataSync()[0];
        });
        let losss = this.actorOpt.minimize(() => {
            const { mean, std } = this.actor.forward(states);
            const u = tf.randomNormal(mean.shape, mean.dataSync()[0], std.dataSync()[0]);
            const z = u.sub(mean).div(std);
            const logProbNormal = tf.scalar(-0.5).mul(z.square())
                .sub(tf.scalar(0.5).mul(tf.log(tf.scalar(2 * Math.PI))))
                .sub(tf.log(std));
            const correction = tf.log(tf.scalar(1).sub(tf.tanh(u).square()).add(tf.scalar(1e-6)));
            const logProb = logProbNormal.sub(correction).sum(-1);
            const ratio = tf.exp(tf.sub(logProb, oldLogProbs));
            const surr1 = tf.mul(ratio, adv);
            const surr2 = tf.mul(
                tf.clipByValue(ratio, 1 - this.clip, 1 + this.clip),
                adv
            );
            const loss = tf.neg(tf.mean(tf.minimum(surr1, surr2)));

            return loss;
        }, true);
        metrics.lossActor = loss.dataSync()[0];

        // ===== Critic update =====

        const returns = tf.add(adv, old_V);
        const targets = returns.clone();
        // 更新 Critic
        let closs = this.criticOpt.minimize(() => {
            const values = this.critic.value(states).reshape([-1]);
            return tf.losses.meanSquaredError(targets, values);
        }, true);

        metrics.lossCritic = closs.dataSync()[0];

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
            }
        });
        const result = {
            lossActor: metrics.lossActor,
            lossCritic: metrics.lossCritic,
            kl: metrics.kl,
            entropy: metrics.entropy,
            advMean: adv.mean().dataSync()[0]
        };
        tf.dispose([states, actions, oldLogProbs, adv, targets, returns, losss, closs]);
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
}