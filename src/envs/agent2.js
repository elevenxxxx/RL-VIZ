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
            biasInitializer: 'randomNormal'
        });
        this.mean = tf.layers.dense({
            units: actionDim,
            activation: "tanh",
            kernelInitializer: 'glorotNormal',  // 明确指定初始化器
            biasInitializer: 'randomNormal'
        });
        this.logStd = tf.variable(tf.zeros([actionDim]));
        this.actionScale = 187;
    }

    forward(state) {
        return tf.tidy(() => {
            const x = this.fc1.apply(state);
            const mean = this.mean.apply(x);
            // console.log('fc1 输出 shape:', x.shape);  // 应该是 [1, 256]
            const log_std = this.logStd.broadcastTo(mean.shape);
            const std = tf.exp(tf.clipByValue(log_std, -20, 2));
            return { mean, std };
        });
    }
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
            const eps = tf.randomNormal(mean.shape);
            const u = tf.add(mean, tf.mul(std, eps));
            //const tanh_u = tf.tanh(mean);
            const tanh_u = tf.tanh(u);
            const a = tf.mul(tf.abs(tanh_u), this.actionScale);
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

    async updateModel(buffer) {
        const states = tf.tensor(buffer.map(b => b.state));
        const old_V = this.critic.value(states).reshape([-1])
        const old_values = old_V.dataSync();
        const old_V_detached = tf.tensor(old_values, old_V.shape);
        const advantages = new Array(buffer.length);
        let x = 0;

        for (let i = buffer.length - 1; i >= 0; i--) {
            const reward = buffer[i].reward;
            const done = buffer[i].done;

            const v = old_values[i];
            const vNext = (done || i === buffer.length - 1)
                ? 0
                : old_values[i + 1];
            //TD差分公式
            const delta =
                reward +
                this.gamma * vNext -
                v;

            x = delta + this.gamma * this.lam * (1 - done) * x;

            advantages[i] = x;
        }

        const actions = tf.tensor(buffer.map(b => b.action));
        const oldLogProbs = tf.tensor(buffer.map(b => b.logProb));
        const adv = tf.tensor(advantages);

        let metrics = {};
        // ===== Actor update =====
        let losss = this.actorOpt.minimize(() => {
            const { mean, std } = this.actor.forward(states);
            const u = actions;
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
        metrics.lossActor = losss.dataSync()[0];
        // ===== Critic update =====
        const returns = tf.add(adv, old_V_detached);
        const targets = returns.clone();
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
            if (buffer.length > 100) {
                console.error("buffer长度大于100:", buffer.length);
                buffer = buffer.slice(0, 100); // 改为取前100个样本
            }
            const metrics = await this.updateModel(buffer);
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
            history.push(record);
        }
        drawlineGraph(history);//绘图
        alert("训练完成,开始评估");
        console.log(history);
        await this.test();
    }
}