import { Game } from "./env.js";
import { piece2id, rc2num, num2rc, encode_action } from "./utils.js";
import { Agent } from "./agent.js";
import { drawlineGraph } from "./graph.js";

const env = new Game();
let agent = new Agent(env);

export function reset() {
    env.reset();
}

export async function train() {
    await agent.train();
}
export async function test() {
    await agent.test();
}
export async function load() {

}
export async function save() {

}
export function drawGraph() {
    let history = [];
    for (let i = 0; i < 800; i++) {
        history.push({
            episode: i,
            reward: Math.random() * 10,
            lossActor: Math.random() * 15,//合法范围波动才对
            lossCritic: Math.random() * 15,//应该逐渐下降才对
            kl: Math.random() * 3,//应该适中
            advMean: Math.random(),//应该接近0
            buffer_len: Math.floor(Math.random() * 100),
        })
    }
    drawlineGraph(history);
}
// reset();
// // 挂到全局
// window.reset = reset;
// window.train = train;
// window.test = test;
// window.load = load;
// window.save = save;
// window.drawGraph = drawGraph;
