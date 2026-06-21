import { Game } from "./env.js";
import { piece2id, rc2num, num2rc ,encode_action} from "./utils.js";
import { Agent } from "./agent.js";

const env = new Game();
let agent=new Agent(env);
let autoplaying = false;
let now_action =encode_action([piece2id('red', '炮', 1),rc2num(7,4)]) ;//当头炮

function reset() {
    now_action = env.reset();
    autoplay(false);
}
function autoplay(x) {

    if (x == true || x == false) {
        autoplaying = x;
    } else {
        autoplaying = !autoplaying;
    }
    document.getElementById('autoplay-btn').innerText = autoplaying ? '停止' : '开始对弈';

    if (autoplaying) {
        start();
    }
}
function sample_action(state) {
    
}

function start() {
    let action = now_action;
    let episode = 0;
    let max_episode = 1000;
    while (env.checkWin() == null && autoplaying) {
        episode++;
        const { next_state, reward, terminated, truncated } = env.step(action);
        next_action = sample_action(next_state);
        action = next_action;
        if (episode >= max_episode) {
            break;
        }
    }
}
async function train() {
    await agent.train();
}
function test(){
    reset();

}

reset();
// 挂到全局
window.reset = reset;
window.autoplay = autoplay;
window.train = train;
window.test = test;
