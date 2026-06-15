import { Game } from "./env.js";
const env = new Game();
let autoplaying = false;
let now_action = null;
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
//状态 动作编码
function encode_state(state) {

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

reset();
// 挂到全局
window.reset = reset;
window.autoplay = autoplay;
