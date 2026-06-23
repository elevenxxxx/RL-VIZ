//await import("https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js")



const trainSteps = [
    "初始化 Q 表",
    "获取当前状态 s",
    "ε-greedy 选择动作",
    "执行动作获得 r, s'",
    "更新 Q(s,a)",
    "判断 episode 结束"
];

const testSteps = [
    "加载训练好的 Q 表",
    "设置 ε = 0",
    "获取当前状态 s",
    "argmax Q(s,a) 选择动作",
    "执行最优动作",
    "直到到达目标"
];

function renderSteps(containerId, title, steps) {
    const el = document.getElementById(containerId);

    el.innerHTML = `
    <div class="step-title">${title}</div>
    <div class="step-flow">
      ${steps.map((s, i) => `
        <div class="step-node" data-index="${i}">
          ${i + 1}. ${s}
        </div>
      `).join("")}
    </div>
  `;
}
function renderFlow(id, title, steps) {
    const el = document.getElementById(id);

    el.innerHTML = `
    <div class="flow-title">${title}</div>
    <div class="flow-row">
      ${steps.map((s, i) => `
        <div class="flow-node" data-i="${i}">
          ${i + 1}. ${s}
        </div>
      `).join("")}
    </div>
  `;
}

async function animateFlow(mode, index, speed = 150) {
    const container = document.getElementById("maze_" + mode + "Steps");
    const nodes = container.querySelectorAll(".flow-node");

    for (let i = 0; i < nodes.length; i++) {
        if (i <= index) {
            nodes[i].classList.add("active");
            nodes[i].classList.add("done");

            // 小延迟制造流动效果
            await new Promise(r => setTimeout(r, speed));
        } else {
            nodes[i].classList.remove("active");
        }
    }
}
await animateFlow("train", 2);
await animateFlow("test", 2);
// renderSteps("maze_trainSteps", "🟦 Training Process", trainSteps);
// renderSteps("maze_testSteps", "🟩 Testing Process", testSteps);

export function setStep(mode, index) {
    const container = document.querySelector(`#maze_${mode}Steps`);

    const nodes = container.querySelectorAll(".step-node");

    nodes.forEach(n => n.classList.remove("active"));

    if (nodes[index]) {
        nodes[index].classList.add("active");
    }
}
