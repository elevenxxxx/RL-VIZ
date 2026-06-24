//await import("https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js")

const trainSteps = [

  "ε-greedy 选择动作",
  "执行动作获得 r, s'",
  "更新 Q(s,a)",

];

const testSteps = [

  "获取当前状态 s",
  "argmax Q(s,a) 选择动作",
  "执行动作",

];

function renderFlow(id, title, steps) {
  const el = document.getElementById(id);

  el.innerHTML = `
    <div class="flow-title">${title}</div>
    <div class="flow-column">
      ${steps.map((s, i) => `
        <div class="flow-node" data-info="${s}" data-index="${i}">
          ${i + 1}. ${s}
        </div>
      `).join("")}
    </div>
  `;
}

async function animateFlow(index, speed = 150) {
  const container = document.getElementById("maze_Steps");
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
export async function setStepRender(mode, index) {
  if (mode === "train") {
    renderFlow("maze_Steps", "训练流程", trainSteps);
    await animateFlow(index);
  }
  else if (mode === "test") {
    renderFlow("maze_Steps", "测试流程", testSteps);
    await animateFlow(index);
  }
}
