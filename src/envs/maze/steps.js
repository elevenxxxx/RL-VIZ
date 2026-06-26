//await import("https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js")

export const trainSteps = [

  "ε-greedy 选择动作",
  "执行动作获得 r, s'",
  "更新 Q(s,a)",

];
export const trainStepsInfo = [
  `<div>
    ε-greedy 贪心策略标准形式：

    <div>
$$
a_t = \\left\\{
\\begin{matrix}
\\arg\\max_{a} Q(s_t, a), & \\text{以概率 } 1-\\epsilon \\\\
\\text{随机动作}, & \\text{以概率 } \\epsilon
\\end{matrix}
\\right.
$$
    </div>

    <div>
也就是有 \\( 1-\\epsilon \\)  概率，在当前状态 \\( s \\) 对应的所有动作 a 中，取使 \\( Q(s,a) \\) 最大的那个 a，如果有多个满足条件则随机选一个；
</div>

    <div>
有 \\( \\epsilon \\)  概率选择随机动作。
    </div>
  </div>`,

  `<div>
智能体执行选择的动作，由环境给出执行后得到的奖励 r 和下一个状态 s'。
</div>`,

  `<div>
$$
Q(s, a)=Q(s, a)\\alpha \\Big[ r + \\gamma \\max_{a'} Q(s', a') - Q(s, a) \\Big]
$$

<div>
其中 \\( \\max_{a'} Q(s', a') \\)  是下一个状态 \\( s' \\) 对应的所有动作 a' 的 Q 值中最大的值。
</div>
</div>`
];
export const testSteps = [

  "argmax Q(s,a) 选择动作",
  "执行动作得到下一个状态s'",

];

export const testStepsInfo = [
  `<div>
    $$
    a_t = 
    \arg\max_{a} Q(s_t, a)
    $$
  </div>

  <div>
    即在当前状态 \\( s_t \\) 对应的所有动作 a 中，取使 \\( Q(s_t, a) \\) 最大的那个 a，如果有多个满足条件则随机选一个。
  </div>`,

  `<div>
    智能体执行选择的动作，不需要关心执行后得到的奖励，只需要关心下一个状态 s'。
  </div>`
];

function renderFlow(id, title, steps) {
  const el = document.getElementById(id);

  el.innerHTML = `
    <div class="flow-title">${title}</div>
    <div class="flow-column">
      ${steps.map((s, i) => `
        <div class="flow-node" data-index="${i}">
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
    renderFlow("maze_Steps", "推理流程", testSteps);
    await animateFlow(index);
  }
}
export function initFlow(mode) {
  if (mode === "train") {
    renderFlow("maze_Steps", "训练流程", trainSteps);
  }
  else if (mode === "test") {
    renderFlow("maze_Steps", "推理流程", testSteps);
  }
}
