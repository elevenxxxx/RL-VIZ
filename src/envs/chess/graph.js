let chart = null;

export function initGraph() {
    const dom = document.getElementById("graph-container");

    chart = echarts.init(dom);

    chart.setOption({
        title: {
            text: "PPO Training Metrics"
        },
        tooltip: {
            trigger: "axis"
        },
        legend: {
            top: 30
        },
        grid: {
            left: "5%",
            right: "5%",
            bottom: "10%",
            containLabel: true
        },
        xAxis: {
            type: "category",
            name: "Episode",
            data: []
        },
        yAxis: [
            {
                type: "value",
                name: "Reward"
            },
            {
                type: "value",
                name: "Loss / KL",
                position: "right"
            }
        ],
        series: []
    });

    window.addEventListener("resize", () => chart.resize());
}

export function drawlineGraph(history) {

    if (!chart) {
        initGraph();
    }

    const episodes = history.map(x => x.episode);

    chart.setOption({
        xAxis: {
            data: episodes
        },
        series: [
            {
                name: "Reward",
                type: "line",
                smooth: true,
                yAxisIndex: 0,
                data: history.map(x => x.reward)
            },
            {
                name: "Actor Loss",
                type: "line",
                smooth: true,
                yAxisIndex: 1,
                data: history.map(x => x.lossActor)
            },
            {
                name: "Critic Loss",
                type: "line",
                smooth: true,
                yAxisIndex: 1,
                data: history.map(x => x.lossCritic)
            },

            {
                name: "AdvMean",
                type: "line",
                smooth: true,
                yAxisIndex: 1,
                data: history.map(x => x.advMean)
            },
            {
                name: "Steps",
                type: "line",
                smooth: true,
                yAxisIndex: 1,
                data: history.map(x => x.buffer_len)
            }
        ]
    });
}