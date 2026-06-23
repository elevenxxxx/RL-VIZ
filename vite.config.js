import { defineConfig } from 'vite'

export default defineConfig(({ command, mode }) => {
    // mode 为 'production' 时使用 /RL-VIZ/，否则使用 /
    const base = mode === 'production' ? '/RL-VIZ/' : '/src/'
    console.log('当前模式:', mode, 'Base路径:', base)
    return {
        base: base,
    }
})