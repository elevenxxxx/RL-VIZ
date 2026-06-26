import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig(({ mode }) => {
    const base = mode === 'production' ? '/RL-VIZ/' : '/'

    return {
        base,
        build: {
            rollupOptions: {
                input: {
                    main: resolve(__dirname, 'index.html'),
                    chess: resolve(__dirname, 'src/pages/chess.html'),
                    maze: resolve(__dirname, 'src/pages/maze.html'),
                    mazeCompare: resolve(__dirname, 'src/pages/maze_compare.html'),
                }
            }
        }
    }
})
