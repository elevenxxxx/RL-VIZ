import { SharedMazeEnv } from "./maze/maze_shared_env.js";
import { QLearningMazeAgent } from "../agents/q_learning_maze_agent.js";
import { PPOMazeAgent } from "../agents/ppo_maze_agent.js";

export const environmentRegistry = {
  maze: {
    key: "maze",
    label: "Maze",
    createEnvironment: (options = {}) => new SharedMazeEnv(options),
  },
  chess: {
    key: "chess",
    label: "Chinese Chess",
    existingPage: "/src/pages/chess.html",
  },
};

export const trainingRegistry = {
  "maze:q_learning": {
    envKey: "maze",
    algorithmKey: "q_learning",
    label: "Maze + Q-learning",
    createEnvironment: environmentRegistry.maze.createEnvironment,
    createAgent: (env, options = {}) => new QLearningMazeAgent(env, options),
  },
  "maze:ppo": {
    envKey: "maze",
    algorithmKey: "ppo",
    label: "Maze + PPO",
    createEnvironment: environmentRegistry.maze.createEnvironment,
    createAgent: (env, options = {}) => new PPOMazeAgent(env, options),
  },
  "chess:ppo": {
    envKey: "chess",
    algorithmKey: "ppo",
    label: "Chess + PPO",
    existingPage: environmentRegistry.chess.existingPage,
  },
};

export function getTrainingRegistration(envKey, algorithmKey) {
  return trainingRegistry[`${envKey}:${algorithmKey}`] ?? null;
}
