const EPISODE_TRACE_TEMPLATE = {
  envType: "generic",
  episode: 0,
  totalReward: 0,
  totalSteps: 0,
  summary: {},
  metadata: {},
  steps: [],
};

const EPISODE_STEP_TEMPLATE = {
  index: 0,
  stateBefore: null,
  stateAfter: null,
  actionId: null,
  actionLabel: "",
  selectedAction: null,
  reward: 0,
  rewardBreakdown: {},
  policyTopActions: [],
  entropy: null,
  done: false,
  info: {},
};

export function createEpisodeTrace(values = {}) {
  return {
    ...EPISODE_TRACE_TEMPLATE,
    ...values,
    summary: {
      ...EPISODE_TRACE_TEMPLATE.summary,
      ...(values.summary ?? {}),
    },
    metadata: {
      ...EPISODE_TRACE_TEMPLATE.metadata,
      ...(values.metadata ?? {}),
    },
    steps: Array.isArray(values.steps) ? values.steps : [],
  };
}

export function createEpisodeStepRecord(values = {}) {
  return {
    ...EPISODE_STEP_TEMPLATE,
    ...values,
    rewardBreakdown: {
      ...EPISODE_STEP_TEMPLATE.rewardBreakdown,
      ...(values.rewardBreakdown ?? {}),
    },
    policyTopActions: Array.isArray(values.policyTopActions) ? values.policyTopActions : [],
    info: {
      ...EPISODE_STEP_TEMPLATE.info,
      ...(values.info ?? {}),
    },
  };
}
