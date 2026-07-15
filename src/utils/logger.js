const START = Date.now();

function ts() {
  return `+${((Date.now() - START) / 1000).toFixed(1)}s`;
}

/** General pipeline log line. */
export function log(msg) {
  console.log(`[${ts()}] ${msg}`);
}

/** Log line tagged with the agent that produced it, for readable CLI output. */
export function logStep(agent, msg) {
  console.log(`[${ts()}] [${agent}] ${msg}`);
}
