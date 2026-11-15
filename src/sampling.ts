/** A sampled result applies near its timestamp only; this does not track objects. */
export function frameAt<T extends { timestamp: number }>(frames: T[], timestamp: number, sampleFps = 4): T | undefined {
  if (!frames.length || !Number.isFinite(timestamp) || sampleFps <= 0) return undefined
  let left = 0, right = frames.length - 1
  while (left < right) {
    const middle = (left + right) >> 1
    if (frames[middle].timestamp < timestamp) left = middle + 1
    else right = middle
  }
  const index = left > 0 && Math.abs(frames[left - 1].timestamp - timestamp) < Math.abs(frames[left].timestamp - timestamp) ? left - 1 : left
  const before = index > 0 ? frames[index].timestamp - frames[index - 1].timestamp : Infinity
  const after = index + 1 < frames.length ? frames[index + 1].timestamp - frames[index].timestamp : Infinity
  const gap = Math.min(before, after, 1 / sampleFps * 1.2)
  return Math.abs(frames[index].timestamp - timestamp) <= gap / 2 + .003 ? frames[index] : undefined
}
