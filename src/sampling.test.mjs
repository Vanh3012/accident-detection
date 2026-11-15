import { test } from 'node:test'
import assert from 'node:assert/strict'
import { frameAt } from './sampling.ts'

test('seeking and replay select by video time, not by a prior playback state', () => {
  const frames = [0, .25, .5, .75].map(timestamp => ({ timestamp }))
  assert.equal(frameAt(frames, .71), frames[3])
  assert.equal(frameAt(frames, .02), frames[0])
  assert.equal(frameAt(frames, .49), frames[2])
})
test('a frame cannot remain active through an analysis gap or after the video', () => {
  const frames = [{timestamp: 0}, {timestamp: 1}]
  assert.equal(frameAt(frames, .5), undefined)
  assert.equal(frameAt(frames, 4), undefined)
  assert.equal(frameAt([], 0), undefined)
})
test('sampling at 1 and 10 FPS uses bounded windows around genuine results', () => {
  const frame = {timestamp: 1}
  assert.equal(frameAt([frame], 1.04, 10), frame)
  assert.equal(frameAt([frame], 1.2, 10), undefined)
  assert.equal(frameAt([frame], 1.4, 1), frame)
  assert.equal(frameAt([frame], 2, 1), undefined)
})
