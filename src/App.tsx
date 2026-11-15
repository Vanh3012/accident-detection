import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { frameAt } from './sampling'
import { AlertTriangle, Check, LoaderCircle, Maximize, Pause, Play, Shield, Upload, Volume2, VolumeX, X } from 'lucide-react'

const API = import.meta.env.VITE_API_URL || ''
type Prediction = { class: string; confidence: number; x: number; y: number; width: number; height: number }
type Accident = { type: string; title: string; class_id: number; confidence: number }
type Frame = { timestamp: number; predictions: Prediction[]; classification: { status: string; accidents: Accident[]; primary: Accident | null } }
type Incident = { id: string; timestamp: number; last_seen: number; type: string; title: string; confidence: number }
type Health = { api_key_configured: boolean; confidence_threshold: number; sample_fps: number }
const time = (seconds: number) => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`
const names: Record<string, string> = { car: 'Car', bike: 'Bike', person: 'Person' }
const colors: Record<string, string> = { car: '#57d39a', bike: '#65aaff', person: '#f4cd66' }
async function api(path: string, options?: RequestInit) {
  const response = await fetch(`${API}${path}`, options)
  const data = await response.json()
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Yêu cầu thất bại.')
  return data
}

export default function App() {
  const video = useRef<HTMLVideoElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const input = useRef<HTMLInputElement>(null)
  const request = useRef(0)
  const audio = useRef<AudioContext | null>(null)
  const lastAlarm = useRef('')
  const [health, setHealth] = useState<Health | null>(null)
  const [id, setId] = useState('')
  const [filename, setFilename] = useState('')
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [status, setStatus] = useState<'idle' | 'analyzing' | 'completed' | 'error'>('idle')
  const [error, setError] = useState('')
  const [progress, setProgress] = useState(0)
  const [frames, setFrames] = useState<Frame[]>([])
  const [events, setEvents] = useState<Incident[]>([])
  const [source, setSource] = useState('')
  const [pollingId, setPollingId] = useState('')
  const [now, setNow] = useState(0)
  const [duration, setDuration] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [volume, setVolume] = useState(0)
  const [sound, setSound] = useState(false)
  const [speed, setSpeed] = useState(1)
  const threshold = health?.confidence_threshold ?? .5

  useEffect(() => { api('/api/health').then(setHealth).catch(() => setHealth(null)) }, [])
  const acceptResults = useCallback((data: { results: Frame[]; events: Incident[]; source: string }) => {
    setFrames(data.results); setEvents(data.events); setSource(data.source); setStatus('completed'); setProgress(100)
  }, [])
  useEffect(() => {
    if (status !== 'analyzing' || !id || pollingId !== id) return
    let cancelled = false
    let timer: number
    const poll = async () => {
      try {
        const data = await api(`/api/analyze/${id}/status`)
        if (cancelled) return
        setProgress(data.progress)
        if (data.status === 'error') throw new Error(data.error || 'Phân tích thất bại.')
        if (data.status === 'completed') {
          const result = await api(`/api/analyze/${id}/results`)
          if (!cancelled) acceptResults(result)
        } else timer = window.setTimeout(poll, 900)
      } catch (err) {
        if (!cancelled) { setError(err instanceof Error ? err.message : 'Mất kết nối.'); setStatus('error') }
      }
    }
    timer = window.setTimeout(poll, 900)
    return () => { cancelled = true; window.clearTimeout(timer) }
  }, [status, id, pollingId, acceptResults])

  const reset = () => {
    request.current++; video.current?.pause(); setId(''); setFilename(''); setFrames([]); setEvents([])
    setStatus('idle'); setProgress(0); setSource(''); setError(''); setNow(0); setDuration(0)
    setBusy(false); setPlaying(false); setPollingId(''); lastAlarm.current = ''
  }
  const upload = async (file?: File) => {
    if (!file || busy || status === 'analyzing') return
    reset(); const token = request.current; setBusy(true)
    try {
      const body = new FormData(); body.append('file', file)
      const data = await api('/api/upload', { method: 'POST', body })
      if (token !== request.current) return
      setId(data.video_id); setFilename(data.filename)
      await analyze(data.video_id, token)
    } catch (err) { if (token === request.current) setError(err instanceof Error ? err.message : 'Upload thất bại.') }
    finally { if (token === request.current) setBusy(false) }
  }
  const analyze = async (videoId = id, token = request.current) => {
    if (!videoId) return
    video.current?.pause(); setPollingId(''); setStatus('analyzing'); setFrames([]); setEvents([]); setSource(''); setProgress(0); setError('')
    try {
      const data = await api(`/api/analyze/${videoId}`, { method: 'POST' })
      if (token !== request.current) return
      if (data.status === 'completed') {
        const result = await api(`/api/analyze/${videoId}/results`)
        if (token === request.current) acceptResults(result)
      } else setPollingId(videoId)
    } catch (err) { if (token === request.current) { setError(err instanceof Error ? err.message : 'Phân tích thất bại.'); setStatus('error') } }
  }

  // Read the video clock every animation frame, including seek, replay and playback speed changes.
  useEffect(() => {
    if (!id) return
    let handle: number
    const tick = () => { if (video.current) setNow(video.current.currentTime); handle = requestAnimationFrame(tick) }
    handle = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(handle)
  }, [id])
  const frame = useMemo(() => status === 'completed' ? frameAt(frames, now, health?.sample_fps || 4) : undefined, [frames, now, status, health])
  const primary = frame?.classification.primary
  const accidents = useMemo(() => {
    const types = new Map<string, Accident>()
    for (const item of frame?.classification.accidents || []) {
      if (!types.has(item.type) || item.confidence > types.get(item.type)!.confidence) types.set(item.type, item)
    }
    return [...types.values()].sort((a, b) => b.confidence - a.confidence)
  }, [frame])
  const counts = { car: 0, bike: 0, person: 0 }
  for (const prediction of frame?.predictions || []) {
    const name = prediction.class.trim().toLowerCase()
    if (prediction.confidence >= threshold && name in counts) counts[name as keyof typeof counts]++
  }
  const draw = useCallback(() => {
    const player = video.current, overlay = canvas.current, ctx = overlay?.getContext('2d')
    if (!player || !overlay || !ctx) return
    const { width, height } = player.getBoundingClientRect(), dpr = window.devicePixelRatio || 1
    overlay.width = Math.round(width * dpr); overlay.height = Math.round(height * dpr)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height)
    if (!frame || !player.videoWidth || !player.videoHeight) return
    const scale = Math.min(width / player.videoWidth, height / player.videoHeight)
    const ox = (width - player.videoWidth * scale) / 2, oy = (height - player.videoHeight * scale) / 2
    for (const p of frame.predictions) {
      if (![p.x, p.y, p.width, p.height, p.confidence].every(Number.isFinite) || p.confidence < threshold || p.width <= 0 || p.height <= 0) continue
      const name = p.class.trim().toLowerCase(), isAccident = name.endsWith('_accident')
      if (!isAccident && !(name in names)) continue
      const x = ox + Math.max(0, p.x - p.width / 2) * scale
      const y = oy + Math.max(0, p.y - p.height / 2) * scale
      const right = ox + Math.min(player.videoWidth, p.x + p.width / 2) * scale
      const bottom = oy + Math.min(player.videoHeight, p.y + p.height / 2) * scale
      if (right <= x || bottom <= y) continue
      const color = isAccident ? '#ff626b' : colors[name]
      const label = `${isAccident ? name.replaceAll('_', ' ') : names[name]} ${Math.round(p.confidence * 100)}%`
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.strokeRect(x, y, right - x, bottom - y)
      ctx.font = '600 12px system-ui'; const labelWidth = ctx.measureText(label).width + 12
      const lx = Math.max(ox, Math.min(x, ox + player.videoWidth * scale - labelWidth)), ly = Math.max(oy, y - 22)
      ctx.fillStyle = color; ctx.fillRect(lx, ly, labelWidth, 21); ctx.fillStyle = '#10151c'; ctx.fillText(label, lx + 6, ly + 15)
    }
  }, [frame, threshold])
  useEffect(() => { draw(); const observer = new ResizeObserver(draw); if (video.current) observer.observe(video.current); return () => observer.disconnect() }, [draw, id])
  useEffect(() => {
    if (!primary || !sound || !playing) return
    const incident = events.find(e => e.type === primary.type && now >= e.timestamp - .15 && now <= e.last_seen + .15)
    if (!incident || incident.id === lastAlarm.current) return
    lastAlarm.current = incident.id
    try {
      audio.current ??= new AudioContext(); const ctx = audio.current
      void ctx.resume(); const osc = ctx.createOscillator(), gain = ctx.createGain()
      osc.frequency.value = 880; gain.gain.value = .1; osc.connect(gain); gain.connect(ctx.destination); osc.start(); osc.stop(ctx.currentTime + .18)
    } catch { /* Audio is optional. */ }
  }, [primary, sound, playing, events, now])
  const seek = (value: number) => { if (video.current) { video.current.currentTime = value; setNow(value); lastAlarm.current = '' } }
  const togglePlay = () => { if (!video.current || status !== 'completed') return; if (video.current.paused) void video.current.play().catch(() => setError('Trình duyệt không phát được codec này. Hãy dùng MP4 H.264.')); else video.current.pause() }
  const heading = primary ? 'PHÁT HIỆN TAI NẠN' : status === 'analyzing' ? 'ĐANG PHÂN TÍCH' : status === 'error' ? 'LỖI PHÂN TÍCH' : frame ? 'NORMAL' : 'CHƯA CÓ KẾT QUẢ'

  return <div className="dashboard">
    <header className="header"><div className="logo"><Shield size={23}/><strong>TrafficGuard <span>AI</span></strong></div><span className={`connection-dot ${health ? 'online' : ''}`}>{health ? 'Backend sẵn sàng' : 'Backend chưa kết nối'}</span></header>
    <main>
      <div className="title-row"><div><h1>Phân tích video giao thông</h1><p>Nhận diện phương tiện và phân loại tai nạn bằng Roboflow.</p></div><button className="secondary" onClick={() => input.current?.click()} disabled={busy || status === 'analyzing'}><Upload size={15}/> Chọn video</button></div>
      <input type="file" ref={input} accept="video/*,.mp4,.mov,.avi,.mkv,.webm,.m4v" hidden onChange={e => { void upload(e.target.files?.[0]); e.target.value = '' }}/>
      <div className="layout">
        <section className="video-column">
          <div className={`player ${dragging ? 'dragging' : ''}`} onDragOver={e => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); void upload(e.dataTransfer.files[0]) }}>
            {id ? <><div className="video-stage"><video ref={video} src={`${API}/api/video/${id}`} playsInline muted={volume === 0} onLoadedMetadata={e => { setDuration(e.currentTarget.duration); e.currentTarget.volume = volume; e.currentTarget.playbackRate = speed; draw() }} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => { setPlaying(false); lastAlarm.current = '' }} onClick={togglePlay}/><canvas ref={canvas}/><button className="remove" aria-label="Đóng video" onClick={reset}><X size={16}/></button>{status === 'analyzing' && <div className="processing"><LoaderCircle className="spin" size={25}/><span>Đang phân tích · {progress}%</span></div>}</div>
            <div className="controls"><button aria-label={playing ? 'Tạm dừng' : 'Phát video'} onClick={togglePlay} disabled={status !== 'completed'}>{playing ? <Pause size={18}/> : <Play size={18}/>}</button><span>{time(now)} / {time(duration)}</span><input aria-label="Tua video" className="seek" type="range" min={0} max={duration || 1} step={.01} value={Math.min(now, duration)} onChange={e => seek(Number(e.target.value))}/><select aria-label="Tốc độ phát" value={speed} onChange={e => { const rate = Number(e.target.value); setSpeed(rate); if (video.current) video.current.playbackRate = rate }}>{[.5, 1, 1.5, 2].map(rate => <option key={rate} value={rate}>{rate}×</option>)}</select><button aria-label="Bật/tắt tiếng video" onClick={() => { const next = volume ? 0 : .8; setVolume(next); if (video.current) video.current.volume = next }}>{volume ? <Volume2 size={17}/> : <VolumeX size={17}/>}</button><input aria-label="Âm lượng" className="volume" type="range" min={0} max={1} step={.05} value={volume} onChange={e => { const value = Number(e.target.value); setVolume(value); if (video.current) video.current.volume = value }}/><button aria-label="Toàn màn hình" onClick={() => { if (document.fullscreenElement) void document.exitFullscreen(); else void video.current?.closest('.player')?.requestFullscreen() }}><Maximize size={17}/></button></div></> : <button className="drop-area" disabled={busy} onClick={() => input.current?.click()}>{busy ? <LoaderCircle size={28} className="spin"/> : <Upload size={28}/>}<strong>{busy ? 'Đang tải video…' : 'Kéo thả hoặc chọn video'}</strong><span>MP4, MOV, AVI, MKV, WEBM · Tối đa 500 MB</span></button>}
          </div>
          {id && <div className="file-row"><div><strong title={filename}>{filename}</strong><small>{status === 'completed' ? source === 'cache' ? 'Kết quả Roboflow đã lưu' : 'Phân tích Roboflow hoàn tất' : status === 'analyzing' ? `Đang tự động phân tích · ${progress}%` : 'Video đã tải lên'}</small></div>{status === 'completed' ? <span className="ready"><Check size={15}/> Sẵn sàng phát</span> : status === 'error' ? <button className="secondary" onClick={() => void analyze()} disabled={busy}>Thử lại</button> : <LoaderCircle className="spin" size={17}/>}</div>}
          {error && <div role="alert" className="error"><AlertTriangle size={17}/>{error}</div>}

        </section>
        <aside><div className={`status-panel ${primary ? 'danger' : frame ? 'normal' : ''}`}><div className="panel-label">TRẠNG THÁI FRAME <span>{time(now)}</span></div><h2>{primary && <AlertTriangle size={19}/>} {heading}</h2><p>{primary?.title || (frame ? 'Không phát hiện tai nạn tại frame lấy mẫu.' : 'Tải và phân tích video để xem kết quả.')}</p>{primary && <div className="confidence"><span>Class {primary.class_id} · Confidence</span><strong>{Math.round(primary.confidence * 100)}%</strong></div>}{accidents.length > 1 && <ul>{accidents.slice(1).map(item => <li key={item.type}>Class {item.class_id} · {item.title} — {Math.round(item.confidence * 100)}%</li>)}</ul>}</div>
          <section className="events" aria-label="Các tình huống tai nạn"><h2>Các tình huống tai nạn <span>{events.length}</span></h2>{events.length ? <div className="event-list">{events.map(event => <button key={event.id} onClick={() => seek(event.timestamp)}><span className="event-time">{time(event.timestamp)}</span><span>{event.title}</span><b>{Math.round(event.confidence * 100)}%</b></button>)}</div> : <p>{status === 'completed' ? 'Không có nhãn tai nạn vượt ngưỡng trong video.' : 'Các mốc xuất hiện sau khi phân tích.'}</p>}</section>
          <div className="objects"><h2>Đối tượng trên frame</h2><div>{Object.entries(counts).map(([name, count]) => <span key={name}><i style={{background: colors[name]}}/>{names[name]}<b>{count}</b></span>)}</div></div>
          <label className="sound"><input type="checkbox" checked={sound} onChange={e => setSound(e.target.checked)}/> Âm thanh cảnh báo</label>
          <p className="sampling">Lấy mẫu {health?.sample_fps ?? 4} FPS. Box và trạng thái chỉ áp dụng gần frame đã phân tích, không phải tracking liên tục.</p>
        </aside>
      </div>
    </main>
  </div>
}
