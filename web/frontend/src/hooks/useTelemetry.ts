import { useState, useEffect, useRef } from 'react'
import useWebSocket, { ReadyState } from 'react-use-websocket'

const WS_URL = 'ws://localhost:8000/ws/telemetry'

export interface Telemetry {
    step: number
    match_id: string
    stats: {
        fps_physics: number
        fps_training: number
        reward_mean: number
        loss: number
        win_rate: number
        active_model: string
        curriculum_stage: number
        generation?: number
        iteration?: number
        agent_id?: number
        env_idx?: number
    }
    logs: string[]
    race_state: {
        pods: Array<{
            id: number
            team: number
            x: number
            y: number
            vx: number
            vy: number
            angle: number
            boost: number
            shield: number
            lap: number
            next_checkpoint: number
            reward: number
            thrust: number
        }>
        checkpoints: Array<{
            x: number
            y: number
            id: number
            radius: number
        }>
    }
}

export type ConnectionStatus = 'Connecting' | 'Open' | 'Closing' | 'Closed' | 'Uninstantiated'

export const useTelemetry = () => {
    const { lastJsonMessage, readyState, sendMessage } = useWebSocket(WS_URL, {
        shouldReconnect: () => true,
        retryOnError: true,
        reconnectAttempts: 10,
        reconnectInterval: 3000,
    })

    const [telemetry, setTelemetry] = useState<Telemetry | null>(null)
    const [history, setHistory] = useState<any[]>([])

    // Replay State - Use Refs for high-frequency updates and buffering without re-renders
    const replayQueue = useRef<ParsedFrame[]>([])
    const replayCursor = useRef(0)
    const lastFrameTime = useRef(0)

    // Merge incoming messages into state
    useEffect(() => {
        if (lastJsonMessage) {
            const msg = lastJsonMessage as any

            if (msg.type === "race_replay" && msg.format === "binary_base64") {
                // New Binary Format
                const frames = parseBinaryReplay(msg.payload, msg.checkpoints, msg.race_id)
                // Append to queue BUFFER
                replayQueue.current.push(...frames)

                // Update Metadata immediately (useful for iteration/gen info)
                setTelemetry(prev => {
                    const currentStats = prev?.stats || {} as any
                    return {
                        ...prev,
                        step: prev?.step || 0,
                        match_id: prev?.match_id || "replay",
                        race_state: prev?.race_state || { pods: [], checkpoints: [] },
                        stats: {
                            ...currentStats,
                            generation: msg.generation,
                            iteration: msg.iteration,
                            agent_id: msg.agent_id,
                            env_idx: msg.env_idx
                        },
                        logs: []
                    } as Telemetry
                })

            } else if (msg.type === "telemetry_stats") {
                // Just update stats without changing race state
                setTelemetry(prev => {
                    if (!prev) {
                        return {
                            step: 0,
                            match_id: "init",
                            stats: msg.payload,
                            logs: [],
                            race_state: { pods: [], checkpoints: [] }
                        }
                    }
                    return { ...prev, stats: msg.payload, logs: [] }
                })
            } else {
                // Legacy / Standard Telemetry
                setTelemetry(prev => {
                    if (!prev) return msg
                    return {
                        ...msg,
                        race_state: msg.race_state || prev.race_state,
                        stats: msg.stats || prev.stats,
                        logs: msg.logs || []
                    }
                })
            }
        }
    }, [lastJsonMessage])

    // Playback Loop
    useEffect(() => {
        let animationFrameId: number;

        const loop = (timestamp: number) => {
            if (lastFrameTime.current === 0) lastFrameTime.current = timestamp

            const diff = timestamp - lastFrameTime.current
            // Target 20 FPS (50ms) for UI updates
            if (diff > 50) {
                lastFrameTime.current = timestamp

                const queue = replayQueue.current
                let cursor = replayCursor.current

                if (queue.length > 0) {
                    // Playback Speed Factor.
                    // Loop runs at 20Hz (50ms).
                    // Backend now sends every step (20Hz).
                    // So we advance 1.0 frames per tick.
                    const PLAYBACK_SPEED = 1.0;
                    const nextCursor = cursor + PLAYBACK_SPEED;

                    // Check boundaries
                    if (Math.floor(nextCursor) >= queue.length - 1) {
                        // End of buffer, hold last frame
                        cursor = Math.max(0, queue.length - 1);
                        replayCursor.current = cursor;
                    } else {
                        replayCursor.current = nextCursor;
                        cursor = nextCursor;
                    }

                    // Buffer Cleanup
                    // Keep a window of past frames to avoid indefinite memory growth
                    // At 20Hz, 500 frames = 25 seconds.
                    // Increased thresholds to handle higher frame rate data
                    if (cursor > 1000 && queue.length > 5000) {
                        const removeCount = 1000;
                        replayQueue.current.splice(0, removeCount);
                        replayCursor.current -= removeCount;
                        cursor -= removeCount;
                    }

                    // Render
                    const idx = Math.floor(cursor)
                    const nextIdx = idx + 1
                    const t = (cursor - idx) / 1.0

                    const frameA = queue[idx]
                    const frameB = queue[nextIdx]

                    if (frameA) {
                        let renderFrame = frameA;

                        if (frameB) {
                            // Only interpolate if within same match AND same race
                            if (frameA.match_id === frameB.match_id && frameA.race_id === frameB.race_id) {
                                renderFrame = interpolateFrame(frameA, frameB, t)
                            }
                        }

                        setTelemetry(current => {
                            if (!current) return null
                            return {
                                ...current,
                                step: renderFrame.step,
                                match_id: renderFrame.match_id,
                                race_state: renderFrame.race_state,
                                logs: []
                            }
                        })
                    }
                }
            }
            animationFrameId = requestAnimationFrame(loop)
        }

        animationFrameId = requestAnimationFrame(loop)
        return () => cancelAnimationFrame(animationFrameId)
    }, [])

    // Update history for graphs
    useEffect(() => {
        if (telemetry?.step && telemetry.step % 10 === 0) {
            setHistory(prev => {
                const next = [...prev, {
                    step: telemetry.step,
                    reward: telemetry.stats.reward_mean,
                    win_rate: telemetry.stats.win_rate
                }]
                if (next.length > 50) next.shift()
                return next
            })
        }
    }, [telemetry?.step])

    const connectionStatus: ConnectionStatus = {
        [ReadyState.CONNECTING]: 'Connecting',
        [ReadyState.OPEN]: 'Open',
        [ReadyState.CLOSING]: 'Closing',
        [ReadyState.CLOSED]: 'Closed',
        [ReadyState.UNINSTANTIATED]: 'Uninstantiated',
    }[readyState] as ConnectionStatus

    return {
        telemetry,
        readyState,
        connectionStatus,
        history,
        stats: telemetry?.stats || {},
        sendMessage
    }
}

// ------ Parsing Logic ------

interface ParsedFrame {
    step: number
    match_id: string
    race_id: string // Start fresh for each race
    race_state: {
        pods: any[]
        checkpoints: any[]
    }
}

function parseBinaryReplay(b64: string, checkpoints: any[], race_id: string = "unknown"): ParsedFrame[] {
    // Decode Base64
    const binaryString = window.atob(b64)
    const len = binaryString.length
    const bytes = new Uint8Array(len)
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i)
    }
    const view = new DataView(bytes.buffer)

    const frames: ParsedFrame[] = []
    let offset = 0
    const FRAME_SIZE = 192 // 16 Header + 4*44 Pods

    while (offset + FRAME_SIZE <= len) {
        // Header
        const magic = view.getUint16(offset, true) // Little Endian
        if (magic !== 0xDEAD) {
            console.error("Invalid Magic", magic)
            break
        }
        // const type = view.getUint8(offset + 2)
        const step = view.getUint32(offset + 3, true)
        // const envIdx = view.getUint16(offset + 7, true)
        // Reserved 7 bytes

        offset += 16

        // Pods
        const pods: any[] = []
        for (let i = 0; i < 4; i++) {
            // 10 floats + 2 shorts = 40 + 4 = 44 bytes
            const x = view.getFloat32(offset, true)
            const y = view.getFloat32(offset + 4, true)
            const vx = view.getFloat32(offset + 8, true)
            const vy = view.getFloat32(offset + 12, true)
            const angle = view.getFloat32(offset + 16, true)
            const thrust = view.getFloat32(offset + 20, true)
            const shield = view.getFloat32(offset + 24, true)
            const boost = view.getFloat32(offset + 28, true)
            const reward = view.getFloat32(offset + 32, true)
            const collision = view.getFloat32(offset + 36, true)
            const lap = view.getUint16(offset + 40, true)
            const next_cp = view.getUint16(offset + 42, true)

            pods.push({
                id: i,
                team: Math.floor(i / 2),
                x, y, vx, vy, angle,
                boost, shield, collision,
                lap, next_checkpoint: next_cp,
                reward, thrust
            })

            offset += 44
        }

        frames.push({
            step,
            match_id: "replay",
            race_id: race_id,
            race_state: {
                pods,
                checkpoints // Static reuse
            }
        })
    }

    return frames
}

// ------ Interpolation Helpers ------

function interpolate(v1: number, v2: number, t: number): number {
    return v1 + (v2 - v1) * t
}

function interpolateAngle(a1: number, a2: number, t: number): number {
    let diff = a2 - a1
    // Normalize diff to -PI to +PI
    while (diff < -Math.PI) diff += Math.PI * 2
    while (diff > Math.PI) diff -= Math.PI * 2
    return a1 + diff * t
}

function interpolateFrame(f1: ParsedFrame, f2: ParsedFrame, t: number): ParsedFrame {
    // Interpolate step (round to nearest integer for display)
    const step = Math.round(interpolate(f1.step, f2.step, t))

    // Interpolate Pods
    const pods = f1.race_state.pods.map((p1, i) => {
        const p2 = f2.race_state.pods[i]
        if (!p2) return p1

        return {
            ...p1,
            x: interpolate(p1.x, p2.x, t),
            y: interpolate(p1.y, p2.y, t),
            vx: interpolate(p1.vx, p2.vx, t),
            vy: interpolate(p1.vy, p2.vy, t),
            angle: interpolateAngle(p1.angle, p2.angle, t),
            thrust: interpolate(p1.thrust, p2.thrust, t),
            reward: interpolate(p1.reward, p2.reward, t),
            // Discrete values, keep from p1 or transition?
            // Usually step changes happening at keyframes, so p1 is safe.
            lap: p1.lap,
            next_checkpoint: p1.next_checkpoint
        }
    })

    return {
        step,
        match_id: f1.match_id,
        race_id: f1.race_id,
        race_state: {
            pods,
            checkpoints: f1.race_state.checkpoints
        }
    }
}
