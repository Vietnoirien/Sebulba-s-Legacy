import React, { useRef, useEffect, useState } from 'react'
import { useGameState, useGameActions } from '../../context/GameStateContext'

// Import assets directly to ensure Vite bundles them
import redCabin from '../../assets/Red_cabin.png'
import redLEngine from '../../assets/Red_L_engine.png'
import redREngine from '../../assets/Red_R_engine.png'
import whiteCabin from '../../assets/White_cabin.png'
import whiteLEngine from '../../assets/White_L_engine.png'
import whiteREngine from '../../assets/White_R_engine.png'
import bgImage from '../../assets/background.jpg'

// Checkpoint Assets
import cpkMainCircle from '../../assets/cpk/main_circle_200*200.png'
import cpkSupport from '../../assets/cpk/light_support.png'
import cpkRedLight from '../../assets/cpk/red_light.png'
import cpkYellowLight from '../../assets/cpk/yellow_light.png'
import cpkBordure from '../../assets/cpk/bordure.png'
import cpkInsideSpinning from '../../assets/cpk/inside_spinning.png'
import cpkInsideStart from '../../assets/cpk/inside_start.png'

import redShield from '../../assets/red_shield.png'
import whiteShield from '../../assets/white_shield.png'


const GAME_WIDTH = 16000
const GAME_HEIGHT = 9000


export const RaceCanvas: React.FC = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const { telemetry } = useGameState()
    const { playbackSpeed, setPlaybackSpeed } = useGameActions()
    const [images, setImages] = useState<Record<string, HTMLImageElement>>({})
    const [debugLines, setDebugLines] = useState(false)
    const [debugText, setDebugText] = useState(false)
    const [showMenu, setShowMenu] = useState(false)

    // Animation state for rotating checkpoints
    const [rotationAngle, setRotationAngle] = useState(0)

    // Flash State Refs
    // Track previous next_checkpoint for each pod to detect crossings
    const lastPodNextCps = useRef<Record<number, number>>({})
    // Track flash intensity (opacity 1.0 -> 0.0) for each CP [red, white]
    const flashes = useRef<Record<number, { red: number, white: number }>>({})

    // Load images on mount
    useEffect(() => {
        const imageSources = {
            redCabin, redLEngine, redREngine,
            whiteCabin, whiteLEngine, whiteREngine,
            bgImage,
            cpkMainCircle, cpkSupport, cpkRedLight, cpkYellowLight,
            cpkBordure, cpkInsideSpinning, cpkInsideStart,
            redShield, whiteShield
        }

        const loaded: Record<string, HTMLImageElement> = {}
        let count = 0
        const total = Object.keys(imageSources).length

        Object.entries(imageSources).forEach(([key, src]) => {
            const img = new Image()
            img.src = src // Check if src is string path
            img.onload = () => {
                loaded[key] = img
                count++
                if (count === total) {
                    setImages(loaded)
                }
            }
        })
    }, [])

    // Rotation loop
    useEffect(() => {
        let frameId: number
        const loop = () => {
            setRotationAngle(prev => (prev + 0.02) % (Math.PI * 2))
            frameId = requestAnimationFrame(loop)
        }
        loop()
        return () => cancelAnimationFrame(frameId)
    }, [])

    // Fullscreen ref
    const containerRef = useRef<HTMLDivElement>(null)

    // Fullscreen Handler
    const toggleFullscreen = () => {
        if (!document.fullscreenElement) {
            containerRef.current?.requestFullscreen()
        } else {
            document.exitFullscreen()
        }
    }

    // Main Draw Loop
    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas || !telemetry || Object.keys(images).length === 0) return

        const ctx = canvas.getContext('2d')
        if (!ctx) return

        // Handle canvas sizing for high DPI
        const dpr = window.devicePixelRatio || 1
        const rect = canvas.getBoundingClientRect()

        if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
            canvas.width = rect.width * dpr
            canvas.height = rect.height * dpr
        }

        // --- Logic: Detect Crossings (Flash Trigger) ---
        if (telemetry.race_state?.pods) {
            telemetry.race_state.pods.forEach((pod, index) => {
                const currentNext = pod.next_checkpoint ?? 1
                const prevNext = lastPodNextCps.current[index]

                // Detect Change
                if (prevNext !== undefined && currentNext !== prevNext) {
                    // Checkpoint Passed = Old Next (The one we were aiming for)
                    const passedCpId = prevNext

                    // Init flash entry if missing
                    if (!flashes.current[passedCpId]) {
                        flashes.current[passedCpId] = { red: 0, white: 0 }
                    }

                    // Trigger Flash
                    if (index === 0) {
                        // Red Team Flag (Pod 0)
                        flashes.current[passedCpId].red = 1.0
                    } else {
                        // White Team/Bots
                        flashes.current[passedCpId].white = 1.0
                    }
                }

                // Update last state
                lastPodNextCps.current[index] = currentNext
            })
        }

        // --- Logic: Decay Flashes ---
        // Decay speed: 0.05 per frame (approx 20 frames / 0.3s)
        const DECAY = 0.05
        Object.keys(flashes.current).forEach(key => {
            const id = Number(key)
            const f = flashes.current[id]
            if (f.red > 0) f.red = Math.max(0, f.red - DECAY)
            if (f.white > 0) f.white = Math.max(0, f.white - DECAY)
        })


        // Clear background
        ctx.clearRect(0, 0, canvas.width, canvas.height)

        // Scaling - Strictly Contain (should be perfect fit due to CSS aspect ratio, but we calculate safely)
        const scaleX = canvas.width / GAME_WIDTH
        const scaleY = canvas.height / GAME_HEIGHT
        const scale = Math.min(scaleX, scaleY)

        const offsetX = (canvas.width - GAME_WIDTH * scale) / 2
        const offsetY = (canvas.height - GAME_HEIGHT * scale) / 2

        ctx.save()
        ctx.translate(offsetX, offsetY)
        ctx.scale(scale, scale)

        // Draw Board Background
        if (images.bgImage) {
            ctx.drawImage(images.bgImage, 0, 0, GAME_WIDTH, GAME_HEIGHT)
        }

        // Draw Checkpoints
        if (telemetry.race_state?.checkpoints) {
            telemetry.race_state.checkpoints.forEach((cp) => {
                // Get Flash State
                const flash = flashes.current[cp.id] || { red: 0, white: 0 }
                drawCheckpoint(ctx, cp, images, rotationAngle, flash)
            })
        }

        // Draw Pods
        if (telemetry.race_state?.pods) {
            telemetry.race_state.pods.forEach((pod) => {
                const color = pod.team === 0 ? 'red' : 'white'
                drawPod(ctx, pod, color, images, debugLines)
            })
        }

        ctx.restore()

        if (debugText && telemetry.race_state?.pods) {
            // UI Scale relative to a baseline width (e.g. 1920 or 1600)
            const uiScale = rect.width / 1600
            drawDebugOverlay(ctx, telemetry.race_state.pods, dpr, uiScale)
        }

        // --- HUD Overlay (Generation / Pop) ---
        // Always Visible
        const gen = telemetry.stats?.generation ?? "0"
        const iter = telemetry.stats?.iteration ?? "0"
        const agent = telemetry.stats?.agent_id ?? "-"
        // @ts-ignore
        const isPareto = telemetry.stats?.is_pareto ?? false

        ctx.save()
        ctx.setTransform(1, 0, 0, 1, 0, 0) // Reset transform to screen space
        ctx.scale(dpr, dpr)

        const labelText = `ITER: ${iter} | GEN: ${gen} | AGENT: ${agent}`

        // Dynamic Sizing
        // Base 24px at 1600w
        const fontSize = Math.max(12, Math.floor(rect.width * 0.015))
        const padding = fontSize * 0.8

        ctx.font = `bold ${fontSize}px monospace`
        const measure = ctx.measureText(labelText)
        const w = measure.width + (padding * 2)
        const h = fontSize + padding

        // Top Right
        const margin = 20
        const cx = rect.width - (w / 2) - margin
        const cy = h / 2 + margin

        // Box
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)'
        ctx.beginPath()
        ctx.roundRect(cx - w / 2, cy - h / 2, w, h, 8)
        ctx.fill()

        // Border
        ctx.strokeStyle = '#00ffff'
        ctx.lineWidth = Math.max(1, fontSize * 0.08)
        ctx.stroke()

        // Text
        ctx.fillStyle = '#ffffff'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        // Add glow
        ctx.shadowColor = '#00ffff'
        ctx.shadowBlur = fontSize * 0.4
        ctx.fillText(labelText, cx, cy)

        if (isPareto) {
            // Draw Badge Style Crown on top-right corner of the box
            const crownSize = fontSize * 1.5
            const badgeX = cx + w / 2
            const badgeY = cy - h / 2

            drawCrown(ctx, badgeX, badgeY, crownSize, '#ffd700')
        }

        ctx.restore()

    }, [telemetry, images, rotationAngle, debugLines, debugText])

    return (
        <div ref={containerRef} className="relative w-full aspect-[16/9] bg-black group">
            <canvas
                ref={canvasRef}
                className="w-full h-full block"
                style={{ imageRendering: 'pixelated' as any }}
            />

            {/* Controls Container */}
            <div className="absolute bottom-4 right-4 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity z-10">
                {/* Config Menu Trigger */}
                <div className="relative">
                    <button
                        onClick={() => setShowMenu(!showMenu)}
                        className="bg-black/60 hover:bg-black/90 text-white/80 hover:text-white p-2 rounded backdrop-blur-sm"
                        title="Settings"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
                    </button>

                    {/* Menu Popup */}
                    {showMenu && (
                        <div className="absolute bottom-full right-0 mb-2 w-48 bg-gray-900 border border-gray-700 rounded shadow-xl p-2 flex flex-col gap-2">
                            <button
                                onClick={() => setDebugLines(!debugLines)}
                                className={`text-left px-3 py-2 rounded text-sm ${debugLines ? 'bg-green-600 text-white' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'}`}
                            >
                                {debugLines ? 'Debug Lines: ON' : 'Debug Lines: OFF'}
                            </button>
                            <button
                                onClick={() => setDebugText(!debugText)}
                                className={`text-left px-3 py-2 rounded text-sm ${debugText ? 'bg-green-600 text-white' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'}`}
                            >
                                {debugText ? 'Debug Text: ON' : 'Debug Text: OFF'}
                            </button>

                            <div className="px-3 py-2 border-t border-gray-700 mt-1 pt-2">
                                <div className="flex justify-between text-gray-400 text-xs mb-1">
                                    <span>Speed</span>
                                    <span>{playbackSpeed.toFixed(1)}x</span>
                                </div>
                                <input
                                    type="range"
                                    min="0.1"
                                    max="2.0"
                                    step="0.1"
                                    value={playbackSpeed}
                                    onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
                                    className="w-full accent-green-500 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                                />
                            </div>
                        </div>
                    )}
                </div>

                {/* Fullscreen Button */}
                <button
                    onClick={toggleFullscreen}
                    className="bg-black/60 hover:bg-black/90 text-white/80 hover:text-white p-2 rounded backdrop-blur-sm"
                    title="Fullscreen"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M8 3H5a2 2 0 0 0-2 2v3" />
                        <path d="M21 8V5a2 2 0 0 0-2-2h-3" />
                        <path d="M3 16v3a2 2 0 0 0 2 2h3" />
                        <path d="M16 21h3a2 2 0 0 0 2-2v-3" />
                        <rect x="7" y="7" width="10" height="10" rx="1" />
                    </svg>
                </button>
            </div>
        </div>
    )
}

function drawCheckpoint(
    ctx: CanvasRenderingContext2D,
    cp: { x: number, y: number, radius: number, id: number },
    images: Record<string, HTMLImageElement>,
    rotation: number,
    flash: { red: number, white: number }
) {
    const {
        cpkMainCircle, cpkSupport, cpkRedLight, cpkYellowLight,
        cpkBordure, cpkInsideSpinning, cpkInsideStart
    } = images

    if (!cpkMainCircle || !cpkSupport) return

    ctx.save()
    ctx.translate(cp.x, cp.y)

    // Scaling Factor
    // Main Circle is 200px. Game size is 1200 (2 * 600 radius).
    // All assets in folder are "placed perfectly aligned" relative to this scale.
    // So we apply a global scale of 6.0
    const SCALE = 6.0

    // Draw Helper
    const drawAsset = (img: HTMLImageElement | undefined, alpha: number = 1.0) => {
        if (!img) return
        ctx.globalAlpha = alpha
        const w = img.width * SCALE
        const h = img.height * SCALE
        ctx.drawImage(img, -w / 2, -h / 2, w, h)
        ctx.globalAlpha = 1.0
    }

    // Layer 1: Support
    drawAsset(cpkSupport)

    // Layer 2: Main Circle
    drawAsset(cpkMainCircle)

    // Layer 3: Inside (Start or Spin)
    if (cp.id === 0) {
        drawAsset(cpkInsideStart)
    } else {
        ctx.save()
        ctx.rotate(rotation)
        drawAsset(cpkInsideSpinning)
        ctx.restore()
    }

    // Number (for non-start checkpoints, or all?)
    // User said "checkpoint number in the middle". 
    // Start (0) has different graphic, no number.
    if (cp.id !== 0) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
        ctx.font = '300px "Russo One", sans-serif'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        // Draw shadow for visibility
        ctx.shadowColor = 'black'
        ctx.shadowBlur = 20
        ctx.fillText(cp.id.toString(), 0, 0)
        ctx.shadowBlur = 0
    }

    // Layer 4: Bordure
    drawAsset(cpkBordure)

    // Layer 5: Lights (FLASH)
    if (flash.red > 0) {
        drawAsset(cpkRedLight, flash.red)
    }
    if (flash.white > 0) {
        drawAsset(cpkYellowLight, flash.white)
    }

    // Draw ID Text (Optional, keeping it for clarity if assets don't have numbers)
    // Assets likely don't have numbers if they are generic.
    // Let's add faint number on top? Or assume 'inside' has it?
    // 'inside_spinning' probably doesn't have number.
    // Let's keep number for debug/clarity but make it subtle or on top.
    // User didn't ask for it, but removing it makes it hard to know which CP is which.
    // I will leave it out for now to respect the "Visuals" request strictly (clean look).
    // If user wants numbers, they can ask.

    ctx.restore()
}

function drawPod(
    ctx: CanvasRenderingContext2D,
    pod: {
        x: number, y: number, vx: number, vy: number, angle: number,
        lap?: number, next_checkpoint?: number, reward?: number, thrust?: number,
        team?: number, collision?: number
    },
    color: 'red' | 'white',
    images: Record<string, HTMLImageElement>,
    debugMode: boolean
) {
    // Determine orientation from velocity vector
    let angle = Math.atan2(pod.vy, pod.vx)
    if (pod.vx === 0 && pod.vy === 0) angle = pod.angle // Fallback

    const cabin = color === 'red' ? images.redCabin : images.whiteCabin
    const leftEngine = color === 'red' ? images.redLEngine : images.whiteLEngine
    const rightEngine = color === 'red' ? images.redREngine : images.whiteREngine

    if (!cabin || !leftEngine || !rightEngine) return

    ctx.save()
    ctx.translate(pod.x, pod.y)

    // Draw Pod Body
    ctx.save()
    ctx.rotate(angle)

    // Draw Shield (Underneath or On Top? Underneath implies "Shield bubble")
    if ((pod as any).collision > 0.5) {
        const shieldImg = pod.team === 0 ? images.redShield : images.whiteShield
        if (shieldImg) {
            // Scale shield to cover pod. Pod parts are scaled by 4.0.
            // Shield asset should be drawn relative to that.
            // Let's assume shield asset is roughly same resolution as cabins.
            // We use scale 5.0 to be slightly larger than 4.0
            const sScale = 5.0
            const sw = shieldImg.width * sScale
            const sh = shieldImg.height * sScale
            ctx.drawImage(shieldImg, -sw / 2, -sh / 2, sw, sh)
        }
    }

    const scale = 4.0
    const cabinW = cabin.width * scale
    const cabinH = cabin.height * scale
    const engineW = leftEngine.width * scale
    const engineH = leftEngine.height * scale

    // Sprite Rotation Correction (Rotate individual components)
    // Apply to both Red and White pods as they share the same asset orientation
    const spriteRotation = Math.PI / 2

    const drawPart = (img: HTMLImageElement, cx: number, cy: number, w: number, h: number) => {
        if (spriteRotation !== 0) {
            ctx.save()
            ctx.translate(cx, cy)
            ctx.rotate(spriteRotation)
            ctx.drawImage(img, -w / 2, -h / 2, w, h)
            ctx.restore()
        } else {
            ctx.drawImage(img, cx - w / 2, cy - h / 2, w, h)
        }
    }

    // Draw Cabin
    // Moved back for separation
    drawPart(cabin, -80, 0, cabinW, cabinH)

    // Engines Layout
    // Adjusted to fit inside hitbox (Radius 400)
    // Scale 4.0, Red Engine max radius Check:
    // Pos(170, 135) + HalfSize(102, 150) -> Edge(272, 285) -> Dist 394 < 400
    const engineX = 170
    const engineY = 135

    // Connection Lines
    ctx.strokeStyle = '#00ffff'
    ctx.lineWidth = 10
    ctx.shadowColor = '#00ffff'
    ctx.shadowBlur = 10

    ctx.beginPath()
    ctx.moveTo(0, 0)
    ctx.lineTo(engineX - engineW / 2, -engineY)
    ctx.moveTo(0, 0)
    ctx.lineTo(engineX - engineW / 2, engineY)

    // Cross-engine strut?
    ctx.moveTo(engineX, -engineY)
    ctx.lineTo(engineX, engineY)

    ctx.stroke()
    ctx.shadowBlur = 0

    // Engines
    drawPart(leftEngine, engineX, -engineY, engineW, engineH)
    drawPart(rightEngine, engineX, engineY, engineW, engineH)

    ctx.restore() // End Rotate

    // --- DEBUG VISUALIZATION ---
    if (debugMode) {
        // Prevent rotation for text/bars ease of reading
        // Vectors are drawn from center (0,0)

        // 1. Velocity (Green)
        const speed = Math.sqrt(pod.vx * pod.vx + pod.vy * pod.vy)
        const vScale = 3.0 // Scale for visibility
        if (speed > 1) {
            ctx.beginPath()
            ctx.strokeStyle = '#00ff00' // Green
            ctx.lineWidth = 15
            ctx.moveTo(0, 0)
            ctx.lineTo(pod.vx * vScale, pod.vy * vScale)
            ctx.stroke()
        }

        // 2. Thrust (Red/Orange)
        if (pod.thrust !== undefined) {
            // Thrust is scalar 0-100. Direction is pod.angle.
            // pod.angle is in degrees.
            const rad = (pod.angle * Math.PI) / 180
            const tLen = pod.thrust * 15.0 // Scale: 100 * 15 = 1500 pixels max
            if (tLen > 1) {
                ctx.beginPath()
                ctx.strokeStyle = '#ff4400' // Orange
                ctx.lineWidth = 10
                ctx.moveTo(0, 0)
                ctx.lineTo(Math.cos(rad) * tLen, Math.sin(rad) * tLen)
                ctx.stroke()
            }
        }
    }

    ctx.restore() // End Translate
}

function drawDebugOverlay(
    ctx: CanvasRenderingContext2D,
    pods: any[],
    dpr: number,
    _scaleFactor: number
) {
    ctx.save()
    ctx.scale(dpr, dpr) // Scale to CSS pixels

    const lineHeight = 24
    const boxWidth = 280
    const boxHeight = pods.length * (6 * lineHeight + 10) + 20

    // Background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.4)'
    ctx.fillRect(10, 10, boxWidth, boxHeight)

    ctx.font = 'bold 15px monospace'
    ctx.textAlign = 'left'
    ctx.textBaseline = 'top'

    let y = 20
    pods.forEach(pod => {
        const speed = Math.sqrt(pod.vx * pod.vx + pod.vy * pod.vy).toFixed(0)

        // Header
        ctx.fillStyle = pod.team === 0 ? '#ff6666' : '#ccccff'
        ctx.fillText(`Pod ${pod.id} [Team ${pod.team}]`, 20, y)
        y += lineHeight

        // Stats
        ctx.fillStyle = '#cccccc'
        ctx.fillText(`  Lap:  ${pod.lap ?? 0}`, 20, y); y += lineHeight
        ctx.fillText(`  Next: ${pod.next_checkpoint ?? 0}`, 20, y); y += lineHeight
        ctx.fillText(`  Rew:  ${(pod.reward ?? 0).toFixed(1)}`, 20, y); y += lineHeight
        ctx.fillText(`  Thr:  ${(pod.thrust ?? 0).toFixed(0)}`, 20, y); y += lineHeight
        ctx.fillText(`  Spd:  ${speed}`, 20, y); y += lineHeight

        y += 10 // Spacing
    })

    ctx.restore()
}

function drawCrown(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, color: string) {
    ctx.save()
    ctx.translate(x, y)

    ctx.fillStyle = color
    ctx.strokeStyle = '#daa520' // GoldenRod
    ctx.lineWidth = 2

    ctx.beginPath()
    // Simple Crown Shape
    // Bottom line
    ctx.moveTo(-size / 2, size / 2)
    ctx.lineTo(size / 2, size / 2)
    // Right side up
    ctx.lineTo(size / 2, 0)
    // Spikes
    ctx.lineTo(size / 4, size / 4) // Dip
    ctx.lineTo(0, -size / 2) // Center Peak
    ctx.lineTo(-size / 4, size / 4) // Dip
    ctx.lineTo(-size / 2, 0) // Left side top

    ctx.closePath()
    ctx.fill()
    ctx.stroke()

    // Gems?
    ctx.fillStyle = 'red'
    ctx.beginPath()
    ctx.arc(0, size / 4, size / 8, 0, Math.PI * 2)
    ctx.fill()

    ctx.restore()
}
