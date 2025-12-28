import React, { useRef, useLayoutEffect, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Grid, useTexture, useGLTF } from '@react-three/drei'
import { SkeletonUtils } from 'three-stdlib'
import { useGameState } from '../../context/GameStateContext'
import * as THREE from 'three'
import bgImage from '../../assets/background.jpg'
// @ts-ignore
import podModelUrl from '../../assets/models/race_pod.glb'

// Constants matching backend (Physics world is roughly 16000x9000)
const SCALE_FACTOR = 0.01
const MAP_WIDTH = 16000 * SCALE_FACTOR // 160
const MAP_HEIGHT = 9000 * SCALE_FACTOR // 90
const MAP_CENTER_X = MAP_WIDTH / 2     // 80
const MAP_CENTER_Z = MAP_HEIGHT / 2    // 45

const TEAM_COLORS = [
    new THREE.Color('#ff2222'), // Team 0: Red
    new THREE.Color('#eeeeee'), // Team 1: White
]

const BackgroundRenderer: React.FC = () => {
    const texture = useTexture(bgImage)

    // Background is 16000x9000. 
    // Position center at [80, -0.2, 45] to align with game coordinates starting at 0,0
    return (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[MAP_CENTER_X, -0.2, MAP_CENTER_Z]}>
            <planeGeometry args={[MAP_WIDTH, MAP_HEIGHT]} />
            <meshBasicMaterial map={texture} toneMapped={false} />
        </mesh>
    )
}

const CheckpointsRenderer: React.FC = () => {
    const { telemetry } = useGameState()
    const meshRef = useRef<THREE.InstancedMesh>(null)
    const checkpoints = telemetry?.race_state?.checkpoints || []

    useLayoutEffect(() => {
        if (!meshRef.current || checkpoints.length === 0) return

        const tempObject = new THREE.Object3D()

        checkpoints.forEach((cp, i) => {
            const x = cp.x * SCALE_FACTOR
            const z = cp.y * SCALE_FACTOR
            // Note: y in 2D is z in 3D (XZ plane)
            // cp.radius is typically 600 -> 6.0 in 3D
            const scale = (cp.radius * SCALE_FACTOR) * 2

            tempObject.position.set(x, 0.05, z) // Very close to ground
            tempObject.scale.set(scale, scale, 1) // Ring geometry is flat XY
            tempObject.rotation.x = -Math.PI / 2 // Rotate to lie flat on XZ
            tempObject.updateMatrix()

            meshRef.current!.setMatrixAt(i, tempObject.matrix)
        })
        meshRef.current.instanceMatrix.needsUpdate = true
    }, [checkpoints])

    return (
        <instancedMesh ref={meshRef} args={[undefined, undefined, checkpoints.length]}>
            {/* Inner Radius 0.9, Outer 1.0 -> Thin Ring effect. 32 segments */}
            <ringGeometry args={[0.85, 1.0, 32]} />
            <meshBasicMaterial color="#555" transparent opacity={0.6} side={THREE.DoubleSide} />
        </instancedMesh>
    )
}

const PodModel: React.FC<{ pod: any, visible: boolean }> = ({ pod, visible }) => {
    // Load Model (Cached)
    const { scene } = useGLTF(podModelUrl)
    // Clone Scene per instance
    const clone = React.useMemo(() => SkeletonUtils.clone(scene), [scene])

    // References for Animation
    const flamesMatRef = useRef<THREE.MeshStandardMaterial | null>(null)
    const arcsMatRef = useRef<THREE.MeshStandardMaterial | null>(null)
    const bodyMatRef = useRef<THREE.MeshStandardMaterial | null>(null)

    // Setup Materials Once
    useLayoutEffect(() => {
        clone.traverse((child) => {
            if ((child as THREE.Mesh).isMesh) {
                const mesh = child as THREE.Mesh
                // Materials might be an array or single. Based on inspect_glb, we have one mesh with multiple matrials.
                if (Array.isArray(mesh.material)) {
                    // Clone materials so we can modify them per instance
                    mesh.material = mesh.material.map((m) => m.clone())

                    // Identify by Name (Mapped from inspect_glb output)
                    // Material 0: pod-skin
                    // Material 1: flames
                    // Material 2: cockpit
                    // Material 3: arcs
                    mesh.material.forEach((mat) => {
                        const m = mat as THREE.MeshStandardMaterial
                        // Safer matching
                        if (m.name.includes('flames')) {
                            flamesMatRef.current = m
                            m.transparent = true
                            m.opacity = 0 // Start invisible
                            m.emissive = new THREE.Color('#ffaa00')
                            m.emissiveIntensity = 2.0
                            m.depthWrite = false
                            m.blending = THREE.AdditiveBlending
                        }
                        if (m.name.includes('arcs')) {
                            arcsMatRef.current = m
                            m.transparent = true
                            m.emissive = new THREE.Color('#00ffff')
                            m.emissiveIntensity = 1.5
                            // Ensure texture wraps for scrolling
                            if (m.map) {
                                m.map.wrapS = THREE.RepeatWrapping
                                m.map.wrapT = THREE.RepeatWrapping
                            }
                        }
                        if (m.name.includes('pod-skin')) {
                            bodyMatRef.current = m
                        }
                    })
                }
            }
        })
    }, [clone])

    // Scale/Pos/Rot
    const groupRef = useRef<THREE.Group>(null)

    useFrame((_state, delta) => {
        if (!groupRef.current) return

        // 1. Position & Rotation
        const x = pod.x * SCALE_FACTOR
        const z = pod.y * SCALE_FACTOR
        groupRef.current.position.set(x, 2, z)

        // Rotation Logic
        groupRef.current.rotation.set(0, 0, 0)

        let heading = pod.angle
        const speed = Math.sqrt(pod.vx * pod.vx + pod.vy * pod.vy)
        if (speed > 5.0) heading = Math.atan2(pod.vy, pod.vx)

        // Rotate around Y axis
        groupRef.current.rotation.y = -heading

        // Base Rotation: +90 degrees on X to lay it flat (User correction)
        const qBase = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2)
        // Heading Rotation: -heading around Y (Standard)
        const qHeading = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -heading)

        // Combine
        qHeading.multiply(qBase)
        groupRef.current.quaternion.copy(qHeading)

        // 3. Logic Updates (Flames, Team Color, Arcs)
        const thrust = (pod as any).thrust ?? 0
        if (flamesMatRef.current) {
            const opacity = Math.max(0, thrust) / 100.0
            flamesMatRef.current.opacity = opacity
            flamesMatRef.current.visible = opacity > 0.01
        }

        // Color -> Team
        if (bodyMatRef.current) {
            const color = TEAM_COLORS[pod.team % TEAM_COLORS.length]
            bodyMatRef.current.color.set(color)
        }

        // Arcs Animation -> Scroll X
        if (arcsMatRef.current && arcsMatRef.current.map) {
            arcsMatRef.current.map.offset.x -= delta * 2.0 // "defil on X axis"
        }
    })

    return <primitive object={clone} ref={groupRef} scale={[3, 3, 3]} visible={visible} />
}

const PodsRenderer: React.FC = () => {
    const { telemetry } = useGameState()
    const pods = telemetry?.race_state?.pods || []

    return (
        <group>
            {pods.map((pod, i) => (
                <PodModel key={i} pod={pod} visible={true} />
            ))}
        </group>
    )
}
// Preload
useGLTF.preload(podModelUrl)

const ShieldsRenderer: React.FC = () => {
    const { telemetry } = useGameState()
    const meshRef = useRef<THREE.InstancedMesh>(null)
    const shieldTimers = useRef<number[]>([])

    useFrame(() => {
        if (!meshRef.current || !telemetry?.race_state?.pods) return

        const pods = telemetry.race_state.pods
        const tempObject = new THREE.Object3D()

        if (shieldTimers.current.length !== pods.length) {
            shieldTimers.current = new Array(pods.length).fill(0)
        }

        pods.forEach((pod, i) => {
            // Check collision flag
            if ((pod as any).collision > 0.5) {
                // Set decay timer to ~15 frames (250ms)
                shieldTimers.current[i] = 15
            }

            // Decrement
            if (shieldTimers.current[i] > 0) {
                shieldTimers.current[i]--
            }

            if (shieldTimers.current[i] > 0) {
                const x = pod.x * SCALE_FACTOR
                const z = pod.y * SCALE_FACTOR

                tempObject.position.set(x, 2, z)
                tempObject.scale.set(3.5, 3.5, 3.5)
                tempObject.updateMatrix()

                meshRef.current!.setMatrixAt(i, tempObject.matrix)
            } else {
                tempObject.scale.set(0, 0, 0)
                tempObject.updateMatrix()
                meshRef.current!.setMatrixAt(i, tempObject.matrix)
            }
        })

        meshRef.current.instanceMatrix.needsUpdate = true
    })

    return (
        <instancedMesh ref={meshRef} args={[undefined, undefined, 4]}>
            <sphereGeometry args={[1, 16, 16]} />
            <meshStandardMaterial
                color="#0088ff"
                transparent
                opacity={0.4}
                emissive="#0044aa"
                emissiveIntensity={0.5}
                depthWrite={false}
            />
        </instancedMesh>
    )
}

// Controls Camera Movement based on mode
const CameraController: React.FC<{
    mode: 'orbit' | 'pod',
    focusedPodIndex: number
}> = ({ mode, focusedPodIndex }) => {
    const { telemetry } = useGameState()
    const { camera } = useThree()

    // Smooth Follow Refs
    const targetPos = useRef(new THREE.Vector3())
    const lookAtPos = useRef(new THREE.Vector3())

    useFrame((_state, delta) => {
        if (mode === 'pod') {
            const pods = telemetry?.race_state?.pods || []
            const pod = pods[focusedPodIndex]

            if (pod) {
                // 1. Determine Target Position (Pod Position)
                const x = pod.x * SCALE_FACTOR
                const z = pod.y * SCALE_FACTOR

                // 2. Determine Desired Camera Position
                // User Request: Follow Speed Vector
                // If moving meaningfully, use velocity vector. Otherwise use facing.
                let heading = pod.angle
                const speed = Math.sqrt(pod.vx * pod.vx + pod.vy * pod.vy)

                // Threshold of 20 to ensure we have a stable vector
                if (speed > 20.0) {
                    heading = Math.atan2(pod.vy, pod.vx)
                }

                // Racing Game Style: Closer and lower
                const dist = 20
                const height = 8

                const camX = x - Math.cos(heading) * dist
                const camZ = z - Math.sin(heading) * dist
                const camY = height

                const desiredPos = new THREE.Vector3(camX, camY, camZ)

                // 3. Smooth Lerp Camera Position
                // Reduce lerp speed slightly to hide micro-stutter from telemetry updates
                camera.position.lerp(desiredPos, delta * 4.0)

                // 4. Smooth Look At
                // Instead of snapping lookAt to the pod center, we lerp the lookAt target
                // This prevents jitter when the pod position updates discretely
                targetPos.current.set(x, 2, z) // Target: slightly above pod center
                lookAtPos.current.lerp(targetPos.current, delta * 10.0) // Fast lerp for responsiveness, but smooths snaps
                camera.lookAt(lookAtPos.current)
            }
        }
    })

    if (mode === 'orbit') {
        return <OrbitControls makeDefault target={[MAP_CENTER_X, 0, MAP_CENTER_Z]} maxPolarAngle={Math.PI / 2} />
    }

    return null
}

const SceneContent: React.FC<{
    mode: 'orbit' | 'pod',
    focusedPodIndex: number
}> = ({ mode, focusedPodIndex }) => {
    return (
        <>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 100, 10]} intensity={1.0} />
            <directionalLight position={[-100, 200, 50]} intensity={1.5} castShadow />

            <Grid
                position={[MAP_CENTER_X, -0.1, MAP_CENTER_Z]}
                args={[200, 200]}
                cellSize={10}
                cellThickness={0.5}
                cellColor="#222"
                sectionSize={50}
                sectionThickness={1}
                sectionColor="#444"
                fadeDistance={250}
                infiniteGrid
            />

            <BackgroundRenderer />
            <CheckpointsRenderer />
            <PodsRenderer />
            <ShieldsRenderer />

            <CameraController mode={mode} focusedPodIndex={focusedPodIndex} />
        </>
    )
}

export const RaceScene3D: React.FC = () => {
    const [cameraMode, setCameraMode] = useState<'orbit' | 'pod'>('orbit')
    const [focusedPodIndex, setFocusedPodIndex] = useState(0)
    const { telemetry } = useGameState()

    const podsCount = telemetry?.race_state?.pods?.length || 0

    const toggleMode = () => {
        setCameraMode(prev => prev === 'orbit' ? 'pod' : 'orbit')
    }

    const nextPod = () => {
        if (podsCount === 0) return
        setFocusedPodIndex(prev => (prev + 1) % podsCount)
    }

    const prevPod = () => {
        if (podsCount === 0) return
        setFocusedPodIndex(prev => (prev - 1 + podsCount) % podsCount)
    }

    return (
        <div className="relative w-full aspect-[16/9] bg-black">
            <Canvas
                camera={{ position: [MAP_CENTER_X, 100, MAP_CENTER_Z + 80], fov: 60 }}
                shadows
                dpr={[1, 2]}
                className="w-full h-full"
            >
                <SceneContent mode={cameraMode} focusedPodIndex={focusedPodIndex} />
            </Canvas>

            {/* Overlay UI */}
            <div className="absolute top-4 left-4 flex gap-2 z-10">
                <div className="text-white/50 text-xs font-mono pointer-events-none select-none">
                    3D MODE {[cameraMode.toUpperCase()]}
                </div>
            </div>

            <div className="absolute top-4 right-4 flex gap-2 z-10">
                <button
                    onClick={toggleMode}
                    className="px-3 py-1 bg-white/10 hover:bg-white/20 text-white text-xs rounded border border-white/20 backdrop-blur-sm transition-colors"
                >
                    {cameraMode === 'orbit' ? 'Switch to Pod View' : 'Switch to Orbit View'}
                </button>
            </div>

            {cameraMode === 'pod' && (
                <div className="absolute bottom-16 left-1/2 -translate-x-1/2 flex items-center gap-4 z-10 glass-panel px-4 py-2 rounded-full border border-white/10">
                    <button
                        onClick={prevPod}
                        className="w-8 h-8 flex items-center justify-center bg-white/5 hover:bg-white/10 rounded-full text-white/70 hover:text-white transition-colors"
                    >
                        ←
                    </button>
                    <div className="text-white/80 text-sm font-mono">
                        POD {focusedPodIndex}
                    </div>
                    <button
                        onClick={nextPod}
                        className="w-8 h-8 flex items-center justify-center bg-white/5 hover:bg-white/10 rounded-full text-white/70 hover:text-white transition-colors"
                    >
                        →
                    </button>
                </div>
            )}

            <div className="absolute bottom-4 right-4 text-right text-white/30 text-[10px] font-mono pointer-events-none select-none z-10">
                {cameraMode === 'orbit' ? 'LMB: Rotate | RMB: Pan | Wheel: Zoom' : 'Camera locked to Pod'}
            </div>
        </div>
    )
}

