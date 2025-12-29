import React, { useRef, useLayoutEffect, useState } from 'react'
import { Canvas, useFrame, useThree, useLoader } from '@react-three/fiber'
import { OrbitControls, Grid, useTexture } from '@react-three/drei'
import { SkeletonUtils, OBJLoader } from 'three-stdlib'
import { useGameState, useGameActions } from '../../context/GameStateContext'
import * as THREE from 'three'
import bgImage from '../../assets/background.jpg'
// @ts-ignore
// Pod 1 Assets
// @ts-ignore
import pod1ObjUrl from '../../assets/models/Pods/pod_1/pod.obj'
// @ts-ignore
import flames1ObjUrl from '../../assets/models/Pods/pod_1/flames.obj'
// @ts-ignore
import arcs1ObjUrl from '../../assets/models/Pods/pod_1/arcs.obj'
// @ts-ignore
import thrusters1ObjUrl from '../../assets/models/Pods/pod_1/thrusters.obj'
import pod1SkinUrl from '../../assets/models/Pods/pod_1/skin.png'

// Pod 2 Assets
// @ts-ignore
import pod2ObjUrl from '../../assets/models/Pods/pod_2/pod.obj'
// @ts-ignore
import flames2ObjUrl from '../../assets/models/Pods/pod_2/flames.obj'
// @ts-ignore
import arcs2ObjUrl from '../../assets/models/Pods/pod_2/arcs.obj'
// @ts-ignore
import thrusters2ObjUrl from '../../assets/models/Pods/pod_2/thrusters.obj'
import pod2SkinUrl from '../../assets/models/Pods/pod_2/skin.png'

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

interface PodAssets {
    podMesh: THREE.Group
    flamesMesh: THREE.Group
    arcsMesh: THREE.Group
    thrustersMesh: THREE.Group
    podTexture: THREE.Texture
}

const GenericPodRender: React.FC<{ pod: any, assets: PodAssets, visible: boolean }> = ({ pod, assets, visible }) => {
    const { podMesh, flamesMesh, arcsMesh, thrustersMesh, podTexture } = assets

    // Clone Scene per instance
    const podClone = React.useMemo(() => SkeletonUtils.clone(podMesh), [podMesh])
    const flamesClone = React.useMemo(() => SkeletonUtils.clone(flamesMesh), [flamesMesh])
    const arcsClone = React.useMemo(() => SkeletonUtils.clone(arcsMesh), [arcsMesh])
    const thrustersClone = React.useMemo(() => SkeletonUtils.clone(thrustersMesh), [thrustersMesh])

    // References for Animation
    const flamesMatRef = useRef<THREE.MeshStandardMaterial | null>(null)
    const arcsMatRef = useRef<THREE.MeshStandardMaterial | null>(null)
    const bodyMatRef = useRef<THREE.MeshStandardMaterial | null>(null)

    // Setup Materials
    useLayoutEffect(() => {
        // 1. Pod Body
        podClone.traverse((child) => {
            if ((child as THREE.Mesh).isMesh) {
                const mesh = child as THREE.Mesh
                // Create a standard material for team coloring if not present
                // Assuming OBJ loads with MeshPhongMaterial or similar
                const newMat = new THREE.MeshStandardMaterial({
                    map: podTexture,
                    color: 0xffffff,
                    roughness: 0.4,
                    metalness: 0.6
                })
                mesh.material = newMat
                bodyMatRef.current = newMat
            }
        })

        // 2. Flames
        flamesClone.traverse((child) => {
            if ((child as THREE.Mesh).isMesh) {
                const mesh = child as THREE.Mesh
                const newMat = new THREE.MeshStandardMaterial({
                    map: podTexture,
                    color: 0xffaa00, // Orange tint
                    emissive: 0xffaa00,
                    emissiveIntensity: 2.0,
                    transparent: true,
                    opacity: 0,
                    depthWrite: false,
                    blending: THREE.AdditiveBlending,
                    side: THREE.DoubleSide
                })
                mesh.material = newMat
                flamesMatRef.current = newMat
            }
        })

        // 3. Arcs
        arcsClone.traverse((child) => {
            if ((child as THREE.Mesh).isMesh) {
                const mesh = child as THREE.Mesh
                const newMat = new THREE.MeshStandardMaterial({
                    map: podTexture,
                    color: 0x00ffff,
                    emissive: 0x00ffff,
                    emissiveIntensity: 1.5,
                    transparent: true,
                    opacity: 0.8,
                    side: THREE.DoubleSide
                })
                mesh.material = newMat
                arcsMatRef.current = newMat
            }
        })

        // 4. Thrusters (Static?)
        thrustersClone.traverse((child) => {
            if ((child as THREE.Mesh).isMesh) {
                const mesh = child as THREE.Mesh
                // Thrusters
                mesh.material = new THREE.MeshStandardMaterial({
                    map: podTexture,
                    color: 0xffffff,
                    roughness: 0.3,
                    metalness: 0.8
                })
            }
        })

    }, [podClone, flamesClone, arcsClone, thrustersClone, podTexture])

    // Scale/Pos/Rot
    const groupRef = useRef<THREE.Group>(null)

    useFrame((_state, _delta) => {
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



        // Base Rotation: -90 degrees on X to align Z-up model to Y-up world
        const qBase = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2)
        // Heading Rotation: -heading around Y (Standard) + -90 degrees offset for "Left" facing fix
        const qHeading = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -heading + Math.PI / 2)

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

        // Arcs Animation -> Scroll X? (Maybe not needed if no texture, but let's keep ref)
        // if (arcsMatRef.current && arcsMatRef.current.map) ...
    })

    return (
        <group ref={groupRef} scale={[3, 3, 3]} visible={visible}>
            <primitive object={podClone} />
            <primitive object={flamesClone} />
            <primitive object={arcsClone} />
            <primitive object={thrustersClone} />
        </group>
    )
}

const Pod1Model: React.FC<{ pod: any, visible: boolean }> = ({ pod, visible }) => {
    const [podMesh, flamesMesh, arcsMesh, thrustersMesh] = useLoader(OBJLoader, [
        pod1ObjUrl, flames1ObjUrl, arcs1ObjUrl, thrusters1ObjUrl
    ]) as THREE.Group[]
    const podTexture = useTexture(pod1SkinUrl)

    return <GenericPodRender pod={pod} visible={visible} assets={{ podMesh, flamesMesh, arcsMesh, thrustersMesh, podTexture }} />
}

const Pod2Model: React.FC<{ pod: any, visible: boolean }> = ({ pod, visible }) => {
    const [podMesh, flamesMesh, arcsMesh, thrustersMesh] = useLoader(OBJLoader, [
        pod2ObjUrl, flames2ObjUrl, arcs2ObjUrl, thrusters2ObjUrl
    ]) as THREE.Group[]
    const podTexture = useTexture(pod2SkinUrl)

    return <GenericPodRender pod={pod} visible={visible} assets={{ podMesh, flamesMesh, arcsMesh, thrustersMesh, podTexture }} />
}

const PodsRenderer: React.FC<{ swapSkins: boolean }> = ({ swapSkins }) => {
    const { telemetry } = useGameState()
    const pods = telemetry?.race_state?.pods || []

    return (
        <group>
            {pods.map((pod, i) => {
                // Standard: Team 0 -> Pod 1, Team 1 -> Pod 2
                // Swap: Team 0 -> Pod 2, Team 1 -> Pod 1
                const isTeam1 = pod.team === 1
                const usePod2 = swapSkins ? !isTeam1 : isTeam1

                return usePod2
                    ? <Pod2Model key={i} pod={pod} visible={true} />
                    : <Pod1Model key={i} pod={pod} visible={true} />
            })}
        </group>
    )
}
// Preload
// useLoader.preload(OBJLoader, podObjUrl) // Preload if needed


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
    focusedPodIndex: number,
    swapSkins: boolean
}> = ({ mode, focusedPodIndex, swapSkins }) => {
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
            <PodsRenderer swapSkins={swapSkins} />
            <ShieldsRenderer />

            <CameraController mode={mode} focusedPodIndex={focusedPodIndex} />
        </>
    )
}



export const RaceScene3D: React.FC = () => {
    const [cameraMode, setCameraMode] = useState<'orbit' | 'pod'>('orbit')
    const [focusedPodIndex, setFocusedPodIndex] = useState(0)
    const [showMenu, setShowMenu] = useState(false)
    const [swapSkins, setSwapSkins] = useState(false)

    const { telemetry } = useGameState()
    const { playbackSpeed, setPlaybackSpeed } = useGameActions()
    const containerRef = useRef<HTMLDivElement>(null)

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

    const toggleFullscreen = () => {
        if (!document.fullscreenElement) {
            containerRef.current?.requestFullscreen()
        } else {
            document.exitFullscreen()
        }
    }

    return (
        <div ref={containerRef} className="relative w-full aspect-[16/9] bg-black group">
            <Canvas
                camera={{ position: [MAP_CENTER_X, 100, MAP_CENTER_Z + 80], fov: 60 }}
                shadows
                dpr={[1, 2]}
                className="w-full h-full"
            >
                <SceneContent mode={cameraMode} focusedPodIndex={focusedPodIndex} swapSkins={swapSkins} />
            </Canvas>

            {/* --- UI OVERLAY --- */}

            {/* Top Left: Mode Indicator */}
            <div className="absolute top-4 left-4 flex gap-2 z-10 pointer-events-none">
                <div className="text-white/50 text-xs font-mono select-none">
                    3D MODE {[cameraMode.toUpperCase()]}
                </div>
            </div>

            {/* Top Right: Controls (Mode, Fullscreen, Config) */}
            <div className="absolute top-4 right-4 flex gap-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                    onClick={toggleMode}
                    className="px-3 py-1 bg-black/60 hover:bg-black/90 text-white/80 hover:text-white text-xs rounded border border-white/10 backdrop-blur-sm transition-colors"
                >
                    {cameraMode === 'orbit' ? 'Cam: Orbit' : 'Cam: Pod'}
                </button>

                <button
                    onClick={toggleFullscreen}
                    className="p-1 px-2 bg-black/60 hover:bg-black/90 text-white/80 hover:text-white rounded border border-white/10 backdrop-blur-sm transition-colors"
                    title="Fullscreen"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3" /><path d="M21 8V5a2 2 0 0 0-2-2h-3" /><path d="M3 16v3a2 2 0 0 0 2 2h3" /><path d="M16 21h3a2 2 0 0 0 2-2v-3" /><rect x="7" y="7" width="10" height="10" rx="1" /></svg>
                </button>

                {/* Config Menu Trigger */}
                <div className="relative">
                    <button
                        onClick={() => setShowMenu(!showMenu)}
                        className="p-1 px-2 bg-black/60 hover:bg-black/90 text-white/80 hover:text-white rounded border border-white/10 backdrop-blur-sm transition-colors"
                        title="Settings"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
                    </button>

                    {/* Menu Popup */}
                    {showMenu && (
                        <div className="absolute top-full right-0 mt-2 w-48 bg-gray-900 border border-gray-700 rounded shadow-xl p-2 flex flex-col gap-2 z-50">
                            <div className="px-3 py-2">
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
            </div>

            {/* Bottom Left: Skin Swap */}
            <div className="absolute bottom-4 left-4 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                    onClick={() => setSwapSkins(!swapSkins)}
                    className={`px-3 py-1 text-xs rounded border backdrop-blur-sm transition-colors ${swapSkins
                        ? 'bg-blue-600/60 border-blue-400 text-white'
                        : 'bg-black/60 border-white/10 text-white/60 hover:text-white'
                        }`}
                >
                    {swapSkins ? 'Swap Skins: ON' : 'Swap Skins: OFF'}
                </button>
            </div>

            {/* Bottom Center: Pod Selector (Refined) */}
            {cameraMode === 'pod' && (
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 z-10 glass-panel px-3 py-1 rounded-full border border-white/5 bg-black/40 backdrop-blur-md opacity-30 hover:opacity-100 transition-opacity">
                    <button
                        onClick={prevPod}
                        className="w-6 h-6 flex items-center justify-center bg-white/5 hover:bg-white/10 rounded-full text-white/50 hover:text-white transition-colors"
                    >
                        ←
                    </button>
                    <div className="text-white/60 text-xs font-mono w-16 text-center">
                        POD {focusedPodIndex}
                    </div>
                    <button
                        onClick={nextPod}
                        className="w-6 h-6 flex items-center justify-center bg-white/5 hover:bg-white/10 rounded-full text-white/50 hover:text-white transition-colors"
                    >
                        →
                    </button>
                </div>
            )}

            {/* Hint Text */}
            <div className="absolute bottom-1 right-2 text-right text-white/20 text-[9px] font-mono pointer-events-none select-none z-0">
                {cameraMode === 'orbit' ? 'LMB: Rotate | RMB: Pan | Wheel: Zoom' : 'Camera locked to Pod'}
            </div>
        </div>
    )
}

