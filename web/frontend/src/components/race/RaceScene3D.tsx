import React, { useRef, useLayoutEffect, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, useTexture, useGLTF, Text, Environment, Sky } from '@react-three/drei'
import { SkeletonUtils } from 'three-stdlib'
import { useGameState, useGameActions } from '../../context/GameStateContext'
import * as THREE from 'three'
// @ts-ignore
import mapGlbUrl from '../../assets/models/maps/StoneQuarry_SPT_mesh.glb'
// Checkpoint Assets
// @ts-ignore
// @ts-ignore
import groundGlbUrl from '../../assets/models/Chckpts/ground.glb'
// @ts-ignore
import cpk0GlbUrl from '../../assets/models/Chckpts/ckp-0.glb'
// @ts-ignore
import cpk1GlbUrl from '../../assets/models/Chckpts/ckp-1.glb'
// @ts-ignore
import pod1CheckGlbUrl from '../../assets/models/Chckpts/pod1-check.glb'
// @ts-ignore
import pod2CheckGlbUrl from '../../assets/models/Chckpts/pod2-check.glb'

import checkpointTextureUrl from '../../assets/models/Chckpts/checkpoint.png'
import groundNormalUrl from '../../assets/models/Chckpts/ckp-normal.png'
// @ts-ignore
// Pod 1 Assets
// @ts-ignore
// @ts-ignore
// Pod 1 Assets
// @ts-ignore
import pod1GlbUrl from '../../assets/models/Pods/pod-1/pod.glb'
// @ts-ignore
import flames1GlbUrl from '../../assets/models/Pods/pod-1/flames.glb'
// @ts-ignore
import arcs1GlbUrl from '../../assets/models/Pods/pod-1/arcs.glb'
// @ts-ignore
import thrusters1GlbUrl from '../../assets/models/Pods/pod-1/thrusters.glb'
import pod1SkinUrl from '../../assets/models/Pods/pod-1/pod-1-skin.png'

// Pod 2 Assets
// @ts-ignore
import pod2GlbUrl from '../../assets/models/Pods/pod-2/pod-2.glb'
// @ts-ignore
import flames2GlbUrl from '../../assets/models/Pods/pod-2/flames-2.glb'
// @ts-ignore
import arcs2GlbUrl from '../../assets/models/Pods/pod-2/arcs-2.glb'
// @ts-ignore
import thrusters2GlbUrl from '../../assets/models/Pods/pod-2/thrusters-2.glb'
import pod2SkinUrl from '../../assets/models/Pods/pod-2/pod-2-skin.png'
import racerFontUrl from '../../fonts/racer/RACER___.TTF'

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
    // Load Map Model
    const { scene } = useGLTF(mapGlbUrl)

    // Memoize the cloned scene to avoid re-cloning on every render
    const mapClone = React.useMemo(() => {
        const clone = SkeletonUtils.clone(scene)

        // Apply Global Scale
        clone.scale.set(1, 1, 1)

        // Fix Orientation? OBJ usually comes in Y-up, but game coords might differ.
        // If it was exported from Blender +Z up, we might need rotation.
        // Based on previous ground plane: rotation={[-Math.PI / 2, 0, 0]}
        // Let's assume the GLB is correct or needs standard X-flip if it was Z-up in Blender
        // Standard GLTF is Y-up. If our map is Z-up in data, we usually rotate X by -90 deg.
        // However, the optimize_assets script didn't rotate.
        // Let's suspect we might need a rotation if it lies flat. 
        // We'll inspect visuals. For now, assume Y-up GLTF standard.

        return clone
    }, [scene])

    return <primitive object={mapClone} position={[MAP_CENTER_X, -82, MAP_CENTER_Z]} />
}

const SingleCheckpoint: React.FC<{
    cp: any,
    assets: any,
    crossings: Record<string, number>
}> = ({ cp, assets, crossings }) => {
    const groupRef = useRef<THREE.Group>(null)

    // Clone Assets for this instance
    const disableCulling = (obj: THREE.Object3D) => {
        obj.traverse((child) => {
            if ((child as THREE.Mesh).isMesh) {
                (child as THREE.Mesh).frustumCulled = false
            }
        })
        return obj
    }

    // Clone Assets for this instance, ensure culling is disabled on the specific instances
    const groundClone = React.useMemo(() => disableCulling(SkeletonUtils.clone(assets.ground)), [assets.ground])
    const cpk0Clone = React.useMemo(() => disableCulling(SkeletonUtils.clone(assets.cpk0)), [assets.cpk0])
    const cpk1Clone = React.useMemo(() => disableCulling(SkeletonUtils.clone(assets.cpk1)), [assets.cpk1])
    const pod1CheckClone = React.useMemo(() => disableCulling(SkeletonUtils.clone(assets.pod1Check)), [assets.pod1Check])
    const pod2CheckClone = React.useMemo(() => disableCulling(SkeletonUtils.clone(assets.pod2Check)), [assets.pod2Check])

    // Rotation Logic
    useFrame((_, delta) => {
        if (!groupRef.current) return
        const speed = 1.0
        groupRef.current.rotation.y += delta * speed
    })

    const x = cp.x * SCALE_FACTOR
    const z = cp.y * SCALE_FACTOR

    const isStart = cp.id === 0
    const Model = isStart ? cpk0Clone : cpk1Clone

    // Crossing Visibility
    const showPod1 = (Date.now() - (crossings[`${cp.id}-0`] || 0)) < 1000
    const showPod2 = (Date.now() - (crossings[`${cp.id}-1`] || 0)) < 1000

    return (
        <group position={[x, 0, z]}>
            <primitive object={groundClone} />

            <group ref={groupRef}>
                <primitive object={Model} />
                {!isStart && (
                    <Text
                        position={[0, 4, 0]}
                        rotation={[0, Math.PI / 4, 0]}
                        fontSize={2}
                        color="white"
                        anchorX="center"
                        anchorY="middle"
                        outlineWidth={0.1}
                        outlineColor="black"
                        font={racerFontUrl}
                    >
                        {cp.id}
                    </Text>
                )}
            </group>

            {showPod1 && (
                <primitive object={pod1CheckClone} position={[0, 1, 0]} />
            )}
            {showPod2 && (
                <primitive object={pod2CheckClone} position={[0, 1, 0]} />
            )}
        </group>
    )
}

const CheckpointsRenderer: React.FC = () => {
    const { telemetry } = useGameState()
    const checkpoints = telemetry?.race_state?.checkpoints || []
    const pods = telemetry?.race_state?.pods || []

    // Load Assets
    const { scene: groundObj } = useGLTF(groundGlbUrl)
    const { scene: cpk0Obj } = useGLTF(cpk0GlbUrl)
    const { scene: cpk1Obj } = useGLTF(cpk1GlbUrl)
    const { scene: pod1CheckObj } = useGLTF(pod1CheckGlbUrl)
    const { scene: pod2CheckObj } = useGLTF(pod2CheckGlbUrl)

    const checkpointTexture = useTexture(checkpointTextureUrl)
    checkpointTexture.flipY = false
    const groundNormal = useTexture(groundNormalUrl)
    groundNormal.flipY = false

    // Process Assets Once
    const processedAssets = React.useMemo(() => {
        // prepare function now handles both "Opaque" and "Two-Pass Glass" modes
        const prepare = (obj: THREE.Group, normalMap?: THREE.Texture, rotateOnX: boolean = false, isGlass: boolean = false) => {

            const setupMaterial = (mesh: THREE.Mesh, side: THREE.Side, renderOrder: number) => {
                mesh.frustumCulled = false
                const usedMaterial = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material

                // Clone material to avoid shared state
                const newMat = usedMaterial.clone() as THREE.MeshStandardMaterial

                // Force texture as requested
                newMat.map = checkpointTexture

                if (normalMap) {
                    newMat.normalMap = normalMap
                } else {
                    newMat.normalMap = null
                }

                newMat.side = side
                newMat.transparent = true // Always enabled for these assets

                if (isGlass) {
                    // Glass Mode: Disable Depth Write to prevent self-occlusion artifacts
                    newMat.depthWrite = false
                    mesh.renderOrder = renderOrder
                } else {
                    // Standard Mode (Ground): Enable Depth Write
                    newMat.depthWrite = true
                    mesh.renderOrder = 0
                }

                mesh.material = newMat
            }

            if (!isGlass) {
                // --- Simple Path (Ground) ---
                const clone = SkeletonUtils.clone(obj)
                if (rotateOnX) clone.rotation.x = -Math.PI / 2

                clone.traverse((child) => {
                    if ((child as THREE.Mesh).isMesh) {
                        // Ground uses DoubleSide usually, or FrontSide. Keeping DoubleSide as per original
                        setupMaterial(child as THREE.Mesh, THREE.DoubleSide, 0)
                    }
                })
                return clone
            } else {
                // --- Two-Pass Path (Glass Checkpoints) ---
                // We create a container and add two copies: BackSide first, then FrontSide.
                // This guarantees the back is drawn before the front, solving the transparency sorting issue.
                const root = new THREE.Group()

                // Pass 1: Back Faces
                const backPass = SkeletonUtils.clone(obj)
                if (rotateOnX) backPass.rotation.x = -Math.PI / 2
                backPass.traverse((child) => {
                    if ((child as THREE.Mesh).isMesh) {
                        setupMaterial(child as THREE.Mesh, THREE.BackSide, 1) // Render Order 1
                    }
                })
                root.add(backPass)

                // Pass 2: Front Faces
                const frontPass = SkeletonUtils.clone(obj)
                if (rotateOnX) frontPass.rotation.x = -Math.PI / 2
                frontPass.traverse((child) => {
                    if ((child as THREE.Mesh).isMesh) {
                        setupMaterial(child as THREE.Mesh, THREE.FrontSide, 2) // Render Order 2
                    }
                })
                root.add(frontPass)

                return root
            }
        }

        return {
            ground: prepare(groundObj, groundNormal, true, false),      // Ground: Normal rendering
            cpk0: prepare(cpk0Obj, undefined, true, true),              // Checkpoints: Two-Pass Glass
            cpk1: prepare(cpk1Obj, undefined, true, true),              // Checkpoints: Two-Pass Glass
            pod1Check: prepare(pod1CheckObj, undefined, false, true),   // Markers: Two-Pass Glass
            pod2Check: prepare(pod2CheckObj, undefined, false, true),   // Markers: Two-Pass Glass
        }
    }, [groundObj, cpk0Obj, cpk1Obj, pod1CheckObj, pod2CheckObj, checkpointTexture, groundNormal])

    // Crossing Logic
    const lastPodNextCps = useRef<Record<number, number>>({})
    const [crossings, setCrossings] = useState<Record<string, number>>({})

    useFrame(() => {
        if (!pods) return

        const newCrossings: Record<string, number> = {}
        let changed = false

        pods.forEach((pod, index) => {
            const currentNext = pod.next_checkpoint ?? 1
            const prevNext = lastPodNextCps.current[index]

            if (prevNext !== undefined && currentNext !== prevNext) {
                // Optimization: ignore undefined (start)
                // Detected crossing of prevNext
                const passedCpId = prevNext
                const key = `${passedCpId}-${pod.team}`
                newCrossings[key] = Date.now()
                changed = true
            }
            lastPodNextCps.current[index] = currentNext
        })

        if (changed) {
            setCrossings(prev => ({ ...prev, ...newCrossings }))
        }
    })

    return (
        <group>
            {checkpoints.map((cp) => (
                <SingleCheckpoint
                    key={cp.id}
                    cp={cp}
                    assets={processedAssets}
                    crossings={crossings}
                />
            ))}
        </group>
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

    // Store latest pod state in ref to avoid stale closures in useFrame
    const podRef = useRef(pod)
    useLayoutEffect(() => {
        podRef.current = pod
    }, [pod])

    // Initial positioning flag
    const isInitialized = useRef(false)

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

    useFrame((_state, delta) => {
        if (!groupRef.current) return

        const currentPod = podRef.current

        // 1. Position & Rotation Logic
        const targetX = currentPod.x * SCALE_FACTOR
        const targetZ = currentPod.y * SCALE_FACTOR
        const targetPos = new THREE.Vector3(targetX, 3, targetZ)

        let heading = currentPod.angle
        const speed = Math.sqrt(currentPod.vx * currentPod.vx + currentPod.vy * currentPod.vy)
        if (speed > 5.0) heading = Math.atan2(currentPod.vy, currentPod.vx)

        const targetQuat = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -heading)

        if (!isInitialized.current) {
            // Hard Snap on first frame
            groupRef.current.position.copy(targetPos)
            groupRef.current.quaternion.copy(targetQuat)
            isInitialized.current = true
        } else {
            // Smooth Interpolation
            // Use a higher lerp factor for responsiveness, but enough to smooth discrete updates
            // delta * 15 means it closes ~90% of the gap in 0.15s (assuming 60fps)
            // Adjust this value if it feels too laggy or too jittery
            const smoothFactor = Math.min(delta * 15.0, 1.0)

            groupRef.current.position.lerp(targetPos, smoothFactor)
            groupRef.current.quaternion.slerp(targetQuat, smoothFactor)
        }

        // 3. Logic Updates (Flames, Team Color, Arcs)
        const thrust = (currentPod as any).thrust ?? 0
        if (flamesMatRef.current) {
            const opacity = Math.max(0, thrust) / 100.0
            flamesMatRef.current.opacity = opacity
            flamesMatRef.current.visible = opacity > 0.01
        }

        // Color -> Team
        if (bodyMatRef.current) {
            const color = TEAM_COLORS[currentPod.team % TEAM_COLORS.length]
            bodyMatRef.current.color.set(color)
        }
    })

    return (
        <group ref={groupRef} visible={visible}>
            <primitive object={podClone} />
            <primitive object={flamesClone} />
            <primitive object={arcsClone} />
            <primitive object={thrustersClone} />
        </group>
    )
}

const PodsRenderer: React.FC<{ swapSkins: boolean }> = ({ swapSkins }) => {
    const { telemetry } = useGameState()
    const pods = telemetry?.race_state?.pods || []

    // ---- LOAD ASSETS ONCE ----
    const pod1Texture = useTexture(pod1SkinUrl)
    pod1Texture.flipY = false
    const pod2Texture = useTexture(pod2SkinUrl)
    pod2Texture.flipY = false

    // POD 1
    const { scene: pod1Mesh } = useGLTF(pod1GlbUrl)
    const { scene: flames1Mesh } = useGLTF(flames1GlbUrl)
    const { scene: arcs1Mesh } = useGLTF(arcs1GlbUrl)
    const { scene: thrusters1Mesh } = useGLTF(thrusters1GlbUrl)

    // POD 2
    const { scene: pod2Mesh } = useGLTF(pod2GlbUrl)
    const { scene: flames2Mesh } = useGLTF(flames2GlbUrl)
    const { scene: arcs2Mesh } = useGLTF(arcs2GlbUrl)
    const { scene: thrusters2Mesh } = useGLTF(thrusters2GlbUrl)

    // Bundle Assets
    const pod1Assets = React.useMemo(() => ({
        podMesh: pod1Mesh, flamesMesh: flames1Mesh, arcsMesh: arcs1Mesh, thrustersMesh: thrusters1Mesh, podTexture: pod1Texture
    }), [pod1Mesh, flames1Mesh, arcs1Mesh, thrusters1Mesh, pod1Texture])

    const pod2Assets = React.useMemo(() => ({
        podMesh: pod2Mesh, flamesMesh: flames2Mesh, arcsMesh: arcs2Mesh, thrustersMesh: thrusters2Mesh, podTexture: pod2Texture
    }), [pod2Mesh, flames2Mesh, arcs2Mesh, thrusters2Mesh, pod2Texture])


    return (
        <group>
            {pods.map((pod, i) => {
                // Standard: Team 0 -> Pod 1, Team 1 -> Pod 2
                // Swap: Team 0 -> Pod 2, Team 1 -> Pod 1
                const isTeam1 = pod.team === 1
                const usePod2 = swapSkins ? !isTeam1 : isTeam1

                return usePod2
                    ? <GenericPodRender key={i} pod={pod} visible={true} assets={pod2Assets} />
                    : <GenericPodRender key={i} pod={pod} visible={true} assets={pod1Assets} />
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
                const dist = 5
                const height = 5

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
                targetPos.current.set(x, 3, z) // Target: slightly above pod center
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


            <BackgroundRenderer />
            <CheckpointsRenderer />
            <PodsRenderer swapSkins={swapSkins} />
            <ShieldsRenderer />

            <Environment preset="sunset" />
            <Sky />
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
                camera={{ position: [MAP_CENTER_X, 100, MAP_CENTER_Z + 80], fov: 60, far: 5000 }}
                shadows
                dpr={[1, 2]}
                className="w-full h-full"
            >
                <React.Suspense fallback={null}>
                    <SceneContent
                        key={telemetry?.match_id}
                        mode={cameraMode}
                        focusedPodIndex={focusedPodIndex}
                        swapSkins={swapSkins}
                    />
                </React.Suspense>
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
                                    max="5.0"
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
