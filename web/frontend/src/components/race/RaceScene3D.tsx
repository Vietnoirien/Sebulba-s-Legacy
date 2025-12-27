import React, { useRef, useLayoutEffect } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Grid, useTexture } from '@react-three/drei'
import { useGameState } from '../../context/GameStateContext'
import * as THREE from 'three'
import bgImage from '../../assets/background.jpg'

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

const PodsRenderer: React.FC = () => {
    const { telemetry } = useGameState()
    const meshRef = useRef<THREE.InstancedMesh>(null)

    useFrame(() => {
        if (!meshRef.current || !telemetry?.race_state?.pods) return

        const pods = telemetry.race_state.pods
        const count = pods.length
        if (count === 0) return

        const tempObject = new THREE.Object3D()

        pods.forEach((pod, i) => {
            const x = pod.x * SCALE_FACTOR
            const z = pod.y * SCALE_FACTOR

            tempObject.position.set(x, 2, z)

            // Orientation Logic
            // 1. Reset Rotation
            tempObject.rotation.set(0, 0, 0)
            // 2. Rotate to lie flat
            tempObject.rotateZ(-Math.PI / 2)

            // 3. Determine Heading Angle from Velocity
            let heading = pod.angle
            const speed = Math.sqrt(pod.vx * pod.vx + pod.vy * pod.vy)
            if (speed > 5.0) {
                heading = Math.atan2(pod.vy, pod.vx)
            }

            // 4. Apply Heading (Rotate around GLOBAL Y)
            const qBase = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), -Math.PI / 2)
            const qHeading = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -heading)

            // Combine
            qHeading.multiply(qBase)
            tempObject.quaternion.copy(qHeading)

            tempObject.scale.set(2, 2, 2)
            tempObject.updateMatrix()
            meshRef.current!.setMatrixAt(i, tempObject.matrix)

            const color = TEAM_COLORS[pod.team % TEAM_COLORS.length]
            meshRef.current!.setColorAt(i, color)
        })

        meshRef.current.instanceMatrix.needsUpdate = true
        if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true
    })

    return (
        <instancedMesh ref={meshRef} args={[undefined, undefined, 4]}>
            <coneGeometry args={[1, 3, 8]} />
            <meshStandardMaterial />
        </instancedMesh>
    )
}

const ThrustRenderer: React.FC = () => {
    const { telemetry } = useGameState()
    const meshRef = useRef<THREE.InstancedMesh>(null)

    useFrame(() => {
        if (!meshRef.current || !telemetry?.race_state?.pods) return

        const pods = telemetry.race_state.pods
        const tempObject = new THREE.Object3D()

        pods.forEach((pod, i) => {
            // Check thrust (0-100) or check 200 for boost
            const thrust = (pod as any).thrust ?? 0

            if (thrust > 1.0) {
                const x = pod.x * SCALE_FACTOR
                const z = pod.y * SCALE_FACTOR

                tempObject.position.set(x, 2, z)

                // Orientation matches Pod
                tempObject.rotation.set(0, 0, 0)
                tempObject.rotateZ(-Math.PI / 2)

                let heading = pod.angle
                const speed = Math.sqrt(pod.vx * pod.vx + pod.vy * pod.vy)
                if (speed > 5.0) heading = Math.atan2(pod.vy, pod.vx)

                const qBase = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), -Math.PI / 2)
                const qHeading = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -heading)
                qHeading.multiply(qBase)
                tempObject.quaternion.copy(qHeading)

                tempObject.translateX(-4.2)
                tempObject.rotateZ(Math.PI)

                const tScale = Math.min(thrust / 100.0, 1.5)

                tempObject.scale.set(1, tScale, 1)

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
            <coneGeometry args={[0.4, 3, 8]} />
            <meshBasicMaterial color="#ffaa00" transparent opacity={0.8} />
        </instancedMesh>
    )
}

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

const SceneContent: React.FC = () => {
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
            <ThrustRenderer />
            <ShieldsRenderer />

            <OrbitControls makeDefault target={[MAP_CENTER_X, 0, MAP_CENTER_Z]} maxPolarAngle={Math.PI / 2} />
        </>
    )
}

export const RaceScene3D: React.FC = () => {
    return (
        <div className="relative w-full aspect-[16/9] bg-black">
            <Canvas
                camera={{ position: [MAP_CENTER_X, 100, MAP_CENTER_Z + 80], fov: 60 }}
                shadows
                dpr={[1, 2]}
                className="w-full h-full"
            >
                <SceneContent />
            </Canvas>

            {/* Overlay UI */}
            <div className="absolute top-4 left-4 text-white/50 text-xs font-mono pointer-events-none select-none z-10">
                3D MODE (Experimental)
            </div>
            <div className="absolute bottom-4 right-4 text-right text-white/30 text-[10px] font-mono pointer-events-none select-none z-10">
                LMB: Rotate | RMB: Pan | Wheel: Zoom
            </div>
        </div>
    )
}
