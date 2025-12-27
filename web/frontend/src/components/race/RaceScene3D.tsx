import React, { useRef, useLayoutEffect } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Grid } from '@react-three/drei'
import { useGameState } from '../../context/GameStateContext'
import * as THREE from 'three'

// Constants matching backend (Physics world is roughly 16000x9000)
const SCALE_FACTOR = 0.01

const TEAM_COLORS = [
    new THREE.Color('#00ffff'), // Team 0: Cyan
    new THREE.Color('#ff00ff'), // Team 1: Magenta (Pink)
]

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
            <meshBasicMaterial color="#555" transparent opacity={0.4} side={THREE.DoubleSide} />
        </instancedMesh>
    )
}

const PodsRenderer: React.FC = () => {
    const { telemetry } = useGameState()
    const meshRef = useRef<THREE.InstancedMesh>(null)

    // We use a ref to accessing telemetry in loop without re-render
    // But useGameState returns current value. 
    // Ideally we want the RAW telemetry object reference which stays stable or use a Ref context.
    // However, `telemetry` from context updates on every frame from the hook.
    // So passing it via prop or context is fine, but we must use it inside useFrame cautiously.

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

            // 2. Rotate to lie flat (Cone Y-up -> Point Z-forward? No, let's point X-forward like backend 0 deg)
            // Backend 0 rad = East (+X).
            // Cone Default: Tip at +Y.
            // Rotate Z -90 deg ( -PI/2 ) -> Tip at +X.
            tempObject.rotateZ(-Math.PI / 2)

            // 3. Determine Heading Angle from Velocity
            // User requested: "Orient in the way of their last speed vector"
            let heading = pod.angle
            const speed = Math.sqrt(pod.vx * pod.vx + pod.vy * pod.vy)
            if (speed > 5.0) { // Threshold to avoid jitter at near-zero speed
                heading = Math.atan2(pod.vy, pod.vx)
            }

            // 4. Apply Heading (Rotate around GLOBAL Y, which is LOCAL X after previous rotation?)
            // No, Euler rotations are applied in order (default XYZ).
            // If we use rotateOnAxis (world) it's easier, or construct Quaternion.

            // Quaternion approach for stability
            // A: Base orientation (Tip +X)
            const qBase = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), -Math.PI / 2)
            // B: Heading rotation (around Y axis)
            const qHeading = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -heading)

            // Combine: Apply Heading * Base
            qHeading.multiply(qBase)
            tempObject.quaternion.copy(qHeading)

            tempObject.scale.set(2, 2, 2)

            tempObject.updateMatrix()
            meshRef.current!.setMatrixAt(i, tempObject.matrix)

            // Color logic
            const color = TEAM_COLORS[pod.team % TEAM_COLORS.length]
            meshRef.current!.setColorAt(i, color)
        })

        meshRef.current.instanceMatrix.needsUpdate = true
        if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true
    })

    return (
        <instancedMesh ref={meshRef} args={[undefined, undefined, 4]}>
            {/* Cone pointing forward? Cone points UP by default. Rotate geometry to point +X */}
            <coneGeometry args={[1, 3, 8]} />
            <meshStandardMaterial />
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

            {/* Ground Plane */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.5, 0]}>
                <planeGeometry args={[1000, 1000]} />
                <meshStandardMaterial color="#080808" roughness={0.8} />
            </mesh>

            {/* Renderers */}
            <CheckpointsRenderer />
            <PodsRenderer />

            <OrbitControls makeDefault maxPolarAngle={Math.PI / 2} />
        </>
    )
}

export const RaceScene3D: React.FC = () => {
    return (
        <div className="relative w-full aspect-[16/9] bg-black">
            {/* Note: Parent container in App.tsx sets height */}
            <Canvas
                camera={{ position: [0, 150, 100], fov: 60 }}
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

            {/* Info Overlay */}
            <div className="absolute bottom-4 right-4 text-right text-white/30 text-[10px] font-mono pointer-events-none select-none z-10">
                LMB: Rotate | RMB: Pan | Wheel: Zoom
            </div>
        </div>
    )
}
