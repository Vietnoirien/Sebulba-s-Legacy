import React, { useEffect, useRef, useState } from 'react'
import { useGameState } from '../../context/GameStateContext'

export const LeaguePanel: React.FC = () => {
    const { telemetry } = useGameState()
    const [logs, setLogs] = useState<string[]>([])
    const endRef = useRef<HTMLDivElement>(null)

    // Append new logs
    useEffect(() => {
        if (telemetry?.logs && telemetry.logs.length > 0) {
            setLogs(prev => {
                const newLogs = [...prev, ...telemetry.logs]
                if (newLogs.length > 100) return newLogs.slice(newLogs.length - 100)
                return newLogs
            })
        }
    }, [telemetry])

    // Auto-scroll
    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [logs])

    return (
        <div className="flex-1 bg-slate-900 rounded-lg border border-slate-700 p-2 font-mono text-[10px] overflow-y-auto text-neon-green/80 shadow-inner">
            <div className="mb-1 opacity-50 border-b border-white/10 pb-1">SYSTEM LOG</div>
            <div className="space-y-0.5">
                {logs.length === 0 && (
                    <>
                        <div>&gt; System Initialized</div>
                        <div>&gt; Connecting to Neural Interface...</div>
                    </>
                )}
                {logs.map((log, i) => (
                    <div key={i} className="whitespace-pre-wrap break-all border-b border-white/5 py-0.5">
                        <span className="text-neon-cyan mr-1">&gt;</span>
                        {log}
                    </div>
                ))}
            </div>
            <div ref={endRef} />
        </div>
    )
}
