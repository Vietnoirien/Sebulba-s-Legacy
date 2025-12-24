import React from 'react'
import { Activity } from 'lucide-react'
import { useGameState } from '../../context/GameStateContext'
import { ReadyState } from 'react-use-websocket'

export const Header: React.FC = () => {
    const { telemetry, readyState, connectionStatus } = useGameState()

    const statusColors = {
        [ReadyState.CONNECTING]: 'text-yellow-400 bg-yellow-900/30 border-yellow-800',
        [ReadyState.OPEN]: 'text-neon-green bg-green-900/30 border-green-800',
        [ReadyState.CLOSING]: 'text-orange-400 bg-orange-900/30 border-orange-800',
        [ReadyState.CLOSED]: 'text-red-400 bg-red-900/30 border-red-800',
        [ReadyState.UNINSTANTIATED]: 'text-gray-400 bg-gray-800 border-gray-700',
    }

    return (
        <header className="h-16 bg-panel-bg border-b border-gray-700 flex justify-between items-center px-6 shadow-lg z-10">
            <div className="flex items-center gap-3">
                <div className="p-2 bg-slate-800 rounded-lg border border-cyan-900 icon-glow">
                    <Activity className="text-neon-cyan" size={24} />
                </div>
                <div>
                    <h1 className="text-lg font-bold tracking-wide text-white uppercase text-shadow-sm">
                        Sebulba's <span className="text-neon-cyan">Legacy</span>
                    </h1>
                    <span className="text-xs text-gray-400 font-mono tracking-widest">ADVANCED RACING AI</span>
                </div>
            </div>

            <div className="flex items-center gap-4 font-mono text-sm">
                <div className="flex items-center gap-2 bg-slate-800 border border-slate-700 px-3 py-1.5 rounded-md">
                    <span className="text-gray-400 uppercase text-[10px]">Model</span>
                    <span className="text-neon-cyan font-bold truncate max-w-[150px]">
                        {telemetry?.stats.active_model || 'N/A'}
                    </span>
                </div>

                <div className="flex items-center gap-2 bg-slate-800 border border-slate-700 px-3 py-1.5 rounded-md">
                    <span className="text-gray-400 uppercase text-[10px]">TPS</span>
                    <span className="text-purple-400 font-bold min-w-[30px] text-right">
                        {telemetry?.stats.fps_physics?.toFixed(0) || 0}
                    </span>
                </div>

                <div className={`px-3 py-1.5 rounded-md border text-xs font-bold uppercase tracking-wider ${statusColors[readyState]}`}>
                    {connectionStatus}
                </div>
            </div>
        </header>
    )
}
