import React, { useState, useEffect } from 'react'
import { useLocalStorage } from '../../hooks/useLocalStorage'
import { Play, Square, RotateCcw, Save, Loader, Trash2 } from 'lucide-react'
import { Button } from '../common/Button'
import { useGameState, useGameActions } from '../../context/GameStateContext'

export const ControlPanel: React.FC = () => {
    const { telemetry } = useGameState()
    const { selectedModel, setSelectedModel } = useGameActions()
    const [checkpoints, setCheckpoints] = useState<any[]>([])
    // const [selectedModel, setSelectedModel] = useState<string>("scratch") // Moved to Context
    // const [selectedModel, setSelectedModel] = useState<string>("scratch") // Moved to Context
    const [curriculumMode, setCurriculumMode] = useLocalStorage("spt2_control_curriculumMode", "auto")
    const [curriculumStage, setCurriculumStage] = useLocalStorage<number>("spt2_control_stage_v2", 0)
    const [maxCheckpoints, setMaxCheckpoints] = useLocalStorage("spt2_control_maxCheckpoints", 5)


    const [generations, setGenerations] = useState<any[]>([])

    // Fetch Checkpoints & Generations on Mount
    useEffect(() => {
        const fetchData = async () => {
            try {
                // Checkpoints
                const resCP = await fetch('http://localhost:8000/api/checkpoints')
                const dataCP = await resCP.json()
                setCheckpoints(dataCP)

                // Generations
                const resGen = await fetch('http://localhost:8000/api/generations')
                const dataGen = await resGen.json()
                setGenerations(dataGen)

            } catch (e) { console.error("Failed to fetch data", e) }
        }
        fetchData()
        const interval = setInterval(fetchData, 5000) // Poll
        return () => clearInterval(interval)
    }, [])

    const handleAction = async (action: 'start' | 'stop' | 'reset' | 'save') => {
        try {
            if (action === 'start') {
                await fetch(`http://localhost:8000/api/${action}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: selectedModel,
                        curriculum_mode: curriculumMode,
                        curriculum_stage: curriculumStage,
                        max_checkpoints: maxCheckpoints,
                    })
                })
            } else {
                await fetch(`http://localhost:8000/api/${action}`, { method: 'POST' })
            }
        } catch (e) {
            console.error(`Failed to ${action}`, e)
        }
    }

    const handleExport = async () => {
        try {
            const res = await fetch('http://localhost:8000/api/export', { method: 'POST' })
            if (res.ok) {
                const blob = await res.blob()
                const url = window.URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = 'submission.py'
                document.body.appendChild(a)
                a.click()
                a.remove()
            }
        } catch (e) {
            console.error("Export failed", e)
        }
    }

    const handleWipe = async () => {
        if (!confirm("ARE YOU SURE? This will delete ALL checkpoints permanently.")) return
        try {
            const res = await fetch('http://localhost:8000/api/checkpoints/reset', { method: 'DELETE' })
            if (res.ok) {
                setCheckpoints([]) // Clear local state immediately
                setGenerations([]) // Clear generations
            }
        } catch (e) {
            console.error("Wipe failed", e)
        }
    }

    return (
        <div className="space-y-4">
            {/* Visual Feedback: Active Model */}
            <div className="bg-slate-800/50 p-2 rounded border border-slate-700">
                <h3 className="text-[10px] text-gray-400 font-mono uppercase">Active Model</h3>
                <div className="flex items-center gap-2">
                    <Loader className={`w-3 h-3 ${telemetry?.stats.active_model ? 'text-teal-400 animate-spin-slow' : 'text-gray-500'}`} />
                    <span className="text-sm font-mono font-bold text-white">
                        {telemetry?.stats.active_model || "None"}
                    </span>
                </div>
            </div>

            <div className="h-px bg-gray-700/50" />

            <h2 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-2 font-mono">Operations</h2>

            {/* Model Selector */}
            <div className="mb-2">
                <label className="text-[10px] text-gray-400 font-mono uppercase block mb-1">Start With...</label>
                <select
                    className="w-full bg-slate-900 text-white text-xs border border-gray-700 rounded p-1 font-mono"
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                >
                    <option value="scratch">Start from Scratch</option>

                    {/* Generations Group */}
                    {generations.length > 0 && (
                        <optgroup label="Entire Generations (Resume Training)">
                            {generations
                                .sort((a, b) => {
                                    const numA = parseInt(a.name.replace(/\D/g, '')) || 0;
                                    const numB = parseInt(b.name.replace(/\D/g, '')) || 0;
                                    return numB - numA;
                                })
                                .map((gen: any) => (
                                    <option key={gen.id} value={gen.id}>
                                        {gen.name} ({gen.agent_count} Agents)
                                    </option>
                                ))}
                        </optgroup>
                    )}

                    {/* Checkpoints Group */}
                    {checkpoints.length > 0 && (
                        <optgroup label="Best Agents (Load Leader)">
                            {checkpoints
                                .sort((a, b) => (b.metrics?.laps_score || 0) - (a.metrics?.laps_score || 0))
                                .map((cp: any) => (
                                    <option key={cp.id} value={cp.id}>
                                        {cp.id} {cp.metrics && cp.metrics.laps_score ? `(Laps: ${cp.metrics.laps_score})` : ``}
                                    </option>
                                ))}
                        </optgroup>
                    )}
                </select>
            </div>

            {/* Curriculum Configuration */}
            <div className="mb-4 space-y-2">
                <div>
                    <label className="text-[10px] text-gray-400 font-mono uppercase block mb-1">Progression Mode</label>
                    <div className="flex bg-slate-900 rounded border border-gray-700 p-0.5">
                        {['auto', 'manual'].map(mode => (
                            <button
                                key={mode}
                                onClick={() => setCurriculumMode(mode)}
                                className={`flex-1 text-[10px] uppercase font-mono py-1 rounded ${curriculumMode === mode ? 'bg-teal-600 text-white' : 'text-gray-500 hover:text-gray-300'}`}
                            >
                                {mode}
                            </button>
                        ))}
                    </div>
                </div>

                <div>
                    <label className="text-[10px] text-gray-400 font-mono uppercase block mb-1">Stage (Manual Only)</label>
                    <select
                        className="w-full bg-slate-900 text-white text-xs border border-gray-700 rounded p-1 font-mono disabled:opacity-50"
                        value={curriculumStage}
                        onChange={(e) => setCurriculumStage(Number(e.target.value))}
                        disabled={curriculumMode === 'auto'}
                    >
                        <option value={0}>Stage 0: Nursery (Safe Mode)</option>
                        <option value={1}>Stage 1: Solo Time Trial</option>
                        <option value={2}>Stage 2: 1v1 Duel</option>
                        <option value={3}>Stage 3: Blocker Academy (PvE)</option>
                        <option value={4}>Stage 4: 2v2 Team</option>
                        <option value={5}>Stage 5: League</option>
                    </select>
                </div>
            </div>

            <div className="mb-4">
                <label className="text-[10px] text-gray-400 font-mono uppercase block mb-1">Max Checkpoints</label>
                <div className="flex items-center gap-2">
                    <input
                        type="range"
                        min="1"
                        max="20"
                        step="1"
                        value={maxCheckpoints}
                        onChange={(e) => setMaxCheckpoints(parseInt(e.target.value))}
                        className="flex-1 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                    />
                    <span className="text-xs text-white font-mono w-6 text-right">{maxCheckpoints}</span>
                </div>
                <p className="text-[9px] text-gray-500 font-mono mt-0.5">Keeps newest per stage (by Date)</p>
            </div>

            <div className="grid grid-cols-2 gap-2">
                <Button
                    variant="primary"
                    icon={Play}
                    onClick={() => handleAction('start')}
                    className="col-span-2 bg-gradient-to-r from-teal-600 to-cyan-600 border-0"
                >
                    INIT SEQUENCE
                </Button>
                <Button
                    variant="danger"
                    icon={Square}
                    onClick={() => handleAction('stop')}
                >
                    HALT
                </Button>
                <Button
                    variant="warning"
                    icon={RotateCcw}
                    onClick={() => handleAction('reset')}
                >
                    RESET
                </Button>
                <Button
                    variant="neutral"
                    icon={Save}
                    onClick={() => handleAction('save')}
                >
                    SNAPSHOT
                </Button>
                <Button
                    variant="primary"
                    icon={Save}
                    onClick={handleExport}
                    className="col-span-2 bg-slate-700 hover:bg-slate-600 text-teal-400 border border-teal-500/30"
                >
                    EXPORT SUBMISSION
                </Button>

                <div className="col-span-2 h-px bg-gray-700/50 my-2" />

                <Button
                    variant="danger"
                    icon={Trash2}
                    onClick={handleWipe}
                    className="col-span-2 bg-red-900/50 hover:bg-red-900/80 text-red-400 border border-red-500/30"
                >
                    WIPE ALL CHECKPOINTS
                </Button>
            </div>
        </div >
    )

}

