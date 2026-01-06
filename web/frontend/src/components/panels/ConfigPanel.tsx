import React, { useState, useEffect } from 'react'
import { Slider } from '../common/Slider'
import { useGameActions } from '../../context/GameStateContext'
import { useLocalStorage } from '../../hooks/useLocalStorage'
import { Save, Upload, Trash2, Play, Target, Cpu, Database, Layers, Rocket } from 'lucide-react'

// Reward Indices (Must match Backend config.py)
const RW = {
    WIN: 0,
    LOSS: 1,
    CHECKPOINT: 2,
    CHECKPOINT_SCALE: 3,
    PROGRESS: 4, // Was VELOCITY
    COLLISION_RUNNER: 5,
    COLLISION_BLOCKER: 6,
    STEP_PENALTY: 7,
    ORIENTATION: 8,
    WRONG_WAY: 9,
    COLLISION_MATE: 10,
    PROXIMITY: 11,
    MAGNET: 12,
    RANK: 13,
    DENIAL: 15
}

interface ConfigPreset {
    name: string
    data: any
}

export const ConfigPanel: React.FC = () => {
    const { sendMessage, selectedModel } = useGameActions()
    const [activeTab, setActiveTab] = useLocalStorage<'stages' | 'rewards' | 'training' | 'presets'>('spt2_config_activeTab_v2', 'stages')

    // --- State ---
    const [rewards, setRewards] = useLocalStorage('spt2_config_rewards_v9', {
        weights: {
            [RW.WIN]: 10000.0,
            [RW.LOSS]: 2000.0,
            [RW.CHECKPOINT]: 500.0,
            [RW.CHECKPOINT_SCALE]: 50.0,
            [RW.PROGRESS]: 0.2,
            [RW.MAGNET]: 10.0,
            [RW.ORIENTATION]: 1.0,
            [RW.WRONG_WAY]: 10.0,
            [RW.COLLISION_BLOCKER]: 5.0,
            [RW.COLLISION_RUNNER]: 0.5,
            [RW.COLLISION_MATE]: 2.0,
            [RW.STEP_PENALTY]: 10.0,
            [RW.PROXIMITY]: 5.0,
            [RW.RANK]: 500.0,
            [RW.DENIAL]: 0.5
        },
        tau: 0.0,
        team_spirit: 0.0
    })

    const [curriculum, setCurriculum] = useLocalStorage('spt2_config_curriculum_v2', {
        stage: 0,
        difficulty: 0.0
    })

    const [hyperparams, setHyperparams] = useLocalStorage('spt2_config_hyperparams', {
        lr: 1e-4,
        ent_coef: 0.01
    })

    const [transitions, setTransitions] = useLocalStorage('spt2_config_transitions_v10', {
        nursery_consistency_threshold: 500.0,
        solo_efficiency_threshold: 45.0,
        solo_consistency_threshold: 1500.0,

        // Stage 2 -> 3
        duel_graduation_difficulty: 0.80,
        duel_graduation_win_rate: 0.65,
        duel_graduation_checks: 5,

        // Stage 3 -> 4
        team_graduation_difficulty: 0.85,
        team_graduation_win_rate: 0.70,
        team_graduation_checks: 5
    })

    // Presets
    const [presets, setPresets] = useState<ConfigPreset[]>([])
    const [selectedPreset, setSelectedPreset] = useLocalStorage<string>("spt2_config_selectedPreset", "")
    const [saveName, setSaveName] = useState("")

    // Fetch Presets on Mount
    useEffect(() => {
        fetchPresets()
    }, [])

    const fetchPresets = async () => {
        try {
            const res = await fetch('http://localhost:8000/api/configs')
            const data = await res.json()
            setPresets(data)
        } catch (e) {
            console.error("Failed to fetch presets", e)
        }
    }

    const handleApply = () => {
        const payload = {
            type: 'config',
            payload: {
                rewards: rewards,
                curriculum: curriculum,
                hyperparameters: hyperparams,
                transitions: transitions
            }
        }
        sendMessage(JSON.stringify(payload))
    }

    const handleLaunch = async () => {
        const config = {
            rewards: rewards,
            curriculum: curriculum,
            hyperparameters: hyperparams,
            transitions: transitions
        }

        try {
            await fetch('http://localhost:8000/api/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: selectedModel, // Use Global Selection
                    curriculum_mode: "auto",
                    curriculum_stage: curriculum.stage,
                    config: config
                })
            })
        } catch (e) {
            console.error("Failed to launch", e)
        }
    }

    const handleSavePreset = async () => {
        if (!saveName) return
        if (saveName.toLowerCase() === 'default') {
            alert("Cannot overwrite default preset")
            return
        }

        const config = {
            rewards: rewards,
            curriculum: curriculum,
            hyperparameters: hyperparams,
            transitions: transitions
        }

        try {
            await fetch('http://localhost:8000/api/configs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: saveName, config })
            })
            setSaveName("")
            fetchPresets()
        } catch (e) {
            console.error("Save failed", e)
        }
    }

    const handleLoadPreset = (name?: string) => {
        const targetName = name || selectedPreset
        if (!targetName) return
        const preset = presets.find(p => p.name === targetName)
        if (!preset) return

        const d = preset.data || {}
        if (d.rewards) setRewards(prev => ({ ...prev, ...d.rewards }))
        if (d.curriculum) setCurriculum(prev => ({ ...prev, ...d.curriculum }))
        if (d.hyperparameters) setHyperparams(prev => ({ ...prev, ...d.hyperparameters }))
        if (d.transitions) setTransitions(prev => ({ ...prev, ...d.transitions }))
    }

    const handleDeletePreset = async (name?: string) => {
        const targetName = name || selectedPreset
        if (!targetName) return
        try {
            await fetch(`http://localhost:8000/api/configs/${targetName}`, { method: 'DELETE' })
            setSelectedPreset("")
            fetchPresets()
        } catch (e) {
            console.error("Delete failed", e)
        }
    }

    // Helper for Sliders
    const setWeight = (idx: number, val: number) => {
        setRewards(prev => ({
            ...prev,
            weights: { ...prev.weights, [idx]: val }
        }))
    }

    const TabButton = ({ id, icon: Icon, label }: { id: typeof activeTab, icon: any, label: string }) => (
        <button
            onClick={() => setActiveTab(id)}
            className={`flex-1 py-2 flex flex-col items-center justify-center gap-1 text-[10px] uppercase font-mono transition-colors border-b-2 ${activeTab === id
                ? 'border-neon-cyan text-neon-cyan bg-neon-cyan/5'
                : 'border-transparent text-gray-500 hover:text-gray-300 hover:bg-gray-800'
                }`}
        >
            <Icon size={14} />
            <span>{label}</span>
        </button>
    )

    // STAGE HELPERS
    const renderStageConfig = () => {
        const st = curriculum.stage;
        return (
            <div className="space-y-4">
                {/* Stage Description for User Context */}
                <div className="p-3 bg-cyan-900/20 border border-cyan-800 rounded">
                    <h4 className="text-neon-cyan text-xs font-bold uppercase mb-1">
                        {st === 0 && "Stage 0: Nursery"}
                        {st === 1 && "Stage 1: Solo Time Trial"}
                        {st === 2 && "Stage 2: Duel (1v1)"}
                        {st === 3 && "Stage 3: Team (2v2)"}
                        {st === 4 && "Stage 4: League"}
                    </h4>
                    <p className="text-gray-400 text-[10px]">
                        {st === 0 && "Goal: Learn basic navigation. Simple tracks, no bots. Graduate by reaching Consistency threshold."}
                        {st === 1 && "Goal: Speed & Efficiency. Complex tracks, no bots. Penalties for slow transitions."}
                        {st === 2 && "Goal: Beating a Rival. 1v1 against a bot. Win Rate matters."}
                        {st === 3 && "Goal: Team Coordination. 2v2. Work with teammate to win."}
                        {st === 4 && "Goal: Self-Play Supremacy. No explicit graduation."}
                    </p>
                </div>

                {st >= 2 && (
                    <div className="bg-slate-800/30 p-3 rounded border border-slate-700">
                        <h3 className="text-[10px] text-gray-500 font-bold uppercase tracking-widest mb-2">Bot Settings</h3>
                        <Slider label="BOT DIFFICULTY" min={0} max={1} step={0.01} value={curriculum.difficulty} valueDisplay={curriculum.difficulty.toFixed(2)}
                            onChange={(e) => setCurriculum(prev => ({ ...prev, difficulty: parseFloat(e.target.value) }))} />
                    </div>
                )}

                <div className="bg-slate-800/30 p-3 rounded border border-slate-700 space-y-3">
                    <h3 className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">Graduation Criteria</h3>

                    {st === 0 && (
                        <Slider label="CONSISTENCY" min={100} max={1000} step={10} value={transitions.nursery_consistency_threshold}
                            onChange={(e) => setTransitions(prev => ({ ...prev, nursery_consistency_threshold: parseFloat(e.target.value) }))} />
                    )}

                    {st === 1 && (
                        <>
                            <Slider label="EFFICIENCY SCORE (LOWER IS BETTER)" min={10} max={60} step={1} value={transitions.solo_efficiency_threshold}
                                onChange={(e) => setTransitions(prev => ({ ...prev, solo_efficiency_threshold: parseFloat(e.target.value) }))} />
                            <Slider label="CONSISTENCY" min={1000} max={3000} step={50} value={transitions.solo_consistency_threshold}
                                onChange={(e) => setTransitions(prev => ({ ...prev, solo_consistency_threshold: parseFloat(e.target.value) }))} />
                        </>
                    )}

                    {st === 2 && (
                        <>
                            <div className="p-2 border border-gray-700 rounded bg-gray-900/50 space-y-3">
                                <h4 className="text-[10px] text-neon-cyan mb-2">COMPETENCE STANDARD</h4>
                                <Slider label="GRADUATION DIFFICULTY" min={0.5} max={1.0} step={0.05} value={transitions.duel_graduation_difficulty} valueDisplay={transitions.duel_graduation_difficulty?.toFixed(2)}
                                    onChange={(e) => setTransitions(prev => ({ ...prev, duel_graduation_difficulty: parseFloat(e.target.value) }))} />
                                <Slider label="MIN WIN RATE" min={0.5} max={1.0} step={0.01} value={transitions.duel_graduation_win_rate} valueDisplay={(transitions.duel_graduation_win_rate * 100).toFixed(0) + "%"}
                                    onChange={(e) => setTransitions(prev => ({ ...prev, duel_graduation_win_rate: parseFloat(e.target.value) }))} />
                                <Slider label="CONSISTENCY CHECKS" min={1} max={10} step={1} value={transitions.duel_graduation_checks}
                                    onChange={(e) => setTransitions(prev => ({ ...prev, duel_graduation_checks: parseFloat(e.target.value) }))} />
                            </div>
                        </>
                    )}

                    {st === 3 && (
                        <>
                            <div className="p-2 border border-gray-700 rounded bg-gray-900/50 space-y-3">
                                <h4 className="text-[10px] text-neon-cyan mb-2">COMPETENCE STANDARD</h4>
                                <Slider label="GRADUATION DIFFICULTY" min={0.5} max={1.0} step={0.05} value={transitions.team_graduation_difficulty} valueDisplay={transitions.team_graduation_difficulty?.toFixed(2)}
                                    onChange={(e) => setTransitions(prev => ({ ...prev, team_graduation_difficulty: parseFloat(e.target.value) }))} />
                                <Slider label="MIN WIN RATE" min={0.5} max={1.0} step={0.01} value={transitions.team_graduation_win_rate} valueDisplay={(transitions.team_graduation_win_rate * 100).toFixed(0) + "%"}
                                    onChange={(e) => setTransitions(prev => ({ ...prev, team_graduation_win_rate: parseFloat(e.target.value) }))} />
                                <Slider label="CONSISTENCY CHECKS" min={1} max={10} step={1} value={transitions.team_graduation_checks}
                                    onChange={(e) => setTransitions(prev => ({ ...prev, team_graduation_checks: parseFloat(e.target.value) }))} />
                            </div>
                        </>
                    )}
                </div>
            </div>
        )
    }

    return (
        <div className="flex flex-col h-full bg-[#1e1e1e]">

            {/* TABS */}
            <div className="flex border-b border-gray-700 shrink-0">
                <TabButton id="stages" icon={Layers} label="Stages" />
                <TabButton id="rewards" icon={Target} label="Rewards" />
                <TabButton id="training" icon={Cpu} label="Train" />
                <TabButton id="presets" icon={Database} label="Sets" />
            </div>

            {/* CONTENT AREA */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-gray-700">

                {/* STAGES TAB */}
                {activeTab === 'stages' && (
                    <div className="space-y-6">
                        <div className="bg-slate-800/30 p-3 rounded border border-slate-700 space-y-1">
                            <label className="text-xs text-gray-400 font-mono block mb-2">ACTIVE STAGE</label>
                            <select
                                className="w-full bg-gray-900 border border-gray-700 text-white text-sm rounded p-2 font-mono focus:border-neon-cyan outline-none"
                                value={curriculum.stage}
                                onChange={(e) => setCurriculum(prev => ({ ...prev, stage: parseInt(e.target.value) }))}
                            >
                                <option value={0}>0: NURSERY</option>
                                <option value={1}>1: SOLO TRIAL</option>
                                <option value={2}>2: DUEL (1v1)</option>
                                <option value={3}>3: TEAM (2v2)</option>
                                <option value={4}>4: LEAGUE</option>
                            </select>
                        </div>

                        {renderStageConfig()}
                    </div>
                )}

                {/* REWARDS TAB */}
                {activeTab === 'rewards' && (
                    <div className="space-y-6">
                        {/* Navigation / Progress */}
                        <div className="bg-slate-800/30 p-3 rounded border border-slate-700 space-y-3">
                            <h3 className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">Navigation & Progress</h3>

                            <Slider label="PROGRESS (DELTA)" min={0} max={10.0} step={0.1} value={rewards.weights[RW.PROGRESS]}
                                onChange={(e) => setWeight(RW.PROGRESS, parseFloat(e.target.value))} />

                            <Slider label="MAGNET (PROXIMITY)" min={0} max={20.0} step={0.5} value={rewards.weights[RW.MAGNET]}
                                onChange={(e) => setWeight(RW.MAGNET, parseFloat(e.target.value))} />

                            <Slider label="ORIENTATION (GUIDE)" min={0} max={5.0} step={0.1} value={rewards.weights[RW.ORIENTATION]}
                                onChange={(e) => setWeight(RW.ORIENTATION, parseFloat(e.target.value))} />

                            <Slider label="CHECKPOINT (BONUS)" min={0} max={5000} step={100} value={rewards.weights[RW.CHECKPOINT]}
                                onChange={(e) => setWeight(RW.CHECKPOINT, parseFloat(e.target.value))} />

                            <Slider label="RANK (OVERTAKE)" min={0} max={2000} step={50} value={rewards.weights[RW.RANK]}
                                onChange={(e) => setWeight(RW.RANK, parseFloat(e.target.value))} />
                        </div>

                        {/* Penalties */}
                        <div className="bg-slate-800/30 p-3 rounded border border-slate-700 space-y-3">
                            <h3 className="text-[10px] text-red-400/70 font-bold uppercase tracking-widest">Penalties</h3>
                            <Slider label="STEP COST (TIME)" min={0} max={5.0} step={0.1} value={rewards.weights[RW.STEP_PENALTY]}
                                onChange={(e) => setWeight(RW.STEP_PENALTY, parseFloat(e.target.value))} />
                            <Slider label="WRONG WAY" min={0} max={20.0} step={1} value={rewards.weights[RW.WRONG_WAY]}
                                onChange={(e) => setWeight(RW.WRONG_WAY, parseFloat(e.target.value))} />
                        </div>

                        {/* Combat / Terminal */}
                        <div className="bg-slate-800/30 p-3 rounded border border-slate-700 space-y-3">
                            <h3 className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">Combat & Outcomes</h3>
                            <Slider label="WIN" min={0} max={20000} step={500} value={rewards.weights[RW.WIN]}
                                onChange={(e) => setWeight(RW.WIN, parseFloat(e.target.value))} />
                            <Slider label="LOSS" min={0} max={10000} step={500} value={rewards.weights[RW.LOSS]}
                                onChange={(e) => setWeight(RW.LOSS, parseFloat(e.target.value))} />
                            <Slider label="HUMILIATION (BLOCKER)" min={0} max={50} step={0.5} value={rewards.weights[RW.COLLISION_BLOCKER]}
                                onChange={(e) => setWeight(RW.COLLISION_BLOCKER, parseFloat(e.target.value))} />
                            <Slider label="DENIAL (DOORMAN)" min={0} max={5.0} step={0.1} value={rewards.weights[RW.DENIAL]}
                                onChange={(e) => setWeight(RW.DENIAL, parseFloat(e.target.value))} />
                            <Slider label="TEAM SPIRIT" min={0} max={1} step={0.05} value={rewards.team_spirit}
                                onChange={(e) => setRewards(prev => ({ ...prev, team_spirit: parseFloat(e.target.value) }))} />
                        </div>
                    </div>
                )}

                {/* TRAINING TAB */}
                {activeTab === 'training' && (
                    <div className="space-y-6">
                        <div className="bg-slate-800/30 p-3 rounded border border-slate-700 space-y-3">
                            <h3 className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">Hyperparameters</h3>
                            <Slider label="LEARNING RATE" min={1e-5} max={1e-3} step={1e-5} value={hyperparams.lr} valueDisplay={hyperparams.lr.toExponential(1)}
                                onChange={(e) => setHyperparams(prev => ({ ...prev, lr: parseFloat(e.target.value) }))} />
                            <Slider label="ENTROPY COEF" min={0} max={0.1} step={0.001} value={hyperparams.ent_coef} valueDisplay={hyperparams.ent_coef.toFixed(3)}
                                onChange={(e) => setHyperparams(prev => ({ ...prev, ent_coef: parseFloat(e.target.value) }))} />
                        </div>
                    </div>
                )}

                {/* PRESETS TAB */}
                {activeTab === 'presets' && (
                    <div className="space-y-4">
                        <div className="bg-slate-800/30 p-3 rounded border border-slate-700 space-y-3">
                            <h3 className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">Save Config</h3>
                            <div className="flex gap-2">
                                <input
                                    type="text"
                                    placeholder="Preset Name..."
                                    className="flex-1 bg-gray-900 border border-gray-700 text-white text-xs px-2 rounded font-mono focus:border-neon-cyan outline-none"
                                    value={saveName}
                                    onChange={(e) => setSaveName(e.target.value)}
                                />
                                <button onClick={handleSavePreset} disabled={!saveName} className="bg-neon-cyan/20 text-neon-cyan p-2 rounded hover:bg-neon-cyan/40 disabled:opacity-50">
                                    <Save size={16} />
                                </button>
                            </div>
                        </div>

                        <div className="space-y-2">
                            <h3 className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">Load Preset</h3>
                            {presets.length === 0 && <p className="text-gray-600 text-xs italic">No presets found.</p>}
                            {presets.map(p => (
                                <div key={p.name} className="flex items-center justify-between bg-gray-900/50 p-2 rounded border border-gray-800 group hover:border-gray-600">
                                    <span className="text-xs font-mono text-gray-300">{p.name}</span>
                                    <div className="flex gap-2 opacity-50 group-hover:opacity-100 transition-opacity">
                                        <button onClick={() => { setSelectedPreset(p.name); handleLoadPreset(p.name); }} title="Load" className="text-neon-cyan hover:text-white">
                                            <Upload size={14} />
                                        </button>
                                        <button onClick={() => { setSelectedPreset(p.name); handleDeletePreset(p.name); }} title="Delete" className="text-red-500 hover:text-white">
                                            <Trash2 size={14} />
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

            </div>

            {/* Sticky Action Footer */}
            <div className="p-3 border-t border-gray-700 shrink-0 bg-[#252526] flex gap-2">
                <button
                    onClick={handleApply}
                    className="flex-1 bg-neon-cyan/80 text-black text-xs font-bold uppercase tracking-wide py-2 rounded flex items-center justify-center gap-2 hover:bg-neon-cyan transition-colors shadow-lg shadow-cyan-500/20"
                >
                    <Play size={14} strokeWidth={2.5} />
                    APPLY LIVE
                </button>
                <button
                    onClick={handleLaunch}
                    className="flex-1 bg-orange-600/80 text-white text-xs font-bold uppercase tracking-wide py-2 rounded flex items-center justify-center gap-2 hover:bg-orange-500 transition-colors shadow-lg shadow-orange-500/20"
                >
                    <Rocket size={14} strokeWidth={2.5} />
                    LAUNCH CUSTOM
                </button>
            </div>
        </div>
    )
}
