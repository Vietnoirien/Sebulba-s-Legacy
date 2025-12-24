import React, { useState } from 'react'
import { Slider } from '../common/Slider'
import { useGameState } from '../../context/GameStateContext'

export const ConfigPanel: React.FC = () => {
    const { sendMessage } = useGameState()
    const [lr, setLr] = useState(0.0003)
    const [ent, setEnt] = useState(0.01)

    const handleUpdate = (key: string, value: number) => {
        // Send via WebSocket
        sendMessage(JSON.stringify({
            type: 'config',
            payload: { [key]: value }
        }))
    }

    return (
        <div className="space-y-2">
            <h2 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-2 font-mono mt-6">Hyperparameters</h2>
            <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700 space-y-4">
                <Slider
                    label="LEARNING RATE"
                    min={0.0001}
                    max={0.001}
                    step={0.0001}
                    value={lr}
                    valueDisplay={lr.toFixed(4)}
                    onChange={(e) => {
                        const val = parseFloat(e.target.value)
                        setLr(val)
                        // TODO: Debounce this
                        handleUpdate('learning_rate', val)
                    }}
                />
                <Slider
                    label="ENTROPY COEF"
                    min={0.0}
                    max={0.1}
                    step={0.005}
                    value={ent}
                    valueDisplay={ent.toFixed(3)}
                    onChange={(e) => {
                        const val = parseFloat(e.target.value)
                        setEnt(val)
                        handleUpdate('ent_coef', val)
                    }}
                />
            </div>
        </div>
    )
}
