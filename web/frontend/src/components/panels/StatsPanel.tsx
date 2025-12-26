import React from 'react'
import { LineChart, Line, YAxis, ResponsiveContainer } from 'recharts'
import { useGameState } from '../../context/GameStateContext'

interface StatCardProps {
    title: string
    dataKey: string
    color: string
    data: any[]
}

const StatCard: React.FC<StatCardProps> = ({ title, dataKey, color, data }) => (
    <div className="h-32 bg-slate-800/50 rounded-lg p-3 border border-slate-700 relative overflow-hidden group">
        <div className="absolute top-2 left-3 z-10">
            <h3 className="text-[10px] text-gray-400 font-mono uppercase tracking-wider">{title}</h3>
            <p className="text-lg font-bold text-white font-mono">
                {data.length > 0 ? data[data.length - 1][dataKey]?.toFixed(3) : 0}
            </p>
        </div>
        <div className="absolute inset-0 opacity-50 group-hover:opacity-100 transition-opacity">
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data}>
                    <Line
                        type="monotone"
                        dataKey={dataKey}
                        stroke={color}
                        strokeWidth={2}
                        dot={false}
                        isAnimationActive={false}
                    />
                    <YAxis hide domain={['auto', 'auto']} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    </div>
)

export const StatsPanel: React.FC = () => {
    const { history, stats } = useGameState()

    return (
        <div className="space-y-3">



            <StatCard
                title="Win Rate"
                dataKey="win_rate"
                color="#d946ef"
                data={history}
            />
        </div>
    )
}
