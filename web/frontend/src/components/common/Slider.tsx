import React from 'react'

interface SliderProps extends React.InputHTMLAttributes<HTMLInputElement> {
    label: string
    valueDisplay?: string | number
}

export const Slider: React.FC<SliderProps> = ({ label, valueDisplay, className = '', ...props }) => {
    return (
        <div className={`space-y-1 ${className}`}>
            <div className="flex justify-between text-xs text-gray-400 font-mono">
                <label>{label}</label>
                <span className="text-neon-cyan">{valueDisplay ?? props.value}</span>
            </div>
            <input
                type="range"
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-neon-cyan hover:accent-cyan-300"
                {...props}
            />
        </div>
    )
}
