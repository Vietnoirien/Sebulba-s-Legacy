import React from 'react'
import { type LucideIcon } from 'lucide-react'

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: 'primary' | 'danger' | 'warning' | 'neutral'
    icon?: LucideIcon
    loading?: boolean
}

export const Button: React.FC<ButtonProps> = ({
    children,
    variant = 'primary',
    icon: Icon,
    loading,
    className = '',
    ...props
}) => {
    const baseStyles = "flex items-center justify-center gap-2 px-3 py-2 rounded text-sm font-bold transition-all transform hover:scale-105 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"

    const variants = {
        primary: "bg-neon-cyan/90 hover:bg-neon-cyan text-black shadow-[0_0_10px_rgba(6,182,212,0.3)]",
        danger: "bg-red-600 hover:bg-red-500 text-white shadow-[0_0_10px_rgba(220,38,38,0.3)]",
        warning: "bg-yellow-600 hover:bg-yellow-500 text-white",
        neutral: "bg-gray-700 hover:bg-gray-600 text-gray-200"
    }

    return (
        <button
            className={`${baseStyles} ${variants[variant]} ${className}`}
            disabled={loading || props.disabled}
            {...props}
        >
            {loading ? (
                <span className="animate-spin">⌛</span>
            ) : Icon && (
                <Icon size={16} />
            )}
            {children}
        </button>
    )
}
