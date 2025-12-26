import React, { createContext, useContext, type ReactNode } from 'react'
import { useTelemetry, type Telemetry, type ConnectionStatus } from '../hooks/useTelemetry'
import { ReadyState } from 'react-use-websocket'

interface GameStateContextType {
    telemetry: Telemetry | null
    readyState: ReadyState
    connectionStatus: ConnectionStatus
    history: any[]
    stats: any
    sendMessage: (message: string) => void

    // UI State
    selectedModel: string
    setSelectedModel: (model: string) => void
}

const GameStateContext = createContext<GameStateContextType | undefined>(undefined)

export const GameStateProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const telemetryData = useTelemetry()
    const [selectedModel, setSelectedModel] = React.useState<string>("scratch")

    const value = {
        ...telemetryData,
        selectedModel,
        setSelectedModel
    }

    return (
        <GameStateContext.Provider value={value}>
            {children}
        </GameStateContext.Provider>
    )
}

export const useGameState = () => {
    const context = useContext(GameStateContext)
    if (context === undefined) {
        throw new Error('useGameState must be used within a GameStateProvider')
    }
    return context
}
