import React, { createContext, useContext, type ReactNode, useMemo } from 'react'
import { useTelemetry, type Telemetry, type ConnectionStatus } from '../hooks/useTelemetry'
import { ReadyState } from 'react-use-websocket'

// 1. Telemetry / Volatile State
interface GameStateContextType {
    telemetry: Telemetry | null
    readyState: ReadyState
    connectionStatus: ConnectionStatus
    history: any[]
    stats: any
}

// 2. Actions / Stable State
interface GameActionContextType {
    sendMessage: (message: string) => void
    selectedModel: string
    setSelectedModel: (model: string) => void
}

const GameStateContext = createContext<GameStateContextType | undefined>(undefined)
const GameActionContext = createContext<GameActionContextType | undefined>(undefined)

export const GameStateProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const telemetryData = useTelemetry()
    const [selectedModel, setSelectedModel] = React.useState<string>("scratch")

    // Stable Action Value
    const actionValue = useMemo(() => ({
        sendMessage: telemetryData.sendMessage,
        selectedModel,
        setSelectedModel
    }), [telemetryData.sendMessage, selectedModel])

    // Volatile State Value
    const stateValue = {
        telemetry: telemetryData.telemetry,
        readyState: telemetryData.readyState,
        connectionStatus: telemetryData.connectionStatus,
        history: telemetryData.history,
        stats: telemetryData.stats
    }

    return (
        <GameActionContext.Provider value={actionValue}>
            <GameStateContext.Provider value={stateValue}>
                {children}
            </GameStateContext.Provider>
        </GameActionContext.Provider>
    )
}

export const useGameState = () => {
    const context = useContext(GameStateContext)
    if (context === undefined) {
        throw new Error('useGameState must be used within a GameStateProvider')
    }
    return context
}

export const useGameActions = () => {
    const context = useContext(GameActionContext)
    if (context === undefined) {
        throw new Error('useGameActions must be used within a GameStateProvider')
    }
    return context
}
