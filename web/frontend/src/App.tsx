import { GameStateProvider, useGameActions } from './context/GameStateContext'
import { DashboardLayout } from './components/layout/DashboardLayout'
import { ControlPanel } from './components/panels/ControlPanel'
import { ConfigPanel } from './components/panels/ConfigPanel'
import { RaceCanvas } from './components/race/RaceCanvas'
import { RaceScene3D } from './components/race/RaceScene3D'
import { LeaguePanel } from './components/panels/LeaguePanel'

const AppContent: React.FC = () => {
  const { viewMode } = useGameActions()

  return (
    <DashboardLayout>
      {/* Left Sidebar: Operations */}
      <aside className="w-80 bg-panel-bg border-r border-gray-700 p-4 flex flex-col gap-6 overflow-y-auto scrollbar-default">
        <ControlPanel />
      </aside>

      {/* Center: Game & Console */}
      <section className="flex-1 bg-[#121212] relative flex flex-col h-full overflow-hidden">
        {/* 1. Game View - Top */}
        <div className="w-full bg-black shadow-lg relative z-10 shrink-0">
          {viewMode === '3d' ? <RaceScene3D /> : <RaceCanvas />}
        </div>

        {/* 2. Console Log - Remaining Space */}
        <div className="flex-1 p-4 w-full max-w-[1800px] mx-auto flex flex-col min-h-0">
          <div className="flex-1 flex flex-col overflow-hidden rounded-lg shadow-md border border-gray-800 bg-black/40 backdrop-blur-sm">
            <LeaguePanel />
          </div>
        </div>
      </section>

      {/* Right Sidebar: Data & Config */}
      <aside className="w-96 bg-panel-bg border-l border-gray-700 flex flex-col overflow-hidden">
        <div className="flex-1 min-h-0 relative">
          <ConfigPanel />
        </div>
      </aside>
    </DashboardLayout>
  )
}

function App() {
  return (
    <GameStateProvider>
      <AppContent />
    </GameStateProvider>
  )
}

export default App
