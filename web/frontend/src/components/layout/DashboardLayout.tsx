import React, { type ReactNode } from 'react'
import { Header } from './Header'

interface DashboardLayoutProps {
    children: ReactNode
}

export const DashboardLayout: React.FC<DashboardLayoutProps> = ({ children }) => {
    return (
        <div className="min-h-screen bg-dark-bg flex flex-col overflow-hidden">
            <Header />
            <div className="flex h-[calc(100vh-64px)]">
                {children}
            </div>
        </div>
    )
}
