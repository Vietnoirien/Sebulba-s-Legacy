import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  assetsInclude: ['**/*.glb', '**/*.obj', '**/*.mtl'],
  server: {
    watch: {
      usePolling: true,
      ignored: ['**/node_modules/**', '**/.git/**', '**/dist/**'],
    },
  },
})
