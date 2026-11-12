import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  base: '/ui/',
  plugins: [react(), tailwindcss()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules')) {
            if (id.includes('/react') || id.includes('/react-dom')) {
              return 'react-vendor';
            }
            if (id.includes('/ai/') || id.includes('/@ai-sdk/') || id.includes('/zod/')) {
              return 'ai-sdk';
            }
            if (id.includes('lucide-react')) {
              return 'icons';
            }
          }
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:4000',
      '/v1': 'http://127.0.0.1:4000',
      '/v1beta': 'http://127.0.0.1:4000',
      '/health': 'http://127.0.0.1:4000',
      '/metrics': 'http://127.0.0.1:4000',
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test-setup.ts'],
  },
});
