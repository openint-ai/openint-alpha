import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const certDir = path.resolve(__dirname, '../certs');
const hasCerts = fs.existsSync(path.join(certDir, 'key.pem')) && fs.existsSync(path.join(certDir, 'cert.pem'));
// Disable HTTPS by default - enable only if explicitly needed
const https = process.env.VITE_HTTPS === 'true' && hasCerts
  ? {
      key: fs.readFileSync(path.join(certDir, 'key.pem')),
      cert: fs.readFileSync(path.join(certDir, 'cert.pem')),
    }
  : false;

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    port: parseInt(process.env.FRONTEND_PORT || '3000', 10),
    strictPort: false, // Allow port to be overridden
    host: '0.0.0.0', // Allow external connections
    https,
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || (hasCerts ? 'https://127.0.0.1:3001' : 'http://127.0.0.1:3001'),
        changeOrigin: true,
        secure: false,
        ws: true,
        // 5 min – preview-multi loads and runs 3 models; first request can be very slow
        timeout: 300000,
      },
    },
  },
});
