import { defineConfig, loadEnv } from 'vite';
import vue from '@vitejs/plugin-vue';

declare const process: { cwd: () => string };

export default ({ mode }: { mode: string }) => {
  const env = loadEnv(mode, process.cwd());
  const port = Number.parseInt(env.VITE_PORT || "5173", 10);

  return defineConfig({
    plugins: [vue()],
    server: {
      host: '0.0.0.0',
      port,
      open: env.VITE_OPEN_BROWSER === 'false' ? false : true,
    },
  });
};
