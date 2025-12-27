/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Saga dark theme colors
        saga: {
          bg: '#09090b',      // zinc-950
          surface: '#18181b', // zinc-900
          border: '#27272a',  // zinc-800
          text: '#fafafa',    // zinc-50
          muted: '#a1a1aa',   // zinc-400
          accent: '#6366f1',  // indigo-500
        }
      }
    },
  },
  plugins: [],
}
