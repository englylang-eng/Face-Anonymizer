/** IMPORTANT: Tailwind build config for the Flask web UI */
module.exports = {
  darkMode: 'class',
  content: ['./web/index.html'],
  theme: {
    extend: {
      colors: {
        primary: '#2D8CFF',
        'background-main': '#FFFFFF',
        'card-border': '#e5e7eb',
      },
    },
  },
  plugins: [],
};
