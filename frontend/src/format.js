// Small formatting helpers shared across the console.
export const format = {
  // Seconds -> H:MM:SS (or MM:SS under an hour), matching the backend labels.
  time(seconds) {
    const s = Math.max(0, Math.round(seconds));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    const pad = (n) => String(n).padStart(2, "0");
    return h ? `${h}:${pad(m)}:${pad(sec)}` : `${pad(m)}:${pad(sec)}`;
  },
};
