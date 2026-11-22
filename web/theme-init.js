// The theme must sit on the page before the first paint, and a module is
// deferred: a stored dark page would flash its light paper first. This reads
// the one stored key — the desktop's own `~/.v2m_theme` name and its two words,
// dark or light — and the system's own preference, which is exactly the rule
// `themeFor` (web/prefs.js) decides again below.
//
// It is a CLASSIC script, loaded from the head, so the browser blocks the first
// paint until it has run — and it lives in its own file so the page's
// Content-Security-Policy can be `script-src 'self' blob:` with no inline
// exception (an inline copy needs `'unsafe-inline'` or a hash that every edit
// silently invalidates).
try {
  const stored = localStorage.getItem("v2m_theme");
  const dark =
    stored === "dark" ||
    (stored === null && matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
} catch {
  // storage can be blocked; the stylesheet's light default still paints
}
