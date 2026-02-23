/**
 * CardioGuard AI - Global Configuration
 * Backend deployed on Render.com
 */

const CARDIOGUARD_CONFIG = {
    API_BASE: "https://cardioguard-api.hb0g.onrender.com"
};

// Auto-detect: if running on Vercel/production, use Render; else use localhost
window.API_BASE = CARDIOGUARD_CONFIG.API_BASE || "http://localhost:8000";

console.log("[CardioGuard] API Base:", window.API_BASE);
