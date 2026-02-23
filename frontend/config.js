/**
 * CardioGuard AI — Global Configuration
 * 
 * After deploying your backend to Render.com,
 * replace RENDER_URL below with your actual Render URL.
 * Example: "https://cardioguard-api.onrender.com"
 *
 * Leave as empty string "" to use localhost:8000 (local dev).
 */

const CARDIOGUARD_CONFIG = {
    // ← PASTE YOUR RENDER URL HERE AFTER DEPLOYING
    API_BASE: ""
};

// Auto-detect: if running on Vercel/production, use env; else use localhost
window.API_BASE = CARDIOGUARD_CONFIG.API_BASE || "http://localhost:8000";

console.log("[CardioGuard] API Base:", window.API_BASE);
