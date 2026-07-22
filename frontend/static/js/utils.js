/* =====================================================
   SHARED JS UTILITIES
   ===================================================== */

const API_BASE = "";  // same origin

// ─── Toast notifications ──────────────────────────
function showToast(message, type = "info", duration = 4000) {
  const container = document.getElementById("toast-container") ||
    (() => {
      const el = document.createElement("div");
      el.id = "toast-container";
      document.body.appendChild(el);
      return el;
    })();

  const icons = { success: "✅", error: "❌", info: "ℹ️", warning: "⚠️" };
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${icons[type] || "📢"}</span><span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = "slideIn 0.3s ease reverse";
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ─── API helper ───────────────────────────────────
async function apiFetch(path, options = {}) {
  try {
    const res = await fetch(API_BASE + path, options);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "API error");
    return data;
  } catch (err) {
    showToast(err.message, "error");
    throw err;
  }
}

// ─── Set active nav link ──────────────────────────
function setActiveNav() {
  const links = document.querySelectorAll(".nav-links a");
  links.forEach(link => {
    if (link.href === window.location.href ||
        window.location.pathname.startsWith(link.getAttribute("href"))) {
      link.classList.add("active");
    }
  });
}

// ─── Health check + KB status badge ──────────────
async function updateKBStatus() {
  try {
    const data = await apiFetch("/api/health");
    const statusEls = document.querySelectorAll(".kb-dot");
    const statusTexts = document.querySelectorAll(".kb-status-text");

    statusEls.forEach(el => {
      el.className = "kb-dot " + (data.vectorstore_ready ? "ready" : "empty");
    });
    statusTexts.forEach(el => {
      el.textContent = data.vectorstore_ready
        ? "Knowledge base active"
        : "No knowledge base";
    });
  } catch (_) {}
}

// ─── Format bytes ─────────────────────────────────
function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(2) + " MB";
}

// ─── Simple markdown renderer ──────────────────────
function renderMarkdown(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, "<code>$1</code>")
    .replace(/^### (.+)$/gm, "<h4>$1</h4>")
    .replace(/^## (.+)$/gm,  "<h3>$1</h3>")
    .replace(/^# (.+)$/gm,   "<h2>$1</h2>")
    .replace(/^\* (.+)$/gm,  "<li>$1</li>")
    .replace(/^- (.+)$/gm,   "<li>$1</li>")
    .replace(/(<li>.*<\/li>)/gs, "<ul>$1</ul>")
    .replace(/\n{2,}/g, "</p><p>")
    .replace(/^(?!<[hul])(.+)/gm, (match) =>
      match.startsWith("<") ? match : `<p>${match}</p>`
    )
    .replace(/<p><\/p>/g, "");
}

// ─── Format timestamp ─────────────────────────────
function formatTime(date = new Date()) {
  return date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
}

// Auto-init nav on load
document.addEventListener("DOMContentLoaded", () => {
  setActiveNav();
  updateKBStatus();
});
