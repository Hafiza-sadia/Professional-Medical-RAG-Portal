/* =====================================================
   MEDIQ AI PORTAL — Shared JS Utilities
   ===================================================== */

const API_BASE = "";

// Toast notifications
function showToast(message, type = "info", duration = 4000) {
  let container = document.getElementById("toast-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "toast-container";
    document.body.appendChild(container);
  }

  const icons = {
    success: "✓",
    error: "✕",
    info: "ℹ",
    warning: "⚠"
  };

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span class="toast-icon">${icons[type] || "ℹ"}</span><span class="toast-text">${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = "slideIn 0.3s ease reverse";
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// Resilient API helper with timeout
async function apiFetch(path, options = {}) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), options.timeoutMs || 35000);

  try {
    const res = await fetch(API_BASE + path, {
      ...options,
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Server error (${res.status})`);
    return data;
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === "AbortError") {
      throw new Error("Request timed out (35s). Groq API is taking longer than expected, please try again.");
    }
    showToast(err.message, "error");
    throw err;
  }
}

// Set active nav link
function setActiveNav() {
  const links = document.querySelectorAll(".nav-links a");
  links.forEach(link => {
    if (link.href === window.location.href ||
        (link.getAttribute("href") !== "/" && window.location.pathname.startsWith(link.getAttribute("href")))) {
      link.classList.add("active");
    }
  });
}

// Health check + KB status badge
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

// Format bytes
function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(2) + " MB";
}

// Robust, safe markdown renderer
// Robust, executive-grade clinical markdown renderer
function renderMarkdown(text) {
  if (!text) return "";

  // 0. Unescape literal newlines and clean input
  let str = text.replace(/\\n/g, "\n");

  // 1. Stash fenced code blocks to prevent unintended replacements
  const codeBlocks = [];
  str = str.replace(/```([\s\S]*?)```/g, (match, code) => {
    const placeholder = `__CODE_BLOCK_${codeBlocks.length}__`;
    codeBlocks.push(`<pre class="code-block"><code>${code.trim()}</code></pre>`);
    return placeholder;
  });

  // 2. Horizontal rules
  str = str.replace(/^[\t ]*(?:[-*_]\s*){3,}$/gm, '<hr class="md-hr"/>');

  // 3. Headers
  str = str.replace(/^#### (.*$)/gim, '<h5 class="md-h5">$1</h5>');
  str = str.replace(/^### (.*$)/gim, '<h4 class="md-h4">$1</h4>');
  str = str.replace(/^## (.*$)/gim, '<h3 class="md-h3">$1</h3>');
  str = str.replace(/^# (.*$)/gim, '<h2 class="md-h2">$1</h2>');

  // 4. Blockquotes
  str = str.replace(/^> (.*$)/gim, '<blockquote class="md-quote">$1</blockquote>');

  // 5. Markdown Tables
  str = str.replace(/((?:^|\n)\|[^\n]+\|\r?\n\|[\s-:|]+\|(?:\r?\n\|[^\n]+\|)+)/g, (match) => {
    const lines = match.trim().split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    if (lines.length < 2) return match;

    const parseRow = (line) => line.split("|").slice(1, -1).map(c => c.trim());
    const headerCells = parseRow(lines[0]);
    const thead = `<thead><tr>${headerCells.map(c => `<th>${c}</th>`).join("")}</tr></thead>`;

    const bodyRows = lines.slice(2).map(line => {
      const cells = parseRow(line);
      return `<tr>${cells.map(c => `<td>${c}</td>`).join("")}</tr>`;
    }).join("");
    const tbody = `<tbody>${bodyRows}</tbody>`;

    return `\n<div class="table-responsive"><table class="md-table">${thead}${tbody}</table></div>\n`;
  });

  // 6. Numbered lists (1. item)
  str = str.replace(/^\s*(\d+)\.\s+(.*$)/gim, '<li class="md-ol-item" value="$1">$2</li>');
  str = str.replace(/((?:<li class="md-ol-item"[^>]*>[\s\S]*?<\/li>\s*)+)/g, '<ol class="md-list md-ol">$1</ol>');

  // 7. Bullet lists (- or * item)
  str = str.replace(/^\s*[-*]\s+(.*$)/gim, '<li class="md-ul-item">$1</li>');
  str = str.replace(/((?:<li class="md-ul-item">[\s\S]*?<\/li>\s*)+)/g, '<ul class="md-list md-ul">$1</ul>');

  // 8. Inline Code
  str = str.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

  // 9. Bold & Italic
  str = str.replace(/\*\*([^\*]+)\*\*/g, '<strong>$1</strong>');
  str = str.replace(/(?<!\*)\*([^\*\n]+)\*(?!\*)/g, '<em>$1</em>');

  // 10. Inline Citation Tags highlight [Source X] or [Source X: ..., Page Y]
  str = str.replace(/\[(Source\s*\d+[^\n\]]*)\]/gi, '<span class="citation-tag">📄 $1</span>');

  // 11. Paragraph breaks
  const parts = str.split(/\n{2,}/);
  let html = parts.map(p => {
    p = p.trim();
    if (!p) return "";
    if (
      p.startsWith("<h") ||
      p.startsWith("<ul") ||
      p.startsWith("<ol") ||
      p.startsWith("<pre") ||
      p.startsWith("<div") ||
      p.startsWith("<table") ||
      p.startsWith("<hr") ||
      p.startsWith("<blockquote")
    ) {
      return p;
    }
    return `<p>${p.replace(/\n/g, "<br/>")}</p>`;
  }).join("");

  // 12. Restore code blocks
  codeBlocks.forEach((codeHtml, idx) => {
    html = html.replace(`__CODE_BLOCK_${idx}__`, codeHtml);
  });

  return html;
}

// function formatTime(date = new Date()) {
  return date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
}

// Auto-init nav on load
document.addEventListener("DOMContentLoaded", () => {
  setActiveNav();
  updateKBStatus();
});
