/**
 * frontend/static/js/chat.js
 *
 * Handles all chat UI interactions:
 *  - Sending messages to POST /api/chat
 *  - Rendering user and agent message bubbles
 *  - Displaying tool call activity badges
 *  - Session (thread) management
 *  - Sidebar toggle and suggestion shortcuts
 */

// ── State ──────────────────────────────────────────────────────────────────────
let threadId  = generateId();
let isLoading = false;

// ── DOM references ─────────────────────────────────────────────────────────────
const messagesEl    = () => document.getElementById("messages");
const inputEl       = () => document.getElementById("msg-input");
const sendBtnEl     = () => document.getElementById("send-btn");
const agentStatusEl = () => document.getElementById("agent-status");
const statusTextEl  = () => document.getElementById("status-text");
const threadBadgeEl = () => document.getElementById("thread-badge");

// ── Initialisation ─────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  updateThreadBadge();
  inputEl().focus();

  // Sync max-results slider label if it exists
  const slider = document.getElementById("max-results");
  if (slider) {
    slider.addEventListener("input", () => {
      document.getElementById("max-results-val").textContent = slider.value;
    });
  }
});

// ── Core: send message ─────────────────────────────────────────────────────────

async function sendMessage() {
  const input = inputEl();
  const text  = input.value.trim();
  if (!text || isLoading) return;

  setLoading(true);
  input.value = "";
  autoResize(input);
  hideWelcome();

  appendMessage("user", text);
  showStatus("Routing to the right agent…");

  try {
    const res = await fetch("/api/chat", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ message: text, thread_id: threadId }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${res.status}`);
    }

    const data = await res.json();
    hideStatus();
    appendMessage("agent", data.reply, data.tool_calls_used || []);

    // Keep thread_id consistent in case the server generated one
    if (data.thread_id) {
      threadId = data.thread_id;
      updateThreadBadge();
    }

  } catch (err) {
    hideStatus();
    appendMessage("agent", `**Error:** ${err.message}\n\nPlease check your \`.env\` configuration and try again.`);
  } finally {
    setLoading(false);
    inputEl().focus();
  }
}

// ── Message rendering ──────────────────────────────────────────────────────────

/**
 * Append a message bubble to the chat.
 * @param {"user"|"agent"} role
 * @param {string}         content    - Raw text (may contain markdown).
 * @param {string[]}       toolCalls  - Tool names used by this agent turn.
 */
function appendMessage(role, content, toolCalls = []) {
  const row = document.createElement("div");
  row.className = `msg-row ${role}`;

  // Role label
  const label = document.createElement("div");
  label.className = `msg-label ${role === "user" ? "user-label" : "agent-label"}`;
  label.textContent = role === "user" ? "You" : "IATA";
  row.appendChild(label);

  // Bubble
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role === "user" ? "user-bubble" : "agent-bubble"}`;
  bubble.innerHTML = role === "agent" ? renderMarkdown(content) : escapeHtml(content);
  row.appendChild(bubble);

  // Tool call badges (agent only)
  if (toolCalls.length > 0) {
    const badges = document.createElement("div");
    badges.className = "tool-badges";
    toolCalls.forEach(name => {
      const badge = document.createElement("span");
      badge.className = "tool-badge";
      badge.textContent = `⚙ ${name}`;
      badges.appendChild(badge);
    });
    row.appendChild(badges);
  }

  messagesEl().appendChild(row);
  scrollToBottom();
}

// ── Markdown renderer ──────────────────────────────────────────────────────────

/**
 * Lightweight markdown-to-HTML converter.
 * Handles: ## headings, **bold**, *italic*, `code`, bullet lists, numbered lists.
 * Intentionally minimal — no external dependencies.
 */
function renderMarkdown(text) {
  // Escape HTML first to prevent injection, then apply markdown
  let html = escapeHtml(text);

  // Headings (## and ###)
  html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  html = html.replace(/^## (.+)$/gm,  "<h2>$1</h2>");

  // Bold and italic
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g,     "<em>$1</em>");

  // Inline code
  html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");

  // Bullet lists (- item or * item)
  html = html.replace(/^[-*] (.+)$/gm, "<li>$1</li>");
  html = html.replace(/(<li>.*<\/li>(\n|$))+/g, match => `<ul>${match}</ul>`);

  // Numbered lists (1. item)
  html = html.replace(/^\d+\. (.+)$/gm, "<li>$1</li>");

  // Double newline → paragraph break
  html = html.replace(/\n\n/g, "<br/><br/>");

  // Single newline → line break (preserve formatting)
  html = html.replace(/\n/g, "<br/>");

  return html;
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ── Status indicator ───────────────────────────────────────────────────────────

const TOOL_LABELS = {
  search_papers:            "🔍 Searching arXiv and Semantic Scholar…",
  summarize_paper:          "📝 Summarizing paper…",
  summarize_multiple_papers:"📝 Summarizing multiple papers…",
};

function showStatus(text) {
  statusTextEl().textContent = text;
  agentStatusEl().style.display = "flex";
}

function hideStatus() {
  agentStatusEl().style.display = "none";
}

// ── UI helpers ─────────────────────────────────────────────────────────────────

function setLoading(state) {
  isLoading = state;
  sendBtnEl().disabled = state;
}

function hideWelcome() {
  const welcome = document.getElementById("welcome");
  if (welcome) welcome.remove();
}

function scrollToBottom() {
  const el = messagesEl();
  el.scrollTop = el.scrollHeight;
}

function updateThreadBadge() {
  const badge = threadBadgeEl();
  if (badge) badge.textContent = threadId.slice(0, 8);
}

// ── Textarea auto-resize ───────────────────────────────────────────────────────

function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 150) + "px";
}

// ── Keyboard handling ──────────────────────────────────────────────────────────

function handleKey(event) {
  // Enter sends; Shift+Enter inserts a newline
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
}

// ── Suggestions ────────────────────────────────────────────────────────────────

function useSuggestion(el) {
  const input = inputEl();
  input.value = el.textContent.trim();
  input.focus();
  autoResize(input);
}

// ── Sidebar ────────────────────────────────────────────────────────────────────

function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("hidden");
}

// ── New chat ───────────────────────────────────────────────────────────────────

async function newChat() {
  // Ask the server to clear Supabase history for this thread
  await fetch(`/api/chat/${threadId}`, { method: "DELETE" }).catch(() => {});

  // Reset local state
  threadId = generateId();
  updateThreadBadge();

  // Rebuild the messages area with the welcome screen
  messagesEl().innerHTML = `
    <div class="welcome" id="welcome">
      <div class="welcome-hex">
        <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M32 4L56 18V46L32 60L8 46V18L32 4Z" stroke="currentColor" stroke-width="1.5"/>
          <path d="M32 16L44 23V37L32 44L20 37V23L32 16Z" fill="currentColor" opacity="0.2"/>
          <path d="M32 24L38 27.5V34.5L32 38L26 34.5V27.5L32 24Z" fill="currentColor" opacity="0.5"/>
        </svg>
      </div>
      <h1 class="welcome-title">IATA Research Agent</h1>
      <p class="welcome-sub">
        Find and understand scientific papers with a multi-agent AI system.<br/>
        Powered by LangGraph, Azure OpenAI, and Supabase.
      </p>
      <div class="welcome-chips">
        <button class="chip" onclick="useSuggestion(this)">Find papers on efficient transformers</button>
        <button class="chip" onclick="useSuggestion(this)">What are the trends in RLHF?</button>
        <button class="chip" onclick="useSuggestion(this)">Summarize recent work on state space models</button>
      </div>
    </div>
  `;
}

// ── Utilities ──────────────────────────────────────────────────────────────────

function generateId() {
  return ([1e7]+-1e3+-4e3+-8e3+-1e11)
    .replace(/[018]/g, c =>
      (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
    );
}
