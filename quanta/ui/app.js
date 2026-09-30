"use strict";

/* Quanta UI. Every number and label on this page comes from the running gateway. */

const $ = (id) => document.getElementById(id);
const state = {
  api: "http://127.0.0.1:8787",
  key: "",
  providers: [],
  config: null,
  session: null,
  history: [],          // OpenAI-format messages resent on every turn
  view: [],             // rendered items (includes errors, which are never resent)
  modelCache: {},
  errors: [],
  busy: false,
  abort: null,
};

const SAMPLE_TOOLS = [{
  type: "function",
  function: {
    name: "get_weather",
    description: "Get the current weather for a city",
    parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
  },
}];

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (v !== false && v != null) node.setAttribute(k, v === true ? "" : v);
  }
  for (const child of children) if (child != null) node.append(child);
  return node;
}

function store(name, value) {
  try { value === null ? localStorage.removeItem(name) : localStorage.setItem(name, value); } catch (_) { /* private mode */ }
}
function load(name) { try { return localStorage.getItem(name) || ""; } catch (_) { return ""; } }

/* ---------- API ---------- */
class ApiFailure extends Error {
  constructor(status, body) {
    const e = (body && body.error) || {};
    super(e.message || `HTTP ${status}`);
    this.status = status; this.code = e.code || "http_" + status;
  }
}

async function api(path, opts = {}) {
  const headers = { Authorization: `Bearer ${state.key}`, ...(opts.body ? { "Content-Type": "application/json" } : {}) };
  let res;
  try {
    res = await fetch(state.api + path, { method: opts.method || "GET", headers, body: opts.body, signal: opts.signal });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new ApiFailure(0, { error: { message: `Cannot reach the API at ${state.api}. Start it with \`quanta serve\`; if it is running, this page's origin (${location.origin}) must be in server.corsOrigins.`, code: "unreachable" } });
  }
  if (opts.raw) return res;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiFailure(res.status, data);
  return data;
}

function noteError(where, error) {
  state.errors.unshift(`${new Date().toLocaleTimeString()} · ${where}: ${error.message}${error.code ? ` (${error.code})` : ""}`);
  state.errors.length = Math.min(state.errors.length, 8);
  const list = $("error-list");
  list.replaceChildren(...state.errors.map((t) => el("li", { text: t })));
  $("errors-card").hidden = false;
}

/* ---------- connection ---------- */
function setConn(stateName, text) {
  $("conn-pill").dataset.state = stateName;
  $("conn-text").textContent = text;
}

async function connect() {
  if (!state.key) { setConn("idle", "Add API key"); renderProviders(); renderSnippets(); return false; }
  try {
    const [providers, config, session] = await Promise.all([api("/v1/providers"), api("/v1/config"), api("/v1/session")]);
    state.providers = providers.data; state.config = config; state.session = session; state.lastError = null;
    setConn("ok", `Connected · ${state.providers.filter((p) => p.enabled).length} CLIs`);
    $("key-status").textContent = "Key accepted by the gateway.";
    renderAll();
    refreshActivity();
    return true;
  } catch (error) {
    state.providers = []; state.config = null; state.lastError = error;
    setConn("bad", error.status === 401 ? "Key rejected" : "Gateway unreachable");
    $("key-status").textContent = error.message;
    noteError("connect", error);
    renderAll();
    return false;
  }
}

/* ---------- rendering ---------- */
const CAP_LABELS = [
  ["streaming", "Streaming"],
  ["tool_calling", "Tool calling"],
  ["subagents", "Subagents"],
  ["session_persistence", "Server-side sessions"],
  ["model_switching", "Model switching"],
  ["provider_switching", "Provider switching"],
];

function renderAll() {
  renderProviders(); renderMarquee(); renderSelectors(); renderSnippets(); renderServing();
}

function renderMarquee() {
  const items = state.providers.length ? state.providers : [{ id: "quanta", enabled: false }];
  const make = () => items.map((p) => el("span", { "data-on": String(!!p.enabled) }, el("i"), `${p.id} · ${p.enabled ? "ready" : p.available === false ? "no headless mode" : "disabled"}`));
  $("marquee").replaceChildren(...make(), ...make(), ...make(), ...make());
}

function providerBanner() {
  const banner = $("providers-banner");
  if (state.providers.length) { banner.hidden = true; return; }
  banner.hidden = false;
  banner.replaceChildren();
  if (!state.key) {
    banner.append(el("strong", { text: "Enter your API key to load providers. " }), "It is server.apiKey in quanta.config.json.");
    const input = el("input", { type: "password", placeholder: "API key", autocomplete: "off", "aria-label": "API key" });
    const go = el("button", { class: "btn btn-primary", type: "submit" }, "Load providers");
    const form = el("form", { class: "banner-form" }, input, go);
    form.addEventListener("submit", (e) => { e.preventDefault(); saveKey(input.value); });
    banner.append(form);
  } else {
    banner.append(el("strong", { text: `${state.lastError ? state.lastError.message : "No provider data."} ` }));
    const retry = el("button", { class: "btn btn-secondary", type: "button" }, "Retry");
    retry.addEventListener("click", () => connect());
    banner.append(retry);
  }
}

async function saveKey(value) {
  value = (value || "").trim();
  if (!value) { $("key-status").textContent = "Paste a key first."; return; }
  state.key = value; store("quanta.key", value);
  $("key-input").value = ""; $("key-input").placeholder = "Key saved in this browser";
  $("key-status").textContent = "Testing…";
  await connect();
}

function renderProviders() {
  providerBanner();
  const grid = $("provider-grid");
  grid.replaceChildren(...state.providers.map((p) => {
    const tagClass = !p.available ? "tag-blocked" : p.enabled ? "tag-on" : "tag-off";
    const tagText = !p.available ? "No headless mode" : p.enabled ? "Enabled" : "Disabled";
    const cfg = state.config && state.config.providers[p.id];
    const caps = el("ul", { class: "caps" }, ...CAP_LABELS.map(([key, label]) => {
      const value = p.capabilities[key];
      const yes = !!value;
      const note = !yes ? (key === "session_persistence" ? "stateless" : "unavailable") : (typeof value === "string" ? "client-side" : null);
      return el("li", {}, el("span", { class: `tick ${yes ? "yes" : "no"}`, "aria-hidden": "true", text: yes ? "✓" : "" }),
        el("span", { text: label }), note ? el("span", { class: "note", text: note }) : null);
    }));
    const result = el("span", { class: "test-result", "aria-live": "polite" });
    const test = el("button", { class: "btn btn-secondary", type: "button", disabled: !p.enabled }, "Test");
    test.addEventListener("click", () => testProvider(p, test, result));
    const use = el("button", { class: "btn btn-primary", type: "button", disabled: !p.enabled }, "Use in playground");
    use.addEventListener("click", () => { $("pg-provider").value = p.id; onProviderChange(); location.hash = "#playground"; });
    return el("article", { class: "card provider-card" },
      el("div", { class: "provider-head" }, el("h3", { class: "card-title", text: p.id }), el("span", { class: `tag ${tagClass}`, text: tagText })),
      el("p", { class: "body-sm", text: p.note || `${p.adapter} adapter${cfg ? ` · ${Math.round(cfg.timeout_ms / 1000)}s timeout · ${cfg.max_concurrent} concurrent` : ""}` }),
      caps,
      el("div", { class: "card-actions" }, test, use, result));
  }));
}

function renderSelectors() {
  const usable = state.providers.filter((p) => p.enabled);
  for (const id of ["pg-provider", "sw-provider"]) {
    const select = $(id); const previous = select.value;
    select.replaceChildren(...usable.map((p) => el("option", { value: p.id, text: p.id })));
    if (usable.some((p) => p.id === previous)) select.value = previous;
  }
  onProviderChange();
}

function renderServing() {
  const s = state.session;
  $("serving-line").textContent = s ? `Serving ${s.models.length} alias${s.models.length === 1 ? "" : "es"}: ${s.models.map((m) => m.id).join(", ")}` : "Connect with your API key to load the current selection.";
}

function baseUrl() { return state.api.replace(/\/$/, "") + "/v1"; }

function renderSnippets() {
  const key = "<your API key>";
  const first = (state.providers.find((p) => p.enabled) || {}).id || "opencode";
  $("snip-hermes").textContent = `model:\n  provider: custom\n  base_url: ${baseUrl()}\n  api_key: ${key}\n  default: ${first}\n  api_mode: chat_completions`;
  $("snip-curl").textContent = `curl ${baseUrl()}/chat/completions \\\n  -H "Authorization: Bearer ${key}" \\\n  -H "Content-Type: application/json" \\\n  -d '{"model":"${first}:default","messages":[{"role":"user","content":"Hello"}]}'`;
  $("snip-switch").textContent = `# same conversation, different CLI and model\n"model": "opencode:opencode/big-pickle"\n"model": "antigravity:gemini-3.8-flash-low"\n"model": "cline:default"`;
  $("base-label").textContent = baseUrl();
}

/* ---------- provider actions ---------- */
async function testProvider(p, button, out) {
  button.disabled = true; out.textContent = "Testing…";
  const started = performance.now();
  try {
    const alias = state.session && state.session.models.find((m) => m.provider === p.id);
    const res = await api("/v1/chat/completions", { method: "POST", body: JSON.stringify({ model: alias ? alias.id : `${p.id}:default`, messages: [{ role: "user", content: "Reply with exactly: pong" }] }) });
    out.textContent = `OK · ${((performance.now() - started) / 1000).toFixed(1)}s · “${(res.choices[0].message.content || "").trim().slice(0, 30)}”`;
  } catch (error) {
    out.textContent = `Failed · ${error.message.slice(0, 120)}`;
    noteError(`test ${p.id}`, error);
  } finally { button.disabled = false; refreshActivity(); }
}

async function loadModels(provider) {
  if (!provider) return;
  if (!state.modelCache[provider]) {
    try { state.modelCache[provider] = (await api(`/v1/providers/${encodeURIComponent(provider)}/models`)).data.map((m) => m.id); }
    catch (_) { state.modelCache[provider] = []; }
  }
}

function onProviderChange() {
  const p = $("pg-provider").value;
  loadModels(p); loadModels($("sw-provider").value);
  updateTarget();
}

function target() {
  const p = $("pg-provider").value;
  return p ? `${p}:${$("pg-model").value.trim() || "default"}` : "";
}
function updateTarget() { $("pg-target").textContent = target() || "—"; }


/* ---------- searchable model box ---------- */
const MAX_SHOWN = 60;
function searchModels(models, query) {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  return models.filter((m) => terms.every((t) => m.toLowerCase().includes(t)));
}

function attachCombo(input, getModels, onPick) {
  const list = el("ul", { class: "combo-list", role: "listbox", hidden: true });
  input.setAttribute("role", "combobox"); input.setAttribute("aria-expanded", "false"); input.setAttribute("aria-autocomplete", "list");
  input.parentElement.classList.add("combo"); input.parentElement.append(list);
  let items = []; let active = -1;

  function close() { list.hidden = true; input.setAttribute("aria-expanded", "false"); active = -1; }
  function choose(value) { input.value = value; close(); onPick(value); }
  function highlight() { [...list.children].forEach((li, i) => li.classList.toggle("active", i === active)); const cur = list.children[active]; if (cur) cur.scrollIntoView({ block: "nearest" }); }

  function render() {
    const all = getModels();
    const query = input.value.trim();
    const matches = query && query.toLowerCase() !== "default" ? searchModels(all, query) : all;
    items = matches.slice(0, MAX_SHOWN); active = -1;
    const rows = items.map((m) => {
      const li = el("li", { role: "option" });
      const at = query ? m.toLowerCase().indexOf(query.toLowerCase().split(/\s+/)[0]) : -1;
      if (at >= 0) { li.append(m.slice(0, at), el("mark", { text: m.slice(at, at + query.split(/\s+/)[0].length) }), m.slice(at + query.split(/\s+/)[0].length)); }
      else li.textContent = m;
      li.addEventListener("mousedown", (e) => { e.preventDefault(); choose(m); });
      return li;
    });
    const status = all.length ? `${matches.length} of ${all.length} models${matches.length > items.length ? ` · showing first ${items.length}, keep typing to narrow` : ""}` : "No model list from this CLI. Type any model id.";
    const head = el("li", { class: "combo-status", "aria-hidden": "true", text: matches.length || !all.length ? status : `No match in ${all.length} models. Press Enter to use "${query}" as a custom id.` });
    list.replaceChildren(head, ...rows);
    list.hidden = false; input.setAttribute("aria-expanded", "true");
    list.querySelectorAll("li[role=option]").forEach((li, i) => { li.dataset.i = i; });
  }

  input.addEventListener("focus", render);
  input.addEventListener("input", () => { render(); onPick(input.value); });
  input.addEventListener("blur", () => setTimeout(close, 120));
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); if (list.hidden) render(); active = Math.min(active + 1, items.length - 1); highlight(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); active = Math.max(active - 1, 0); highlight(); }
    else if (e.key === "Enter" && !list.hidden) { if (active >= 0) { e.preventDefault(); choose(items[active]); } else close(); }
    else if (e.key === "Escape") close();
  });
}

/* ---------- playground ---------- */
function renderThread() {
  const thread = $("thread");
  if (!state.view.length) { thread.replaceChildren(el("p", { class: "empty body-sm", text: "No messages yet. Choose a provider and say something." })); return; }
  thread.replaceChildren(...state.view.map(renderItem));
  thread.scrollTop = thread.scrollHeight;
}

function renderItem(item) {
  if (item.kind === "user") {
    return el("div", { class: "msg user" }, el("div", { class: "bubble", text: item.text }));
  }
  if (item.kind === "toolresult") {
    return el("div", { class: "msg" }, el("div", { class: "msg-meta" }, el("span", { class: "badge plain", text: `tool result · ${item.name}` })), el("div", { class: "toolresult", text: item.text }));
  }
  if (item.kind === "error") {
    return el("div", { class: "msg error" }, el("div", { class: "bubble", text: item.text }), el("div", { class: "msg-meta" }, el("span", { class: "badge plain", text: item.target })));
  }
  const meta = el("div", { class: "msg-meta" }, el("span", { class: "badge", text: item.target }),
    item.ms != null ? el("span", { class: "badge plain", text: `${(item.ms / 1000).toFixed(1)}s` }) : null,
    item.usage ? el("span", { class: "badge plain", text: `${item.usage.total_tokens} tokens` }) : null,
    item.streaming ? el("span", { class: "badge plain", text: "streaming" }) : null);
  const wrap = el("div", { class: "msg assistant" }, meta);
  if (item.text) wrap.append(el("div", { class: "bubble", text: item.text }));
  for (const call of item.toolCalls || []) wrap.append(renderToolCall(call));
  return wrap;
}

function renderToolCall(call) {
  let pretty = call.function.arguments;
  try { pretty = JSON.stringify(JSON.parse(pretty), null, 2); } catch (_) { /* partial while streaming */ }
  const node = el("div", { class: "toolcall" },
    el("span", { class: "badge plain", text: `tool call · ${call.function.name}` }),
    el("pre", { text: pretty }));
  if (!call.answered && !state.busy) {
    const input = el("input", { type: "text", placeholder: "Tool result (the client executes tools; type what it returned)", "aria-label": "Tool result" });
    const send = el("button", { class: "btn btn-primary", type: "submit" }, "Send result");
    const form = el("form", {}, input, send);
    form.addEventListener("submit", (e) => { e.preventDefault(); answerToolCall(call, input.value || "{}"); });
    node.append(form);
  }
  return node;
}

function buildMessages() {
  const system = $("pg-system").value.trim();
  return [...(system ? [{ role: "system", content: system }] : []), ...state.history];
}

function setBusy(busy) {
  state.busy = busy;
  $("pg-send").disabled = busy; $("pg-stop").hidden = !busy;
}

async function answerToolCall(call, text) {
  call.answered = true;
  state.history.push({ role: "tool", tool_call_id: call.id, content: text });
  state.view.push({ kind: "toolresult", name: call.function.name, text });
  renderThread();
  await complete();
}

async function complete() {
  const model = target();
  if (!model) { $("pg-status").textContent = "No enabled provider. Connect first."; return; }
  const useStream = $("pg-stream").checked;
  const body = { model, messages: buildMessages(), stream: useStream };
  if ($("pg-tools").checked) body.tools = SAMPLE_TOOLS;
  const item = { kind: "assistant", target: model, text: "", toolCalls: [], streaming: useStream };
  state.view.push(item); setBusy(true); renderThread();
  state.abort = new AbortController();
  $("pg-status").textContent = `Waiting for ${model}…`;
  const started = performance.now();
  try {
    if (useStream) await readStream(body, item); else await readOnce(body, item);
    item.ms = performance.now() - started; item.streaming = false;
    const assistant = { role: "assistant", content: item.text || null };
    if (item.toolCalls.length) assistant.tool_calls = item.toolCalls.map((c) => ({ id: c.id, type: "function", function: c.function }));
    state.history.push(assistant);
    $("pg-status").textContent = item.toolCalls.length ? "The model requested a tool call. Answer it below to continue." : "";
  } catch (error) {
    state.view.splice(state.view.indexOf(item), 1);
    if (error.name === "AbortError") { $("pg-status").textContent = "Stopped."; }
    else {
      state.view.push({ kind: "error", target: model, text: error.message });
      noteError(model, error); $("pg-status").textContent = "";
    }
  } finally {
    state.abort = null; setBusy(false); renderThread(); refreshActivity();
  }
}

async function readOnce(body, item) {
  const res = await api("/v1/chat/completions", { method: "POST", body: JSON.stringify(body), signal: state.abort.signal });
  const choice = res.choices[0];
  item.text = choice.message.content || "";
  item.toolCalls = (choice.message.tool_calls || []).map((c) => ({ id: c.id, function: { name: c.function.name, arguments: c.function.arguments } }));
  item.usage = res.usage || null;
}

async function readStream(body, item) {
  body.stream_options = { include_usage: true };
  const res = await api("/v1/chat/completions", { method: "POST", body: JSON.stringify(body), signal: state.abort.signal, raw: true });
  if (!res.ok) throw new ApiFailure(res.status, await res.json().catch(() => ({})));
  const reader = res.body.getReader(); const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, cut); buffer = buffer.slice(cut + 2);
      const data = frame.split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trim()).join("");
      if (!data || data === "[DONE]") continue;
      const event = JSON.parse(data);
      if (event.error) throw new ApiFailure(502, event);
      if (event.usage) item.usage = event.usage;
      const delta = event.choices && event.choices[0] && event.choices[0].delta;
      if (!delta) continue;
      if (delta.content) item.text += delta.content;
      for (const part of delta.tool_calls || []) {
        const slot = (item.toolCalls[part.index] ||= { id: "", function: { name: "", arguments: "" } });
        if (part.id) slot.id = part.id;
        if (part.function && part.function.name) slot.function.name += part.function.name;
        if (part.function && part.function.arguments) slot.function.arguments += part.function.arguments;
      }
      renderThread();
    }
  }
}

function sendMessage(text) {
  state.history.push({ role: "user", content: text });
  state.view.push({ kind: "user", text });
  renderThread();
  return complete();
}

/* ---------- activity ---------- */
async function refreshActivity() {
  if (!state.key || !state.providers.length) return;
  try {
    const { data } = await api("/v1/requests");
    const rows = data.length ? data.slice(0, 25).map((r) => el("tr", {},
      el("td", { text: new Date(r.time * 1000).toLocaleTimeString() }),
      el("td", { class: "mono", text: r.model }),
      el("td", { text: r.provider }),
      el("td", { text: `${r.stream ? "stream" : "single"}${r.tools ? " + tools" : ""}` }),
      el("td", { text: `${(r.duration_ms / 1000).toFixed(1)}s` }),
      el("td", { class: r.status === "ok" ? "result-ok" : "result-bad", text: r.status === "ok" ? `ok${r.tool_calls ? ` · ${r.tool_calls} tool call${r.tool_calls > 1 ? "s" : ""}` : ""}` : `${r.status} ${r.code || ""}`.trim() })))
      : [el("tr", {}, el("td", { colspan: "6", class: "body-sm", text: "Nothing yet." }))];
    $("activity-rows").replaceChildren(...rows);
  } catch (_) { /* surfaced by connect() */ }
}

/* ---------- wiring ---------- */
function copyText(text, button) {
  const done = () => { const old = button.textContent; button.textContent = "Copied"; setTimeout(() => { button.textContent = old; }, 1200); };
  (navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject()).then(done, () => {
    const area = el("textarea", { readonly: true }); area.value = text; document.body.append(area); area.select();
    try { document.execCommand("copy"); done(); } catch (_) { /* ignore */ } area.remove();
  });
}

async function init() {
  try { const cfg = await (await fetch("config.json")).json(); state.api = cfg.api || state.api; } catch (_) { /* default */ }
  const fragment = new URLSearchParams(location.hash.slice(1));
  if (fragment.get("h")) {  // one-time nonce from `quanta serve` GUI mode; the key is fetched over loopback
    const nonce = fragment.get("h");
    history.replaceState(null, "", location.pathname + location.search);
    try {
      const res = await fetch("handoff?n=" + encodeURIComponent(nonce));
      if (res.ok) store("quanta.key", (await res.json()).key);
    } catch (_) { /* fall back to pasting the key */ }
  }
  state.key = load("quanta.key");
  if (state.key) $("key-input").placeholder = "Key saved in this browser";
  renderSnippets(); renderMarquee(); renderProviders();

  $("burger").addEventListener("click", () => {
    const open = $("nav-links").classList.toggle("open"); $("burger").setAttribute("aria-expanded", String(open));
  });
  $("nav-links").addEventListener("click", () => { $("nav-links").classList.remove("open"); });
  $("copy-base").addEventListener("click", (e) => copyText(baseUrl(), e.currentTarget.querySelector(".copy-hint")));
  for (const b of document.querySelectorAll("[data-copy]")) b.addEventListener("click", () => copyText($(b.dataset.copy).textContent, b));

  $("key-form").addEventListener("submit", (e) => { e.preventDefault(); saveKey($("key-input").value); });
  $("key-forget").addEventListener("click", () => {
    state.key = ""; store("quanta.key", null); state.providers = []; state.config = null; state.session = null;
    $("key-status").textContent = "Key removed from this browser."; $("key-input").placeholder = "Paste the server.apiKey from quanta.config.json";
    setConn("idle", "Not connected"); renderAll();
  });

  $("pg-provider").addEventListener("change", () => { $("pg-model").value = ""; onProviderChange(); });
  $("sw-provider").addEventListener("change", () => loadModels($("sw-provider").value));
  attachCombo($("pg-model"), () => state.modelCache[$("pg-provider").value] || [], updateTarget);
  attachCombo($("sw-model"), () => state.modelCache[$("sw-provider").value] || [], () => {});
  $("pg-model").addEventListener("input", updateTarget);
  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = $("pg-input").value.trim(); if (!text || state.busy) return;
    $("pg-input").value = ""; sendMessage(text);
  });
  $("pg-input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("composer").requestSubmit(); } });
  $("pg-stop").addEventListener("click", () => state.abort && state.abort.abort());
  $("pg-clear").addEventListener("click", () => { if (state.busy) return; state.history = []; state.view = []; $("pg-status").textContent = ""; renderThread(); });

  $("switch-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const provider = $("sw-provider").value, model = $("sw-model").value.trim();
    if (!provider || !model) { noteError("switch", { message: "Choose a provider and enter a model id." }); return; }
    try {
      await api("/v1/session/switch", { method: "POST", body: JSON.stringify({ provider, model }) });
      state.session = await api("/v1/session"); renderServing();
    } catch (error) { noteError("switch", error); }
  });

  setInterval(() => { if (!document.hidden) refreshActivity(); }, 6000);
  await connect();
}

init();
