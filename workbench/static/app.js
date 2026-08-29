"use strict";

const DEMOS = ["cache", "memory", "rbac", "evaluation"];
const DESCRIPTIONS = {
  cache: "Cache reuse with exact-first lookup and guarded semantic matching.",
  memory: "Expiring conversational state beside durable, provenance-rich preferences.",
  rbac: "Permission filtering inside Redis before retrieved evidence reaches the model.",
  evaluation: "Frozen retrieval results shared by generation and scoring.",
};

const state = {
  demo: "cache",
  eventSource: null,
  activeRun: null,
};

const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
const panels = Array.from(document.querySelectorAll('[role="tabpanel"]'));
const forms = Array.from(document.querySelectorAll(".demo-form"));
const runState = document.querySelector("#run-state");
const traceLog = document.querySelector("#trace-log");
const results = document.querySelector("#results");

function selectDemo(demo, { focus = false, updateUrl = true } = {}) {
  const selected = DEMOS.includes(demo) ? demo : "cache";
  state.demo = selected;
  tabs.forEach((tab) => {
    const active = tab.dataset.demo === selected;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    if (active && focus) tab.focus();
  });
  panels.forEach((panel) => {
    panel.hidden = panel.id !== `panel-${selected}`;
  });
  document.querySelector("#tab-description").textContent = DESCRIPTIONS[selected];
  if (updateUrl) {
    const url = new URL(window.location.href);
    url.searchParams.set("demo", selected);
    window.history.replaceState({ demo: selected }, "", url);
  }
}

tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => selectDemo(tab.dataset.demo));
  tab.addEventListener("keydown", (event) => {
    let nextIndex = null;
    if (event.key === "ArrowRight") nextIndex = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft") nextIndex = (index - 1 + tabs.length) % tabs.length;
    if (event.key === "Home") nextIndex = 0;
    if (event.key === "End") nextIndex = tabs.length - 1;
    if (nextIndex !== null) {
      event.preventDefault();
      selectDemo(tabs[nextIndex].dataset.demo, { focus: true });
    }
  });
});

window.addEventListener("popstate", () => {
  const demo = new URL(window.location.href).searchParams.get("demo");
  selectDemo(demo, { updateUrl: false });
});

document.querySelectorAll(".preset").forEach((button) => {
  button.addEventListener("click", () => {
    const field = document.getElementById(button.dataset.target);
    if (!field) return;
    field.value = button.dataset.value;
    field.focus();
  });
});

function resetRecorder() {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  document.querySelectorAll(".flight-path li").forEach((item) => {
    item.dataset.status = "idle";
    item.querySelector("small").textContent = "Waiting";
  });
  traceLog.replaceChildren();
  const empty = document.createElement("p");
  empty.className = "trace-empty";
  empty.textContent = "Connecting to the flight recorder…";
  traceLog.append(empty);
  results.hidden = true;
  runState.className = "run-state running";
  runState.lastChild.textContent = "Request in flight";
}

function setFormsDisabled(disabled) {
  forms.forEach((form) => {
    form.querySelectorAll("button, input, select, textarea").forEach((control) => {
      control.disabled = disabled;
    });
  });
}

function appendEvent(event) {
  const empty = traceLog.querySelector(".trace-empty");
  if (empty) empty.remove();
  const stage = document.querySelector(`.flight-path li[data-stage="${event.stage}"]`);
  if (stage) {
    stage.dataset.status = event.status;
    stage.querySelector("small").textContent = event.title;
  }

  const entry = document.createElement("article");
  entry.className = "trace-entry";
  const sequence = document.createElement("span");
  sequence.className = "seq";
  sequence.textContent = String(event.sequence).padStart(2, "0");
  const stageName = document.createElement("span");
  stageName.className = "stage";
  stageName.textContent = event.stage;
  const observation = document.createElement("span");
  observation.className = "observation";
  const strong = document.createElement("strong");
  strong.textContent = event.title;
  observation.append(strong, document.createTextNode(` — ${event.detail}`));
  const time = document.createElement("time");
  time.textContent = `${Number(event.at_ms).toFixed(1)} ms`;
  entry.append(sequence, stageName, observation, time);
  traceLog.append(entry);
  traceLog.scrollTop = traceLog.scrollHeight;
}

function renderResult(result) {
  document.querySelector("#result-eyebrow").textContent = result.eyebrow || "RUN RESULT";
  document.querySelector("#result-heading").textContent = result.headline || "Result";
  document.querySelector("#result-answer").textContent = result.answer || "No answer returned.";

  const sourceList = document.querySelector("#source-list");
  sourceList.replaceChildren();
  (result.sources || []).forEach((source) => {
    const card = document.createElement("div");
    card.className = "source-card";
    const title = document.createElement("strong");
    title.textContent = source.title;
    const locator = document.createElement("span");
    locator.textContent = `${source.locator} · similarity ${Number(source.similarity).toFixed(3)}`;
    card.append(title, locator);
    sourceList.append(card);
  });

  const metricGrid = document.querySelector("#metric-grid");
  metricGrid.replaceChildren();
  (result.metrics || []).forEach((metric) => {
    const wrapper = document.createElement("div");
    const label = document.createElement("dt");
    label.textContent = metric.label;
    const value = document.createElement("dd");
    value.textContent = metric.value;
    wrapper.append(label, value);
    metricGrid.append(wrapper);
  });

  const chart = result.chart || { title: "Comparison", unit: "", series: [] };
  document.querySelector("#chart-title").textContent = chart.title;
  document.querySelector("#chart-note").textContent = chart.annotation || "Values are derived from this local demonstration run.";
  const chartRoot = document.querySelector("#chart");
  chartRoot.replaceChildren();
  const values = chart.series.map((item) => Number(item.value));
  const maximum = Math.max(...values, 1);
  chartRoot.setAttribute(
    "aria-label",
    `${chart.title}: ${chart.series.map((item) => `${item.label} ${item.value} ${chart.unit}`).join(", ")}`,
  );
  chartRoot.setAttribute("role", "img");
  chart.series.forEach((item) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    const label = document.createElement("span");
    label.textContent = item.label;
    const track = document.createElement("span");
    track.className = "bar-track";
    const fill = document.createElement("span");
    fill.className = "bar-fill";
    fill.style.setProperty("--bar-width", `${Math.max(2, (Number(item.value) / maximum) * 100)}%`);
    track.append(fill);
    const value = document.createElement("span");
    value.className = "bar-value";
    value.textContent = `${item.value} ${chart.unit}`.trim();
    row.append(label, track, value);
    chartRoot.append(row);
  });

  const comparison = result.comparison || { columns: [], rows: [] };
  const table = document.querySelector("#comparison-table");
  table.replaceChildren();
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  comparison.columns.forEach((column) => {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = column;
    headerRow.append(th);
  });
  thead.append(headerRow);
  const tbody = document.createElement("tbody");
  comparison.rows.forEach((rowData) => {
    const row = document.createElement("tr");
    rowData.forEach((value, index) => {
      const cell = document.createElement(index === 0 ? "th" : "td");
      if (index === 0) cell.scope = "row";
      cell.textContent = value;
      row.append(cell);
    });
    tbody.append(row);
  });
  table.append(thead, tbody);

  const notes = document.querySelector("#result-notes");
  notes.replaceChildren();
  (result.notes || []).forEach((note) => {
    const item = document.createElement("li");
    item.textContent = note;
    notes.append(item);
  });
  results.hidden = false;
}

function completeRun(snapshot) {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  setFormsDisabled(false);
  if (snapshot.status === "complete" && snapshot.result) {
    runState.className = "run-state complete";
    runState.lastChild.textContent = "Run complete";
    renderResult(snapshot.result);
  } else {
    runState.className = "run-state error";
    runState.lastChild.textContent = snapshot.error || "Run failed safely";
  }
  refreshInspector();
}

async function submitRun(form) {
  const payload = Object.fromEntries(new FormData(form).entries());
  payload.demo = form.dataset.demo;
  resetRecorder();
  setFormsDisabled(true);
  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || "The run could not be started.");
    state.activeRun = body.run_id;
    const source = new EventSource(`/api/runs/${encodeURIComponent(body.run_id)}/events`);
    state.eventSource = source;
    source.addEventListener("flight", (event) => appendEvent(JSON.parse(event.data)));
    source.addEventListener("complete", (event) => completeRun(JSON.parse(event.data)));
    source.onerror = () => {
      if (state.eventSource !== source) return;
      source.close();
      state.eventSource = null;
      pollRun(body.run_id);
    };
  } catch (error) {
    setFormsDisabled(false);
    runState.className = "run-state error";
    runState.lastChild.textContent = error.message;
  }
}

async function pollRun(runId) {
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
    const snapshot = await response.json();
    if (!response.ok) throw new Error(snapshot.error || "Run status unavailable.");
    if (snapshot.status === "running") {
      window.setTimeout(() => pollRun(runId), 400);
      return;
    }
    completeRun(snapshot);
  } catch (error) {
    setFormsDisabled(false);
    runState.className = "run-state error";
    runState.lastChild.textContent = error.message;
  }
}

forms.forEach((form) => {
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submitRun(form);
  });
});

function formatTTL(ttl) {
  if (ttl === -1) return "persistent";
  if (ttl === -2) return "expired";
  if (ttl < 60) return `${ttl} s`;
  return `${Math.ceil(ttl / 60)} min`;
}

function displayRedisName(value) {
  const name = String(value);
  if (/^(cache|workbench|idx):/.test(name)) return name;
  return name.replace(/^[^:]+:/, "");
}

async function refreshInspector() {
  const keyBody = document.querySelector("#key-table");
  const indexList = document.querySelector("#index-list");
  try {
    const response = await fetch("/api/redis");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Inspector unavailable");
    document.querySelector("#privacy-note").textContent = data.privacy;
    document.querySelector("#key-count").textContent = `${data.keys.length} ${data.keys.length === 1 ? "key" : "keys"}`;
    document.querySelector("#index-count").textContent = `${data.indexes.length} ${data.indexes.length === 1 ? "index" : "indexes"}`;
    keyBody.replaceChildren();
    if (data.keys.length === 0) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 4;
      cell.textContent = "Run a demo to create scoped keys.";
      row.append(cell);
      keyBody.append(row);
    }
    data.keys.forEach((item) => {
      const row = document.createElement("tr");
      [displayRedisName(item.key), item.type, formatTTL(item.ttl_seconds), String(item.memory_bytes)].forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      });
      keyBody.append(row);
    });
    indexList.replaceChildren();
    if (data.indexes.length === 0) {
      const empty = document.createElement("p");
      empty.textContent = "Indexes appear after cache or RAG runs.";
      indexList.append(empty);
    }
    data.indexes.forEach((item) => {
      const card = document.createElement("div");
      card.className = "index-card";
      const name = document.createElement("code");
      name.textContent = displayRedisName(item.name);
      const count = document.createElement("span");
      count.textContent = `${item.documents} indexed document${item.documents === 1 ? "" : "s"}`;
      card.append(name, count);
      indexList.append(card);
    });
  } catch (error) {
    keyBody.replaceChildren();
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 4;
    cell.textContent = error.message;
    row.append(cell);
    keyBody.append(row);
    indexList.replaceChildren();
    const message = document.createElement("p");
    message.textContent = "Redis Search metadata is unavailable.";
    indexList.append(message);
  }
}

document.querySelector("#refresh-inspector").addEventListener("click", refreshInspector);

const resetDialog = document.querySelector("#reset-dialog");
document.querySelector("#open-reset").addEventListener("click", () => resetDialog.showModal());
resetDialog.addEventListener("close", async () => {
  if (resetDialog.returnValue !== "confirm") return;
  const button = document.querySelector("#open-reset");
  button.disabled = true;
  button.textContent = "Resetting…";
  try {
    const response = await fetch("/api/workbench", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm: "reset" }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Reset failed");
    results.hidden = true;
    runState.className = "run-state complete";
    runState.lastChild.textContent = `${data.keys_deleted} scoped keys reset`;
    await refreshInspector();
  } catch (error) {
    runState.className = "run-state error";
    runState.lastChild.textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = "Reset workbench data";
  }
});

async function loadStatus() {
  const redisStatus = document.querySelector("#redis-status");
  const modelStatus = document.querySelector("#model-status");
  const backendSummary = document.querySelector("#backend-summary");
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    redisStatus.classList.toggle("online", status.ready);
    redisStatus.classList.toggle("offline", !status.ready);
    redisStatus.lastChild.textContent = status.ready ? status.redis : "Redis unavailable";
    modelStatus.classList.toggle("online", status.model_mode === "live");
    modelStatus.classList.toggle("demo", status.model_mode === "demo");
    modelStatus.lastChild.textContent = status.model_display;
    backendSummary.textContent = status.model_mode === "live"
      ? `Real Redis · live ${status.model_name} calls`
      : "Real Redis · simulated model responses";
  } catch {
    redisStatus.classList.add("offline");
    redisStatus.lastChild.textContent = "Server unavailable";
    modelStatus.classList.add("offline");
    modelStatus.lastChild.textContent = "Model unavailable";
    backendSummary.textContent = "System status unavailable";
  }
}

const requestedDemo = new URL(window.location.href).searchParams.get("demo");
selectDemo(requestedDemo, { updateUrl: true });
loadStatus();
refreshInspector();
