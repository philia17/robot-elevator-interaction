"use strict";

const BUILDINGS = ["pmb", "db", "ieb", "trent", "lib", "pb", "tb", "iamet"];
const QUICK_LABELS = ["B2", "B1", "G", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "open", "close", "up", "down", "alarm", "keyhole"];
const STATUS_OPTIONS = [
  ["all", "All"],
  ["unannotated", "Unannotated"],
  ["ai_review", "Needs AI review"],
  ["completed", "Completed"],
  ["excluded", "Deleted"],
];
const META_CHOICES = {
  building: BUILDINGS.map(value => [value, value]),
  location: [["inside", "Inside"], ["outside", "Outside"], ["unknown", "Unknown"]],
  view_angle: [["front", "Front"], ["oblique", "Oblique"], ["unknown", "Unknown"]],
  fill_light: [["on", "On"], ["off", "Off"], ["unknown", "Unknown"]],
  polarizer: [["installed", "Installed"], ["not_installed", "Not installed"], ["unknown", "Unknown"]],
};
const HANDLE_NAMES = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];
const SAVE_DELAY_MS = 450;

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const canvas = $("#annotation-canvas");
const ctx = canvas.getContext("2d");

const state = {
  items: [],
  filtered: [],
  datasetStats: {},
  currentId: null,
  annotationCache: new Map(),
  filters: { building: "all", session: "all", status: "all" },
  viewMode: "rgb",
  rgbImage: null,
  depthImage: null,
  selectedBox: -1,
  interaction: null,
  loadToken: 0,
  saveTimers: new Map(),
  saveVersions: new Map(),
  pendingSaves: new Map(),
  toastTimer: null,
};

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

function itemEndpoint(item, suffix) {
  return `/api/items/${encodeURIComponent(item.id)}/${suffix}`;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = "";
    try {
      const payload = await response.json();
      detail = payload.detail || payload.error || payload.message || "";
    } catch (_) {
      detail = (await response.text()).slice(0, 180);
    }
    throw new ApiError(detail || `${response.status} ${response.statusText}`, response.status);
  }
  if (response.status === 204) return null;
  const text = await response.text();
  return text ? JSON.parse(text) : null;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[ch]);
}

function currentItem() {
  return state.items.find(item => item.id === state.currentId) || null;
}

function currentAnnotation() {
  return state.currentId ? state.annotationCache.get(state.currentId) || null : null;
}

function buildingFilterFromUrl() {
  const building = new URLSearchParams(window.location.search).get("building");
  return BUILDINGS.includes(building) ? building : "all";
}

function syncBuildingFilterToUrl() {
  const url = new URL(window.location.href);
  if (state.filters.building === "all") url.searchParams.delete("building");
  else url.searchParams.set("building", state.filters.building);
  window.history.replaceState(window.history.state, "", `${url.pathname}${url.search}${url.hash}`);
}

function normalizeStatus(value) {
  const status = String(value || "").toLowerCase().replaceAll("-", "_");
  if (["done", "complete", "completed", "reviewed"].includes(status)) return "completed";
  if (["ai", "ai_draft", "ai_review", "pending_review"].includes(status)) return "ai_review";
  if (["exclude", "excluded", "deleted"].includes(status)) return "excluded";
  return "unannotated";
}

function normalizeBox(raw = {}) {
  const coords = raw.bbox_xyxy || raw.bbox || [0, 0, 0, 0];
  const bbox = coords.slice(0, 4).map(Number);
  const known = new Set(["bbox_xyxy", "bbox", "coarse_type", "label", "lit", "source", "confidence", "reviewed"]);
  return {
    extra: Object.fromEntries(Object.entries(raw).filter(([key]) => !known.has(key))),
    bbox_xyxy: bbox.length === 4 && bbox.every(Number.isFinite) ? bbox : [0, 0, 0, 0],
    coarse_type: raw.coarse_type || inferCoarseType(raw.label || ""),
    label: raw.label == null ? "" : String(raw.label),
    lit: Boolean(raw.lit),
    source: raw.source || "manual",
    confidence: Number.isFinite(Number(raw.confidence)) ? Number(raw.confidence) : null,
    reviewed: raw.reviewed == null ? raw.source !== "ai" : Boolean(raw.reviewed),
  };
}

function normalizeAnnotation(rawPayload, item) {
  const raw = rawPayload && rawPayload.annotation ? rawPayload.annotation : (rawPayload || {});
  const boxes = raw.ground_truth || raw.boxes || raw.annotations || [];
  const rawMeta = raw.image_meta || raw.metadata || {};
  const sourceMeta = item.metadata || {};
  const metadata = {
    ...sourceMeta,
    ...rawMeta,
    building: rawMeta.building || sourceMeta.building || item.building || "",
    location: rawMeta.location || sourceMeta.location || "unknown",
    view_angle: rawMeta.view_angle || sourceMeta.view_angle || sourceMeta.angle || "unknown",
    fill_light: rawMeta.fill_light || sourceMeta.fill_light || "unknown",
    polarizer: rawMeta.polarizer || sourceMeta.polarizer || "unknown",
    camera_model: rawMeta.camera_model || sourceMeta.camera_model || sourceMeta.camera || "Gemini 335",
    notes: rawMeta.notes ?? sourceMeta.notes ?? "",
  };
  if (metadata.distance_auto_cm == null) {
    metadata.distance_auto_cm = sourceMeta.distance_auto_cm ?? sourceMeta.distance_cm_auto ?? rawMeta.distance_cm ?? sourceMeta.distance_cm ?? null;
  }
  if (metadata.distance_override_cm == null && rawMeta.distance_override_cm != null) {
    metadata.distance_override_cm = rawMeta.distance_override_cm;
  }
  const known = new Set(["annotation", "ground_truth", "boxes", "annotations", "image_meta", "metadata", "status", "review_status", "reviewed", "image_id", "image_stem"]);
  const status = item.excluded ? "excluded" : normalizeStatus(raw.review_status || raw.status || item.status);
  return {
    extra: Object.fromEntries(Object.entries(raw).filter(([key]) => !known.has(key))),
    ground_truth: boxes.map(normalizeBox),
    metadata,
    status,
    reviewed: Boolean(raw.reviewed || status === "completed"),
    dirty: false,
  };
}

function annotationPayload(item, annotation) {
  const boxes = annotation.ground_truth.map(box => {
    const output = {
      ...box.extra,
      bbox_xyxy: box.bbox_xyxy.map(value => Math.round(value * 100) / 100),
      coarse_type: box.coarse_type || inferCoarseType(box.label),
      label: box.label,
      lit: Boolean(box.lit),
      source: box.source || "manual",
      reviewed: Boolean(box.reviewed),
    };
    if (box.confidence != null) output.confidence = box.confidence;
    return output;
  });
  const autoDistance = nullableNumber(annotation.metadata.distance_auto_cm);
  const overrideDistance = nullableNumber(annotation.metadata.distance_override_cm);
  const metadata = {
    ...annotation.metadata,
    distance_auto_cm: autoDistance,
    distance_override_cm: overrideDistance,
    distance_cm: overrideDistance ?? autoDistance,
  };
  return {
    ...annotation.extra,
    image_id: item.id,
    image_stem: item.stem,
    status: annotation.status,
    review_status: annotation.status === "completed" ? "reviewed" : annotation.status === "ai_review" ? "ai_draft" : "unreviewed",
    reviewed: annotation.reviewed,
    ground_truth: boxes,
    image_meta: metadata,
  };
}

function nullableNumber(value) {
  if (value === "" || value == null) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function derivedStatus(item) {
  if (item.excluded) return "excluded";
  const annotation = state.annotationCache.get(item.id);
  if (!annotation) return normalizeStatus(item.status);
  if (annotation.status === "completed" || annotation.reviewed) return "completed";
  if (annotation.ground_truth.some(box => box.source === "ai" && !box.reviewed)) return "ai_review";
  return normalizeStatus(annotation.status);
}

function inferCoarseType(label) {
  const text = String(label || "").trim().toLowerCase();
  if (/^(b\d+|g|\-?\d+)$/.test(text)) return "floor_button_candidate";
  if (["open", "close"].includes(text)) return "door_button_candidate";
  if (["up", "down"].includes(text)) return "call_button_candidate";
  if (text === "keyhole") return "keyhole";
  return "button_candidate";
}

function showToast(message, kind = "") {
  const toast = $("#toast");
  clearTimeout(state.toastTimer);
  toast.textContent = message;
  toast.className = `toast visible ${kind}`.trim();
  state.toastTimer = setTimeout(() => { toast.className = "toast"; }, 2600);
}

function setSaveState(kind, text) {
  const element = $("#save-state");
  element.className = `save-state ${kind}`;
  element.lastChild.textContent = text;
}

function buildStaticControls() {
  $("#building-filters").innerHTML = [["all", "All"], ...BUILDINGS.map(value => [value, value])]
    .map(([value, label]) => `<button type="button" data-building="${value}">${label}</button>`).join("");
  $("#status-filters").innerHTML = STATUS_OPTIONS.map(([value, label]) => `
    <button type="button" class="status-filter" data-status="${value}">
      <i class="dot"></i><span>${label}</span><span class="count" data-status-count="${value}">0</span>
    </button>`).join("");
  $("#quick-labels").innerHTML = QUICK_LABELS.map((label, index) =>
    `<button type="button" data-label="${label}" title="${index < 9 ? `Shortcut ${index + 1}` : label}">${label}</button>`
  ).join("");
  buildChoiceGroup("#meta-building", "building", META_CHOICES.building);
  buildChoiceGroup("#meta-location", "location", META_CHOICES.location);
  buildChoiceGroup("#meta-view-angle", "view_angle", META_CHOICES.view_angle);
  buildChoiceGroup("#meta-fill-light", "fill_light", META_CHOICES.fill_light);
  buildChoiceGroup("#meta-polarizer", "polarizer", META_CHOICES.polarizer);
  $("#bulk-fields").innerHTML = [
    ["building", "Building"], ["location", "Location"], ["view_angle", "View angle"],
    ["fill_light", "Fill light"], ["polarizer", "Polarizer"], ["camera_model", "Camera model"],
    ["distance_override_cm", "Distance override"], ["notes", "Notes"],
  ].map(([key, label]) => `<label class="bulk-check"><input type="checkbox" name="bulk-field" value="${key}"><span>${label}</span></label>`).join("");
}

function buildChoiceGroup(selector, key, choices) {
  $(selector).innerHTML = choices.map(([value, label]) =>
    `<button type="button" data-meta-key="${key}" data-value="${value}">${label}</button>`
  ).join("");
}

async function initialize() {
  buildStaticControls();
  bindEvents();
  state.filters.building = buildingFilterFromUrl();
  syncBuildingFilterToUrl();
  try {
    const payload = await fetchJson("/api/dataset");
    state.datasetStats = payload?.stats || {};
    state.items = (payload?.items || []).map((item, index) => ({
      ...item,
      id: String(item.id ?? item.stem ?? index),
      building: item.building || item.metadata?.building || "unknown",
      session: item.session || "unknown",
      stem: item.stem || String(item.id ?? index),
      excluded: Boolean(item.excluded),
    }));
    $("#dataset-summary").textContent = `${state.items.length} RGB-D pairs · local workspace`;
    populateSessionFilter();
    applyFilters(false);
    if (state.filtered.length) await selectItem(state.filtered[0].id, false);
    else showCanvasMessage("No images match the current filters");
  } catch (error) {
    console.error(error);
    $("#dataset-summary").textContent = "Failed to load dataset";
    showCanvasMessage(`Unable to load dataset: ${error.message}`, true);
    showToast(`Failed to load dataset: ${error.message}`, "error");
  }
}

function bindEvents() {
  $("#building-filters").addEventListener("click", event => {
    const button = event.target.closest("button[data-building]");
    if (!button) return;
    state.filters.building = button.dataset.building;
    state.filters.session = "all";
    syncBuildingFilterToUrl();
    populateSessionFilter();
    applyFilters(true);
  });
  $("#status-filters").addEventListener("click", event => {
    const button = event.target.closest("button[data-status]");
    if (!button) return;
    state.filters.status = button.dataset.status;
    applyFilters(true);
  });
  $("#session-filter").addEventListener("change", event => {
    state.filters.session = event.target.value;
    applyFilters(true);
  });
  $("#clear-filters").addEventListener("click", () => {
    state.filters = { building: "all", session: "all", status: "all" };
    syncBuildingFilterToUrl();
    populateSessionFilter();
    applyFilters(true);
  });
  $("#item-list").addEventListener("click", event => {
    const row = event.target.closest("[data-item-id]");
    if (row) selectItem(row.dataset.itemId);
  });
  $("#btn-prev").addEventListener("click", () => navigate(-1));
  $("#btn-next").addEventListener("click", () => navigate(1));
  $("#btn-complete").addEventListener("click", toggleCompleted);
  $("#btn-accept-ai").addEventListener("click", acceptAllAi);
  $("#btn-clear-ai").addEventListener("click", clearAiDrafts);
  $("#btn-exclude").addEventListener("click", excludeCurrent);
  $("#btn-restore").addEventListener("click", restoreCurrent);
  $("#btn-bulk").addEventListener("click", openBulkDialog);
  $("#bulk-scope").addEventListener("change", updateBulkSummary);
  $("#bulk-fields").addEventListener("change", updateBulkSummary);
  $("#confirm-bulk").addEventListener("click", event => {
    event.preventDefault();
    applyBulkMetadata();
  });
  $$(".segmented [data-view]").forEach(button => button.addEventListener("click", () => setViewMode(button.dataset.view)));
  $("#quick-labels").addEventListener("click", event => {
    const button = event.target.closest("[data-label]");
    if (button) applyLabel(button.dataset.label);
  });
  $("#apply-custom-label").addEventListener("click", () => applyLabel($("#custom-label").value));
  $("#custom-label").addEventListener("keydown", event => {
    if (event.key === "Enter") { event.preventDefault(); applyLabel(event.target.value); }
  });
  $("#toggle-lit").addEventListener("click", toggleSelectedLit);
  $("#delete-box").addEventListener("click", deleteSelectedBox);
  document.body.addEventListener("click", event => {
    const button = event.target.closest("[data-meta-key]");
    if (button) setMetadata(button.dataset.metaKey, button.dataset.value);
  });
  $("#camera-model").addEventListener("input", event => setMetadata("camera_model", event.target.value, false));
  $("#distance-override").addEventListener("input", event => {
    setMetadata("distance_override_cm", event.target.value === "" ? null : event.target.value, false);
    renderDistance();
  });
  $("#meta-notes").addEventListener("input", event => setMetadata("notes", event.target.value, false));
  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointercancel", onPointerUp);
  window.addEventListener("resize", renderCanvas);
  window.addEventListener("keydown", handleKeyboard);
  window.addEventListener("pagehide", saveBeforeExit);
}

function populateSessionFilter() {
  const sessions = [...new Set(state.items
    .filter(item => state.filters.building === "all" || item.building === state.filters.building)
    .map(item => item.session))].sort();
  const select = $("#session-filter");
  select.innerHTML = `<option value="all">All sessions</option>${sessions.map(session =>
    `<option value="${escapeHtml(session)}">${escapeHtml(session)}</option>`).join("")}`;
  if (!sessions.includes(state.filters.session)) state.filters.session = "all";
  select.value = state.filters.session;
}

function itemsBeforeStatusFilter() {
  return state.items.filter(item =>
    (state.filters.building === "all" || item.building === state.filters.building) &&
    (state.filters.session === "all" || item.session === state.filters.session)
  );
}

function applyFilters(selectFirst = false) {
  const scoped = itemsBeforeStatusFilter();
  state.filtered = scoped.filter(item => state.filters.status === "all" || derivedStatus(item) === state.filters.status);
  updateFilterUi(scoped);
  renderItemList();
  updateProgress();
  if (selectFirst && !state.filtered.some(item => item.id === state.currentId)) {
    if (state.filtered.length) selectItem(state.filtered[0].id);
    else {
      state.currentId = null;
      state.rgbImage = null;
      state.depthImage = null;
      renderAll();
      showCanvasMessage("No images match the current filters");
    }
  } else {
    updateNavigation();
  }
}

function updateFilterUi(scoped) {
  $$("[data-building]").forEach(button => button.classList.toggle("active", button.dataset.building === state.filters.building));
  $$(".status-filter").forEach(button => button.classList.toggle("active", button.dataset.status === state.filters.status));
  const counts = Object.fromEntries(STATUS_OPTIONS.map(([key]) => [key, 0]));
  counts.all = scoped.length;
  scoped.forEach(item => { counts[derivedStatus(item)] += 1; });
  $$('[data-status-count]').forEach(element => { element.textContent = counts[element.dataset.statusCount] || 0; });
  $("#visible-count").textContent = state.filtered.length;
}

function renderItemList() {
  $("#item-list").innerHTML = state.filtered.map(item => {
    const status = derivedStatus(item);
    return `<button type="button" class="item-row ${status} ${item.id === state.currentId ? "active" : ""}" data-item-id="${escapeHtml(item.id)}">
      <i class="state-dot"></i><span title="${escapeHtml(item.stem)}">${escapeHtml(item.stem)}</span><small>${escapeHtml(item.building)}</small>
    </button>`;
  }).join("") || `<div class="muted" style="padding:16px;text-align:center">No results</div>`;
  $("#item-list .active")?.scrollIntoView({ block: "nearest" });
}

function updateProgress() {
  const completed = state.filtered.filter(item => derivedStatus(item) === "completed").length;
  const total = state.filtered.filter(item => derivedStatus(item) !== "excluded").length;
  const percent = total ? Math.round(completed / total * 100) : 0;
  $("#progress-percent").textContent = `${percent}%`;
  $("#progress-bar").style.width = `${percent}%`;
  $("#progress-detail").textContent = `${completed} / ${total} complete`;
}

async function selectItem(id, flushPrevious = true) {
  if (!id || id === state.currentId && currentAnnotation()) return;
  const previous = currentItem();
  if (flushPrevious && previous) await flushSave(previous);
  const item = state.items.find(candidate => candidate.id === id);
  if (!item) return;
  state.currentId = id;
  state.selectedBox = -1;
  state.interaction = null;
  state.rgbImage = null;
  state.depthImage = null;
  const token = ++state.loadToken;
  renderAll();
  showCanvasMessage("Loading image…");
  try {
    const annotationPromise = state.annotationCache.has(id)
      ? Promise.resolve(state.annotationCache.get(id))
      : loadAnnotation(item);
    const rgbPromise = loadImage(item.image_url || itemEndpoint(item, "image"));
    const depthPromise = loadImage(item.depth_preview_url || itemEndpoint(item, "depth-preview")).catch(() => null);
    const [annotation, rgb, depth] = await Promise.all([annotationPromise, rgbPromise, depthPromise]);
    if (token !== state.loadToken) return;
    state.annotationCache.set(id, annotation);
    state.rgbImage = rgb;
    state.depthImage = depth;
    hideCanvasMessage();
    renderAll();
  } catch (error) {
    if (token !== state.loadToken) return;
    console.error(error);
    if (!state.annotationCache.has(id)) state.annotationCache.set(id, normalizeAnnotation({}, item));
    showCanvasMessage(`Failed to load image: ${error.message}`, true);
    renderAll(false);
  }
}

async function loadAnnotation(item) {
  try {
    const raw = await fetchJson(item.annotation_url || itemEndpoint(item, "annotation"));
    return normalizeAnnotation(raw, item);
  } catch (error) {
    if (error.status === 404) return normalizeAnnotation({}, item);
    throw error;
  }
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Unable to read ${url}`));
    image.src = url;
  });
}

async function navigate(delta) {
  const index = state.filtered.findIndex(item => item.id === state.currentId);
  const next = index + delta;
  if (next < 0 || next >= state.filtered.length) return;
  await selectItem(state.filtered[next].id);
}

function updateNavigation() {
  const index = state.filtered.findIndex(item => item.id === state.currentId);
  $("#position").textContent = index >= 0 ? `${index + 1} / ${state.filtered.length}` : `0 / ${state.filtered.length}`;
  $("#btn-prev").disabled = index <= 0;
  $("#btn-next").disabled = index < 0 || index >= state.filtered.length - 1;
}

function renderAll(includeCanvas = true) {
  renderItemList();
  updateProgress();
  updateNavigation();
  renderHeader();
  renderMetadata();
  renderBoxEditor();
  if (includeCanvas) renderCanvas();
}

function renderHeader() {
  const item = currentItem();
  const annotation = currentAnnotation();
  const status = item ? derivedStatus(item) : "unannotated";
  $("#image-title").textContent = item?.stem || "—";
  $("#session-name").textContent = item?.session || "—";
  const pill = $("#item-status");
  pill.className = `status-pill ${status}`;
  pill.textContent = Object.fromEntries(STATUS_OPTIONS)[status] || "Unannotated";
  $("#btn-complete").classList.toggle("completed", status === "completed");
  $("#btn-complete").textContent = status === "completed" ? "✓ Completed (click to undo)" : "✓ Complete image";
  const pendingAi = annotation?.ground_truth.filter(box => box.source === "ai" && !box.reviewed).length || 0;
  $("#btn-accept-ai").disabled = !pendingAi || item?.excluded;
  $("#btn-clear-ai").disabled = !pendingAi || item?.excluded;
  $("#btn-complete").disabled = !item || item.excluded;
  $("#btn-exclude").disabled = !item || item.excluded;
  $("#excluded-banner").classList.toggle("hidden", !item?.excluded);
  const agxDeleted = Boolean(item?.agx_deleted);
  $("#excluded-message").textContent = agxDeleted
    ? "Deleted locally and permanently on AGX"
    : "RGB and depth images moved to Trash";
  $("#btn-restore").disabled = !item?.excluded || agxDeleted;
  $("#btn-restore").title = agxDeleted
    ? "Restore is unavailable because the AGX pair was permanently deleted"
    : "Restore this local RGB-D pair";
}

function renderMetadata() {
  const item = currentItem();
  const annotation = currentAnnotation();
  const metadata = annotation?.metadata || {};
  $$('[data-meta-key]').forEach(button => {
    button.classList.toggle("active", String(metadata[button.dataset.metaKey] ?? "") === button.dataset.value);
    button.disabled = !item || item.excluded;
  });
  $("#camera-model").value = metadata.camera_model || "";
  $("#camera-model").disabled = !item || item.excluded;
  $("#meta-notes").value = metadata.notes || "";
  $("#meta-notes").disabled = !item || item.excluded;
  const overrideDistance = nullableNumber(metadata.distance_override_cm);
  $("#distance-override").value = overrideDistance ?? "";
  $("#distance-override").disabled = !item || item.excluded;
  renderDistance();
  $("#source-stem").textContent = item?.stem || "—";
  $("#source-time").textContent = metadata.captured_at || metadata.timestamp || metadata.created_at || item?.timestamp || "—";
}

function renderDistance() {
  const metadata = currentAnnotation()?.metadata || {};
  const autoDistance = nullableNumber(metadata.distance_auto_cm);
  const overrideDistance = nullableNumber(metadata.distance_override_cm);
  $("#distance-auto").textContent = autoDistance == null ? "—" : `${autoDistance.toFixed(1)} cm`;
  const finalDistance = overrideDistance ?? autoDistance;
  $("#distance-final").textContent = finalDistance == null
    ? "Final distance: —"
    : `Final distance: ${finalDistance.toFixed(1)} cm (${overrideDistance != null ? "manual" : "automatic"})`;
}

function renderBoxEditor() {
  const item = currentItem();
  const annotation = currentAnnotation();
  const boxes = annotation?.ground_truth || [];
  if (state.selectedBox >= boxes.length) state.selectedBox = -1;
  $("#box-count").textContent = boxes.length;
  $("#selection-hint").textContent = state.selectedBox < 0 ? "Select a box to edit its label" : `Editing box #${state.selectedBox + 1}`;
  $("#box-list").innerHTML = boxes.map((box, index) => {
    const draft = box.source === "ai" && !box.reviewed;
    const confidence = box.confidence == null ? "" : ` ${Math.round(box.confidence * 100)}%`;
    return `<button type="button" class="box-chip ${index === state.selectedBox ? "active" : ""} ${draft ? "ai-draft" : ""}" data-box-index="${index}">
      <span>#${index + 1} ${escapeHtml(box.label || "Unlabeled")}</span><em>${draft ? `AI${confidence}` : box.lit ? "Lit" : ""}</em>
    </button>`;
  }).join("") || `<span class="muted">No boxes yet. Drag on the image to create one.</span>`;
  $$("[data-box-index]").forEach(button => button.addEventListener("click", () => {
    state.selectedBox = Number(button.dataset.boxIndex);
    renderBoxEditor();
    renderCanvas();
  }));
  const selected = boxes[state.selectedBox];
  $$("#quick-labels [data-label]").forEach(button => button.classList.toggle("active", selected?.label === button.dataset.label));
  $("#custom-label").value = selected && !QUICK_LABELS.includes(selected.label) ? selected.label : "";
  $("#toggle-lit").textContent = selected?.lit ? "Light: on" : "Light: off";
  [$("#custom-label"), $("#apply-custom-label"), $("#toggle-lit"), $("#delete-box")].forEach(element => {
    element.disabled = !selected || item?.excluded;
  });
}

function setViewMode(mode) {
  if (mode === "depth" && !state.depthImage) {
    showToast("No depth preview is available for this image", "error");
    return;
  }
  state.viewMode = mode;
  $$(".segmented [data-view]").forEach(button => button.classList.toggle("active", button.dataset.view === mode));
  renderCanvas();
}

function renderCanvas() {
  const rgb = state.rgbImage;
  if (!rgb) return;
  const displayed = state.viewMode === "depth" && state.depthImage ? state.depthImage : rgb;
  if (canvas.width !== rgb.naturalWidth || canvas.height !== rgb.naturalHeight) {
    canvas.width = rgb.naturalWidth;
    canvas.height = rgb.naturalHeight;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(displayed, 0, 0, canvas.width, canvas.height);
  const annotation = currentAnnotation();
  annotation?.ground_truth.forEach((box, index) => drawBox(box, index === state.selectedBox));
  if (state.interaction?.mode === "draw" && state.interaction.previewBBox) {
    drawBox({ bbox_xyxy: state.interaction.previewBBox, label: "New box", source: "manual", reviewed: true }, true, true);
  }
}

function displayScale() {
  const rect = canvas.getBoundingClientRect();
  return rect.width && canvas.width ? rect.width / canvas.width : 1;
}

function drawBox(box, selected, preview = false) {
  const [x1, y1, x2, y2] = box.bbox_xyxy;
  const scale = displayScale();
  const lineWidth = Math.max(1.5 / scale, selected ? 3 / scale : 2 / scale);
  const isDraft = box.source === "ai" && !box.reviewed;
  const color = preview ? "#ffffff" : isDraft ? "#f5c04a" : box.lit ? "#7dffcb" : "#53a0ff";
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  if (isDraft || preview) ctx.setLineDash([9 / scale, 6 / scale]);
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.setLineDash([]);
  if (selected && !preview) drawHandles(box.bbox_xyxy, scale, color);
  const label = box.label || (isDraft ? "AI ?" : "?");
  const fontSize = 12 / scale;
  ctx.font = `600 ${fontSize}px -apple-system, sans-serif`;
  const confidence = isDraft && box.confidence != null ? ` ${Math.round(box.confidence * 100)}%` : "";
  const text = `${label}${confidence}`;
  const width = ctx.measureText(text).width + 9 / scale;
  const height = 19 / scale;
  const top = Math.max(0, y1 - height);
  ctx.fillStyle = color;
  ctx.fillRect(x1, top, width, height);
  ctx.fillStyle = "#07101b";
  ctx.fillText(text, x1 + 4.5 / scale, top + 14 / scale);
  ctx.restore();
}

function handlePoints(bbox) {
  const [x1, y1, x2, y2] = bbox;
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  return [[x1, y1], [mx, y1], [x2, y1], [x2, my], [x2, y2], [mx, y2], [x1, y2], [x1, my]];
}

function drawHandles(bbox, scale, color) {
  const size = 7 / scale;
  ctx.fillStyle = "#f7fbff";
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5 / scale;
  handlePoints(bbox).forEach(([x, y]) => {
    ctx.fillRect(x - size / 2, y - size / 2, size, size);
    ctx.strokeRect(x - size / 2, y - size / 2, size, size);
  });
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(canvas.width, (event.clientX - rect.left) * canvas.width / rect.width)),
    y: Math.max(0, Math.min(canvas.height, (event.clientY - rect.top) * canvas.height / rect.height)),
  };
}

function hitHandle(point, bbox) {
  const radius = 9 / displayScale();
  const points = handlePoints(bbox);
  for (let index = 0; index < points.length; index += 1) {
    if (Math.hypot(point.x - points[index][0], point.y - points[index][1]) <= radius) return HANDLE_NAMES[index];
  }
  return null;
}

function hitBox(point) {
  const boxes = currentAnnotation()?.ground_truth || [];
  for (let index = boxes.length - 1; index >= 0; index -= 1) {
    const [x1, y1, x2, y2] = boxes[index].bbox_xyxy;
    if (point.x >= x1 && point.x <= x2 && point.y >= y1 && point.y <= y2) return index;
  }
  return -1;
}

function onPointerDown(event) {
  const item = currentItem();
  const annotation = currentAnnotation();
  if (!item || item.excluded || !annotation || !state.rgbImage) return;
  event.preventDefault();
  canvas.setPointerCapture(event.pointerId);
  const point = canvasPoint(event);
  const selected = annotation.ground_truth[state.selectedBox];
  const handle = selected ? hitHandle(point, selected.bbox_xyxy) : null;
  if (handle) {
    state.interaction = { mode: "resize", start: point, boxIndex: state.selectedBox, handle, original: selected.bbox_xyxy.slice() };
    canvas.className = "is-resizing";
    return;
  }
  const hitIndex = hitBox(point);
  if (hitIndex >= 0) {
    state.selectedBox = hitIndex;
    state.interaction = { mode: "move", start: point, boxIndex: hitIndex, original: annotation.ground_truth[hitIndex].bbox_xyxy.slice() };
    canvas.className = "is-moving";
    renderBoxEditor();
    renderCanvas();
    return;
  }
  state.selectedBox = -1;
  state.interaction = { mode: "draw", start: point, previewBBox: [point.x, point.y, point.x, point.y] };
  renderBoxEditor();
  renderCanvas();
}

function onPointerMove(event) {
  const interaction = state.interaction;
  const annotation = currentAnnotation();
  if (!interaction || !annotation) return;
  const point = canvasPoint(event);
  if (interaction.mode === "draw") {
    interaction.previewBBox = normalizeBbox([interaction.start.x, interaction.start.y, point.x, point.y]);
  } else if (interaction.mode === "move") {
    const box = annotation.ground_truth[interaction.boxIndex];
    const dx = point.x - interaction.start.x;
    const dy = point.y - interaction.start.y;
    const [x1, y1, x2, y2] = interaction.original;
    const width = x2 - x1;
    const height = y2 - y1;
    const nextX1 = Math.max(0, Math.min(canvas.width - width, x1 + dx));
    const nextY1 = Math.max(0, Math.min(canvas.height - height, y1 + dy));
    box.bbox_xyxy = [nextX1, nextY1, nextX1 + width, nextY1 + height];
  } else if (interaction.mode === "resize") {
    const box = annotation.ground_truth[interaction.boxIndex];
    box.bbox_xyxy = resizedBbox(interaction.original, interaction.handle, point);
  }
  renderCanvas();
}

function onPointerUp(event) {
  const interaction = state.interaction;
  const annotation = currentAnnotation();
  if (!interaction || !annotation) return;
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  canvas.className = "";
  if (interaction.mode === "draw") {
    const bbox = interaction.previewBBox;
    const minSize = 6 / displayScale();
    if (bbox && bbox[2] - bbox[0] >= minSize && bbox[3] - bbox[1] >= minSize) {
      annotation.ground_truth.push({
        extra: {}, bbox_xyxy: bbox, coarse_type: "button_candidate", label: "", lit: false,
        source: "manual", confidence: null, reviewed: true,
      });
      state.selectedBox = annotation.ground_truth.length - 1;
      markEdited();
      setTimeout(() => $("#custom-label").focus(), 0);
    }
  } else {
    markEdited();
  }
  state.interaction = null;
  renderAll();
}

function normalizeBbox(bbox) {
  return [Math.min(bbox[0], bbox[2]), Math.min(bbox[1], bbox[3]), Math.max(bbox[0], bbox[2]), Math.max(bbox[1], bbox[3])];
}

function resizedBbox(original, handle, point) {
  let [x1, y1, x2, y2] = original;
  const min = 4 / displayScale();
  if (handle.includes("w")) x1 = Math.max(0, Math.min(x2 - min, point.x));
  if (handle.includes("e")) x2 = Math.min(canvas.width, Math.max(x1 + min, point.x));
  if (handle.includes("n")) y1 = Math.max(0, Math.min(y2 - min, point.y));
  if (handle.includes("s")) y2 = Math.min(canvas.height, Math.max(y1 + min, point.y));
  return [x1, y1, x2, y2];
}

function applyLabel(label) {
  const annotation = currentAnnotation();
  const box = annotation?.ground_truth[state.selectedBox];
  const clean = String(label || "").trim();
  if (!box || !clean) return;
  box.label = clean;
  box.coarse_type = inferCoarseType(clean);
  if (box.source === "ai") box.reviewed = true;
  $("#custom-label").value = "";
  markEdited();
  renderAll();
}

function toggleSelectedLit() {
  const box = currentAnnotation()?.ground_truth[state.selectedBox];
  if (!box) return;
  box.lit = !box.lit;
  if (box.source === "ai") box.reviewed = true;
  markEdited();
  renderAll();
}

function deleteSelectedBox() {
  const annotation = currentAnnotation();
  if (!annotation || state.selectedBox < 0) return;
  annotation.ground_truth.splice(state.selectedBox, 1);
  state.selectedBox = Math.min(state.selectedBox, annotation.ground_truth.length - 1);
  markEdited();
  renderAll();
}

function acceptAllAi() {
  const annotation = currentAnnotation();
  if (!annotation) return;
  annotation.ground_truth.forEach(box => { if (box.source === "ai") box.reviewed = true; });
  annotation.status = "unannotated";
  markEdited();
  renderAll();
  showToast("Accepted all AI boxes for this image");
}

function clearAiDrafts() {
  const annotation = currentAnnotation();
  if (!annotation) return;
  const before = annotation.ground_truth.length;
  annotation.ground_truth = annotation.ground_truth.filter(box => box.source !== "ai" || box.reviewed);
  state.selectedBox = -1;
  if (annotation.ground_truth.length !== before) {
    markEdited();
    renderAll();
    showToast(`Cleared ${before - annotation.ground_truth.length} AI draft boxes`);
  }
}

function toggleCompleted() {
  const annotation = currentAnnotation();
  const item = currentItem();
  if (!annotation || !item || item.excluded) return;
  const completed = derivedStatus(item) === "completed";
  if (!completed) {
    const missingLabel = annotation.ground_truth.findIndex(box => !String(box.label || "").trim());
    if (missingLabel >= 0) {
      state.selectedBox = missingLabel;
      renderBoxEditor();
      renderCanvas();
      showToast(`Box #${missingLabel + 1} does not have a label`, "error");
      return;
    }
  }
  annotation.reviewed = !completed;
  annotation.status = completed ? (annotation.ground_truth.some(box => box.source === "ai" && !box.reviewed) ? "ai_review" : "unannotated") : "completed";
  if (!completed) annotation.ground_truth.forEach(box => { box.reviewed = true; });
  markEdited(true);
  renderAll();
  showToast(completed ? "Completion status undone" : "Image marked complete");
}

function setMetadata(key, value, rerender = true) {
  const annotation = currentAnnotation();
  const item = currentItem();
  if (!annotation || !item || item.excluded) return;
  annotation.metadata[key] = value;
  if (key === "building") item.building = value;
  markEdited();
  if (rerender) renderMetadata();
  renderHeader();
  renderItemList();
  if (key === "building") {
    populateSessionFilter();
    applyFilters(false);
  }
}

function markEdited(preserveStatus = false) {
  const item = currentItem();
  const annotation = currentAnnotation();
  if (!item || !annotation) return;
  annotation.dirty = true;
  if (!preserveStatus && annotation.status === "completed") {
    annotation.reviewed = false;
    annotation.status = annotation.ground_truth.some(box => box.source === "ai" && !box.reviewed) ? "ai_review" : "unannotated";
  } else if (annotation.ground_truth.some(box => box.source === "ai" && !box.reviewed)) {
    annotation.status = "ai_review";
  }
  scheduleSave(item);
  updateFilterUi(itemsBeforeStatusFilter());
  updateProgress();
  setSaveState("saving", "Waiting to auto-save…");
}

function scheduleSave(item) {
  clearTimeout(state.saveTimers.get(item.id));
  const timer = setTimeout(() => {
    state.saveTimers.delete(item.id);
    persistItem(item);
  }, SAVE_DELAY_MS);
  state.saveTimers.set(item.id, timer);
}

async function flushSave(item) {
  if (!item) return;
  const timer = state.saveTimers.get(item.id);
  if (timer) {
    clearTimeout(timer);
    state.saveTimers.delete(item.id);
    await persistItem(item);
  } else if (state.pendingSaves.has(item.id)) {
    await state.pendingSaves.get(item.id);
  }
}

function persistItem(item) {
  const annotation = state.annotationCache.get(item.id);
  if (!annotation || !annotation.dirty) return Promise.resolve();
  const inFlight = state.pendingSaves.get(item.id);
  if (inFlight) return inFlight.then(() => persistItem(item));
  const version = (state.saveVersions.get(item.id) || 0) + 1;
  state.saveVersions.set(item.id, version);
  annotation.dirty = false;
  setSaveState("saving", "Saving…");
  const request = fetchJson(itemEndpoint(item, "annotation"), {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(annotationPayload(item, annotation)),
  }).then(() => {
    item.status = annotation.status;
    if (state.saveVersions.get(item.id) === version) setSaveState("saved", "Auto-saved");
  }).catch(error => {
    annotation.dirty = true;
    setSaveState("error", "Save failed");
    showToast(`Save failed: ${error.message}`, "error");
    console.error(error);
  }).finally(() => {
    if (state.pendingSaves.get(item.id) === request) state.pendingSaves.delete(item.id);
  });
  state.pendingSaves.set(item.id, request);
  return request;
}

function saveBeforeExit() {
  const item = currentItem();
  const annotation = currentAnnotation();
  if (!item || !annotation?.dirty) return;
  fetch(itemEndpoint(item, "annotation"), {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(annotationPayload(item, annotation)),
    keepalive: true,
  }).catch(() => {});
}

async function excludeCurrent() {
  const item = currentItem();
  if (!item || item.excluded) return;
  await flushSave(item);
  try {
    const saved = await fetchJson(itemEndpoint(item, "exclude"), { method: "POST" });
    const agxDeleted = Boolean(saved?.source?.agx_deleted_at);
    if (agxDeleted) {
      const deletedIndex = state.items.findIndex(candidate => candidate.id === item.id);
      state.items = state.items.filter(candidate => candidate.id !== item.id);
      state.annotationCache.delete(item.id);
      state.currentId = null;
      state.rgbImage = null;
      state.depthImage = null;
      populateSessionFilter();
      applyFilters(false);
      const next = state.filtered[Math.min(Math.max(deletedIndex, 0), state.filtered.length - 1)];
      if (next) await selectItem(next.id, false);
      else renderAll();
      showToast("Photo permanently deleted from this Mac and AGX");
      return;
    }
    item.excluded = true;
    const annotation = currentAnnotation();
    if (annotation) annotation.status = "excluded";
    updateFilterUi(itemsBeforeStatusFilter());
    renderAll();
    showToast("The RGB and matching depth image were moved to Trash and can be restored");
  } catch (error) {
    showToast(`Failed to delete photo: ${error.message}`, "error");
  }
}

async function restoreCurrent() {
  const item = currentItem();
  if (!item || !item.excluded) return;
  try {
    await fetchJson(itemEndpoint(item, "restore"), { method: "POST" });
    item.excluded = false;
    const annotation = currentAnnotation();
    if (annotation) annotation.status = annotation.reviewed ? "completed" : "unannotated";
    updateFilterUi(itemsBeforeStatusFilter());
    renderAll();
    showToast("Photo restored");
  } catch (error) {
    showToast(`Failed to restore photo: ${error.message}`, "error");
  }
}

function openBulkDialog() {
  if (!currentAnnotation()) return;
  $$('input[name="bulk-field"]').forEach(input => { input.checked = false; });
  $("#bulk-scope").value = "session";
  updateBulkSummary();
  $("#bulk-dialog").showModal();
}

function bulkTargets() {
  const item = currentItem();
  const scope = $("#bulk-scope").value;
  if (!item) return [];
  if (scope === "session") return state.items.filter(candidate => candidate.session === item.session && !candidate.excluded);
  if (scope === "following") {
    const index = state.filtered.findIndex(candidate => candidate.id === item.id);
    return state.filtered.slice(Math.max(0, index)).filter(candidate => !candidate.excluded);
  }
  return state.filtered.filter(candidate => !candidate.excluded);
}

function updateBulkSummary() {
  const checked = $$('input[name="bulk-field"]:checked').length;
  $("#bulk-summary").textContent = `Will overwrite ${checked} fields on ${bulkTargets().length} images using the current image`;
  $("#confirm-bulk").disabled = !checked || !bulkTargets().length;
}

async function applyBulkMetadata() {
  const annotation = currentAnnotation();
  if (!annotation) return;
  const fields = $$('input[name="bulk-field"]:checked').map(input => input.value);
  const targets = bulkTargets();
  if (!fields.length || !targets.length) return;
  const metadata = Object.fromEntries(fields.map(key => [key, annotation.metadata[key] ?? null]));
  const button = $("#confirm-bulk");
  button.disabled = true;
  button.textContent = "Applying…";
  try {
    await fetchJson("/api/bulk-metadata", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ids: targets.map(item => item.id),
        image_meta: metadata,
        item_ids: targets.map(item => item.id),
        metadata,
      }),
    });
    targets.forEach(item => {
      item.metadata = { ...(item.metadata || {}), ...metadata };
      const cached = state.annotationCache.get(item.id);
      if (cached) cached.metadata = { ...cached.metadata, ...metadata };
      if (metadata.building) item.building = metadata.building;
    });
    $("#bulk-dialog").close();
    populateSessionFilter();
    applyFilters(false);
    renderAll();
    showToast(`Applied to ${targets.length} images`);
  } catch (error) {
    showToast(`Bulk apply failed: ${error.message}`, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Apply";
  }
}

function handleKeyboard(event) {
  if ($("#bulk-dialog").open) return;
  const tag = event.target.tagName;
  const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(tag) || event.target.isContentEditable;
  if (typing) return;
  if (["ArrowLeft", "a", "A"].includes(event.key)) { event.preventDefault(); navigate(-1); return; }
  if (["ArrowRight", "d", "D"].includes(event.key)) { event.preventDefault(); navigate(1); return; }
  if (["Delete", "Backspace"].includes(event.key)) { event.preventDefault(); deleteSelectedBox(); return; }
  if (event.key === "Enter") { event.preventDefault(); toggleCompleted(); return; }
  if (event.key.toLowerCase() === "r") { setViewMode(state.viewMode === "rgb" ? "depth" : "rgb"); return; }
  if (/^[1-9]$/.test(event.key) && state.selectedBox >= 0) {
    event.preventDefault();
    applyLabel(QUICK_LABELS[Number(event.key) - 1]);
  }
}

function showCanvasMessage(message, error = false) {
  const element = $("#canvas-empty");
  element.textContent = message;
  element.classList.remove("hidden");
  element.classList.toggle("error", error);
  canvas.classList.add("hidden");
}

function hideCanvasMessage() {
  $("#canvas-empty").classList.add("hidden");
  canvas.classList.remove("hidden");
}

initialize();
