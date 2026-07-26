// Pipeline Console — frontend logic.
// Talks to the FastAPI backend (/api/*) and drives the stage rail + report panel.

const STAGES = [
  { key: "intake", label: "Patient intake" },
  { key: "nlp", label: "Medical NLP" },
  { key: "risk", label: "Risk prediction (ML)" },
  { key: "retrieval", label: "Evidence retrieval (RAG)" },
  { key: "drug_safety", label: "Drug safety" },
  { key: "guideline", label: "Guideline verification" },
  { key: "reasoning", label: "Clinical reasoning" },
  { key: "report", label: "Report generation" },
];

const $ = (id) => document.getElementById(id);

const railEl = $("rail");
const patientSelect = $("patientSelect");
const mockToggle = $("mockToggle");
const runBtn = $("runBtn");
const runBtnLabel = $("runBtnLabel");
const modeHint = $("modeHint");
const scanline = document.querySelector(".scanline");

function renderRail() {
  railEl.innerHTML = STAGES.map((s, i) => `
    <li class="rail__item" data-key="${s.key}" id="rail-${s.key}">
      <span class="rail__num">${String(i + 1).padStart(2, "0")}</span>
      <span class="rail__body">
        <span class="rail__label">${s.label}</span>
        <div class="rail__trace" id="trace-${s.key}"></div>
      </span>
    </li>
  `).join("");
}
renderRail();

function resetRail() {
  STAGES.forEach((s) => {
    const item = $(`rail-${s.key}`);
    item.classList.remove("is-active", "is-done");
    $(`trace-${s.key}`).textContent = "";
  });
}

async function checkHealth() {
  const dot = $("apiStatusDot");
  const text = $("apiStatusText");
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.local_model_ready) {
      dot.className = "dot dot--ok";
      text.textContent = "backend online · local model loaded";
      modeHint.style.display = "none";
    } else {
      dot.className = "dot dot--warn";
      text.textContent = "backend online · no local model found (mock mode only)";
      mockToggle.checked = true;
      mockToggle.disabled = true;
    }
  } catch (err) {
    dot.className = "dot dot--warn";
    text.textContent = "backend unreachable";
  }
}

async function loadPatients() {
  const res = await fetch("/api/patients");
  const patients = await res.json();
  patientSelect.innerHTML = patients.map((p) => `
    <option value="${p.patient_id}">${p.name} — ${p.age}${p.sex}, ${p.patient_id}</option>
  `).join("");
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function animateTrace(traceLines) {
  scanline.classList.add("active");
  for (let i = 0; i < STAGES.length; i++) {
    const stage = STAGES[i];
    const item = $(`rail-${stage.key}`);
    item.classList.add("is-active");
    await sleep(320);
    item.classList.remove("is-active");
    item.classList.add("is-done");
    const line = traceLines[i] || "";
    $(`trace-${stage.key}`).textContent = line.replace(/^\[.*?\]\s*/, "");
    await sleep(90);
  }
  scanline.classList.remove("active");
}

function riskClass(category) {
  if (category === "high") return "high";
  if (category === "moderate") return "moderate";
  return "low";
}

function priorityClass(priority) {
  if (priority === "urgent") return "urgent";
  if (priority === "elevated") return "elevated";
  return "routine";
}

function renderDefList(el, pairs) {
  el.innerHTML = pairs.map(([label, value]) => `
    <dt>${label}</dt>
    <dd>${value}</dd>
  `).join("");
}

function renderTags(el, items, emptyText) {
  if (!items || items.length === 0) {
    el.innerHTML = "";
    return;
  }
  el.innerHTML = items.map((i) => `<li>${i}</li>`).join("");
}

function renderReport(data) {
  const profile = data.patient_profile || {};
  const entities = data.extracted_entities || {};
  const risk = data.risk_result || {};
  const evidence = data.retrieved_evidence || [];
  const drugSafety = data.drug_safety_result || {};
  const verification = data.guideline_verification || {};
  const reasoning = data.clinical_reasoning || {};

  $("reportEyebrow").textContent = data.mock_mode ? "Case report · mock mode" : "Case report";
  $("patientName").textContent = `${profile.name || "Unknown"} · ${profile.patient_id || ""}`;
  $("patientMeta").textContent = `${profile.age ?? "—"} yr · ${profile.sex ?? "—"} · ${(profile.comorbidities || []).join(", ") || "no listed comorbidities"}`;

  // --- RISK SCORE UI ---
  const rClass = riskClass(risk.risk_category);
  const riskValueEl = $("riskValue");
  const percentage = Math.round((risk.risk_score || 0) * 100);
  riskValueEl.textContent = `${percentage}%`;
  riskValueEl.className = `risk-badge__value ${rClass}`;

  const fillEl = $("riskbarFill");
  fillEl.className = `riskbar__fill ${rClass}`;
  fillEl.style.width = `${Math.round((risk.risk_score || 0) * 100)}%`;

  // --- PROFILE & ENTITIES ---
  renderDefList($("entitiesList"), [
    ["Symptoms", (entities.symptoms || []).join(", ") || "None extracted"],
    ["Mentioned conditions", (entities.mentioned_conditions || []).join(", ") || "None"],
    ["Mentioned medications", (entities.mentioned_medications || []).join(", ") || "None"],
    ["Risk Factors", (entities.risk_factors || []).join(", ") || "None"],
    ["Family History", (entities.family_history || []).join(", ") || "None"],
  ]);

  // NEW: Confidence Badge Logic
  const confidence = entities.extraction_confidence || "medium"; // Default to medium if missing
  const confColor = confidence === "high" ? "#10b981" : confidence === "low" ? "#ef4444" : "#f59e0b";

  renderDefList($("entitiesList"), [
    ["Symptoms", (entities.symptoms || []).join(", ") || "None extracted"],
    ["Mentioned conditions", (entities.mentioned_conditions || []).join(", ") || "None"],
    ["Mentioned medications", (entities.mentioned_medications || []).join(", ") || "None"],
    ["Notable flags", (entities.notable_flags || []).join(", ") || "None"],
    ["Extraction Confidence", `<span style="background:${confColor}; color:white; padding:2px 8px; border-radius:12px; font-size:0.85em; font-weight:bold;">${confidence.toUpperCase()}</span>`],
  ]);

  renderTags($("riskFactors"), risk.top_factors || []);

  // --- DRUG SAFETY ---
  const alerts = drugSafety.interactions || [];
  const alertsEl = $("drugAlerts");
  if (alerts.length === 0) {
    alertsEl.innerHTML = `<li class="alert-none">No interaction flags found among ${(drugSafety.medications_considered || []).length} medication(s) checked.</li>`;
  } else {
    alertsEl.innerHTML = alerts.map((a) => `
      <li><strong>${a.drug_a} + ${a.drug_b}</strong> (${a.severity} severity) — ${a.description}</li>
    `).join("");
  }

  // --- EVIDENCE ---
  const evEl = $("evidenceList");
  if (evidence.length === 0) {
    evEl.innerHTML = `<p class="prose">No relevant evidence retrieved.</p>`;
  } else {
    evEl.innerHTML = evidence.map((e) => `
      <div class="evidence__item">
        <div class="evidence__source"><span>${e.source}</span><span>relevance ${e.score}</span></div>
        <p class="evidence__snippet">${e.snippet}</p>
      </div>
    `).join("");
  }

  $("verificationNotes").textContent = verification.notes || "—";
  renderTags($("verificationCitations"), verification.citations || []);

  // --- CLINICAL REASONING & CITATIONS ---
  const chip = $("priorityChip");
  chip.textContent = (reasoning.priority || "—").toUpperCase();
  chip.className = `priority-chip ${priorityClass(reasoning.priority)}`;
  $("assessmentText").textContent = reasoning.assessment || "—";
  $("recommendationsList").innerHTML = (reasoning.recommendations || [])
    .map((r) => `<li>${r}</li>`).join("") || "<li>No recommendations generated.</li>";

  // NEW: Citations List
  const citations = reasoning.citations || [];
  const citationsEl = $("citationsList"); // Make sure this ID exists in HTML, or we append to recommendations
  if (citations.length > 0) {
    const citationsHtml = `
      <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #e5e7eb;">
        <h4 style="font-size: 0.85em; text-transform: uppercase; color: #6b7280; margin-bottom: 10px;">Sources & Citations</h4>
        <ul style="list-style-type: none; padding-left: 0;">
          ${citations.map(c => `<li style="font-size: 0.9em; color: #4b5563; margin-bottom: 6px;">📄 ${c}</li>`).join("")}
        </ul>
      </div>
    `;
    // Append citations to the recommendations list container
    $("recommendationsList").parentElement.innerHTML += citationsHtml;
  }

  $("traceLog").textContent = (data.trace || []).join("\n");
}

async function runPipeline() {
  const patientId = patientSelect.value;
  if (!patientId) return;

  runBtn.disabled = true;
  runBtnLabel.textContent = "Running…";
  resetRail();

  $("reportEmpty").hidden = true;
  $("reportBody").hidden = true;

  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patient_id: patientId, mock: mockToggle.checked }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    await animateTrace(data.trace || []);
    renderReport(data);

    $("reportBody").hidden = false;
  } catch (err) {
    $("reportEmpty").hidden = false;
    $("reportEmpty").innerHTML = `
      <p class="report__empty-eyebrow">Run failed</p>
      <h1>Something went wrong.</h1>
      <p class="report__empty-body">${err.message}</p>
    `;
  } finally {
    runBtn.disabled = false;
    runBtnLabel.textContent = "Run pipeline";
  }
}

runBtn.addEventListener("click", runPipeline);

(async function init() {
  await Promise.all([checkHealth(), loadPatients()]);
})();
