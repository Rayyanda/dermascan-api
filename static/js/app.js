// DermaScan AI Studio — dashboard client logic. Talks only to /api/*.

const API = "";

// ---------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------
document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`view-${btn.dataset.view}`).classList.add("active");

    if (btn.dataset.view === "dashboard") loadDashboard();
    if (btn.dataset.view === "dataset") loadDataset();
    if (btn.dataset.view === "training") loadTraining();
    if (btn.dataset.view === "models") loadModels();
  });
});

function badge(status) {
  return `<span class="badge badge-${status}">${status}</span>`;
}

function fmtPct(x) {
  return x === null || x === undefined ? "—" : `${(x * 100).toFixed(1)}%`;
}

function fmtDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString();
}

// ---------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------
async function loadDashboard() {
  const res = await fetch(`${API}/api/dashboard/summary`);
  const data = await res.json();

  document.getElementById("stat-total-images").textContent = data.total_dataset_images;
  document.getElementById("stat-total-runs").textContent = data.total_training_runs;
  document.getElementById("stat-total-predictions").textContent = data.total_predictions_served;
  document.getElementById("stat-deployed-model").textContent = data.active_model
    ? data.active_model.filename
    : "none deployed";

  document.getElementById("pipeline-status").textContent = data.active_model
    ? "model live"
    : "no model deployed";

  const runBody = document.getElementById("last-run-body");
  if (!data.last_training_run) {
    runBody.innerHTML = `<p class="muted">No training runs yet. Head to <strong>Training</strong> to start one.</p>`;
  } else {
    const r = data.last_training_run;
    runBody.innerHTML = `
      <table class="data-table">
        <tbody>
          <tr><td>Status</td><td>${badge(r.status)}</td></tr>
          <tr><td>Val Accuracy</td><td class="mono">${fmtPct(r.val_accuracy)}</td></tr>
          <tr><td>Val Loss</td><td class="mono">${r.val_loss ?? "—"}</td></tr>
          <tr><td>Epochs</td><td class="mono">${r.epochs}</td></tr>
          <tr><td>Started</td><td>${fmtDate(r.started_at)}</td></tr>
        </tbody>
      </table>`;
  }
}

// ---------------------------------------------------------------------
// Dataset Manager
// ---------------------------------------------------------------------
async function loadDataset() {
  const classesRes = await fetch(`${API}/api/dataset/classes`);
  const { classes } = await classesRes.json();
  const select = document.getElementById("upload-class");
  if (!select.dataset.filled) {
    select.innerHTML = classes.map((c) => `<option value="${c}">${c}</option>`).join("");
    select.dataset.filled = "1";
  }

  const statsRes = await fetch(`${API}/api/dataset/stats`);
  const stats = await statsRes.json();

  const tbody = document.querySelector("#class-table tbody");
  tbody.innerHTML = stats.per_class.map((c) => `
    <tr>
      <td class="mono">${c.class_label}</td>
      <td>${c.full_label}</td>
      <td class="mono">${c.image_count}</td>
      <td>${c.image_count >= stats.min_images_per_class_required ? "✅" : "—"}</td>
    </tr>
  `).join("");

  document.getElementById("ready-status").textContent = stats.ready_to_train
    ? `Ready to train — ${stats.total_images} total images.`
    : `Not ready yet — every class needs at least ${stats.min_images_per_class_required} images.`;
}

document.getElementById("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const classLabel = document.getElementById("upload-class").value;
  const files = document.getElementById("upload-files").files;
  if (!files.length) return;

  const formData = new FormData();
  formData.append("class_label", classLabel);
  for (const f of files) formData.append("files", f);

  const resultBox = document.getElementById("upload-result");
  resultBox.textContent = "Uploading...";

  const res = await fetch(`${API}/api/dataset/upload`, { method: "POST", body: formData });
  const data = await res.json();
  resultBox.textContent = `Uploaded: ${data.uploaded}, Skipped: ${data.skipped}\n${data.details.join("\n")}`;
  loadDataset();
});

document.getElementById("sync-btn").addEventListener("click", async () => {
  const resultBox = document.getElementById("sync-result");
  resultBox.textContent = "Scanning data/dataset/ for new images...";

  const res = await fetch(`${API}/api/dataset/sync`, { method: "POST" });
  const data = await res.json();
  resultBox.textContent = `Registered: ${data.registered} new image(s). Unknown-class folders: ${data.unknown_class_folders}\n${data.details.slice(0, 50).join("\n")}${data.details.length > 50 ? `\n...and ${data.details.length - 50} more` : ""}`;
  loadDataset();
});

// ---------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------
async function loadTraining() {
  const res = await fetch(`${API}/api/training/history`);
  const runs = await res.json();

  const tbody = document.querySelector("#training-table tbody");
  tbody.innerHTML = runs.map((r) => `
    <tr>
      <td class="mono">#${r.id}</td>
      <td>${badge(r.status)}</td>
      <td class="mono">${r.epochs}</td>
      <td class="mono">${fmtPct(r.val_accuracy)}</td>
      <td class="mono">${r.val_loss ?? "—"}</td>
      <td>${fmtDate(r.started_at)}</td>
    </tr>
  `).join("") || `<tr><td colspan="6" class="muted">No runs yet.</td></tr>`;
}

document.getElementById("training-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = {
    epochs: parseInt(document.getElementById("train-epochs").value, 10),
    batch_size: parseInt(document.getElementById("train-batch").value, 10),
    learning_rate: parseFloat(document.getElementById("train-lr").value),
  };

  const msgBox = document.getElementById("training-message");
  msgBox.textContent = "Starting training run...";

  const res = await fetch(`${API}/api/training/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const err = await res.json();
    msgBox.textContent = `Error: ${err.detail}`;
    return;
  }

  const run = await res.json();
  msgBox.textContent = `Training run #${run.id} started (status: ${run.status}). This runs in the background — refresh Training History to check progress.`;
  loadTraining();
});

// ---------------------------------------------------------------------
// Models & Deployment
// ---------------------------------------------------------------------
async function loadModels() {
  const res = await fetch(`${API}/api/models`);
  const versions = await res.json();

  const tbody = document.querySelector("#models-table tbody");
  tbody.innerHTML = versions.map((m) => `
    <tr>
      <td class="mono">${m.filename}</td>
      <td class="mono">${fmtPct(m.val_accuracy)}</td>
      <td>${fmtDate(m.created_at)}</td>
      <td>${m.is_deployed ? badge("deployed") : "—"}</td>
      <td>
        ${m.is_deployed
          ? ""
          : `<button class="btn btn-secondary deploy-btn" data-id="${m.id}">Deploy</button>`}
      </td>
    </tr>
  `).join("") || `<tr><td colspan="5" class="muted">No models trained yet.</td></tr>`;

  document.querySelectorAll(".deploy-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      await fetch(`${API}/api/models/${btn.dataset.id}/deploy`, { method: "POST" });
      loadModels();
    });
  });
}

// ---------------------------------------------------------------------
// Initial load
// ---------------------------------------------------------------------
loadDashboard();
