// DermaScan frontend logic

const state = {
  sessionId: null,
  images: {},
  answers: {},
  questions: [],
  location: null,
};

const $ = (id) => document.getElementById(id);

// --- Geolocation (best effort, non-blocking) ---
if (navigator.geolocation) {
  navigator.geolocation.getCurrentPosition(
    (pos) => { state.location = { lat: pos.coords.latitude, lng: pos.coords.longitude }; },
    () => {},
    { timeout: 4000 }
  );
}

// --- Upload handling ---
const dropzone = $("dropzone");
const fileInput = $("file-input");
const cameraInput = $("camera-input");

$("browse-btn").addEventListener("click", () => fileInput.click());
$("camera-btn").addEventListener("click", () => cameraInput.click());
fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));
cameraInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); })
);
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

async function handleFile(file) {
  if (!file) return;
  const form = new FormData();
  form.append("image", file);
  if (state.sessionId) form.append("session_id", state.sessionId);
  if (state.location) {
    form.append("lat", state.location.lat);
    form.append("lng", state.location.lng);
  }

  showAnalysis();
  $("disease-name").textContent = "Analysing…";

  try {
    const res = await fetch("/api/analyze", { method: "POST", body: form });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderResult(data);
  } catch (err) {
    $("disease-name").textContent = "Something went wrong";
    $("disease-desc").textContent = err.message + " Please try another image.";
  }
}

function showAnalysis() {
  $("analysis-section").classList.remove("hidden");
  $("analysis-section").scrollIntoView({ behavior: "smooth" });
}

// --- Render prediction ---
function renderResult(data) {
  state.sessionId = data.session_id;
  state.images = data.images;

  setImage("original");
  document.querySelectorAll(".img-tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.view === "original")
  );

  const p = data.prediction;
  $("disease-name").textContent = p.disease_name;
  $("disease-desc").textContent = p.description;
  $("confidence").textContent = `${p.confidence}% confidence`;

  const pill = $("severity-pill");
  pill.textContent = p.severity;
  pill.className = "pill sev-" + p.severity;

  $("meter-fill").style.width = p.confidence + "%";

  const top3 = $("top3");
  top3.innerHTML = "";
  p.top3.forEach((t) => {
    const row = document.createElement("div");
    row.className = "top3-row";
    row.innerHTML = `<span>${t.name}</span><span>${t.confidence}%</span>`;
    top3.appendChild(row);
  });

  const prec = $("precautions");
  prec.innerHTML = "";
  p.precautions.forEach((line) => {
    const li = document.createElement("li");
    li.textContent = line;
    prec.appendChild(li);
  });

  renderDerm(data.recommendation);
  loadQuestions();
}

function setImage(view) {
  const img = $("stage-img");
  img.src = state.images[view];
  const caps = {
    original: "Uploaded image.",
    processed: "After OpenCV preprocessing (hair removal, colour normalisation, denoise).",
    gradcam: "Grad-CAM focus map — regions that most influenced the model.",
  };
  $("img-caption").textContent = caps[view];
}

document.querySelectorAll(".img-tab").forEach((tab) =>
  tab.addEventListener("click", () => {
    document.querySelectorAll(".img-tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    setImage(tab.dataset.view);
  })
);

// --- Dermatologist card ---
function renderDerm(rec) {
  const card = $("derm-card");
  card.classList.remove("hidden");
  $("derm-message").textContent = rec.message;
  $("maps-link").href = rec.maps_url;
  $("derm-urgent").classList.toggle("hidden", !rec.urgent);

  const list = $("clinic-list");
  list.innerHTML = "";
  rec.fallback_clinics.forEach((c) => {
    const li = document.createElement("li");
    li.textContent = `${c.name} — ${c.phone}`;
    list.appendChild(li);
  });
}

// --- Chatbot questionnaire ---
async function loadQuestions() {
  if (state.questions.length === 0) {
    const res = await fetch("/api/questions");
    state.questions = await res.json();
  }
  const box = $("questionnaire");
  box.innerHTML = "";
  state.answers = {};

  state.questions.forEach((q) => {
    const block = document.createElement("div");
    block.className = "q-block";
    block.innerHTML = `<div class="q-text">${q.text}</div>`;
    const opts = document.createElement("div");
    opts.className = "q-options";
    q.options.forEach((opt) => {
      const b = document.createElement("button");
      b.className = "q-opt";
      b.textContent = opt;
      b.addEventListener("click", () => {
        opts.querySelectorAll(".q-opt").forEach((o) => o.classList.remove("selected"));
        b.classList.add("selected");
        state.answers[q.id] = opt;
        $("get-advice").disabled = Object.keys(state.answers).length < state.questions.length;
      });
      opts.appendChild(b);
    });
    block.appendChild(opts);
    box.appendChild(block);
  });

  $("chat-reply").classList.add("hidden");
  $("get-advice").disabled = true;
}

$("get-advice").addEventListener("click", async () => {
  const btn = $("get-advice");
  btn.disabled = true;
  btn.textContent = "Thinking…";
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, answers: state.answers }),
    });
    const data = await res.json();
    const reply = $("chat-reply");
    reply.classList.remove("hidden");
    reply.className = "chat-reply" + (data.urgency === "high" ? " urg-high" : "");
    // Render **bold** markers
    reply.innerHTML = data.reply
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  } finally {
    btn.textContent = "Get guidance";
    btn.disabled = false;
  }
});

// --- New scan ---
$("new-scan").addEventListener("click", () => {
  $("analysis-section").classList.add("hidden");
  fileInput.value = "";
  cameraInput.value = "";
  $("upload-section").scrollIntoView({ behavior: "smooth" });
});
