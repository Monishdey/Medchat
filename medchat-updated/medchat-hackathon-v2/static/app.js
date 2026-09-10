/* MedChat front end. Plain JavaScript, no framework, no build step. */

const $ = (id) => document.getElementById(id);

const thread = $("thread");
const input = $("input");
const sendBtn = $("sendBtn");
const menu = $("menu");

let activeModel = "T1";
let attachment = null;   // { file, preview, found, text }
let busy = false;

/* ------------------------------------------------------------ helpers */

function classKey(label) {
  if (label.includes("Primary")) return "primary_hypothyroid";
  if (label.includes("Compensated")) return "compensated";
  return "negative";
}

function clearWelcome() {
  const w = $("welcome");
  if (w) w.remove();
}

function scrollDown() {
  thread.parentElement.scrollTop = thread.parentElement.scrollHeight;
}

function addUser(text, imageUrl) {
  clearWelcome();
  const div = document.createElement("div");
  div.className = "msg user";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (imageUrl) {
    const img = document.createElement("img");
    img.src = imageUrl;
    bubble.appendChild(img);
  }
  bubble.appendChild(document.createTextNode(text));
  div.appendChild(bubble);
  thread.appendChild(div);
  scrollDown();
}

function addThinking() {
  const div = document.createElement("div");
  div.className = "msg bot thinking";
  div.id = "thinking";
  div.textContent = "Checking...";
  thread.appendChild(div);
  scrollDown();
}

function removeThinking() {
  const t = $("thinking");
  if (t) t.remove();
}

/* ------------------------------------------------------- render replies */

/* ------------------------------------------------------- render replies */

/* ------------------------------------------------------- render replies */

function addText(answer, source, related) {
  const div = document.createElement("div");
  div.className = "msg bot";
  
  const textContainer = document.createElement("span");
  div.appendChild(textContainer);
  
  thread.appendChild(div);
  
  let i = 0;
  const typingSpeed = 25; // Speed in milliseconds (lower is faster)

  function typeWriter() {
    if (i < answer.length) {
      textContainer.textContent += answer.charAt(i);
      i++;
      scrollDown(); 
      setTimeout(typeWriter, typingSpeed);
    } else {
      if (source) {
        const s = document.createElement("div");
        s.className = "src";
        s.textContent = "From the knowledge base: " + source;
        div.appendChild(s);
      }

      if (related && related.length) {
        const wrap = document.createElement("div");
        wrap.className = "related";
        related.forEach((q) => {
          const b = document.createElement("button");
          b.textContent = q;
          b.onclick = () => { input.value = q; send(); };
          wrap.appendChild(b);
        });
        div.appendChild(wrap);
      }
      scrollDown();
    }
  }
  
  typeWriter();
}

function addPrediction(result) {
  const key = classKey(result.label);

  const card = document.createElement("div");
  card.className = "card " + key;

  const top = document.createElement("div");
  top.className = "card-top";
  top.innerHTML =
    "<span>MODEL T1</span><span>" +
    (result.confidence * 100).toFixed(1) + "% confident</span>";
  card.appendChild(top);

  const body = document.createElement("div");
  body.className = "card-body";

  const verdict = document.createElement("div");
  verdict.className = "verdict " + key;
  verdict.textContent = result.label;
  body.appendChild(verdict);

  result.probabilities.forEach((p) => {
    const k = classKey(p.label);
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML =
      '<div class="bar-label"><span>' + p.label + "</span><span>" +
      (p.value * 100).toFixed(1) + '%</span></div>' +
      '<div class="bar-track"><div class="bar-fill ' + k +
      '" style="width:' + Math.max(p.value * 100, 1.5) + '%"></div></div>';
    body.appendChild(row);
  });

  const tags = document.createElement("div");
  tags.className = "tags";
  Object.entries(result.values_used).forEach(([k, v]) => {
    const t = document.createElement("span");
    t.className = "tag";
    t.innerHTML = k + " <b>" + v + "</b>";
    tags.appendChild(t);
  });
  body.appendChild(tags);

  const reasons = document.createElement("div");
  reasons.className = "reasons";
  result.reasons.forEach((r) => {
    const d = document.createElement("div");
    d.textContent = "— " + r;
    reasons.appendChild(d);
  });
  body.appendChild(reasons);

  if (result.missing && result.missing.length) {
    const note = document.createElement("div");
    note.className = "note";
    note.textContent =
      "Ran without " + result.missing.join(", ") +
      ". A typical value was used instead, so this is less precise than it could be.";
    body.appendChild(note);
  }

  card.appendChild(body);

  const wrap = document.createElement("div");
  wrap.className = "msg bot";
  wrap.appendChild(card);

  const followUp = document.createElement("div");
  followUp.textContent =
    "This is a screening estimate from a model trained on old research data, " +
    "not a diagnosis. Please take these numbers to a doctor.";
  wrap.appendChild(followUp);

  thread.appendChild(wrap);
  scrollDown();
}

/* ---------------------------------------------------------------- send */

async function send() {
  const typed = input.value.trim();
  if ((!typed && !attachment) || busy) return;

  let message = typed;
  let preview = null;

  if (attachment) {
    preview = attachment.preview;
    const pairs = Object.entries(attachment.found)
      .filter(([k, v]) => String(v).trim() !== "")
      .map(([k, v]) => (k === "sex" ? v : k + " " + v))
      .join(", ");
    // Feed the values read off the image into the same text pipeline.
    message = (pairs ? pairs + ". " : "") + typed;
    if (!pairs && !typed) message = "I uploaded a report but no values were found.";
  }

  addUser(typed || "(uploaded a lab report)", preview);
  // Blob URLs do not survive a page reload, so we store a marker
  // rather than a dead link.
  remember("user", { text: typed || "(uploaded a lab report)",
                     hadImage: !!preview });

  input.value = "";
  input.style.height = "auto";
  clearAttachment();
  busy = true;
  sendBtn.disabled = true;
  addThinking();

  // In clinician mode, two or more non-empty lines means a batch of patients,
  // so it goes to the triage endpoint instead of the normal chat route.
  const lines = message.split("\n").map((l) => l.trim())
                       .filter((l) => l && !l.startsWith("#"));
  const isBatch = doctorMode && lines.length >= 2;

  try {
    const res = await fetch(isBatch ? "/api/triage" : "/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(
        isBatch
          ? { text: message }
          : { message: message, model_id: activeModel, mode: doctorMode ? "doctor" : "patient" }
      ),
    });
    const data = await res.json();
    removeThinking();

    if (data.type === "triage") {
      addWorklist(data);
      remember("bot", { kind: "triage", data: data });
    } else if (data.type === "prediction") {
      addPrediction(data);
      remember("bot", { kind: "prediction", data: data });
    } else {
      addText(data.answer, data.source, data.related);
      remember("bot", { kind: "text", text: data.answer,
                        source: data.source, related: data.related });
    }
  } catch (err) {
    removeThinking();
    addText("Could not reach the server. Is app.py still running?");
  }

  busy = false;
  sendBtn.disabled = false;
  input.focus();
}

/* ---------------------------------------------------------- attachment */

function clearAttachment() {
  attachment = null;
  const bar = $("attachBar");
  bar.className = "attach-bar hidden";
  bar.innerHTML = "";
}

async function handleFile(file) {
  const bar = $("attachBar");
  bar.classList.remove("hidden");
  bar.textContent = "Reading the image...";

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();

    if (data.error) {
      bar.textContent = data.error;
      setTimeout(clearAttachment, 4000);
      return;
    }

    attachment = {
      preview: URL.createObjectURL(file),
      found: data.found,
      details: data.details || [],
      warnings: data.warnings || [],
      text: data.text,
    };

    renderReview(file.name);
  } catch (err) {
    bar.textContent = "Upload failed.";
    setTimeout(clearAttachment, 3000);
  }
}

/*
  Show every value we read off the image as an EDITABLE box.

  This is the safety net. OCR regularly drops a decimal point, turning 3.3 into
  33 — which is the difference between a healthy thyroid and a severely
  underactive one. Nothing reaches the model until the user has seen it.
*/
function renderReview(filename) {
  const bar = $("attachBar");
  bar.innerHTML = "";
  bar.className = "attach-bar review";

  const head = document.createElement("div");
  head.className = "review-head";
  head.innerHTML =
    "<img src='" + attachment.preview + "' alt=''>" +
    "<div><div class='review-title'>" + filename + "</div>" +
    "<div class='review-sub'>Check these against your report, then edit anything wrong.</div></div>";
  const x = document.createElement("span");
  x.className = "x";
  x.textContent = "\u00d7";
  x.onclick = clearAttachment;
  head.appendChild(x);
  bar.appendChild(head);

  const keys = Object.keys(attachment.found);
  if (keys.length === 0) {
    const none = document.createElement("div");
    none.className = "review-sub";
    none.textContent =
      "No lab values recognised. Try a sharper photo taken straight on, " +
      "or just type your values into the box below.";
    bar.appendChild(none);
    return;
  }

  const grid = document.createElement("div");
  grid.className = "review-grid";

  keys.forEach((key) => {
    const detail = attachment.details.find((d) => d.field === key);

    const cell = document.createElement("div");
    cell.className = "review-cell" + (detail && detail.check ? " flagged" : "");

    const label = document.createElement("label");
    label.textContent = key;

    const box = document.createElement("input");
    box.type = "text";
    box.value = attachment.found[key];
    box.dataset.key = key;
    box.oninput = () => { attachment.found[key] = box.value; };

    cell.append(label, box);

    // If we converted the units, say so. The user sees 98 on their report and
    // 126 here, and needs to know why.
    if (detail && detail.converted) {
      const note = document.createElement("div");
      note.className = "review-conv";
      note.textContent = "was " + detail.raw + " " + detail.unit;
      cell.appendChild(note);
    }

    grid.appendChild(cell);
  });

  bar.appendChild(grid);

  attachment.warnings.forEach((w) => {
    const warn = document.createElement("div");
    warn.className = "review-warn";
    warn.textContent = w;
    bar.appendChild(warn);
  });
}

/* -------------------------------------------------------------- wiring */

sendBtn.onclick = send;

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
});

$("attachBtn").onclick = () => $("fileInput").click();
$("fileInput").onchange = (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
  e.target.value = "";
};

document.querySelectorAll(".starter").forEach((b) => {
  b.onclick = () => { input.value = b.textContent; send(); };
});

/* model switcher */
$("switchBtn").onclick = (e) => { e.stopPropagation(); menu.classList.toggle("hidden"); };
document.addEventListener("click", () => menu.classList.add("hidden"));

fetch("/api/models").then((r) => r.json()).then((data) => {
  menu.innerHTML = '<div class="menu-title">Disease model</div>';
  data.models.forEach((m) => {
    const btn = document.createElement("button");
    btn.className = "menu-item" + (m.ready ? "" : " locked");
    btn.disabled = !m.ready;
    btn.innerHTML =
      '<span class="tick">' + (m.id === activeModel && m.ready ? "\u2713" : "") + "</span>" +
      "<span><span class='name'>" + m.name + "</span> " +
      '<span class="badge">' + m.area + "</span>" +
      (m.ready ? "" : ' <span class="badge soon">Coming soon</span>') +
      '<div class="note">' + m.note + "</div></span>";
    if (m.ready) {
      btn.onclick = () => {
        activeModel = m.id;
        $("activeId").textContent = m.id;
        $("activeArea").textContent = m.area;
        menu.classList.add("hidden");
      };
    }
    menu.appendChild(btn);
  });
});

/* scorecard */
fetch("/api/scorecard").then((r) => r.json()).then((s) => {
  $("scoreBtn").textContent = "T1 \u00b7 F1 " + s.macro_f1;

  const feats = s.top_features
    .map((f) =>
      '<div class="feat-row"><span class="feat-name">' + f.name + "</span>" +
      '<span class="feat-track"><span class="feat-fill" style="width:' +
      f.importance * 100 + '%"></span></span>' +
      '<span class="feat-name" style="width:34px;text-align:right">' +
      (f.importance * 100).toFixed(1) + "</span></div>")
    .join("");

  $("modalBody").innerHTML =
    '<div class="stat-grid">' +
    '<div><div class="stat-label">Macro F1</div><div class="stat-value">' + s.macro_f1 + "</div></div>" +
    '<div><div class="stat-label">Cross-val</div><div class="stat-value">' + s.cv_mean + "</div></div>" +
    '<div><div class="stat-label">Tested on</div><div class="stat-value">' + s.n_test + "</div></div>" +
    "</div>" +
    '<div class="stat-label" style="margin-bottom:8px">What the model looks at most</div>' +
    feats +
    '<div class="modal-foot">Trained on ' + s.n_train +
    " patient records and tested on " + s.n_test +
    " it had never seen. Macro F1 is used instead of accuracy because most " +
    "patients in the data are healthy, so a lazy model could score 92% by " +
    "always answering healthy.</div>";
});

$("scoreBtn").onclick = () => $("modal").classList.remove("hidden");
$("modalClose").onclick = () => $("modal").classList.add("hidden");
$("modal").onclick = (e) => { if (e.target.id === "modal") $("modal").classList.add("hidden"); };

input.focus();
$("modal").onclick = (e) => { if (e.target.id === "modal") $("modal").classList.add("hidden"); };

input.focus();

/* -------- PASTE THE NEW CODE STARTING HERE -------- */
/* ------------------------------------------------------------- language switcher */
const translations = {
  en: {
    title: "What can I help you look at?",
    sub: "Paste your thyroid blood results and I'll assess them, or ask me anything about how the thyroid works. You can also upload a photo of a lab report.",
    newChat: "New chat",
    placeholder: "Type your results, or ask a question...",
    start1: "What is TSH and why does it matter?",
    start2: "What is subclinical hypothyroidism?",
    start3: "How accurate is this model?",
    disclaimer: "MedChat is a student project. It cannot diagnose you. Please talk to a doctor.",
    history: "History",
    modeTitle: "Clinician mode",
    modeSub: "Batch triage & clinical notes",
    clearAll: "Clear all history"
  },
  hi: {
    title: "मैं आपकी क्या मदद कर सकता हूँ?",
    sub: "अपनी थायरॉयड रक्त रिपोर्ट यहाँ पेस्ट करें, या थायरॉयड के बारे में कुछ भी पूछें। आप लैब रिपोर्ट की फोटो भी अपलोड कर सकते हैं।",
    newChat: "नई चैट",
    placeholder: "अपने परिणाम टाइप करें, या कोई प्रश्न पूछें...",
    start1: "TSH क्या है और यह क्यों महत्वपूर्ण है?",
    start2: "सबक्लिनिकल हाइपोथायरायडिज्म क्या है?",
    start3: "यह मॉडल कितना सटीक है?",
    disclaimer: "MedChat एक छात्र प्रोजेक्ट है। यह आपका निदान नहीं कर सकता। कृपया डॉक्टर से बात करें।",
    history: "इतिहास",
    modeTitle: "चिकित्सक मोड",
    modeSub: "बैच ट्राइएज और क्लिनिकल नोट्स",
    clearAll: "सारा इतिहास मिटाएँ"
  },
  as: {
    title: "মই আপোনাক কিহত সহায় কৰিব পাৰোঁ?",
    sub: "আপোনাৰ থাইৰয়ড তেজৰ ৰিজাল্ট পেষ্ট কৰক আৰু মই পৰীক্ষা কৰিম, বা থাইৰয়ডৰ বিষয়ে যিকোনো কথা সোধক। আপুনি লেব ৰিপৰ্টৰ ফটোও আপলোড কৰিব পাৰে।",
    newChat: "নতুন চেট",
    placeholder: "আপোনাৰ ৰিজাল্ট টাইপ কৰক, বা এটা প্ৰশ্ন সোধক...",
    start1: "TSH কি আৰু ই কিয় গুৰুত্বপূৰ্ণ?",
    start2: "চাবক্লিনিকেল হাইপোথাইৰয়ডিজম কি?",
    start3: "এই মডেলটো কিমান সঠিক?",
    disclaimer: "MedChat এটা ছাত্ৰ প্ৰজেক্ট। ই আপোনাৰ ৰোগ নিৰ্ণয় কৰিব নোৱাৰে। অনুগ্ৰহ কৰি ডাক্তৰৰ সৈতে কথা পাতক।",
    history: "ইতিহাস",
    modeTitle: "চিকিৎসক ম’ড",
    modeSub: "বেচ ট্ৰাইএজ আৰু ক্লিনিকেল নোট",
    clearAll: "সকলো ইতিহাস মচক"
  },
  bn: {
    title: "আমি আপনাকে কী সাহায্য করতে পারি?",
    sub: "আপনার থাইরয়েড রক্তের ফলাফল পেস্ট করুন এবং আমি সেগুলি মূল্যায়ন করব, বা থাইরয়েড সম্পর্কে যে কোনও কিছু জিজ্ঞাসা করুন। আপনি ল্যাব রিপোর্টের ছবিও আপলোড করতে পারেন।",
    newChat: "নতুন চ্যাট",
    placeholder: "আপনার ফলাফল টাইপ করুন, বা একটি প্রশ্ন জিজ্ঞাসা করুন...",
    start1: "TSH কী এবং এটি কেন গুরুত্বপূর্ণ?",
    start2: "সাবক্লিনিক্যাল হাইপোথাইরয়েডিজম কী?",
    start3: "এই মডেলটি কতটা নির্ভুল?",
    disclaimer: "MedChat একটি স্টুডেন্ট প্রজেক্ট। এটি আপনাকে ডায়াগনোজ করতে পারে খন। অনুগ্রহ করে একজন ডাক্তারের সাথে কথা বলুন।",
    history: "ইতিহাস",
    modeTitle: "চিকিৎসক মোড",
    modeSub: "ব্যাচ ট্রায়াজ ও ক্লিনিক্যাল নোট",
    clearAll: "সব ইতিহাস মুছুন"
  }
};

const langSelect = document.getElementById("langSelect");

/* Current language, so we can re-apply it after "New chat" rebuilds the
   welcome block and its element ids. */
let currentLang = "en";

function applyLanguage(lang) {
  const t = translations[lang];
  if (!t) return;
  currentLang = lang;

  const set = (id, value) => {
    const el = document.getElementById(id);
    if (el && value) el.textContent = value;
  };

  set("welcomeTitle", t.title);
  set("welcomeSub", t.sub);
  set("resetBtn", t.newChat);
  set("start1", t.start1);
  set("start2", t.start2);
  set("start3", t.start3);

  // Sidebar strings (added with the history panel).
  set("newChatLabel", t.newChat);
  set("historyLabel", t.history);
  set("modeTitle", t.modeTitle);
  set("modeSub", t.modeSub);
  set("clearAllBtn", t.clearAll);

  const box = document.getElementById("input");
  // Clinician mode owns the placeholder while it is on.
  if (box && !(typeof doctorMode !== "undefined" && doctorMode)) {
    box.placeholder = t.placeholder;
  }

  const disclaimer = document.querySelector(".disclaimer");
  if (disclaimer && !(typeof doctorMode !== "undefined" && doctorMode)) {
    disclaimer.textContent = t.disclaimer;
  }

  document.documentElement.lang = lang;
}

if (langSelect) {
  langSelect.addEventListener("change", (e) => applyLanguage(e.target.value));
}
/* ==========================================================================
   PART 2 — SIDEBAR, CHAT HISTORY, AND CLINICIAN MODE
   ==========================================================================

   HOW HISTORY IS STORED
   ---------------------
   In the browser's localStorage, which means it lives on YOUR computer and
   never touches the server. Refreshing keeps it, clearing your browser
   removes it, and nobody else can see it. For a health app that is the
   right trade-off: you get history without us building a patient database.
*/

const STORE_KEY = "medchat.history.v1";
const MODE_KEY = "medchat.mode.v1";

let history = [];        // all saved conversations
let currentId = null;    // the one on screen
let doctorMode = false;

/* ------------------------------------------------------------- storage */

function loadHistory() {
  try {
    history = JSON.parse(localStorage.getItem(STORE_KEY) || "[]");
  } catch {
    history = [];
  }
}

function saveHistory() {
  try {
    // Keep the 50 most recent so storage cannot grow without limit.
    localStorage.setItem(STORE_KEY, JSON.stringify(history.slice(0, 50)));
  } catch {
    // Storage full or blocked (private browsing). History just won't persist.
  }
}

function titleFrom(text) {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length > 42 ? clean.slice(0, 42) + "..." : clean || "New chat";
}

function timeAgo(stamp) {
  const mins = Math.floor((Date.now() - stamp) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return mins + "m ago";
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return hrs + "h ago";
  const days = Math.floor(hrs / 24);
  return days === 1 ? "yesterday" : days + "d ago";
}

/* Record one exchange into the current conversation. */
function remember(role, payload) {
  if (!currentId) {
    currentId = "c" + Date.now();
    history.unshift({
      id: currentId,
      title: role === "user" ? titleFrom(payload.text || "") : "New chat",
      created: Date.now(),
      doctor: doctorMode,
      turns: [],
    });
  }

  const chat = history.find((c) => c.id === currentId);
  if (!chat) return;

  chat.turns.push({ role, ...payload });
  chat.updated = Date.now();

  // Title comes from the first thing the user said.
  if (role === "user" && chat.turns.filter((t) => t.role === "user").length === 1) {
    chat.title = titleFrom(payload.text || "");
  }

  saveHistory();
  renderHistory();
}

/* ------------------------------------------------------------- sidebar */

function renderHistory() {
  const list = $("historyList");
  list.innerHTML = "";

  if (history.length === 0) {
    const empty = document.createElement("div");
    empty.className = "history-empty";
    empty.textContent =
      "Your past chats appear here. They are saved in this browser only, never on a server.";
    list.appendChild(empty);
    return;
  }

  history.forEach((chat) => {
    const item = document.createElement("button");
    item.className = "history-item" + (chat.id === currentId ? " active" : "");

    const title = document.createElement("span");
    title.className = "h-title";
    title.textContent = chat.title;

    const meta = document.createElement("span");
    meta.className = "h-meta";
    if (chat.doctor) {
      const tag = document.createElement("span");
      tag.className = "h-doc";
      tag.textContent = "CLINICIAN";
      meta.appendChild(tag);
    }
    meta.appendChild(
      document.createTextNode(timeAgo(chat.updated || chat.created))
    );

    item.append(title, meta);
    item.onclick = () => openChat(chat.id);

    const del = document.createElement("button");
    del.className = "h-del";
    del.textContent = "\u00d7";
    del.title = "Delete this chat";
    del.onclick = (e) => {
      e.stopPropagation();
      history = history.filter((c) => c.id !== chat.id);
      if (currentId === chat.id) startNewChat();
      saveHistory();
      renderHistory();
    };

    item.appendChild(del);
    list.appendChild(item);
  });
}

/* Replay a stored conversation back onto the screen. */
function openChat(id) {
  const chat = history.find((c) => c.id === id);
  if (!chat) return;

  currentId = id;
  setDoctorMode(!!chat.doctor, false);

  thread.innerHTML = "";
  chat.turns.forEach((turn) => {
    if (turn.role === "user") {
      addUser(turn.text, null);   // images are not persisted
    } else if (turn.kind === "prediction") {
      addPrediction(turn.data);
    } else if (turn.kind === "triage") {
      addWorklist(turn.data);
    } else {
      addTextInstant(turn.text, turn.source, turn.related);
    }
  });

  renderHistory();
  closeSidebarOnMobile();
  scrollDown();
}

function startNewChat() {
  currentId = null;
  location.hash = "";
  thread.innerHTML = "";
  const welcome = document.createElement("div");
  welcome.className = "welcome";
  welcome.id = "welcome";
  welcome.innerHTML = WELCOME_HTML;
  thread.appendChild(welcome);
  wireStarters();
  applyLanguage(currentLang);   // welcome block was rebuilt, re-translate it
  renderHistory();
  closeSidebarOnMobile();
}

/* Keep a pristine copy of the welcome block so New chat can restore it. */
const WELCOME_HTML = $("welcome") ? $("welcome").innerHTML : "";

function wireStarters() {
  document.querySelectorAll(".starter").forEach((b) => {
    b.onclick = () => { input.value = b.textContent; send(); };
  });
}

/* A no-animation version of addText, used when replaying history.
   Retyping every old message letter by letter would be unbearable. */
function addTextInstant(answer, source, related) {
  clearWelcome();
  const div = document.createElement("div");
  div.className = "msg bot";

  const span = document.createElement("span");
  span.textContent = answer;
  div.appendChild(span);

  if (source) {
    const s = document.createElement("div");
    s.className = "src";
    s.textContent = "From the knowledge base: " + source;
    div.appendChild(s);
  }
  if (related && related.length) {
    const wrap = document.createElement("div");
    wrap.className = "related";
    related.forEach((q) => {
      const b = document.createElement("button");
      b.textContent = q;
      b.onclick = () => { input.value = q; send(); };
      wrap.appendChild(b);
    });
    div.appendChild(wrap);
  }
  thread.appendChild(div);
  scrollDown();
}

/* --------------------------------------------------------- sidebar open/close */

function sidebarIsOverlay() { return window.innerWidth <= 820; }

function closeSidebarOnMobile() {
  if (sidebarIsOverlay()) {
    $("sidebar").classList.remove("open");
    $("scrim").classList.add("hidden");
  }
}

$("sideOpen").onclick = () => {
  if (sidebarIsOverlay()) {
    $("sidebar").classList.add("open");
    $("scrim").classList.remove("hidden");
  } else {
    $("sidebar").classList.remove("closed");
    $("shell").classList.remove("wide");
  }
};

$("sideClose").onclick = () => {
  if (sidebarIsOverlay()) {
    $("sidebar").classList.remove("open");
    $("scrim").classList.add("hidden");
  } else {
    $("sidebar").classList.add("closed");
    $("shell").classList.add("wide");
  }
};

$("scrim").onclick = closeSidebarOnMobile;
$("newChatBtn").onclick = startNewChat;

$("clearAllBtn").onclick = () => {
  if (!history.length) return;
  if (!confirm("Delete all saved chats from this browser? This cannot be undone.")) return;
  history = [];
  saveHistory();
  startNewChat();
};

/* ------------------------------------------------------ clinician mode */

function setDoctorMode(on, persist = true) {
  doctorMode = on;
  $("modeToggle").setAttribute("aria-checked", on ? "true" : "false");
  $("doctorBadge").classList.toggle("hidden", !on);

  input.placeholder = on
    ? "Paste one patient per line for batch triage, or a single panel..."
    : "Type your results, or ask a question...";

  const disclaimer = $("disclaimerText");
  if (disclaimer) {
    disclaimer.textContent = on
      ? "Decision support only. Screening model on historical data, not validated prospectively. Clinical judgement required."
      : "MedChat is a student project. It cannot diagnose you. Please talk to a doctor.";
  }

  if (persist) {
    try { localStorage.setItem(MODE_KEY, on ? "1" : "0"); } catch {}
  }
}

$("modeToggle").onclick = () => setDoctorMode(!doctorMode);

/* ------------------------------------------------------- triage rendering */

function addWorklist(data) {
  clearWelcome();

  const wrap = document.createElement("div");
  wrap.className = "msg bot";

  const block = document.createElement("div");
  block.className = "worklist";

  const head = document.createElement("div");
  head.className = "wl-head";
  head.innerHTML =
    "<h3>Triage worklist</h3><span>" +
    data.assessed + " of " + data.total + " assessed</span>";
  block.appendChild(head);

  data.worklist.forEach((row, index) => {
    const card = document.createElement("div");
    card.className = "wl-row " + row.band_key;

    const main = document.createElement("div");
    main.className = "wl-main";
    main.innerHTML =
      '<span class="wl-rank">' + (index + 1) + "</span>" +
      '<span class="wl-id"></span>' +
      '<span class="wl-pred"></span>' +
      '<span class="wl-conf">' + (row.confidence * 100).toFixed(0) + "%</span>" +
      '<span class="wl-band ' + row.band_key + '">' + row.band + "</span>";
    main.querySelector(".wl-id").textContent = row.label;
    main.querySelector(".wl-pred").textContent = row.prediction;
    card.appendChild(main);

    if (row.flags.length) {
      const flags = document.createElement("div");
      flags.className = "wl-flags";
      row.flags.forEach((f) => {
        const line = document.createElement("div");
        line.className = "wl-flag";
        line.textContent = "\u26a0 " + f;
        flags.appendChild(line);
      });
      card.appendChild(flags);
    }

    // The clinical note is hidden until the row is clicked, so a 30-row
    // worklist stays scannable.
    const detail = document.createElement("div");
    detail.className = "wl-detail hidden";

    const tags = document.createElement("div");
    tags.className = "tags";
    Object.entries(row.values).forEach(([k, v]) => {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.innerHTML = k + " <b>" + v + "</b>";
      tags.appendChild(tag);
    });
    detail.appendChild(tags);

    const note = document.createElement("div");
    note.className = "note-box";
    note.textContent = row.note;
    detail.appendChild(note);

    const copy = document.createElement("button");
    copy.className = "copy-btn";
    copy.textContent = "Copy clinical note";
    copy.onclick = () => {
      navigator.clipboard.writeText(row.note).then(() => {
        copy.textContent = "Copied";
        copy.classList.add("done");
        setTimeout(() => {
          copy.textContent = "Copy clinical note";
          copy.classList.remove("done");
        }, 1600);
      });
    };
    detail.appendChild(copy);

    card.appendChild(detail);
    main.onclick = () => detail.classList.toggle("hidden");

    block.appendChild(card);
  });

  if (data.skipped && data.skipped.length) {
    const skipped = document.createElement("div");
    skipped.className = "wl-skipped";
    skipped.textContent =
      "Not assessed (" + data.skipped.length + "): " +
      data.skipped.map((s) => s.label).join(", ") +
      " — each row needs TSH plus at least one of T3, TT4, T4U or FTI. " +
      "The model does not guess from partial panels, in bulk or otherwise.";
    block.appendChild(skipped);
  }

  wrap.appendChild(block);
  thread.appendChild(wrap);
  scrollDown();
}

/* ---------------------------------------------------------------- boot */

loadHistory();
try { setDoctorMode(localStorage.getItem(MODE_KEY) === "1", false); } catch {}
renderHistory();
