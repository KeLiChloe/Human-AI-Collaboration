(() => {
  const STORAGE_KEY = "theory-rating-session";
  const DEFAULT_SEED = "1024";
  const dims = window.RATING_CONFIG.dimensions;

  // Exact snake_case feature names from the survey (main-effect predictors).
  const FEATURE_NAMES = [
    "social_science",
    "natural_science",
    "engineering_and_technology",
    "num_authors",
    "female",
    "asian",
    "black",
    "hispanic_and_other",
    "white",
    "authors_race_diversity_score",
    "country_race_diversity_score",
    "news_inequality_mentions_3_years",
    "paper_inequality_mentions_3_years",
  ].slice().sort((a, b) => b.length - a.length);

  const FEATURE_NAME_RE = new RegExp(
    `\\b(?:${FEATURE_NAMES.map((n) => n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\b`,
    "g"
  );

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /** Wrap exact feature-name hits in small-caps spans for display. */
  function formatTheoryHtml(text) {
    const escaped = escapeHtml(text);
    return escaped.replace(FEATURE_NAME_RE, (m) => `<span class="feature-textsc">${m}</span>`);
  }

  function formatSelectionFeature(feat) {
    const escaped = escapeHtml(feat || "");
    return escaped.replace(FEATURE_NAME_RE, (m) => `<span class="feature-textsc">${m}</span>`);
  }

  function renderSelections(selections) {
    if (!Array.isArray(selections) || !selections.length) {
      return `<p class="selection-empty">No feature selections recorded.</p>`;
    }
    const rows = selections
      .map((s, i) => {
        const rank = s.rank != null ? s.rank : i + 1;
        const rawSign = (s.sign || "").trim();
        const signClass =
          rawSign === "-" || rawSign === "–" || rawSign.toLowerCase() === "negative"
            ? "sel-sign sel-sign-neg"
            : "sel-sign";
        const sign = rawSign ? ` (${escapeHtml(rawSign)})` : "";
        return `<li><span class="sel-rank">${rank}.</span> ${formatSelectionFeature(
          s.feature
        )}<span class="${signClass}">${sign}</span></li>`;
      })
      .join("");
    return `
      <div class="selection-block">
        <h4 class="selection-title">Respondent selected features</h4>
        <ol class="selection-list">${rows}</ol>
      </div>
    `;
  }

  const loginPanel = document.getElementById("login-panel");
  const workspace = document.getElementById("workspace");
  const theoryList = document.getElementById("theory-list");
  const sessionMeta = document.getElementById("session-meta");
  const sessionIdLabel = document.getElementById("session-id-label");
  const progressPill = document.getElementById("progress-pill");
  const doneBanner = document.getElementById("done-banner");
  const loginForm = document.getElementById("login-form");
  const loginError = document.getElementById("login-error");
  const logoutBtn = document.getElementById("logout-btn");
  const pageTabs = document.getElementById("page-tabs");
  const pageTitle = document.getElementById("page-title");
  const prevBtn = document.getElementById("prev-page-btn");
  const nextBtn = document.getElementById("next-page-btn");
  const submitBtn = document.getElementById("submit-btn");
  const submitStatus = document.getElementById("submit-status");

  let identifier = "";
  let seed = "";
  let pages = [];
  let pageIndex = 0;
  let submitted = false;
  let lastProgress = { completed: 0, total: 100, ready: false, submitted: false };
  let saveTimers = {};

  function showError(msg) {
    loginError.hidden = !msg;
    loginError.textContent = msg || "";
  }

  function unfinishedLabels() {
    const missing = [];
    pages.forEach((page) => {
      page.theories.forEach((t) => {
        if (!t.started) {
          const n = t.page_position || t.position;
          if (n != null) missing.push(`Theory ${n}`);
        }
      });
    });
    return missing;
  }

  function showIncompleteSubmitMessage() {
    const missing = unfinishedLabels();
    submitStatus.hidden = false;
    submitStatus.classList.add("submit-status-error");
    submitStatus.textContent = missing.length
      ? `Not ready to submit. Still need: ${missing.join(", ")}`
      : "Not ready to submit.";
  }

  function clearSubmitStatusError() {
    submitStatus.classList.remove("submit-status-error");
  }

  function formatSubmittedAt(raw) {
    if (!raw) return "";
    const trimmed = String(raw).replace(/\.\d+(?=[+-]\d{2}:\d{2}|Z)/, "");
    const d = new Date(trimmed);
    if (Number.isNaN(d.getTime())) return trimmed;
    return d.toISOString().replace(/\.\d{3}Z$/, "Z").replace("Z", "+00:00");
  }

  function updateProgress(progress) {
    lastProgress = progress;
    progressPill.textContent = `${progress.completed} / ${progress.total}`;
    submitted = !!progress.submitted;
    doneBanner.hidden = !submitted;
    submitBtn.hidden = pageIndex !== pages.length - 1;
    submitBtn.disabled = false;
    submitBtn.textContent = submitted ? "Update submission" : "Submit all ratings";
    if (submitted && progress.ready) {
      clearSubmitStatusError();
      submitStatus.hidden = false;
      submitStatus.textContent = `Last submitted at ${formatSubmittedAt(progress.submitted_at)}. You can revise and submit again.`;
    } else if (progress.ready) {
      clearSubmitStatusError();
      submitStatus.hidden = true;
      submitStatus.textContent = "";
    } else if (submitStatus.classList.contains("submit-status-error")) {
      // Keep the red incomplete message in sync after more ratings.
      showIncompleteSubmitMessage();
    } else {
      clearSubmitStatusError();
      submitStatus.hidden = true;
      submitStatus.textContent = "";
    }
    setLocked(false);
  }

  function setLocked(locked) {
    theoryList.querySelectorAll('input[type="range"]').forEach((input) => {
      input.disabled = locked;
    });
  }

  function scorePayload(card) {
    const scores = {};
    for (const d of dims) {
      const input = card.querySelector(`input[data-dim="${d.key}"]`);
      scores[d.key] = Number(input.value);
    }
    return scores;
  }

  function updateTheoryState(theoryId, { ratings, started } = {}) {
    pages.forEach((page) => {
      page.theories.forEach((t) => {
        if (t.id !== theoryId) return;
        if (ratings) t.ratings = { ...ratings };
        if (started) t.started = true;
      });
    });
  }

  /** Copy visible slider values into pages[] before re-rendering another category. */
  function syncCurrentPageFromDom() {
    const page = pages[pageIndex];
    if (!page || !theoryList) return;
    theoryList.querySelectorAll(".theory-card").forEach((card) => {
      const theoryId = card.dataset.theoryId;
      const ratings = scorePayload(card);
      updateTheoryState(theoryId, {
        ratings,
        started: card.dataset.started === "1",
      });
    });
  }

  async function persistCard(card, { immediate = false } = {}) {
    const theoryId = card.dataset.theoryId;
    const status = card.querySelector(".card-status");
    const scores = scorePayload(card);

    const run = async () => {
      status.className = "card-status";
      status.textContent = "Saving…";
      try {
        const res = await fetch("/api/ratings", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ identifier, seed, theory_id: theoryId, scores }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Save failed");
        status.className = "card-status saved";
        status.textContent = data.progress && data.progress.submitted ? "Saved & submitted" : "Saved";
        card.dataset.started = "1";
        updateTheoryState(theoryId, { ratings: scores, started: true });
        updateProgress(data.progress);
      } catch (err) {
        status.className = "card-status error";
        status.textContent = err.message || "Could not save";
      }
    };

    clearTimeout(saveTimers[theoryId]);
    if (immediate) {
      await run();
    } else {
      // Short debounce so every drag still persists quickly without flooding.
      saveTimers[theoryId] = setTimeout(run, 120);
    }
  }

  function scaleLabel(score) {
    const n = Number(score);
    if (n <= 2) return "poor";
    if (n <= 4) return "weak";
    if (n <= 6) return "moderate";
    if (n <= 8) return "strong";
    return "excellent";
  }

  function formatScore(score) {
    return `${score} (${scaleLabel(score)})`;
  }

  function renderTheory(item) {
    const card = document.createElement("article");
    card.className = "theory-card";
    card.dataset.theoryId = item.id;

    const ratings = item.ratings || {};
    const started = !!item.started;
    const dimHtml = dims
      .map((d) => {
        const value = Number.isInteger(ratings[d.key]) ? ratings[d.key] : 1;
        const desc = escapeHtml(d.description || "");
        const dimLabel = escapeHtml(d.label);
        return `
          <div class="dim-row">
            <div class="dim-top">
              <label for="${item.id}-${d.key}">${dimLabel}</label>
              <div class="dim-end">
                <span class="score-value" data-for="${d.key}">${formatScore(value)}</span>
                <button
                  type="button"
                  class="dim-help"
                  aria-label="About ${dimLabel}"
                  tabindex="0"
                >?</button>
                <span class="dim-tooltip" role="tooltip">${desc}</span>
              </div>
            </div>
            <input
              id="${item.id}-${d.key}"
              type="range"
              min="1"
              max="10"
              step="1"
              value="${value}"
              data-dim="${d.key}"
            />
          </div>
        `;
      })
      .join("");

    const idx = item.page_position || item.position;
    const label =
      item.display_label ||
      `Theory ${idx} (${item.task === "gender" ? "Gender" : "Race"} · ${
        item.effect === "soi" ? "Interactions" : "Main"
      })`;
    card.innerHTML = `
      <div class="theory-body">
        <div class="theory-head">
          <span class="theory-index">${escapeHtml(label)}</span>
        </div>
        ${renderSelections(item.selections)}
        <div class="theory-text"></div>
      </div>
      <aside class="rate-panel">
        <div class="rate-panel-head">
          <h3>Scores</h3>
          <span class="scale-hint">1–10</span>
        </div>
        ${dimHtml}
        <div class="card-status ${started ? "saved" : ""}">
          ${started ? "Saved" : ""}
        </div>
      </aside>
    `;
    card.querySelector(".theory-text").innerHTML = formatTheoryHtml(item.text);
    card.dataset.started = started ? "1" : "0";
    card.dataset.label = label;

    card.querySelectorAll(".dim-help").forEach((btn) => {
      btn.addEventListener("click", (e) => e.preventDefault());
    });

    card.querySelectorAll('input[type="range"]').forEach((input) => {
      const valueEl = card.querySelector(`.score-value[data-for="${input.dataset.dim}"]`);
      input.addEventListener("input", () => {
        valueEl.textContent = formatScore(input.value);
        // Auto-save on every drag tick (debounced briefly).
        persistCard(card, { immediate: false });
      });
      input.addEventListener("change", () => {
        // Final value when the user releases the slider.
        persistCard(card, { immediate: true });
      });
      input.addEventListener("pointerup", () => {
        persistCard(card, { immediate: true });
      });
    });

    return card;
  }

  function renderPage() {
    syncCurrentPageFromDom();
    const page = pages[pageIndex];
    if (!page) return;
    pageTitle.textContent = `Page ${pageIndex + 1} of ${pages.length}: ${page.title}`;
    theoryList.innerHTML = "";
    page.theories.forEach((t) => theoryList.appendChild(renderTheory(t)));

    pageTabs.innerHTML = "";
    pages.forEach((p, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `page-tab${i === pageIndex ? " active" : ""}`;
      btn.textContent = p.short || p.title;
      btn.addEventListener("click", () => {
        pageIndex = i;
        renderPage();
      });
      pageTabs.appendChild(btn);
    });

    prevBtn.hidden = pageIndex === 0;
    nextBtn.hidden = pageIndex >= pages.length - 1;
    submitBtn.hidden = pageIndex !== pages.length - 1;
    submitBtn.disabled = false;
    submitBtn.textContent = submitted ? "Update submission" : "Submit all ratings";
    if (submitted && !submitStatus.classList.contains("submit-status-error")) {
      submitStatus.hidden = false;
    } else if (!submitStatus.classList.contains("submit-status-error")) {
      submitStatus.hidden = true;
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function openSession(rawSeed, rawId) {
    showError("");
    const res = await fetch("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seed: rawSeed, identifier: rawId }),
    });
    const data = await res.json();
    if (!res.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map((d) => d.msg || d).join("; ")
        : data.detail || "Could not start session";
      throw new Error(detail);
    }

    identifier = data.identifier;
    seed = data.seed || DEFAULT_SEED;
    pages = data.pages || [];
    pageIndex = 0;
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ identifier, seed }));
    sessionIdLabel.textContent = `${identifier} · seed ${seed}`;
    sessionMeta.hidden = false;
    loginPanel.hidden = true;
    workspace.hidden = false;
    updateProgress(data.progress);
    renderPage();
  }

  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      await openSession(DEFAULT_SEED, document.getElementById("identifier").value);
    } catch (err) {
      showError(err.message || "Login failed");
    }
  });

  prevBtn.addEventListener("click", () => {
    if (pageIndex > 0) {
      pageIndex -= 1;
      renderPage();
    }
  });

  nextBtn.addEventListener("click", () => {
    if (pageIndex < pages.length - 1) {
      pageIndex += 1;
      renderPage();
    }
  });

  submitBtn.addEventListener("click", async () => {
    const missing = unfinishedLabels();
    if (missing.length) {
      showIncompleteSubmitMessage();
      return;
    }
    submitBtn.disabled = true;
    clearSubmitStatusError();
    submitStatus.hidden = false;
    submitStatus.textContent = "Submitting…";
    try {
      const res = await fetch("/api/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ identifier, seed }),
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = data.detail;
        if (detail && typeof detail === "object" && detail.missing) {
          showIncompleteSubmitMessage();
          return;
        }
        throw new Error(
          typeof detail === "string" ? detail : detail?.message || "Submit failed"
        );
      }
      updateProgress(data.progress);
      clearSubmitStatusError();
      const q = new URLSearchParams({ identifier, seed });
      window.location.assign(`/submitted?${q.toString()}`);
      return;
    } catch (err) {
      submitStatus.classList.add("submit-status-error");
      submitStatus.hidden = false;
      submitStatus.textContent = err.message || "Submit failed";
    } finally {
      submitBtn.disabled = false;
    }
  });

  logoutBtn.addEventListener("click", () => {
    localStorage.removeItem(STORAGE_KEY);
    identifier = "";
    seed = "";
    pages = [];
    pageIndex = 0;
    submitted = false;
    theoryList.innerHTML = "";
    sessionMeta.hidden = true;
    workspace.hidden = true;
    doneBanner.hidden = true;
    submitStatus.hidden = true;
    loginPanel.hidden = false;
    document.getElementById("identifier").focus();
  });

  try {
    const params = new URLSearchParams(window.location.search);
    const identifierFromUrl = (params.get("identifier") || "").trim();
    // Prefill identifier when returning from the submitted page.
    document.getElementById("identifier").value = identifierFromUrl;
  } catch (_) {
    localStorage.removeItem(STORAGE_KEY);
    document.getElementById("identifier").value = "";
  }
})();
