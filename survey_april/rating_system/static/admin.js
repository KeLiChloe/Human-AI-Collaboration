(() => {
  const form = document.getElementById("admin-form");
  const tokenInput = document.getElementById("token");
  const seedInput = document.getElementById("seed-filter");
  const errorEl = document.getElementById("admin-error");
  const results = document.getElementById("results");
  const tbody = document.querySelector("#sessions-table tbody");
  const csvBtn = document.getElementById("csv-btn");
  const clearAllBtn = document.getElementById("clear-all-btn");

  const STORAGE_KEY = "theory-rating-admin-token";

  function showError(msg) {
    errorEl.hidden = !msg;
    errorEl.textContent = msg || "";
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function qs(extra = {}) {
    const params = new URLSearchParams({ token: tokenInput.value.trim() });
    const seed = seedInput.value.trim();
    if (seed) params.set("seed", seed);
    Object.entries(extra).forEach(([k, v]) => {
      if (v != null && String(v).trim() !== "") params.set(k, String(v).trim());
    });
    return params.toString();
  }

  function exportUrl({ identifier, seed } = {}) {
    const params = new URLSearchParams({ token: tokenInput.value.trim() });
    if (seed) params.set("seed", seed);
    if (identifier) params.set("identifier", identifier);
    return `/api/admin/export.csv?${params.toString()}`;
  }

  function requireToken() {
    const token = tokenInput.value.trim();
    if (!token) {
      showError("Enter admin token");
      return null;
    }
    localStorage.setItem(STORAGE_KEY, token);
    return token;
  }

  async function loadSessions(e) {
    if (e) e.preventDefault();
    showError("");
    if (!requireToken()) return;
    try {
      const res = await fetch(`/api/admin/sessions?${qs()}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Unauthorized");
      tbody.innerHTML = "";
      if (!data.sessions.length) {
        tbody.innerHTML = `<tr><td colspan="6">No sessions yet.</td></tr>`;
      } else {
        data.sessions.forEach((s) => {
          const tr = document.createElement("tr");
          const href = exportUrl({ identifier: s.identifier, seed: s.seed });
          tr.innerHTML = `
            <td>${escapeHtml(s.identifier)}</td>
            <td>${escapeHtml(s.seed)}</td>
            <td>${s.completed} / ${s.total}</td>
            <td>${s.submitted_at ? "Yes" : "No"}</td>
            <td>${escapeHtml(s.updated_at || "")}</td>
            <td class="admin-row-actions">
              <a class="ghost-btn download-link" href="${href}">Download CSV</a>
              <button
                type="button"
                class="ghost-btn danger-btn clear-session-btn"
                data-identifier="${escapeHtml(s.identifier)}"
                data-seed="${escapeHtml(s.seed)}"
              >Clear session</button>
            </td>
          `;
          tbody.appendChild(tr);
        });
      }
      results.hidden = false;
    } catch (err) {
      results.hidden = true;
      showError(err.message || "Failed to load");
    }
  }

  async function clearSession(identifier, seed) {
    showError("");
    if (!requireToken()) return;
    const ok = window.confirm(
      `Clear session for "${identifier}" (seed ${seed})?\n\nThis deletes their ratings and assignment. Cannot be undone.`
    );
    if (!ok) return;
    try {
      const params = new URLSearchParams({
        token: tokenInput.value.trim(),
        identifier,
        seed,
      });
      const res = await fetch(`/api/admin/session?${params.toString()}`, {
        method: "DELETE",
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(
          typeof data.detail === "string" ? data.detail : data.detail?.message || "Clear failed"
        );
      }
      await loadSessions();
    } catch (err) {
      showError(err.message || "Clear failed");
    }
  }

  async function clearAll() {
    showError("");
    if (!requireToken()) return;
    const ok = window.confirm(
      "Clear ALL sessions, assignments, and ratings?\n\nThis wipes the entire database. Cannot be undone."
    );
    if (!ok) return;
    const ok2 = window.confirm("Really wipe all data?");
    if (!ok2) return;
    try {
      const res = await fetch(`/api/admin/clear?${qs()}`, { method: "DELETE" });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(
          typeof data.detail === "string" ? data.detail : data.detail?.message || "Clear failed"
        );
      }
      await loadSessions();
    } catch (err) {
      showError(err.message || "Clear failed");
    }
  }

  form.addEventListener("submit", loadSessions);
  csvBtn.addEventListener("click", () => {
    if (!requireToken()) return;
    window.location.href = exportUrl({ seed: seedInput.value.trim() || undefined });
  });
  clearAllBtn.addEventListener("click", clearAll);

  tbody.addEventListener("click", (e) => {
    const btn = e.target.closest(".clear-session-btn");
    if (!btn) return;
    clearSession(btn.dataset.identifier, btn.dataset.seed);
  });

  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved) tokenInput.value = saved;
})();
