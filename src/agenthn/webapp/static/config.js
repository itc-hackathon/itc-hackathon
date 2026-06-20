/* AgentHN runtime config — loaded before app.js.
 *
 * `backend`: base URL of the LIVE FastAPI server (the GPU box behind an
 *   ngrok / cloudflared https tunnel), e.g. "https://abc123.ngrok-free.app".
 *   Leave "" when the page is served by the FastAPI app itself (same origin).
 *   scripts/capture_fixtures.sh rewrites this line when $AGENTHN_BACKEND is set.
 *
 * Fallback: the site pings `${backend}/api/health` on load and before every
 *   demo action. If that fails — or the hard `cutoff` below has passed — the
 *   demos replay the recorded fixtures in /fixtures instead of calling the API.
 *   Health is the primary trigger; the cutoff is just a backstop so the page
 *   degrades correctly even if the server dies early or runs late.
 */
window.AGENTHN_CONFIG = {
  backend: "",
  // Hard fallback backstop: 9:00pm PT on 2026-06-20. Past this, always replay.
  cutoff: Date.parse("2026-06-20T21:00:00-07:00"),
};
