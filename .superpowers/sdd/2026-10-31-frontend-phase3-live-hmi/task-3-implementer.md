# Task 3 implementation report

Implemented the scoped live HMI mechanical integration. Removed `prepare:data` from all UI
commands and removed Vite `publicDir` so precomputed JSON is never exposed by the app. Replaced
the obsolete Phase 2 Playwright flow with an explicit TEST-ONLY contract-simulator E2E, including
dummy upload, event-stage assertions, alarm acknowledgement, locator, engineer details, test-only
label, commit attestation, and dark/light 1920×1080 screenshots. Added a separate real-service E2E
that fails clearly unless both `REAL_RUNNER_BASE_URL` and `REAL_TWIN_VIDEO` are supplied.

Checks:

- `npm test`: 22 passed
- `npm run typecheck`: passed
- `npm run build`: passed
- TEST-ONLY Playwright: 1 passed with `PLAYWRIGHT_BROWSERS_PATH=/tmp/frontend-playwright-browsers`
- screenshots: `phase3-test-only-dark-1920x1080.png`, `phase3-test-only-light-1920x1080.png`

The real E2E was not run. Read-only lead service inspection at clean `59d674f` still has no
reviewed Varied V3/V4 deployment factory/start command; only `your_deployment.runner:create_adapter`
is documented. No video was accessed. @salasl DECISION NEEDED for that reviewed factory and command.
