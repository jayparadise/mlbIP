# In-Play Pricing Terminal

Interactive terminal for live MLB player-prop pricing with comps anchoring.
Vite + React, no other dependencies.

## Run locally
```bash
npm install
npm run dev        # http://localhost:5173
```

## Deploy to Vercel

### Option A — Vercel CLI (fastest)
```bash
npm i -g vercel
cd inplay-terminal
vercel              # first run: link/create the project, accept defaults
vercel --prod       # promote to production
```
Vercel auto-detects Vite: build command `vite build`, output dir `dist`.

### Option B — Git + dashboard
1. `git init && git add . && git commit -m "init"` and push to GitHub/GitLab.
2. In the Vercel dashboard: **Add New → Project → Import** the repo.
3. Framework preset auto-detects as **Vite**. Leave defaults:
   - Build command: `npm run build`
   - Output directory: `dist`
4. **Deploy.**

### Option C — Prebuilt drag-and-drop
```bash
npm install && npm run build      # produces ./dist
```
Then drag the `dist` folder into the Vercel dashboard (or `vercel deploy --prebuilt`).

No environment variables are required — all pricing math runs client-side.
