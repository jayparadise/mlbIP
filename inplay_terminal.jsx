import React, { useState, useMemo } from "react";

/* ============================================================================
   IN-PLAY PROP PRICING — TRADING TERMINAL
   JS port of inplay_engine.py. Same math: S_final = S_locked + S_remaining,
   compound mean/variance, NB/Poisson fit, hook survival for pitchers,
   comps anchoring, 7% margin, pre-game-book gating until first event.
   ========================================================================== */

// ---------- distributions ----------
function logGamma(x) {
  const g = 7;
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  x -= 1;
  let a = c[0];
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
function nbLogPmf(k, r, p) {
  return logGamma(k + r) - logGamma(k + 1) - logGamma(r) + r * Math.log(p) + k * Math.log(1 - p);
}
// CDF array where cdf[k] = P(S<=k)
function countCdf(mean, variance, kmax = 60) {
  const pmf = new Array(kmax + 1).fill(0);
  if (mean <= 1e-9) { pmf[0] = 1; }
  else if (variance > mean * 1.05) {            // Negative Binomial (overdispersed)
    const v = Math.max(variance, mean * 1.0000001);
    const p = mean / v, r = (mean * mean) / (v - mean);
    for (let k = 0; k <= kmax; k++) pmf[k] = Math.exp(nbLogPmf(k, r, p));
  } else if (variance < mean * 0.95) {          // Binomial (underdispersed)
    let p = 1 - variance / mean;
    p = Math.min(Math.max(p, 1e-3), 0.999);
    let n = Math.max(1, Math.round(mean / p));
    p = mean / n;                               // preserve the mean exactly
    for (let k = 0; k <= Math.min(n, kmax); k++)
      pmf[k] = Math.exp(logGamma(n + 1) - logGamma(k + 1) - logGamma(n - k + 1) + k * Math.log(p) + (n - k) * Math.log(1 - p));
  } else {                                        // Poisson
    pmf[0] = Math.exp(-mean);
    for (let k = 1; k <= kmax; k++) pmf[k] = pmf[k - 1] * mean / k;
  }
  let tot = pmf.reduce((a, b) => a + b, 0);
  const cdf = []; let run = 0;
  for (let k = 0; k <= kmax; k++) { run += pmf[k] / tot; cdf.push(Math.min(run, 1)); }
  return cdf;
}

// ---------- pre-game expected opportunities ----------
function teamRemainingPA(gs, battingHome, q = 0.32) {
  const paPerInnMean = 3 / (1 - q);
  const paPerInnVar = 3 * q / Math.pow(1 - q, 2);
  const reg = 9;
  let inningsLeft = gs.inning > reg ? 0 : Math.max(reg - gs.inning, 0);
  let curFrac = 0;
  const teamBattingNow = (battingHome && !gs.top) || (!battingHome && gs.top);
  if (teamBattingNow && gs.inning <= 9) curFrac = Math.max(0, (3 - gs.outs) / 3);
  let meanPA = inningsLeft * paPerInnMean + curFrac * paPerInnMean;
  let varPA = (inningsLeft + curFrac) * paPerInnVar;
  if (battingHome && gs.inning >= 9 && gs.homeScore > gs.awayScore)
    meanPA = Math.max(0, meanPA - 0.5 * paPerInnMean);
  // extra innings only matter once we're in/through the 9th and still tied —
  // before that the 9th is already counted as a normal inning.
  if (gs.inning >= 9 && gs.awayScore === gs.homeScore) {
    const pEx = 0.33;
    meanPA += pEx * paPerInnMean;
    varPA += pEx * (1 - pEx) * Math.pow(paPerInnMean, 2);
  }
  return [meanPA, varPA];
}
function hitterRemainingPA(gs, slot, battingHome, q = 0.32, pullProb = 0) {
  const [teamMean, teamVar] = teamRemainingPA(gs, battingHome, q);
  const dueUp = battingHome ? gs.homeDueUp : gs.awayDueUp;
  const offset = ((slot - dueUp) % 9 + 9) % 9;
  let meanPA = 0;
  for (let j = 0; j < 15; j++) {
    const idx = offset + 1 + 9 * j;
    if (idx <= teamMean) meanPA += 1;
    else { meanPA += Math.max(0, Math.min(1, teamMean - (idx - 1))); break; }
  }
  const marg = meanPA - Math.floor(meanPA);
  let varPA = marg * (1 - marg) + teamVar / 81;
  meanPA *= (1 - pullProb);
  return [meanPA, varPA];
}
function pregameXpaBySlot(slot, battingHome = false, q = 0.32) {
  const fresh = { inning: 1, top: !battingHome, outs: 0, awayScore: 0, homeScore: 0, awayDueUp: 0, homeDueUp: 0 };
  return hitterRemainingPA(fresh, slot, battingHome, q, 0)[0];
}

// ---------- pitcher hook survival ----------
function hookHazard(projPitch, tto, ps) {
  if (!ps.isStarter) {
    const base = 0.12, over = Math.max(0, ps.battersFaced - 4) * 0.06;
    return Math.min(0.9, base + over);
  }
  const limit = ps.hookPitchLimit;
  const z = (projPitch - limit) / 11;
  const hPitch = 1 / (1 + Math.exp(-z));
  const ttoPen = Math.max(0, tto - 2) * 0.06;
  let state = 0;
  if (ps.scoreDiff <= -4) state += 0.08;
  if (ps.scoreDiff >= 6) state -= 0.05;
  return Math.max(0, Math.min(0.95, 0.005 + 0.9 * hPitch + ttoPen + state));
}
function pitcherRemainingBF(ps) {
  const ppb = 3.9;
  const svs = [];
  let survive = 1, projPitch = ps.pitchesThrown;
  for (let i = 1; i < 40; i++) {
    projPitch += ppb;
    const ttoI = ps.tto + i / 9;
    survive *= (1 - hookHazard(projPitch, ttoI, ps));
    svs.push(survive);
    if (survive < 0.01) break;
  }
  let meanBF = 0, varBF = 0;
  svs.forEach((s) => (meanBF += s));
  for (let i = 0; i < svs.length; i++) {
    varBF += svs[i] * (1 - svs[i]);
    for (let j = i + 1; j < svs.length; j++) varBF += 2 * (svs[j] - svs[i] * svs[j]);
  }
  return [meanBF, Math.max(varBF, 0)];
}
function pregameBF(hookPitchLimit = 95) {
  return pitcherRemainingBF({ pitchesThrown: 0, battersFaced: 0, tto: 0, isStarter: true, hookPitchLimit, scoreDiff: 0 })[0];
}

// ---------- calibration ----------
function calibrateHitter(projHits, projTB, projHR, xpa) {
  const pHit = projHits / xpa, pHR = projHR / xpa;
  const nonHR = Math.max(pHit - pHR, 1e-9);
  const [s1, s2, s3] = [0.62, 0.26, 0.04];
  const den = s1 + s2 + s3;
  let p1 = nonHR * s1 / den, p2 = nonHR * s2 / den, p3 = nonHR * s3 / den;
  const targetTB = projTB / xpa;
  let scale = (targetTB - p1 - 4 * pHR) / Math.max(2 * p2 + 3 * p3, 1e-9);
  scale = Math.max(0.2, Math.min(3, scale));
  p2 *= scale; p3 *= scale;
  const tbMean = p1 + 2 * p2 + 3 * p3 + 4 * pHR;
  const tbE2 = p1 + 4 * p2 + 9 * p3 + 16 * pHR;
  return { pHit, p1b: p1, p2b: p2, p3b: p3, pHR, tbMean, tbVar: tbE2 - tbMean * tbMean };
}
function calibratePitcher(projK, projHA, bf) {
  return { pK: projK / bf, pHit: projHA / bf };
}

// ---------- compound moments + pricing ----------
const compoundCount = (mN, vN, p) => [mN * p, mN * p * (1 - p) + vN * p * p];
const compoundTB = (mN, vN, r) => [mN * r.tbMean, mN * r.tbVar + vN * r.tbMean * r.tbMean];

function pMilestone(cdf, locked, kPlus) {
  const need = kPlus - locked;
  if (need <= 0) return 1;
  const idx = need - 1;
  if (idx >= cdf.length) return 0;
  return Math.max(0, Math.min(1, 1 - cdf[idx]));
}
function pOver(cdf, locked, line) {
  const k = Math.floor(line - locked);
  const pu = k < 0 ? 0 : (k < cdf.length ? cdf[k] : 1);
  return Math.max(0, Math.min(1, 1 - pu));
}
const toDecimal = (p, margin) => (p <= 0 ? null : +(1 / Math.min(0.999, p * (1 + margin))).toFixed(2));
const decToProb = (d) => (!d || d <= 1 ? null : 1 / d);
function devig(over, under) {
  const po = decToProb(over), pu = decToProb(under);
  if (po && pu) { const s = po + pu; return [po / s, pu / s]; }
  if (po) return [Math.min(po / 1.03, 0.999), null];
  if (pu) return [null, Math.min(pu / 1.03, 0.999)];
  return [null, null];
}
function anchor( our, comp, weight, maxDev) {
  if (comp == null) return our;
  const blended = (1 - weight) * our + weight * comp;
  return Math.max(comp - maxDev, Math.min(comp + maxDev, blended));
}

// ---------- derive the full-game mean (projection) from the book market ----------
// The O/U line + de-vigged over price is the real read on expected performance.
// Solve for the mean whose distribution (spread set by std dev) reproduces the
// market's fair P(over line). Monotonic in mean -> bisection.
function impliedMeanFromOU(line, over, under, std) {
  const po = decToProb(over), pu = decToProb(under);
  let fair;
  if (po && pu) fair = po / (po + pu);          // two-way de-vig
  else if (po) fair = po / 1.03;                // one-sided: shave ~half the hold
  else return null;
  fair = Math.min(0.995, Math.max(0.005, fair));
  let lo = 0.001, hi = 30;
  for (let i = 0; i < 46; i++) {
    const mid = (lo + hi) / 2;
    const v = std && std > 0 ? std * std : mid;
    const p = pOver(countCdf(mid, v, 90), 0, line);
    if (p < fair) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}
// derive the mean from a 1+ (or n+) milestone price, e.g. home runs
function impliedMeanFrom1Plus(price, std) {
  const pr = decToProb(price);
  if (!pr) return null;
  const fair = Math.min(0.97, Math.max(0.003, pr / 1.03));
  let lo = 0.001, hi = 6;
  for (let i = 0; i < 42; i++) {
    const mid = (lo + hi) / 2;
    const v = std && std > 0 ? std * std : mid;
    const p1 = 1 - countCdf(mid, v, 40)[0];
    if (p1 < fair) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

// ============================================================================
//  UI
// ============================================================================
const C = {
  bg: "#0b0e14", panel: "#12161f", panel2: "#161b26", line: "#232a38",
  ink: "#e6ebf5", dim: "#8a94a7", faint: "#5a6273",
  accent: "#4dd0a7", accentDim: "#2a6b58", comp: "#e0a34d", warn: "#e05d5d",
  up: "#4dd0a7", down: "#e07a7a", mono: "'JetBrains Mono','SF Mono',ui-monospace,Menlo,monospace",
};

function Field({ label, children }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <span style={{ fontSize: 10, letterSpacing: 0.8, textTransform: "uppercase", color: C.faint, fontFamily: C.mono }}>{label}</span>
      {children}
    </label>
  );
}
const inputStyle = {
  background: C.panel2, border: `1px solid ${C.line}`, color: C.ink,
  padding: "7px 9px", borderRadius: 5, fontSize: 13, fontFamily: C.mono, width: "100%", boxSizing: "border-box",
};
function Stepper({ value, set, min = 0, max = 99, step = 1 }) {
  return (
    <div style={{ display: "flex", alignItems: "stretch", border: `1px solid ${C.line}`, borderRadius: 5, overflow: "hidden" }}>
      <button onClick={() => set(Math.max(min, +(value - step).toFixed(2)))} style={stepBtn}>−</button>
      <div style={{ flex: 1, textAlign: "center", padding: "7px 0", fontFamily: C.mono, fontSize: 13, background: C.panel2 }}>{value}</div>
      <button onClick={() => set(Math.min(max, +(value + step).toFixed(2)))} style={stepBtn}>+</button>
    </div>
  );
}
const stepBtn = { background: C.panel, border: "none", color: C.dim, width: 30, cursor: "pointer", fontSize: 16, fontFamily: C.mono };

// compact decimal-safe text input (raw string, sanitized on change)
function MiniInput({ value, onChange }) {
  return (
    <input value={value ?? ""} inputMode="decimal" placeholder="—"
      onChange={(e) => { const v = e.target.value; if (v === "" || /^\d*\.?\d*$/.test(v)) onChange(v); }}
      style={{ background: C.panel2, border: `1px solid ${C.line}`, borderRadius: 4, color: C.ink,
        fontFamily: C.mono, fontSize: 12, padding: "6px 4px", width: "100%", boxSizing: "border-box", textAlign: "center" }} />
  );
}

function Toggle({ on, set, labels = ["OFF", "ON"] }) {
  return (
    <div style={{ display: "flex", border: `1px solid ${C.line}`, borderRadius: 5, overflow: "hidden", fontFamily: C.mono, fontSize: 12 }}>
      {labels.map((l, i) => (
        <button key={l} onClick={() => set(i === 1)}
          style={{ flex: 1, padding: "7px 0", cursor: "pointer", border: "none",
            background: (on ? 1 : 0) === i ? C.accentDim : C.panel2,
            color: (on ? 1 : 0) === i ? C.ink : C.dim }}>{l}</button>
      ))}
    </div>
  );
}

// price cell: our fair/price, editable comp, anchored our-price
function PriceRow({ row, suspended, setComp }) {
  const { label, ourFair, ourPrice, compKey, compRaw, anchoredPrice } = row;
  const compNum = parseFloat(compRaw);
  const deviates = !isNaN(compNum) && anchoredPrice && Math.abs(anchoredPrice - compNum) / compNum > 0.08;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1.15fr 1.1fr 0.9fr 0.9fr", alignItems: "center",
      padding: "7px 12px", borderBottom: `1px solid ${C.line}`, fontFamily: C.mono, fontSize: 13,
      opacity: suspended ? 0.35 : 1 }}>
      <span style={{ color: C.dim }}>{label}</span>
      <span style={{ color: C.faint, fontSize: 11 }}>
        {ourFair != null ? `${(ourFair * 100).toFixed(1)}%` : "—"}
        <span style={{ color: C.ink, marginLeft: 8, fontSize: 13 }}>{suspended ? "SUSP" : (ourPrice ?? "—")}</span>
      </span>
      <input
        value={compRaw ?? ""} placeholder="—" inputMode="decimal"
        onChange={(e) => setComp(compKey, e.target.value)}
        style={{ background: "transparent", border: `1px solid ${C.line}`, borderRadius: 4,
          color: C.comp, fontFamily: C.mono, fontSize: 12.5, width: 54, padding: "3px 5px", textAlign: "center" }} />
      <span style={{ color: suspended ? C.faint : C.accent, fontWeight: 600, textAlign: "right" }}>
        {suspended ? "—" : (anchoredPrice ?? "—")}
        {deviates && !suspended && <span title="model far from comp" style={{ color: C.warn, marginLeft: 5 }}>⚠</span>}
      </span>
    </div>
  );
}

function MarketCard({ title, proj, line, rows, suspended, note }) {
  return (
    <div style={{ background: C.panel, border: `1px solid ${C.line}`, borderRadius: 8, overflow: "hidden" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center",
        padding: "10px 12px", borderBottom: `1px solid ${C.line}`, background: C.panel2 }}>
        <span style={{ fontFamily: C.mono, fontSize: 12, letterSpacing: 0.5, color: C.ink, textTransform: "uppercase" }}>
          {title}
          {proj != null && <span style={{ color: C.dim, marginLeft: 8, textTransform: "none" }}>proj {proj.toFixed(2)}</span>}
          {line != null && <span style={{ color: C.accent, marginLeft: 8 }}>O/U {line}</span>}
        </span>
        {suspended && <span style={{ fontFamily: C.mono, fontSize: 10, color: C.warn, border: `1px solid ${C.warn}`, padding: "2px 6px", borderRadius: 4 }}>SUSPENDED</span>}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1.15fr 1.1fr 0.9fr 0.9fr", padding: "6px 12px",
        fontFamily: C.mono, fontSize: 9.5, letterSpacing: 0.6, color: C.faint, textTransform: "uppercase", borderBottom: `1px solid ${C.line}` }}>
        <span>Market</span><span>Model fair / px</span><span>Comp px</span><span style={{ textAlign: "right" }}>Our price</span>
      </div>
      {rows}
      {note && <div style={{ padding: "8px 12px", fontFamily: C.mono, fontSize: 10, color: C.faint }}>{note}</div>}
    </div>
  );
}

export default function App() {
  const [kind, setKind] = useState("hitter");

  // ---- editable player inputs (type any player; no dropdown) ----
  // projections (means) and std devs are RAW STRINGS so decimals type freely
  const [proj, setProj] = useState({ hits: "1.5", tb: "2.2", hr: "0.2", k: "6.14", ha: "5.5" });
  const [std, setStd]   = useState({ hits: "0.9", tb: "1.67", hr: "0.5", k: "2.4", ha: "2.0" });
  // pre-game book O/U line + prices — the REAL read on expected performance.
  // the engine derives each stat's mean by inverting these (see impliedMean*).
  const [book, setBook] = useState({
    hits_line: "0.5", hits_over: "1.43", hits_under: "2.74",
    tb_line: "1.5", tb_over: "1.92", tb_under: "1.82",
    hr1: "5.00",
    k_line: "5.5", k_over: "1.80", k_under: "2.00",
    ha_line: "5.5", ha_over: "1.90", ha_under: "1.90",
  });
  const hookLimit = 95;
  const setPr = (k, v) => { if (v === "" || /^\d*\.?\d*$/.test(v)) setProj((p) => ({ ...p, [k]: v })); };
  const setSd = (k, v) => { if (v === "" || /^\d*\.?\d*$/.test(v)) setStd((p) => ({ ...p, [k]: v })); };
  const setBk = (k, v) => { if (v === "" || /^\d*\.?\d*$/.test(v)) setBook((p) => ({ ...p, [k]: v })); };
  const fnum = (obj, k, def = 0) => { const v = parseFloat(obj[k]); return isNaN(v) ? def : v; };
  const bnum = (k) => { const v = parseFloat(book[k]); return isNaN(v) ? null : v; };

  // game state
  const [inning, setInning] = useState(6);
  const [top, setTop] = useState(true);
  const [outs, setOuts] = useState(1);
  const [awayScore, setAwayScore] = useState(2);
  const [homeScore, setHomeScore] = useState(3);
  const [slot, setSlot] = useState(3);          // 1-based in UI
  const [dueUp, setDueUp] = useState(2);         // 1-based in UI
  const [battingHome, setBattingHome] = useState(false);

  // hitter locked
  const [lockHits, setLockHits] = useState(1);
  const [lockTB, setLockTB] = useState(2);
  const [lockHR, setLockHR] = useState(0);

  // pitcher state
  const [pitches, setPitches] = useState(80);
  const [bf, setBF] = useState(20);
  const [tto, setTto] = useState(2.2);
  const [lockK, setLockK] = useState(6);
  const [lockHA, setLockHA] = useState(4);
  const [pScoreDiff, setPScoreDiff] = useState(1);

  // engine flags
  const [firstEvent, setFirstEvent] = useState(true);
  const [suspended, setSuspended] = useState(false);
  const [reachBase, setReachBase] = useState(0.32);
  const [pullProb, setPullProb] = useState(0);
  const margin = 0.07;

  // comps + anchoring
  const [useComps, setUseComps] = useState(true);
  const [weight, setWeight] = useState(0.5);
  const [maxDev, setMaxDev] = useState(0.06);
  // comp inputs are RAW STRINGS (so decimals type naturally). Keys are stable
  // per-market ids; the O/U line is chosen dynamically so its comp key is not
  // tied to a specific line number.
  const [comps, setComps] = useState({
    hits_ou_over: "2.20", hits_ou_under: "1.67",
    tb_ou_over: "1.95", tb_ou_under: "1.80",
    hr_m1: "5.00",
    k_ou_over: "2.10", k_ou_under: "1.72",
    ha_ou_over: "1.90", ha_ou_under: "1.90",
  });
  // accept only empty / digits / a single decimal point while typing
  const setComp = (k, v) => {
    if (v === "" || /^\d*\.?\d*$/.test(v)) setComps((c) => ({ ...c, [k]: v }));
  };
  const num = (k) => {
    const v = parseFloat(comps[k]);
    return isNaN(v) ? null : v;
  };

  const gs = { inning, top, outs, awayScore, homeScore, awayDueUp: dueUp - 1, homeDueUp: dueUp - 1 };
  const pregameActive = !firstEvent;

  const result = useMemo(() => {
    // overdispersion factor phi so the model's full-game variance matches the
    // std dev you entered. IMPORTANT: std dev can only WIDEN the distribution
    // (phi >= 1), never tighten it below the natural sampling variance — a hit
    // count built from per-PA coin flips has a hard variance floor.
    const phiFor = (userStd, modelFullVar) => {
      if (!userStd || userStd <= 0) return 1;
      const f = (userStd * userStd) / Math.max(modelFullVar, 1e-6);
      return Math.min(4, Math.max(1.0, f));
    };
    // choose the O/U line from a ladder so the price stays near even money.
    const pickBalancedLine = (cdf, locked, ladder) => {
      let best = ladder[0], bestDiff = Infinity;
      for (const L of ladder) {
        const diff = Math.abs(pOver(cdf, locked, L) - 0.5);
        if (diff < bestDiff) { bestDiff = diff; best = L; }
      }
      return best;
    };
    // main two-way O/U at the auto-selected line. In pre-game mode we serve the
    // book O/U line + prices you entered, verbatim, until the first event.
    const ouMain = (cdf, locked, ladder, prefix, bookLineKey, bookOverKey, bookUnderKey) => {
      const line = pickBalancedLine(cdf, locked, ladder);
      const fo = pOver(cdf, locked, line);
      const fu = 1 - fo;
      const [cofair, cufair] = useComps ? devig(num(prefix + "_over"), num(prefix + "_under")) : [null, null];
      let aOver = toDecimal(anchor(fo, useComps ? cofair : null, weight, maxDev), margin);
      let aUnder = toDecimal(anchor(fu, useComps ? cufair : null, weight, maxDev), margin);
      let shownLine = line;
      if (pregameActive) {                       // book passthrough
        shownLine = fnum(book, bookLineKey, line);
        aOver = fnum(book, bookOverKey) || aOver;
        aUnder = fnum(book, bookUnderKey) || aUnder;
      }
      return { line: shownLine, rows: [
        { label: `Over ${shownLine}`, ourFair: pregameActive ? null : fo, ourPrice: pregameActive ? "book" : toDecimal(fo, margin), compKey: prefix + "_over", compRaw: comps[prefix + "_over"] ?? "", anchoredPrice: aOver },
        { label: `Under ${shownLine}`, ourFair: pregameActive ? null : fu, ourPrice: pregameActive ? "book" : toDecimal(fu, margin), compKey: prefix + "_under", compRaw: comps[prefix + "_under"] ?? "", anchoredPrice: aUnder },
      ] };
    };
    const msRows = (cdf, locked, ladder, prefix) => ladder.map((kPlus) => {
      const p = pMilestone(cdf, locked, kPlus);
      const compKey = `${prefix}_m${kPlus}`;
      const cfair = useComps ? devig(num(compKey), null)[0] : null;
      const a = anchor(p, cfair, weight, maxDev);
      return { label: `${kPlus}+`, ourFair: p, ourPrice: toDecimal(p, margin), compKey, compRaw: comps[compKey] ?? "", anchoredPrice: toDecimal(a, margin) };
    });

    if (kind === "hitter") {
      const xpa = pregameXpaBySlot(slot - 1, battingHome, reachBase);
      // MEANS come from the market: invert book O/U line+price (HR from 1+ price)
      const hitsMean = impliedMeanFromOU(fnum(book, "hits_line", 0.5), bnum("hits_over"), bnum("hits_under"), fnum(std, "hits")) ?? fnum(proj, "hits", 1);
      const tbMean = impliedMeanFromOU(fnum(book, "tb_line", 1.5), bnum("tb_over"), bnum("tb_under"), fnum(std, "tb")) ?? fnum(proj, "tb", 1);
      const hrMean = impliedMeanFrom1Plus(bnum("hr1"), fnum(std, "hr")) ?? fnum(proj, "hr", 0.1);
      const rates = calibrateHitter(hitsMean, tbMean, hrMean, xpa);
      // full-game (t=0) model variances for phi
      const freshGS = { inning: 1, top: !battingHome, outs: 0, awayScore: 0, homeScore: 0, awayDueUp: 0, homeDueUp: 0 };
      const [mNf, vNf] = hitterRemainingPA(freshGS, slot - 1, battingHome, reachBase, 0);
      const phiH = phiFor(fnum(std, "hits"), compoundCount(mNf, vNf, rates.pHit)[1]);
      const phiT = phiFor(fnum(std, "tb"), compoundTB(mNf, vNf, rates)[1]);
      const phiR = phiFor(fnum(std, "hr"), compoundCount(mNf, vNf, rates.pHR)[1]);

      const [mN, vN] = hitterRemainingPA(gs, slot - 1, battingHome, reachBase, pullProb);
      let [hm, hv] = compoundCount(mN, vN, rates.pHit); hv *= phiH;
      const cdfH = countCdf(hm, hv);
      let [tm, tv] = compoundTB(mN, vN, rates); tv *= phiT;
      const cdfT = countCdf(tm, tv, 80);
      let [rm, rv] = compoundCount(mN, vN, rates.pHR); rv *= phiR;
      const cdfR = countCdf(rm, rv);

      const hOU = ouMain(cdfH, lockHits, [0.5, 1.5, 2.5, 3.5], "hits_ou", "hits_line", "hits_over", "hits_under");
      const tOU = ouMain(cdfT, lockTB, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], "tb_ou", "tb_line", "tb_over", "tb_under");
      return {
        xpa, xRPA: mN,
        cards: [
          { title: "Hits", proj: hitsMean, line: hOU.line, rows: [...hOU.rows, ...msRows(cdfH, lockHits, [1, 2, 3, 4], "hits")] },
          { title: "Total Bases", proj: tbMean, line: tOU.line, rows: [...tOU.rows, ...msRows(cdfT, lockTB, [1, 2, 3, 4, 5, 6, 7, 8], "tb")] },
          { title: "Home Runs", proj: hrMean, rows: msRows(cdfR, lockHR, [1, 2], "hr") },
        ],
      };
    } else {
      const bfPre = pregameBF(hookLimit);
      const kMean = impliedMeanFromOU(fnum(book, "k_line", 5.5), bnum("k_over"), bnum("k_under"), fnum(std, "k")) ?? fnum(proj, "k", 1);
      const haMean = impliedMeanFromOU(fnum(book, "ha_line", 5.5), bnum("ha_over"), bnum("ha_under"), fnum(std, "ha")) ?? fnum(proj, "ha", 1);
      const rates = calibratePitcher(kMean, haMean, bfPre);
      const freshPS = { pitchesThrown: 0, battersFaced: 0, tto: 0, isStarter: true, hookPitchLimit: hookLimit, scoreDiff: 0 };
      const [mBFf, vBFf] = pitcherRemainingBF(freshPS);
      const phiK = phiFor(fnum(std, "k"), compoundCount(mBFf, vBFf, rates.pK)[1]);
      const phiA = phiFor(fnum(std, "ha"), compoundCount(mBFf, vBFf, rates.pHit)[1]);

      const ps = { pitchesThrown: pitches, battersFaced: bf, tto, isStarter: true, hookPitchLimit: hookLimit, scoreDiff: pScoreDiff };
      const [mBF, vBF] = pitcherRemainingBF(ps);
      let [km, kv] = compoundCount(mBF, vBF, rates.pK); kv *= phiK;
      const cdfK = countCdf(km, kv, 30);
      let [am, av] = compoundCount(mBF, vBF, rates.pHit); av *= phiA;
      const cdfA = countCdf(am, av, 30);

      const kOU = ouMain(cdfK, lockK, [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5], "k_ou", "k_line", "k_over", "k_under");
      const aOU = ouMain(cdfA, lockHA, [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], "ha_ou", "ha_line", "ha_over", "ha_under");
      return {
        bfPre, xRemBF: mBF,
        cards: [
          { title: "Strikeouts", proj: kMean, line: kOU.line, rows: [...kOU.rows, ...msRows(cdfK, lockK, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12], "k")] },
          { title: "Hits Allowed", proj: haMean, line: aOU.line, rows: [...aOU.rows, ...msRows(cdfA, lockHA, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "ha")] },
        ],
      };
    }
    // eslint-disable-next-line
  }, [kind, proj, std, book, inning, top, outs, awayScore, homeScore, slot, dueUp,
      battingHome, lockHits, lockTB, lockHR, pitches, bf, tto, lockK, lockHA, pScoreDiff,
      reachBase, pullProb, pregameActive, useComps, weight, maxDev, comps]);

  return (
    <div style={{ background: C.bg, minHeight: "100vh", color: C.ink, fontFamily: "system-ui,-apple-system,sans-serif", padding: 18 }}>
      {/* header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 18, borderBottom: `1px solid ${C.line}`, paddingBottom: 14 }}>
        <div>
          <div style={{ fontFamily: C.mono, fontSize: 11, letterSpacing: 2, color: C.accent }}>IN-PLAY · MLB PLAYER PROPS</div>
          <h1 style={{ margin: "4px 0 0", fontSize: 24, fontWeight: 650, letterSpacing: -0.5 }}>Live Pricing Terminal</h1>
        </div>
        <div style={{ fontFamily: C.mono, fontSize: 11, color: C.dim, textAlign: "right", lineHeight: 1.7 }}>
          <div>margin <span style={{ color: C.ink }}>7.0%</span></div>
          <div>engine <span style={{ color: pregameActive ? C.comp : C.accent }}>{pregameActive ? "PRE-GAME BOOK" : "LIVE MODEL"}</span></div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "340px 1fr", gap: 18, alignItems: "start" }}>
        {/* ---------------- LEFT: controls ---------------- */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div style={{ background: C.panel, border: `1px solid ${C.line}`, borderRadius: 8, padding: 14, display: "flex", flexDirection: "column", gap: 12 }}>
            <Toggle on={kind === "pitcher"} set={(v) => setKind(v ? "pitcher" : "hitter")} labels={["HITTER", "PITCHER"]} />
            <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 1, color: C.faint, textTransform: "uppercase" }}>Pre-match inputs</div>
            {/* header */}
            <div style={{ display: "grid", gridTemplateColumns: "1.3fr 1fr 1fr 0.9fr 0.9fr 0.9fr", gap: 6, alignItems: "end" }}>
              {["Stat", "Proj·ref", "StdDev", "Bk line", "Bk O", "Bk U"].map((h) => (
                <span key={h} style={{ fontFamily: C.mono, fontSize: 8.5, letterSpacing: 0.4, color: C.faint, textTransform: "uppercase" }}>{h}</span>
              ))}
              {(kind === "hitter"
                ? [["Hits", "hits", "hits_line", "hits_over", "hits_under"],
                   ["Total bases", "tb", "tb_line", "tb_over", "tb_under"]]
                : [["Strikeouts", "k", "k_line", "k_over", "k_under"],
                   ["Hits allowed", "ha", "ha_line", "ha_over", "ha_under"]]
              ).map(([label, sk, lk, ok, uk]) => (
                <React.Fragment key={sk}>
                  <span style={{ fontFamily: C.mono, fontSize: 11, color: C.dim, alignSelf: "center" }}>{label}</span>
                  <MiniInput value={proj[sk]} onChange={(v) => setPr(sk, v)} />
                  <MiniInput value={std[sk]} onChange={(v) => setSd(sk, v)} />
                  <MiniInput value={book[lk]} onChange={(v) => setBk(lk, v)} />
                  <MiniInput value={book[ok]} onChange={(v) => setBk(ok, v)} />
                  <MiniInput value={book[uk]} onChange={(v) => setBk(uk, v)} />
                </React.Fragment>
              ))}
              {kind === "hitter" && (
                <React.Fragment>
                  <span style={{ fontFamily: C.mono, fontSize: 11, color: C.dim, alignSelf: "center" }}>Home runs</span>
                  <MiniInput value={proj.hr} onChange={(v) => setPr("hr", v)} />
                  <MiniInput value={std.hr} onChange={(v) => setSd("hr", v)} />
                  <span style={{ fontFamily: C.mono, fontSize: 9, color: C.faint, alignSelf: "center" }}>1+ px</span>
                  <MiniInput value={book.hr1} onChange={(v) => setBk("hr1", v)} />
                  <span />
                </React.Fragment>
              )}
            </div>
            <div style={{ fontFamily: C.mono, fontSize: 9.5, color: C.faint, lineHeight: 1.6 }}>
              Engine derives each stat's expected value from the book O/U <b>line + price</b> (HR from the 1+ price). <b>Proj</b> is your internal reference only — not used for pricing. StdDev shapes the spread. See each market header for the derived projection.
            </div>
          </div>

          {/* engine flags */}
          <div style={{ background: C.panel, border: `1px solid ${C.line}`, borderRadius: 8, padding: 14, display: "flex", flexDirection: "column", gap: 12 }}>
            <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 1, color: C.faint, textTransform: "uppercase" }}>Engine</div>
            <Field label="Player has had first in-game event">
              <Toggle on={firstEvent} set={setFirstEvent} labels={["NOT YET · book", "YES · live"]} />
            </Field>
            <Field label="Market suspended (at-bat / pitch live)">
              <Toggle on={suspended} set={setSuspended} labels={["OPEN", "SUSPENDED"]} />
            </Field>
          </div>

          {/* game state */}
          <div style={{ background: C.panel, border: `1px solid ${C.line}`, borderRadius: 8, padding: 14, display: "flex", flexDirection: "column", gap: 12 }}>
            <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 1, color: C.faint, textTransform: "uppercase" }}>Game State</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <Field label="Inning"><Stepper value={inning} set={setInning} min={1} max={15} /></Field>
              <Field label="Half"><Toggle on={!top} set={(v) => setTop(!v)} labels={["TOP", "BOT"]} /></Field>
              <Field label="Outs"><Stepper value={outs} set={setOuts} min={0} max={2} /></Field>
              <Field label="Batting side"><Toggle on={battingHome} set={setBattingHome} labels={["AWAY", "HOME"]} /></Field>
              <Field label="Away runs"><Stepper value={awayScore} set={setAwayScore} min={0} max={30} /></Field>
              <Field label="Home runs"><Stepper value={homeScore} set={setHomeScore} min={0} max={30} /></Field>
            </div>
          </div>

          {kind === "hitter" ? (
            <>
              <div style={{ background: C.panel, border: `1px solid ${C.line}`, borderRadius: 8, padding: 14, display: "flex", flexDirection: "column", gap: 12 }}>
                <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 1, color: C.faint, textTransform: "uppercase" }}>Lineup</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                  <Field label="His slot (1–9)"><Stepper value={slot} set={setSlot} min={1} max={9} /></Field>
                  <Field label="Due up (1–9)"><Stepper value={dueUp} set={setDueUp} min={1} max={9} /></Field>
                </div>
                <Field label={`Pull / pinch-hit prob (${(pullProb * 100).toFixed(0)}%)`}>
                  <input type="range" min={0} max={0.8} step={0.05} value={pullProb} onChange={(e) => setPullProb(parseFloat(e.target.value))} style={{ width: "100%", accentColor: C.accent }} />
                </Field>
              </div>
              <div style={{ background: C.panel, border: `1px solid ${C.line}`, borderRadius: 8, padding: 14, display: "flex", flexDirection: "column", gap: 12 }}>
                <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 1, color: C.faint, textTransform: "uppercase" }}>Accumulated (locked)</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
                  <Field label="Hits"><Stepper value={lockHits} set={setLockHits} min={0} max={7} /></Field>
                  <Field label="Total bases"><Stepper value={lockTB} set={setLockTB} min={0} max={14} /></Field>
                  <Field label="Home runs"><Stepper value={lockHR} set={setLockHR} min={0} max={4} /></Field>
                </div>
              </div>
            </>
          ) : (
            <div style={{ background: C.panel, border: `1px solid ${C.line}`, borderRadius: 8, padding: 14, display: "flex", flexDirection: "column", gap: 12 }}>
              <div style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 1, color: C.faint, textTransform: "uppercase" }}>Pitcher State</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                <Field label="Pitch count"><Stepper value={pitches} set={setPitches} min={0} max={130} step={5} /></Field>
                <Field label="Batters faced"><Stepper value={bf} set={setBF} min={0} max={40} /></Field>
                <Field label="Times thru order"><Stepper value={tto} set={setTto} min={0} max={4} step={0.1} /></Field>
                <Field label="Score diff (his team)"><Stepper value={pScoreDiff} set={setPScoreDiff} min={-15} max={15} /></Field>
                <Field label="K locked"><Stepper value={lockK} set={setLockK} min={0} max={20} /></Field>
                <Field label="Hits allowed locked"><Stepper value={lockHA} set={setLockHA} min={0} max={20} /></Field>
              </div>
            </div>
          )}

          {/* comps */}
          <div style={{ background: C.panel, border: `1px solid ${C.line}`, borderRadius: 8, padding: 14, display: "flex", flexDirection: "column", gap: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontFamily: C.mono, fontSize: 10, letterSpacing: 1, color: C.comp, textTransform: "uppercase" }}>Comps Anchor</span>
              <div style={{ width: 120 }}><Toggle on={useComps} set={setUseComps} labels={["OFF", "ON"]} /></div>
            </div>
            <Field label={`Anchor weight — ${(weight * 100).toFixed(0)}% toward comp`}>
              <input type="range" min={0} max={1} step={0.05} value={weight} onChange={(e) => setWeight(parseFloat(e.target.value))} style={{ width: "100%", accentColor: C.comp }} />
            </Field>
            <Field label={`Max deviation from comp — ${(maxDev * 100).toFixed(0)} pts of prob`}>
              <input type="range" min={0.01} max={0.2} step={0.01} value={maxDev} onChange={(e) => setMaxDev(parseFloat(e.target.value))} style={{ width: "100%", accentColor: C.comp }} />
            </Field>
            <div style={{ fontFamily: C.mono, fontSize: 10, color: C.faint }}>Enter competitor decimal prices below each market. Two-way inputs are de-vigged before anchoring.</div>
          </div>
        </div>

        {/* ---------------- RIGHT: board ---------------- */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {/* live opportunity strip */}
          <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
            {kind === "hitter" ? (
              <>
                <Stat label="pre-game xPA (modeled)" value={result.xpa.toFixed(2)} />
                <Stat label="live xRPA (remaining)" value={result.xRPA.toFixed(2)} accent />
                <Stat label="engine" value={pregameActive ? "book" : "live"} accent={!pregameActive} />
              </>
            ) : (
              <>
                <Stat label="pre-game BF (modeled)" value={result.bfPre.toFixed(2)} />
                <Stat label="live xRem BF" value={result.xRemBF.toFixed(2)} accent />
                <Stat label="engine" value={pregameActive ? "book" : "live"} accent={!pregameActive} />
              </>
            )}
          </div>

          {pregameActive && (
            <div style={{ background: "#1c1608", border: `1px solid ${C.comp}`, borderRadius: 8, padding: "12px 14px", fontFamily: C.mono, fontSize: 12, color: C.comp }}>
              Player has not had a first in-game event yet → serving the <b>pre-game book price</b>. The live model output below is what will take over the instant the first event lands.
            </div>
          )}

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            {result.cards.map((card, i) => (
              <div key={i} style={{ gridColumn: card.title === "Home Runs" ? "span 2" : "span 1" }}>
                <MarketCard
                  title={card.title}
                  proj={card.proj}
                  line={card.line}
                  suspended={suspended}
                  rows={card.rows.map((r, j) => (
                    <PriceRow key={j} row={r} suspended={suspended} setComp={setComp} />
                  ))}
                />
              </div>
            ))}
          </div>

          <div style={{ fontFamily: C.mono, fontSize: 11, color: C.faint, lineHeight: 1.7, borderTop: `1px solid ${C.line}`, paddingTop: 12 }}>
            <span style={{ color: C.dim }}>Column key:</span> <span style={{ color: C.faint }}>model</span> = fair % and raw price off the live distribution ·
            <span style={{ color: C.comp }}> comp</span> = competitor decimal you entered ·
            <span style={{ color: C.accent }}> our price</span> = model anchored toward comp (weight {(weight * 100).toFixed(0)}%, capped {(maxDev * 100).toFixed(0)}pts) then loaded 7%.
            {" "}Achieved milestones lock to 1.00; suspended markets take no bets.
          </div>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value, accent }) {
  return (
    <div style={{ background: C.panel, border: `1px solid ${C.line}`, borderRadius: 8, padding: "10px 16px", minWidth: 150 }}>
      <div style={{ fontFamily: C.mono, fontSize: 9.5, letterSpacing: 0.6, color: C.faint, textTransform: "uppercase" }}>{label}</div>
      <div style={{ fontFamily: C.mono, fontSize: 22, fontWeight: 600, color: accent ? C.accent : C.ink, marginTop: 2 }}>{value}</div>
    </div>
  );
}
