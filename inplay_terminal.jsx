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
  else if (variance <= mean * 1.05) {           // Poisson
    pmf[0] = Math.exp(-mean);
    for (let k = 1; k <= kmax; k++) pmf[k] = pmf[k - 1] * mean / k;
  } else {                                        // Negative Binomial
    const v = Math.max(variance, mean * 1.0000001);
    const p = mean / v, r = (mean * mean) / (v - mean);
    for (let k = 0; k <= kmax; k++) pmf[k] = Math.exp(nbLogPmf(k, r, p));
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
  if (gs.inning >= 8 && gs.awayScore === gs.homeScore) {
    const pEx = 0.35;
    meanPA += pEx * 2 * paPerInnMean;
    varPA += pEx * (1 - pEx) * Math.pow(2 * paPerInnMean, 2);
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

// ---------- sample players (from the pre-game screenshots) ----------
const HITTERS = {
  "Paul Goldschmidt": { hits: 1.5, tb: 2.2, hr: 0.2, team: "NYY" },
  "Ben Rice":         { hits: 1.5, tb: 4.5, hr: 0.4, team: "NYY" },
  "Riley Greene":     { hits: 1.25, tb: 2.5, hr: 0.25, team: "DET" },
  "Cody Bellinger":   { hits: 1.5, tb: 2.5, hr: 0.25, team: "NYY" },
  "Jazz Chisholm Jr.":{ hits: 1.22, tb: 3.0, hr: 0.3, team: "NYY" },
  "Kerry Carpenter":  { hits: 1.11, tb: 3.0, hr: 0.3, team: "DET" },
};
const PITCHERS = {
  "Will Warren":  { k: 6.14, ha: 5.5, team: "NYY", limit: 95 },
  "Troy Melton":  { k: 5.01, ha: 5.91, team: "DET", limit: 90 },
};

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
  const { label, ourFair, ourPrice, compKey, compPrice, anchoredPrice } = row;
  const deviates = compPrice && anchoredPrice && Math.abs(anchoredPrice - compPrice) / compPrice > 0.08;
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
        value={compPrice ?? ""} placeholder="—"
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

function MarketCard({ title, rows, suspended, note }) {
  return (
    <div style={{ background: C.panel, border: `1px solid ${C.line}`, borderRadius: 8, overflow: "hidden" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center",
        padding: "10px 12px", borderBottom: `1px solid ${C.line}`, background: C.panel2 }}>
        <span style={{ fontFamily: C.mono, fontSize: 12, letterSpacing: 0.5, color: C.ink, textTransform: "uppercase" }}>{title}</span>
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
  const [hitterName, setHitterName] = useState("Paul Goldschmidt");
  const [pitcherName, setPitcherName] = useState("Will Warren");

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
  // comp inputs keyed by market id: {over, under} decimal prices
  const [comps, setComps] = useState({
    hits15_over: 2.2, hits15_under: 1.67,
    tb15_over: 1.95, tb15_under: 1.8,
    hr1: 5.0,
    k65_over: 2.1, k65_under: 1.72,
    ha55_over: 1.9, ha55_under: 1.9,
  });
  const setComp = (k, v) => setComps((c) => ({ ...c, [k]: v === "" ? null : parseFloat(v) }));

  const gs = { inning, top, outs, awayScore, homeScore, awayDueUp: dueUp - 1, homeDueUp: dueUp - 1 };

  const result = useMemo(() => {
    // helper to build a two-way OU row
    const ouRow = (label, cdf, locked, line, compKey) => {
      const fo = pOver(cdf, locked, line);
      const fu = 1 - fo;
      const [cofair, cufair] = useComps ? devig(comps[compKey + "_over"], comps[compKey + "_under"]) : [null, null];
      const aOver = anchor(fo, useComps ? cofair : null, weight, maxDev);
      const aUnder = anchor(fu, useComps ? cufair : null, weight, maxDev);
      return [
        { label: label + " Over", ourFair: fo, ourPrice: toDecimal(fo, margin), compKey: compKey + "_over", compPrice: comps[compKey + "_over"], anchoredPrice: toDecimal(aOver, margin) },
        { label: label + " Under", ourFair: fu, ourPrice: toDecimal(fu, margin), compKey: compKey + "_under", compPrice: comps[compKey + "_under"], anchoredPrice: toDecimal(aUnder, margin) },
      ];
    };
    const msRow = (label, cdf, locked, kPlus, compKey) => {
      const p = pMilestone(cdf, locked, kPlus);
      const cfair = useComps ? devig(comps[compKey], null)[0] : null;
      const a = anchor(p, cfair, weight, maxDev);
      return { label, ourFair: p, ourPrice: toDecimal(p, margin), compKey, compPrice: comps[compKey] ?? null, anchoredPrice: toDecimal(a, margin) };
    };

    if (kind === "hitter") {
      const P = HITTERS[hitterName];
      const xpa = pregameXpaBySlot(slot - 1, battingHome, reachBase);
      const rates = calibrateHitter(P.hits, P.tb, P.hr, xpa);
      const [mN, vN] = hitterRemainingPA(gs, slot - 1, battingHome, reachBase, pullProb);
      const [hm, hv] = compoundCount(mN, vN, rates.pHit);
      const cdfH = countCdf(hm, hv);
      const [tm, tv] = compoundTB(mN, vN, rates);
      const cdfT = countCdf(tm, tv, 80);
      const [rm, rv] = compoundCount(mN, vN, rates.pHR);
      const cdfR = countCdf(rm, rv);
      return {
        xpa, xRPA: mN,
        cards: [
          { title: "Hits", rows: [...ouRow("1.5", cdfH, lockHits, 1.5, "hits15"),
            msRow("1+", cdfH, lockHits, 1, "hits1"), msRow("2+", cdfH, lockHits, 2, "hits2"), msRow("3+", cdfH, lockHits, 3, "hits3")] },
          { title: "Total Bases", rows: [...ouRow("1.5", cdfT, lockTB, 1.5, "tb15"),
            msRow("2+", cdfT, lockTB, 2, "tb2"), msRow("3+", cdfT, lockTB, 3, "tb3"), msRow("4+", cdfT, lockTB, 4, "tb4")] },
          { title: "Home Runs", rows: [msRow("1+", cdfR, lockHR, 1, "hr1"), msRow("2+", cdfR, lockHR, 2, "hr2")] },
        ],
      };
    } else {
      const P = PITCHERS[pitcherName];
      const bfPre = pregameBF(P.limit);
      const rates = calibratePitcher(P.k, P.ha, bfPre);
      const ps = { pitchesThrown: pitches, battersFaced: bf, tto, isStarter: true, hookPitchLimit: P.limit, scoreDiff: pScoreDiff };
      const [mBF, vBF] = pitcherRemainingBF(ps);
      const [km, kv] = compoundCount(mBF, vBF, rates.pK);
      const cdfK = countCdf(km, kv, 30);
      const [am, av] = compoundCount(mBF, vBF, rates.pHit);
      const cdfA = countCdf(am, av, 30);
      return {
        bfPre, xRemBF: mBF,
        cards: [
          { title: "Strikeouts", rows: [...ouRow("6.5", cdfK, lockK, 6.5, "k65"),
            msRow("4+", cdfK, lockK, 4, "k4"), msRow("6+", cdfK, lockK, 6, "k6"), msRow("8+", cdfK, lockK, 8, "k8")] },
          { title: "Hits Allowed", rows: [...ouRow("5.5", cdfA, lockHA, 5.5, "ha55"),
            msRow("4+", cdfA, lockHA, 4, "ha4"), msRow("6+", cdfA, lockHA, 6, "ha6")] },
        ],
      };
    }
    // eslint-disable-next-line
  }, [kind, hitterName, pitcherName, inning, top, outs, awayScore, homeScore, slot, dueUp,
      battingHome, lockHits, lockTB, lockHR, pitches, bf, tto, lockK, lockHA, pScoreDiff,
      reachBase, pullProb, useComps, weight, maxDev, comps]);

  const pregameActive = !firstEvent;
  const P = kind === "hitter" ? HITTERS[hitterName] : PITCHERS[pitcherName];

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
            {kind === "hitter" ? (
              <Field label="Player">
                <select style={inputStyle} value={hitterName} onChange={(e) => setHitterName(e.target.value)}>
                  {Object.keys(HITTERS).map((n) => <option key={n}>{n}</option>)}
                </select>
              </Field>
            ) : (
              <Field label="Player">
                <select style={inputStyle} value={pitcherName} onChange={(e) => setPitcherName(e.target.value)}>
                  {Object.keys(PITCHERS).map((n) => <option key={n}>{n}</option>)}
                </select>
              </Field>
            )}
            <div style={{ fontFamily: C.mono, fontSize: 10.5, color: C.faint, lineHeight: 1.6 }}>
              {kind === "hitter"
                ? `book proj — hits ${P.hits} · TB ${P.tb} · HR ${P.hr}`
                : `book proj — K ${P.k} · hits allowed ${P.ha} · hook ${P.limit}p`}
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
