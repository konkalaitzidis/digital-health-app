// frontend/adamma-frontend/App.js
import React, { useEffect, useRef, useState } from "react";
import {
  SafeAreaView, Text, View, StyleSheet, StatusBar, Pressable,
  ScrollView, KeyboardAvoidingView, Platform, TextInput
} from "react-native";
import { Accelerometer } from "expo-sensors";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Network from "expo-network";

const CLASSES = ["Sedentary", "Light", "Moderate", "Vigorous"];
const COLORS = {
  Sedentary: "#9CA3AF",
  Light: "#10B981",
  Moderate: "#3B82F6",
  Vigorous: "#EF4444",
};

const FS = 20;
const WIN_SEC = 5;
const WIN = FS * WIN_SEC;           // 100
const OVERLAP = 0.5;
const STEP = Math.floor(WIN * (1 - OVERLAP)); // 50
const STORAGE_KEY = "ADAMMA_BACKEND_BASE";

// Fallback default; Auto-Detect/Deep Scan will replace this
const DEFAULT_BACKEND_BASE = "http://10.200.30.140:8000";

// ---------- helpers ----------
async function fetchWithTimeout(url, options = {}, ms = 600) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), ms);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(t);
  }
}

async function pingBase(base, timeoutMs = 700) {
  try {
    const clean = (base || "").replace(/\/+$/, "");
    const res = await fetchWithTimeout(`${clean}/ping`, {}, timeoutMs);
    return res.ok;
  } catch {
    return false;
  }
}

// limited-concurrency mapper
async function mapLimit(items, limit, worker) {
  const results = new Array(items.length);
  let i = 0, active = 0;
  return new Promise(resolve => {
    const next = () => {
      if (i >= items.length && active === 0) return resolve(results);
      while (active < limit && i < items.length) {
        const idx = i++;
        active++;
        Promise.resolve(worker(items[idx], idx))
          .then(r => { results[idx] = r; })
          .finally(() => { active--; next(); });
      }
    };
    next();
  });
}

function majorityVote(arr) {
  const counts = {};
  for (const x of arr) counts[x] = (counts[x] || 0) + 1;
  let best = null, bestC = -1;
  for (const k of Object.keys(counts)) {
    if (counts[k] > bestC) { best = k; bestC = counts[k]; }
  }
  return best;
}

export default function App() {
  const [backendBase, setBackendBase] = useState(DEFAULT_BACKEND_BASE);
  const [showSettings, setShowSettings] = useState(false);

  const [current, setCurrent] = useState("Sedentary");
  const [timers, setTimers] = useState({ Sedentary:0, Light:0, Moderate:0, Vigorous:0 });
  const [status, setStatus] = useState("Starting…");

  const bufferRef = useRef([]);            // [{accel_x, accel_y, accel_z}]
  const postingRef = useRef(false);
  const lastPostTsRef = useRef(0);
  const predsRef = useRef([]);             // smoothing
  const tickRef = useRef(null);
  const ignoreUntilTsRef = useRef(0);
  const firstBootRef = useRef(true);

  // Load saved backend on mount
  useEffect(() => {
    (async () => {
      try {
        const saved = await AsyncStorage.getItem(STORAGE_KEY);
        if (saved) setBackendBase(saved);
      } catch {}
    })();
  }, []);

  // First boot: verify current base; if unreachable, try Auto-Detect (quick)
  useEffect(() => {
    if (!firstBootRef.current) return;
    firstBootRef.current = false;
    (async () => {
      const ok = await pingBase(backendBase);
      if (ok) setStatus(`Backend OK: ${backendBase}`);
      else await autodetectBackend({ deep:false });
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backendBase]);

  // Accelerometer stream @ ~20 Hz
  useEffect(() => {
    Accelerometer.setUpdateInterval(50); // ms → ~20 Hz
    const sub = Accelerometer.addListener(({ x, y, z }) => {
      bufferRef.current.push({ accel_x: x, accel_y: y, accel_z: z });

      const now = Date.now();
      if (now < ignoreUntilTsRef.current) return;

      if (bufferRef.current.length >= WIN && !postingRef.current) {
        if (now - lastPostTsRef.current < 1000) return; // throttle 1s
        const windowSamples = bufferRef.current.slice(-WIN);
        bufferRef.current = bufferRef.current.slice(-(WIN - STEP));

        postingRef.current = true;
        lastPostTsRef.current = now;
        classify(windowSamples).finally(() => { postingRef.current = false; });
      }
    });
    setStatus((s) => (s.startsWith("Backend") ? s : "Sensor ON"));
    return () => { sub && sub.remove(); setStatus("Sensor OFF"); };
  }, []);

  // Per-second timers
  useEffect(() => {
    tickRef.current && clearInterval(tickRef.current);
    tickRef.current = setInterval(() => {
      setTimers(prev => ({ ...prev, [current]: prev[current] + 1 }));
    }, 1000);
    return () => clearInterval(tickRef.current);
  }, [current]);

  function predictUrl() {
    const base = (backendBase || DEFAULT_BACKEND_BASE).replace(/\/+$/, "");
    return `${base}/predict`;
  }

  // ---- Auto-Detect / Deep Scan ----
  async function autodetectBackend({ deep = false } = {}) {
    try {
      setStatus(deep ? "Deep scanning LAN…" : "Detecting backend…");

      // Keep current base if OK
      if (await pingBase(backendBase)) {
        setStatus(`Backend OK: ${backendBase}`);
        return;
      }

      const ip = await Network.getIpAddressAsync(); // e.g., "192.168.1.23"
      const parts = (ip || "").split(".");
      if (parts.length !== 4) throw new Error(`Bad device IP: ${ip}`);
      const [a, b, c, dStr] = parts;
      const d = parseInt(dStr, 10);
      const subnet = `${a}.${b}.${c}`;

      const quickCandidates = [
        `${subnet}.1`,
        `${subnet}.${Math.max(2, d - 1)}`,
        `${subnet}.${d}`,
        `${subnet}.${Math.min(254, d + 1)}`,
        `${subnet}.10`,
        `${subnet}.20`,
        `${subnet}.30`,
        `${subnet}.40`,
        `${subnet}.50`,
        `${subnet}.100`,
        `${subnet}.140`, // your known box
        `${subnet}.154`,
      ];

      // Quick pass
      for (const host of quickCandidates) {
        const base = `http://${host}:8000`;
        if (await pingBase(base, 700)) {
          setBackendBase(base);
          try { await AsyncStorage.setItem(STORAGE_KEY, base); } catch {}
          setStatus(`Backend detected: ${base}`);
          return;
        }
      }

      if (!deep) {
        setStatus("Quick detect failed. Try Deep Scan.");
        return;
      }

      // Deep scan /24 with concurrency
      const allHosts = [];
      for (let x = 2; x <= 254; x++) allHosts.push(`${subnet}.${x}`);
      const likely = new Set(quickCandidates);
      allHosts.sort((h1, h2) => (likely.has(h2) ? 1 : 0) - (likely.has(h1) ? 1 : 0));

      const foundRef = { base: null };
      await mapLimit(allHosts, 48, async (host) => {
        if (foundRef.base) return null;
        const base = `http://${host}:8000`;
        const ok = await pingBase(base, 500);
        if (ok && !foundRef.base) foundRef.base = base;
        return ok ? base : null;
      });

      if (foundRef.base) {
        setBackendBase(foundRef.base);
        try { await AsyncStorage.setItem(STORAGE_KEY, foundRef.base); } catch {}
        setStatus(`Backend detected: ${foundRef.base}`);
      } else {
        setStatus("Deep scan didn’t find a backend. Enter URL or use ngrok.");
      }
    } catch (e) {
      setStatus(`Detect failed: ${e?.message || e}`);
    }
  }

  async function testBackend() {
    const base = (backendBase || DEFAULT_BACKEND_BASE).replace(/\/+$/, "");
    const ok = await pingBase(base);
    setStatus(ok ? `Ping OK: ${base}` : `Ping FAIL: ${base}`);
  }

  // ---- Prediction ----
  async function classify(samples) {
    const url = predictUrl();
    setStatus(`Predicting… (${samples.length})`);
    console.log("POST /predict →", url, "samples:", samples.length);

    const payloadSamples = samples.map(s => ({
      accel_x: s.accel_x ?? s.x ?? 0,
      accel_y: s.accel_y ?? s.y ?? 0,
      accel_z: s.accel_z ?? s.z ?? 0,
    }));

    const controller = new AbortController();
    const timeout = setTimeout(() => {
      console.log("POST /predict timeout → aborting");
      controller.abort();
    }, 10000); // reduce to 4–6s after network is stable

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ samples: payloadSamples }),
        signal: controller.signal,
      });

      console.log("POST /predict status", res.status);

      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        console.log("predict non-200:", res.status, txt);
        setStatus(`API ${res.status}`);
        return;
      }

      const data = await res.json();
      const cls = data?.met_class || data?.class || "Sedentary";

      predsRef.current.push(cls);
      if (predsRef.current.length > 3) predsRef.current.shift();
      const smooth = majorityVote(predsRef.current);

      setCurrent(smooth);
      setStatus(`OK: ${smooth}`);
    } catch (e) {
      console.log("predict error:", e?.message || e);
      setStatus("Offline or timeout (keeping last class)");
    } finally {
      clearTimeout(timeout);
    }
  }

  function handleReset() {
    setTimers({ Sedentary:0, Light:0, Moderate:0, Vigorous:0 });
    predsRef.current = [];
    bufferRef.current = [];
    lastPostTsRef.current = 0;
    ignoreUntilTsRef.current = Date.now() + 1500;
    setCurrent("Sedentary");
    setStatus("Reset ✓");
  }

  async function handleSaveBackend() {
    let base = (backendBase || "").trim();
    if (base && !/^https?:\/\//.test(base)) base = `http://${base}`;
    setBackendBase(base);
    try { await AsyncStorage.setItem(STORAGE_KEY, base); } catch {}
    setShowSettings(false);
  }

  const fmt = s => `${Math.floor(s/60)}m ${s%60}s`;

  const total = timers.Sedentary + timers.Light + timers.Moderate + timers.Vigorous;
  const active = timers.Light + timers.Moderate + timers.Vigorous;
  const mvpa = timers.Moderate + timers.Vigorous;
  const pct = (part, whole) => (whole > 0 ? Math.round((part / whole) * 100) : 0);
  const activePct = pct(active, total);
  const mvpaPct = pct(mvpa, total);

  const SummaryRow = ({ label, value }) => (
    <View style={styles.row}>
      <Text style={styles.summaryLabel}>{label}</Text>
      <Text style={styles.value}>{fmt(value)}</Text>
    </View>
  );

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" />
      <KeyboardAvoidingView style={{ flex:1 }} behavior={Platform.OS === "ios" ? "padding" : undefined}>
        <ScrollView contentContainerStyle={styles.scroll}>
          <View style={styles.headerRow}>
            <Text style={styles.title}>Live MET Tracker</Text>
            <Pressable onPress={() => setShowSettings(s => !s)} style={({ pressed }) => [styles.settingsBtn, pressed && { opacity: 0.85 }]}>
              <Text style={styles.settingsText}>⚙︎ Settings</Text>
            </Pressable>
          </View>

          {showSettings && (
            <View style={styles.card}>
              <Text style={styles.section}>Backend</Text>
              <Text style={styles.subtle}>
                Auto-detect finds a local backend at port 8000 on your Wi-Fi. You can also paste a LAN IP or an ngrok HTTPS URL.
              </Text>
              <TextInput
                value={backendBase}
                onChangeText={setBackendBase}
                placeholder="http://YOUR-IP:8000  or  https://<ngrok>.ngrok.io"
                autoCapitalize="none"
                autoCorrect={false}
                style={styles.input}
              />
              <View style={{ flexDirection:"row", gap:8, flexWrap:"wrap" }}>
                <Pressable onPress={handleSaveBackend} style={styles.btn}><Text style={styles.btnText}>Save</Text></Pressable>
                <Pressable onPress={() => autodetectBackend({ deep:false })} style={styles.btn}><Text style={styles.btnText}>Auto-Detect</Text></Pressable>
                <Pressable onPress={() => autodetectBackend({ deep:true })} style={styles.btn}><Text style={styles.btnText}>Deep Scan</Text></Pressable>
                <Pressable onPress={testBackend} style={styles.btn}><Text style={styles.btnText}>Test</Text></Pressable>
              </View>
              <Text style={[styles.subtle, { marginTop:6 }]}>Predict URL will be: {predictUrl()}</Text>
            </View>
          )}

          <View style={styles.card}>
            <Text style={styles.section}>Current</Text>
            <View style={[styles.currentPill, { backgroundColor: COLORS[current] + "22", borderColor: COLORS[current] }]}>
              <View style={[styles.dot, { backgroundColor: COLORS[current] }]} />
              <Text style={[styles.current, { color: COLORS[current] }]}>{current}</Text>
            </View>
            <Text style={styles.subtle}>Sampling ~{FS} Hz • window {WIN_SEC}s • 50% overlap</Text>
            <Text style={styles.subtle}>Backend: {backendBase || DEFAULT_BACKEND_BASE}</Text>
            <Text style={styles.subtle}>Status: {status}</Text>

            <Pressable onPress={handleReset} style={({ pressed }) => [styles.btn, pressed && { opacity: 0.8 }]}>
              <Text style={styles.btnText}>Reset Day</Text>
            </Pressable>
          </View>

          <View style={styles.card}>
            <Text style={styles.section}>Today</Text>
            {CLASSES.map(c => (
              <View key={c} style={styles.row}>
                <View style={styles.rowLeft}>
                  <View style={[styles.dot, { backgroundColor: COLORS[c] }]} />
                  <Text style={[styles.label, { color: COLORS[c] }]}>{c}</Text>
                </View>
                <Text style={styles.value}>{fmt(timers[c])}</Text>
              </View>
            ))}
          </View>

          <View style={styles.card}>
            <Text style={styles.section}>Session Summary</Text>
            <SummaryRow label="Total time" value={total} />
            <SummaryRow label="Active time (L+M+V)" value={active} />
            <View style={styles.row}>
              <Text style={styles.summaryLabel}>Active %</Text>
              <Text style={styles.value}>{activePct}%</Text>
            </View>
            <SummaryRow label="MVPA (M+V)" value={mvpa} />
            <View style={styles.row}>
              <Text style={styles.summaryLabel}>MVPA %</Text>
              <Text style={styles.value}>{mvpaPct}%</Text>
            </View>
          </View>

          <View style={styles.footer}>
            <Text style={styles.footerText}>Created by Konstantinos Kalaitzidis</Text>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex:1, backgroundColor:"#fff" },
  scroll: { padding:20, paddingBottom:28 },

  headerRow: { flexDirection:"row", justifyContent:"space-between", alignItems:"center", marginBottom:8 },

  title: { fontSize:20, fontWeight:"700", textAlign:"left" },
  settingsBtn: { paddingHorizontal:12, paddingVertical:6, backgroundColor:"#111827", borderRadius:8 },
  settingsText: { color:"#fff", fontWeight:"700" },

  container: { flex:1, padding:20, backgroundColor:"#fff", justifyContent:"space-between" },
  content: { flexGrow:1, gap:16 },

  card: { borderWidth:1, borderColor:"#e5e7eb", borderRadius:12, padding:16, gap:10, backgroundColor:"#fff", marginBottom:12 },
  section: { fontSize:12, color:"gray", textTransform:"uppercase", letterSpacing:0.5 },

  current: { fontSize:20, fontWeight:"800" },
  currentPill: {
    flexDirection:"row",
    alignItems:"center",
    gap:8,
    borderWidth:1,
    paddingVertical:8,
    paddingHorizontal:12,
    borderRadius:9999,
    alignSelf:"flex-start",
  },

  dot: { width:10, height:10, borderRadius:5 },

  row: { flexDirection:"row", justifyContent:"space-between", alignItems:"center", paddingVertical:10 },
  rowLeft: { flexDirection:"row", alignItems:"center", gap:8 },

  label: { fontSize:16, fontWeight:"700" },
  summaryLabel: { fontSize:16, fontWeight:"600" },

  value: { fontSize:16, fontVariant:["tabular-nums"] },
  subtle: { color:"gray", fontSize:12 },

  btn: {
    marginTop:6,
    alignSelf:"flex-start",
    backgroundColor:"#111827",
    paddingVertical:10,
    paddingHorizontal:14,
    borderRadius:10,
  },
  btnText: { color:"#fff", fontWeight:"700" },

  footer: { paddingVertical:10, alignItems:"center" },
  footerText: { fontSize:12, color:"#000", opacity:0.7 },

  input: { borderWidth:1, borderColor:"#d1d5db", borderRadius:8, paddingHorizontal:10, paddingVertical:8, fontSize:14, marginTop:6 },
});
