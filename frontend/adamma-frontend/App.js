// frontend/adamma-frontend/App.js
import React, { useEffect, useRef, useState } from "react";
import {
  SafeAreaView, Text, View, StyleSheet, StatusBar, Pressable,
  ScrollView, KeyboardAvoidingView, Platform, TextInput
} from "react-native";
import { Accelerometer } from "expo-sensors";
import AsyncStorage from "@react-native-async-storage/async-storage";

const CLASSES = ["Sedentary", "Light", "Moderate", "Vigorous"];
const COLORS = {
  Sedentary: "#9CA3AF", // gray-400
  Light: "#10B981",     // emerald-500
  Moderate: "#3B82F6",  // blue-500
  Vigorous: "#EF4444",  // red-500
};

const FS = 20;               // ~Hz (match training)
const WIN_SEC = 5;
const WIN = FS * WIN_SEC;    // 100
const OVERLAP = 0.5;
const STEP = Math.floor(WIN * (1 - OVERLAP)); // 50
const STORAGE_KEY = "ADAMMA_BACKEND_BASE";    // e.g., http://192.168.1.45:8000

// Known-good default (exactly what previously worked)
const DEFAULT_BACKEND_BASE = "http://192.168.1.154:8000"

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

  const bufferRef = useRef([]);            // rolling [{accel_x, accel_y, accel_z}]
  const postingRef = useRef(false);
  const lastPostTsRef = useRef(0);
  const predsRef = useRef([]);             // for smoothing
  const tickRef = useRef(null);
  const ignoreUntilTsRef = useRef(0);      // temporarily ignore classify after reset

  // Load any saved backend base on mount
  useEffect(() => {
    (async () => {
      try {
        const saved = await AsyncStorage.getItem(STORAGE_KEY);
        if (saved) setBackendBase(saved);
      } catch {}
    })();
  }, []);

  // Start accelerometer stream
  useEffect(() => {
    Accelerometer.setUpdateInterval(1000 / FS);
    const sub = Accelerometer.addListener(({ x, y, z }) => {
      bufferRef.current.push({ accel_x: x, accel_y: y, accel_z: z });

      const now = Date.now();
      // gate during reset ignore window
      if (now < ignoreUntilTsRef.current) return;

      if (bufferRef.current.length >= WIN && !postingRef.current) {
        // throttle: at least 1000 ms between posts
        if (now - lastPostTsRef.current < 1000) return;

        const windowSamples = bufferRef.current.slice(-WIN);
        bufferRef.current = bufferRef.current.slice(-(WIN - STEP)); // keep overlap
        classify(windowSamples);
        lastPostTsRef.current = now;
      }
    });

    setStatus("Sensor ON");
    return () => { sub && sub.remove(); setStatus("Sensor OFF"); };
  }, []);

  // Per-second timers for the current class
  useEffect(() => {
    tickRef.current && clearInterval(tickRef.current);
    tickRef.current = setInterval(() => {
      setTimers(prev => ({ ...prev, [current]: prev[current] + 1 }));
    }, 1000);
    return () => clearInterval(tickRef.current);
  }, [current]);

  // Build final predict URL from base (add /predict, strip trailing slash)
  function predictUrl() {
    const base = (backendBase || DEFAULT_BACKEND_BASE).replace(/\/+$/, "");
    return `${base}/predict`;
  }

  async function classify(samples) {
    try {
      postingRef.current = true;
      setStatus("Predicting…");
      const res = await fetch(predictUrl(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ samples })
      });
      if (!res.ok) {
        const txt = await res.text().catch(()=> "");
        console.log("predict non-200:", res.status, txt);
        setStatus(`API ${res.status}`);
        return;
      }

      const data = await res.json();
      const cls = data?.met_class || "Sedentary";

      // Smoothing over last 3 predictions (unchanged)
      predsRef.current.push(cls);
      if (predsRef.current.length > 3) predsRef.current.shift();
      const smooth = majorityVote(predsRef.current);

      setCurrent(smooth);
      setStatus(`OK: ${smooth}`);
    } catch (e) {
      console.log("predict error:", e?.message || e);
      setStatus("Offline (keeping last class)");
    } finally {
      postingRef.current = false;
    }
  }

  function handleReset() {
    // Zero timers
    setTimers({ Sedentary:0, Light:0, Moderate:0, Vigorous:0 });
    // Clear prediction history and buffers
    predsRef.current = [];
    bufferRef.current = [];
    lastPostTsRef.current = 0;
    // Ignore classify briefly to avoid instant re-population
    ignoreUntilTsRef.current = Date.now() + 1500;
    // Reset UI
    setCurrent("Sedentary");
    setStatus("Reset ✓");
  }

  async function handleSaveBackend() {
    let base = backendBase.trim();
    if (base && !/^https?:\/\//.test(base)) base = `http://${base}`;
    setBackendBase(base);
    try { await AsyncStorage.setItem(STORAGE_KEY, base); } catch {}
    setShowSettings(false);
  }

  const fmt = s => `${Math.floor(s/60)}m ${s%60}s`;

  // --- Session summary ---
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
          {/* Header with Settings */}
          <View style={styles.headerRow}>
            <Text style={styles.title}>Live MET Tracker</Text>
            <Pressable onPress={() => setShowSettings(s => !s)} style={({ pressed }) => [styles.settingsBtn, pressed && { opacity: 0.85 }]}>
              <Text style={styles.settingsText}>⚙︎ Settings</Text>
            </Pressable>
          </View>

          {showSettings && (
            <View style={styles.card}>
              <Text style={styles.section}>Backend</Text>
              <Text style={styles.subtle}>Enter your computer’s URL (e.g., http://192.168.1.45:8000)</Text>
              <TextInput
                value={backendBase}
                onChangeText={setBackendBase}
                placeholder="http://192.168.1.45:8000"
                autoCapitalize="none"
                autoCorrect={false}
                style={styles.input}
              />
              <View style={{ flexDirection:"row", gap:8, flexWrap:"wrap" }}>
                <Pressable onPress={handleSaveBackend} style={styles.btn}><Text style={styles.btnText}>Save</Text></Pressable>
              </View>
              <Text style={[styles.subtle, { marginTop:6 }]}>Predict URL will be: {predictUrl()}</Text>
            </View>
          )}

          {/* Main content */}
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

          {/* Footer */}
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

  // Same styling as before, with containers scrollable to fit small screens
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
