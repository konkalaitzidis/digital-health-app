// frontend/adamma-frontend/App.js
import React, { useEffect, useRef, useState } from "react";
import { SafeAreaView, Text, View, StyleSheet, StatusBar, Alert } from "react-native";
import { Accelerometer } from "expo-sensors";

const CLASSES = ["Sedentary", "Light", "Moderate", "Vigorous"];
const FS = 20;            // ~Hz (match training)
const WIN_SEC = 5;
const WIN = FS * WIN_SEC; // 100
const OVERLAP = 0.5;
const STEP = Math.floor(WIN * (1 - OVERLAP)); // 50
const API_URL = "http://130.229.165.254:8000/predict"; // <- Change to your server

export default function App() {
  const [current, setCurrent] = useState("Sedentary");
  const [timers, setTimers] = useState({ Sedentary:0, Light:0, Moderate:0, Vigorous:0 });
  const [status, setStatus] = useState("Starting…");

  const bufferRef = useRef([]);    // rolling [{accel_x, accel_y, accel_z}, ...]
  const postingRef = useRef(false); // prevent overlapping requests
  const tickRef = useRef(null);     // 1 Hz timers

  // Start accelerometer
  useEffect(() => {
    Accelerometer.setUpdateInterval(1000 / FS);
    const sub = Accelerometer.addListener(({ x, y, z }) => {
      bufferRef.current.push({ accel_x: x, accel_y: y, accel_z: z });

      // When enough samples, send the LAST full window; then keep overlap in buffer
      if (bufferRef.current.length >= WIN && !postingRef.current) {
        const windowSamples = bufferRef.current.slice(-WIN);
        // Keep the last (WIN - STEP) samples -> 50% overlap
        bufferRef.current = bufferRef.current.slice(- (WIN - STEP));
        classify(windowSamples);
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

  async function classify(samples) {
    try {
      postingRef.current = true;
      setStatus("Predicting…");
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ samples })
      });
      if (!res.ok) {
        const msg = await res.text();
        throw new Error(`API ${res.status}: ${msg}`);
      }
      const data = await res.json();
      const cls = data?.met_class || "Sedentary";
      setCurrent(cls);
      setStatus(`OK: ${cls}`);
    } catch (e) {
      setStatus("Offline (keeping last class)");
      // Optional: Alert once on first failure; then comment this out for quiet mode
      // Alert.alert("Prediction error", String(e).slice(0, 200));
    } finally {
      postingRef.current = false;
    }
  }

  const fmt = s => `${Math.floor(s/60)}m ${s%60}s`;

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />
      <Text style={styles.title}>ADAMMA — Live MET</Text>

      <View style={styles.card}>
        <Text style={styles.section}>Current</Text>
        <Text style={styles.current}>{current}</Text>
        <Text style={styles.subtle}>Sampling ~{FS} Hz • window {WIN_SEC}s • 50% overlap</Text>
        <Text style={styles.subtle}>Status: {status}</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.section}>Today</Text>
        {CLASSES.map(c => (
          <View key={c} style={styles.row}>
            <Text style={styles.label}>{c}</Text>
            <Text style={styles.value}>{fmt(timers[c])}</Text>
          </View>
        ))}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex:1, padding:20, gap:16, backgroundColor:"#fff" },
  title: { fontSize:20, fontWeight:"700", textAlign:"center" },
  card: { borderWidth:1, borderColor:"#ddd", borderRadius:12, padding:16, gap:6 },
  section: { fontSize:12, color:"gray", textTransform:"uppercase" },
  current: { fontSize:24, fontWeight:"800" },
  row: { flexDirection:"row", justifyContent:"space-between", paddingVertical:8 },
  label: { fontSize:16, fontWeight:"600" },
  value: { fontSize:16, fontVariant:["tabular-nums"] },
  subtle: { color:"gray", fontSize:12 }
});
