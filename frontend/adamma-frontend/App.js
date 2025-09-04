import React, { useEffect, useState } from "react";
import { SafeAreaView, Text, View, StyleSheet } from "react-native";

const CLASSES = ["Sedentary", "Light", "Moderate", "Vigorous"];

export default function App() {
  const [current, setCurrent] = useState("Moderate");
  const [timers, setTimers] = useState({ Sedentary: 0, Light: 0, Moderate: 0, Vigorous: 0 });

  // Day-1 placeholder: pretend we're in "Moderate" and increment every 5s
  useEffect(() => {
    const id = setInterval(() => {
      setTimers(prev => ({ ...prev, Moderate: prev.Moderate + 5 }));
    }, 5000);
    return () => clearInterval(id);
  }, []);

  const fmt = s => `${Math.floor(s / 60)}m ${s % 60}s`;

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>ADAMMA — MET Tracker</Text>

      <View style={styles.card}>
        <Text style={styles.section}>Current</Text>
        <Text style={styles.current}>{current}</Text>
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
  container: { flex: 1, padding: 20, gap: 16 },
  title: { fontSize: 20, fontWeight: "700", textAlign: "center" },
  card: { borderWidth: 1, borderRadius: 12, padding: 16, gap: 6 },
  section: { fontSize: 12, color: "gray", textTransform: "uppercase" },
  current: { fontSize: 22, fontWeight: "700" },
  row: { flexDirection: "row", justifyContent: "space-between", paddingVertical: 6 },
  label: { fontSize: 16, fontWeight: "600" },
  value: { fontSize: 16 }
});
