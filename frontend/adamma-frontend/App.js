import React, { useEffect, useState } from "react";
import { SafeAreaView, Text, View, StyleSheet } from "react-native";

const CLASSES = ["Sedentary", "Light", "Moderate", "Vigorous"];

export default function App() {
  const [timers, setTimers] = useState({ Sedentary: 0, Light: 0, Moderate: 0, Vigorous: 0 });

  useEffect(() => {
    const id = setInterval(() => {
      // Dummy increment for Day 1 visualization
      setTimers(prev => ({ ...prev, Sedentary: prev.Sedentary + 5 }));
    }, 5000);
    return () => clearInterval(id);
  }, []);

  const fmt = s => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}m ${sec}s`;
  };

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>ADAMMA — MET Tracker (Day 1 Shell)</Text>
      <View style={styles.card}>
        {CLASSES.map(c => (
          <View key={c} style={styles.row}>
            <Text style={styles.label}>{c}</Text>
            <Text style={styles.value}>{fmt(timers[c])}</Text>
          </View>
        ))}
      </View>
      <Text style={styles.note}>TEST / Dummy timers increment every 5s.</Text>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, justifyContent: "center" },
  title: { fontSize: 20, fontWeight: "700", textAlign: "center", marginBottom: 16 },
  card: { borderWidth: 1, borderRadius: 12, padding: 16 },
  row: { flexDirection: "row", justifyContent: "space-between", paddingVertical: 8 },
  label: { fontSize: 16, fontWeight: "600" },
  value: { fontSize: 16 },
  note: { textAlign: "center", marginTop: 12, color: "gray" }
});
