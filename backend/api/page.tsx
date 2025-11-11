"use client";
import { useState } from "react";

export default function Home() {
  const [title, setTitle] = useState("");
  const [abstract, setAbstract] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  async function generate() {
    if (!abstract.trim()) {
      alert("Paste an abstract first!");
      return;
    }

    setLoading(true);
    const payload = {
      title: title || "not specified",
      authors: [],
      year: null,
      venue: null,
      abstract: abstract,
    };

    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    setResult(data);
    setLoading(false);
  }

  return (
    <div style={{ maxWidth: 700, margin: "40px auto", fontFamily: "sans-serif" }}>
      <h1>Compyle.ai — Summary Generator</h1>

      <label>Title</label>
      <input
        style={{ width: "100%", padding: 8, marginBottom: 12 }}
        value={title}
        onChange={(e) => setTitle(e.target.value)}
      />

      <label>Abstract</label>
      <textarea
        rows={8}
        style={{ width: "100%", padding: 8 }}
        value={abstract}
        onChange={(e) => setAbstract(e.target.value)}
      ></textarea>

      <button
        onClick={generate}
        style={{
          marginTop: 12,
          padding: "10px 16px",
          background: "black",
          color: "white",
          border: "none",
          borderRadius: 8,
        }}
      >
        {loading ? "Generating..." : "Generate Summary"}
      </button>

      <pre style={{ whiteSpace: "pre-wrap", marginTop: 20 }}>
        {result ? JSON.stringify(result, null, 2) : "No result yet"}
      </pre>
    </div>
  );
}
