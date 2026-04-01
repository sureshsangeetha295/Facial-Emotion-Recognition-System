// landing.js — Updated for EngageAI (Student Engagement Monitor)

document.addEventListener("DOMContentLoaded", () => {

  // Smooth scroll for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener("click", e => {
      const target = document.querySelector(a.getAttribute("href"));
      if (target) { 
        e.preventDefault(); 
        target.scrollIntoView({ behavior: "smooth" }); 
      }
    });
  });

  // Hero mockup: cycle emotions (kept your original logic but updated names)
  const emotions = [
    { name: "Happiness", pct: 87, color: "#f59e0b", probs: [0.02,0.01,0.01,0.87,0.05,0.03,0.01] },
    { name: "Sadness",   pct: 65, color: "#3b82f6", probs: [0.05,0.02,0.08,0.03,0.12,0.65,0.05] },
    { name: "Surprise",  pct: 74, color: "#f97316", probs: [0.03,0.01,0.06,0.10,0.04,0.02,0.74] },
    { name: "Anger",     pct: 72, color: "#ef4444", probs: [0.72,0.05,0.04,0.03,0.08,0.06,0.02] },
    { name: "Neutral",   pct: 76, color: "#6b7280", probs: [0.04,0.02,0.03,0.08,0.76,0.05,0.02] },
  ];

  let ei = 0;

  function cycleMockup() {
    const e = emotions[ei++ % emotions.length];
    const nameEl  = document.querySelector(".res-name");
    const confEl  = document.querySelector(".res-conf");
    
    if (nameEl) nameEl.textContent = e.name;
    if (confEl) confEl.textContent = e.pct + "% confidence";

    document.querySelectorAll(".bar-fill").forEach((bar, i) => {
      bar.style.width = Math.round(e.probs[i] * 100) + "%";
    });
  }

  // Auto cycle the mockup every 2.8 seconds
  setInterval(cycleMockup, 2800);

  // FAQ accordion (if any on landing)
  document.querySelectorAll(".faq-item").forEach(item => {
    item.addEventListener("click", () => item.classList.toggle("open"));
  });

  console.log("%cEngageAI Landing Page Loaded Successfully", "color:#e8440a;font-weight:bold");
});