 const btn = document.getElementById("predictBtn");
const featuresInput = document.getElementById("features");
const resultBox = document.getElementById("result");

btn.addEventListener("click", async () => {
  const raw = featuresInput.value.trim();
  if (!raw) {
    resultBox.textContent = "Please enter features.";
    return;
  }

  const features = raw.split(",").map(v => Number(v.trim()));

  try {
    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features })
    });

    if (!res.ok) {
      const text = await res.text();
      resultBox.textContent = "Error: " + text;
      return;
    }

    const data = await res.json();
    const label = data.prediction === 1 ? "FRAUD" : "NOT FRAUD";
    const prob = (data.fraud_probability * 100).toFixed(2);

    resultBox.textContent = `Prediction: ${label}
Fraud probability: ${prob}%`;
  } catch (err) {
    resultBox.textContent = "Request failed: " + err;
  }
});