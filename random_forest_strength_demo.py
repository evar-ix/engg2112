import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "combined_concrete.csv"
TARGET = "cs"
RANDOM_STATE = 42
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

FEATURE_COLUMNS = [
    "cement",
    "ggbs",
    "flyash",
    "silica_fume",
    "limestone_powder",
    "quartz_powder",
    "nano_silica",
    "water",
    "superplasticizer",
    "coarse_agg",
    "fine_agg",
    "temperature",
    "age",
    "is_uhpc",
    "binder",
    "water_binder_ratio",
]

MIX_INPUTS = [
    ("cement", "Cement", "kg/m3", 500.0),
    ("ggbs", "GGBS / slag", "kg/m3", 0.0),
    ("flyash", "Fly ash", "kg/m3", 0.0),
    ("silica_fume", "Silica fume", "kg/m3", 0.0),
    ("limestone_powder", "Limestone powder", "kg/m3", 0.0),
    ("quartz_powder", "Quartz powder", "kg/m3", 0.0),
    ("nano_silica", "Nano silica", "kg/m3", 0.0),
    ("water", "Water", "kg/m3", 160.0),
    ("superplasticizer", "Superplasticizer", "kg/m3", 5.0),
    ("coarse_agg", "Coarse aggregate", "kg/m3", 800.0),
    ("fine_agg", "Fine aggregate", "kg/m3", 700.0),
    ("temperature", "Curing temperature", "C", 20.0),
    ("age", "Curing age", "days", 28.0),
]

BINDER_COMPONENTS = [
    "cement",
    "ggbs",
    "flyash",
    "silica_fume",
    "limestone_powder",
    "quartz_powder",
    "nano_silica",
]


def train_random_forest():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(X, y)
    return model


def coerce_float(value, field_name):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number.") from exc


def build_mix_values(raw_values):
    values = {}
    for key, label, _unit, default in MIX_INPUTS:
        values[key] = coerce_float(raw_values.get(key, default), label)

    values["is_uhpc"] = 1 if raw_values.get("is_uhpc") in {1, "1", True, "true", "yes", "on"} else 0
    values["binder"] = sum(values[component] for component in BINDER_COMPONENTS)

    if values["binder"] <= 0:
        raise ValueError("Binder must be greater than 0.")
    if values["water"] < 0:
        raise ValueError("Water cannot be negative.")
    if values["age"] <= 0:
        raise ValueError("Curing age must be greater than 0.")

    values["water_binder_ratio"] = values["water"] / values["binder"]
    return values


def predict_strength(model, values):
    row = pd.DataFrame([{column: values[column] for column in FEATURE_COLUMNS}])
    return float(model.predict(row)[0])


def predict_from_payload(model, payload):
    values = build_mix_values(payload)
    prediction = predict_strength(model, values)
    return {
        "prediction": prediction,
        "binder": values["binder"],
        "water_binder_ratio": values["water_binder_ratio"],
        "is_uhpc": values["is_uhpc"],
    }


def input_cards_html():
    cards = []
    for key, label, unit, default in MIX_INPUTS:
        cards.append(
            f"""
            <label class="field">
              <span>{label}</span>
              <input type="number" step="any" min="0" name="{key}" value="{default:g}" required>
              <small>{unit}</small>
            </label>
            """
        )
    return "\n".join(cards)


def page_html():
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Concrete Strength Predictor</title>
  <style>
    :root {{
      --ink: #1d2525;
      --muted: #5f6968;
      --line: #d7dedb;
      --panel: #f7faf8;
      --paper: #ffffff;
      --green: #2f6f73;
      --rose: #b84a62;
      --gold: #d88c32;
      --focus: #1b7f83;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      color: var(--ink);
      background: #eef3f0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    main {{
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(300px, 1fr) minmax(320px, 420px);
      gap: 0;
    }}

    .workspace {{
      padding: 28px;
    }}

    .topbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 20px;
    }}

    h1 {{
      margin: 0;
      font-size: 30px;
      line-height: 1.15;
      letter-spacing: 0;
    }}

    .subtitle {{
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 14px;
    }}

    .model-pill {{
      border: 1px solid var(--line);
      background: var(--paper);
      border-radius: 8px;
      padding: 10px 12px;
      min-width: 170px;
      text-align: right;
      font-size: 13px;
      color: var(--muted);
    }}

    .model-pill strong {{
      display: block;
      color: var(--ink);
      font-size: 14px;
    }}

    form {{
      display: grid;
      gap: 18px;
    }}

    .section {{
      border-top: 1px solid var(--line);
      padding-top: 18px;
    }}

    .section-heading {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }}

    h2 {{
      margin: 0;
      font-size: 16px;
      letter-spacing: 0;
    }}

    .count {{
      color: var(--muted);
      font-size: 13px;
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}

    .field {{
      min-height: 96px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
      padding: 11px;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 7px;
    }}

    .field span {{
      font-size: 13px;
      font-weight: 700;
    }}

    .field small {{
      color: var(--muted);
      font-size: 12px;
    }}

    input[type="number"] {{
      width: 100%;
      min-width: 0;
      border: 0;
      border-bottom: 1px solid var(--line);
      color: var(--ink);
      font: inherit;
      font-size: 18px;
      padding: 4px 0 6px;
      outline: none;
      background: transparent;
    }}

    input[type="number"]:focus {{
      border-bottom-color: var(--focus);
    }}

    .toggle-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      border: 1px solid var(--line);
      background: var(--paper);
      border-radius: 8px;
      padding: 14px;
    }}

    .toggle-copy strong {{
      display: block;
      font-size: 14px;
      margin-bottom: 3px;
    }}

    .toggle-copy span {{
      color: var(--muted);
      font-size: 13px;
    }}

    .switch {{
      position: relative;
      width: 62px;
      height: 34px;
      flex: 0 0 auto;
    }}

    .switch input {{
      opacity: 0;
      width: 0;
      height: 0;
    }}

    .slider {{
      position: absolute;
      inset: 0;
      cursor: pointer;
      background: #ccd5d2;
      border-radius: 999px;
      transition: 0.2s ease;
    }}

    .slider::before {{
      content: "";
      position: absolute;
      width: 26px;
      height: 26px;
      left: 4px;
      top: 4px;
      background: var(--paper);
      border-radius: 50%;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.18);
      transition: 0.2s ease;
    }}

    .switch input:checked + .slider {{
      background: var(--rose);
    }}

    .switch input:checked + .slider::before {{
      transform: translateX(28px);
    }}

    .actions {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding-bottom: 24px;
    }}

    button {{
      border: 0;
      border-radius: 8px;
      padding: 12px 16px;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
    }}

    .primary {{
      color: white;
      background: var(--green);
    }}

    .secondary {{
      color: var(--ink);
      background: transparent;
      border: 1px solid var(--line);
    }}

    .primary:disabled {{
      cursor: wait;
      opacity: 0.65;
    }}

    .result-panel {{
      background: #203232;
      color: white;
      padding: 28px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 24px;
    }}

    .status {{
      color: #c7d6d1;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}

    .prediction {{
      font-size: 58px;
      line-height: 1;
      font-weight: 900;
      letter-spacing: 0;
      overflow-wrap: anywhere;
    }}

    .prediction span {{
      font-size: 22px;
      color: #c7d6d1;
      margin-left: 6px;
    }}

    .result-copy {{
      margin: 12px 0 0;
      color: #dbe8e4;
      line-height: 1.5;
    }}

    .metrics {{
      display: grid;
      gap: 10px;
    }}

    .metric {{
      border-top: 1px solid rgba(255, 255, 255, 0.18);
      padding-top: 12px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
    }}

    .metric span {{
      color: #c7d6d1;
    }}

    .metric strong {{
      text-align: right;
    }}

    .error {{
      display: none;
      border-left: 4px solid var(--rose);
      background: #fff4f6;
      color: #692131;
      padding: 12px;
      border-radius: 8px;
      font-size: 14px;
    }}

    .error.visible {{
      display: block;
    }}

    @media (max-width: 980px) {{
      main {{
        grid-template-columns: 1fr;
      }}

      .result-panel {{
        order: -1;
      }}

      .grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}

    @media (max-width: 620px) {{
      .workspace,
      .result-panel {{
        padding: 18px;
      }}

      .topbar,
      .section-heading,
      .actions {{
        align-items: stretch;
        flex-direction: column;
      }}

      .model-pill {{
        text-align: left;
      }}

      .grid {{
        grid-template-columns: 1fr;
      }}

      .prediction {{
        font-size: 44px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="workspace">
      <div class="topbar">
        <div>
          <h1>Concrete Strength Predictor</h1>
          <p class="subtitle">Estimate compressive strength from mix design and curing inputs.</p>
        </div>
        <div class="model-pill">
          <strong>Random Forest</strong>
          Trained from combined_concrete.csv
        </div>
      </div>

      <form id="prediction-form">
        <div class="section">
          <div class="section-heading">
            <h2>Mix And Curing Inputs</h2>
            <span class="count">13 values</span>
          </div>
          <div class="grid">
            {input_cards_html()}
          </div>
        </div>

        <div class="toggle-row">
          <div class="toggle-copy">
            <strong>UHPC mix</strong>
            <span>Include the UHPC flag used by the training dataset.</span>
          </div>
          <label class="switch" aria-label="UHPC mix">
            <input type="checkbox" name="is_uhpc">
            <span class="slider"></span>
          </label>
        </div>

        <div id="error" class="error"></div>

        <div class="actions">
          <button class="primary" type="submit">Predict Strength</button>
          <button class="secondary" type="button" id="reset-button">Reset Defaults</button>
        </div>
      </form>
    </section>

    <aside class="result-panel">
      <div>
        <div class="status" id="status">Ready</div>
        <div class="prediction" id="prediction">--<span>MPa</span></div>
        <p class="result-copy" id="result-copy">Enter the mix values, then run the prediction.</p>
      </div>

      <div class="metrics">
        <div class="metric">
          <span>Binder content</span>
          <strong id="binder">-- kg/m3</strong>
        </div>
        <div class="metric">
          <span>Water-to-binder ratio</span>
          <strong id="ratio">--</strong>
        </div>
        <div class="metric">
          <span>Concrete type</span>
          <strong id="type">Normal</strong>
        </div>
      </div>
    </aside>
  </main>

  <script>
    const form = document.querySelector("#prediction-form");
    const resetButton = document.querySelector("#reset-button");
    const errorBox = document.querySelector("#error");
    const submitButton = document.querySelector(".primary");
    const statusLabel = document.querySelector("#status");
    const prediction = document.querySelector("#prediction");
    const resultCopy = document.querySelector("#result-copy");
    const binder = document.querySelector("#binder");
    const ratio = document.querySelector("#ratio");
    const type = document.querySelector("#type");

    const defaults = Object.fromEntries(
      Array.from(form.elements)
        .filter((element) => element.name)
        .map((element) => [element.name, element.type === "checkbox" ? element.checked : element.value])
    );

    function payloadFromForm() {{
      const data = {{}};
      for (const element of form.elements) {{
        if (!element.name) continue;
        data[element.name] = element.type === "checkbox" ? element.checked : Number(element.value);
      }}
      return data;
    }}

    function setError(message) {{
      errorBox.textContent = message;
      errorBox.classList.toggle("visible", Boolean(message));
    }}

    function updateResult(data) {{
      prediction.innerHTML = `${{data.prediction.toFixed(2)}}<span>MPa</span>`;
      binder.textContent = `${{data.binder.toFixed(2)}} kg/m3`;
      ratio.textContent = data.water_binder_ratio.toFixed(3);
      type.textContent = data.is_uhpc ? "UHPC" : "Normal";
      statusLabel.textContent = "Prediction";
      resultCopy.textContent = "This estimate comes from the Random Forest model trained on the combined concrete dataset.";
    }}

    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      setError("");
      submitButton.disabled = true;
      statusLabel.textContent = "Calculating";

      try {{
        const response = await fetch("/api/predict", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(payloadFromForm()),
        }});
        const data = await response.json();
        if (!response.ok) {{
          throw new Error(data.error || "Prediction failed.");
        }}
        updateResult(data);
      }} catch (error) {{
        statusLabel.textContent = "Input needed";
        setError(error.message);
      }} finally {{
        submitButton.disabled = false;
      }}
    }});

    resetButton.addEventListener("click", () => {{
      for (const element of form.elements) {{
        if (!element.name) continue;
        if (element.type === "checkbox") {{
          element.checked = Boolean(defaults[element.name]);
        }} else {{
          element.value = defaults[element.name];
        }}
      }}
      setError("");
      statusLabel.textContent = "Ready";
      prediction.innerHTML = "--<span>MPa</span>";
      resultCopy.textContent = "Enter the mix values, then run the prediction.";
      binder.textContent = "-- kg/m3";
      ratio.textContent = "--";
      type.textContent = "Normal";
    }});
  </script>
</body>
</html>
"""


class StrengthDemoHandler(BaseHTTPRequestHandler):
    model = None

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self.send_html(page_html())
        elif path == "/health":
            self.send_json({"status": "ok"})
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/predict":
            self.send_error(404, "Not found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length) or "{}")
            result = predict_from_payload(self.model, payload)
            self.send_json(result)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=400)
        except json.JSONDecodeError:
            self.send_json({"error": "Request body must be valid JSON."}, status=400)

    def send_html(self, html):
        encoded = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_json(self, payload, status=200):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        return


def run_server(host=DEFAULT_HOST, port=DEFAULT_PORT):
    print("Training Random Forest model from combined_concrete.csv...")
    StrengthDemoHandler.model = train_random_forest()

    server = ThreadingHTTPServer((host, port), StrengthDemoHandler)
    url = f"http://{host}:{port}"
    print(f"Concrete strength demo running at {url}")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Run the local concrete strength prediction website.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT, type=int)
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
