"use strict";

const descriptions = {
  sac: "Soft Actor-Critic learns continuous price and fee adjustments from market rewards.",
  fixed: "Keep the initial sampled prices and fee unchanged throughout each episode.",
  random: "Sample bounded price and fee changes at every market step.",
  adaptive: "Adjust prices from the previous step's trading participation. Keep the initial fee."
};
let latestHistory = [];
const byId = (id) => document.getElementById(id);
const format = (value, digits = 2) => Number(value).toLocaleString("en-US", {maximumFractionDigits: digits});

function updateStrategyHelp() {
  byId("strategy-help").textContent = descriptions[byId("method").value];
}

function addProduct() {
  const wrap = byId("product-fields");
  const index = wrap.children.length;
  if (index >= 12) return;
  const field = document.createElement("fieldset");
  field.className = "product-field";
  field.innerHTML = `<legend>Product ${index + 1}</legend><label for="name-${index}">Name</label>
    <input id="name-${index}" name="name_${index}" value="Product ${index + 1}" maxlength="60" required><div class="field-grid"></div>`;
  const grid = field.querySelector(".field-grid");
  for (const [key, label, value] of [["buy_low", "Buy min", 10], ["buy_high", "Buy max", 20], ["sell_low", "Sell min", 25], ["sell_high", "Sell max", 50]]) {
    const cell = document.createElement("div");
    cell.innerHTML = `<label for="${key}-${index}">${label}</label><input id="${key}-${index}" name="${key}_${index}" type="number" min="0.01" step="0.01" value="${value}" required>`;
    grid.appendChild(cell);
  }
  wrap.appendChild(field);
  byId("product-count").value = index + 1;
}

function drawChart(id, values, color) {
  const canvas = byId(id);
  const width = canvas.getBoundingClientRect().width;
  if (!width || !values.length) return;
  const height = 160;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const left = 55, right = width - 12, top = 16, bottom = height - 27;
  const min = Math.min(...values), max = Math.max(...values);
  const margin = Math.max((max - min) * 0.15, Math.abs(max) * 0.015, 0.01);
  const low = min - margin, high = max + margin;
  const x = (index) => left + index * (right - left) / Math.max(values.length - 1, 1);
  const y = (value) => bottom - (value - low) * (bottom - top) / (high - low);
  ctx.font = "10px system-ui";
  ctx.fillStyle = "#7b8880";
  for (let i = 0; i <= 3; i++) {
    const value = low + (high - low) * i / 3;
    ctx.beginPath(); ctx.strokeStyle = "#e8ede6"; ctx.lineWidth = 1;
    ctx.moveTo(left, y(value)); ctx.lineTo(right, y(value)); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(format(value, 2), left - 9, y(value) + 3);
  }
  ctx.textAlign = "left"; ctx.fillText("1", left, height - 7);
  ctx.textAlign = "right"; ctx.fillText(`${values.length} episodes`, right, height - 7);
  ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2;
  values.forEach((value, index) => index ? ctx.lineTo(x(index), y(value)) : ctx.moveTo(x(index), y(value)));
  ctx.stroke();
  if (values.length === 1) { ctx.beginPath(); ctx.arc(x(0), y(values[0]), 3, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill(); }
  canvas.setAttribute("aria-label", `${id === "return-chart" ? "Episode return" : "Mean price index"}: ${values.map(v => format(v)).join(", ")}`);
}

function redrawCharts() {
  drawChart("return-chart", latestHistory.map(row => row.return), "#237d65");
  drawChart("cpi-chart", latestHistory.map(row => row.mean_cpi), "#a47839");
}

function updateStatus(data) {
  byId("run-status").textContent = data.message;
  byId("state-pill").textContent = data.state.toUpperCase();
  byId("start-button").disabled = data.running;
  byId("stop-button").hidden = !data.running;
  byId("metric-episodes").textContent = data.total_episodes ? `${data.current_episode} / ${data.total_episodes}` : "—";
  latestHistory = data.history || [];
  const row = latestHistory.at(-1);
  if (row) {
    byId("metric-return").textContent = format(row.return);
    byId("metric-cpi").textContent = format(row.mean_cpi, 3);
    byId("metric-gold").textContent = format(row.player_gold_mean, 1);
  } else {
    for (const id of ["metric-return", "metric-cpi", "metric-gold"]) byId(id).textContent = "—";
  }
  byId("chart-empty").hidden = latestHistory.length > 0;
  byId("chart-content").hidden = !latestHistory.length;
  redrawCharts();
  const info = data.latest;
  const rows = byId("market-rows");
  rows.replaceChildren();
  if (info) {
    info.buy_prices.forEach((price, index) => {
      const tr = document.createElement("tr");
      for (const value of [info.product_names[index], format(price), format(info.sell_prices[index]),
        info.buy_count[index], info.sell_count[index], `${info.buy_volume[index]} / ${info.sell_volume[index]}`]) {
        const td = document.createElement("td"); td.textContent = value; tr.appendChild(td);
      }
      rows.appendChild(tr);
    });
    byId("market-extra").textContent = `Fee ${format(info.fee_rate * 100, 1)}% · Last-step trade ${format(info.trade_volume)} units · Player-product participation ${format(info.participation * 100, 1)}%`;
  } else {
    const tr = document.createElement("tr"); const td = document.createElement("td");
    td.colSpan = 6; td.className = "muted"; td.textContent = "No completed episode yet."; tr.appendChild(td); rows.appendChild(tr);
    byId("market-extra").textContent = "";
  }
  const artifacts = byId("artifact-links");
  artifacts.replaceChildren();
  if (data.artifacts.length && !data.running) {
    for (const name of data.artifacts) {
      const link = document.createElement("a");
      link.href = `/artifacts/${encodeURIComponent(data.run_id)}/${encodeURIComponent(name)}/`;
      link.textContent = `${name} ↓`;
      artifacts.appendChild(link);
    }
  } else {
    const label = document.createElement("span"); label.className = "muted";
    label.textContent = "Available when the run finishes."; artifacts.appendChild(label);
  }
}

async function poll() {
  let running = false;
  try {
    const response = await fetch("/training_status_api/", {cache: "no-store"});
    if (!response.ok) throw new Error("Status unavailable");
    const data = await response.json();
    running = data.running;
    updateStatus(data);
  } catch (error) {
    byId("run-status").textContent = "Connection interrupted. Retrying…";
  } finally {
    window.setTimeout(poll, running ? 600 : 2000);
  }
}

byId("method").addEventListener("change", updateStrategyHelp);
byId("add-product").addEventListener("click", addProduct);
byId("remove-product").addEventListener("click", () => {
  const wrap = byId("product-fields");
  if (wrap.children.length > 1) wrap.lastElementChild.remove();
  byId("product-count").value = wrap.children.length;
});
byId("stop-button").addEventListener("click", async () => {
  try {
    const response = await fetch("/stop_training/", {method: "POST", headers: {"X-CSRFToken": document.querySelector("[name=csrfmiddlewaretoken]").value}});
    if (!response.ok) throw new Error("Stop request failed");
    byId("run-status").textContent = "Stopping…";
  } catch (error) { byId("run-status").textContent = "Could not stop the run. Please retry."; }
});
new ResizeObserver(redrawCharts).observe(byId("chart-content"));
updateStrategyHelp();
poll();
