// static/js/new_training.js

/**
 * 1. 動態產生「商品價格上下限」欄位
 */
function generateFields() {
    const cnt = parseInt(document.getElementById("productCount").value || 0, 10);
    const wrap = document.getElementById("fieldsContainer");
    wrap.innerHTML = "";
    for (let i = 0; i < cnt; i++) {
      wrap.insertAdjacentHTML(
        "beforeend",
        `<!-- 使用反引號包整段 HTML，否則內部的 ${i} 會被跑錯 -->
        <div class="fieldset-custom">
          <div class="legend-custom"><i class="bi bi-box"></i> 商品 ${i}</div>
          <div class="row mb-3">
            <div class="col-md-6">
              <label class="form-label">
                <i class="bi bi-arrow-down-circle text-success"></i> 買進價格下限
              </label>
              <input type="number" class="form-control" name="buy_low_${i}" required step="0.01" />
            </div>
            <div class="col-md-6">
              <label class="form-label">
                <i class="bi bi-arrow-up-circle text-danger"></i> 買進價格上限
              </label>
              <input type="number" class="form-control" name="buy_high_${i}" required step="0.01" />
            </div>
          </div>
          <div class="row">
            <div class="col-md-6">
              <label class="form-label">
                <i class="bi bi-arrow-down-circle text-success"></i> 賣出價格下限
              </label>
              <input type="number" class="form-control" name="sell_low_${i}" required step="0.01" />
            </div>
            <div class="col-md-6">
              <label class="form-label">
                <i class="bi bi-arrow-up-circle text-danger"></i> 賣出價格上限
              </label>
              <input type="number" class="form-control" name="sell_high_${i}" required step="0.01" />
            </div>
          </div>
        </div>`
      );
    }
  }
  
  /**
   * 2. 每秒輪詢 /training_status_api/，更新右側各區塊
   */
  function pollStatus() {
    fetch("/training_status_api/?t=" + Date.now())
      .then((r) => r.json())
      .then((d) => {
        // 更新上方數值
        document.getElementById("currentEpisode").innerHTML = `<span class="fw-bold">${d.current_episode}</span> / ${d.total_episodes}`;
        document.getElementById("reward").textContent = d.reward.toFixed(2);
        document.getElementById("statusMsg").innerHTML = `<span class="loading-spinner"></span> ${d.message}`;
        document.getElementById("cpi").textContent = d.cpi.toFixed(3);
        document.getElementById("playerGold").textContent = d.player_gold_avg.toFixed(1);
  
        // 商品即時價格
        const pc = document.getElementById("productContainer");
        pc.innerHTML = "";
        d.avg_buy_price.forEach((buy, i) => {
          const sell = d.avg_sell_price[i];
          pc.insertAdjacentHTML(
            "beforeend",
            `<div class="col-md-4 mb-3">
              <div class="card h-100 border-0" style="background: linear-gradient(135deg,#a8edea 0%,#fed6e3 100%);">
                <div class="card-body text-center">
                  <h5 class="card-title"><i class="bi bi-box"></i> 商品 ${i}</h5>
                  <p class="card-text">
                    <span class="badge bg-success mb-1">買進: $${buy}</span><br/>
                    <span class="badge bg-danger">賣出: $${sell}</span>
                  </p>
                </div>
              </div>
            </div>`
          );
        });
  
        // 初始價格上下限
        const bc = document.getElementById("finalBoundsContainer");
        bc.innerHTML = "";
        if (d.final_buy_bounds) {
          d.final_buy_bounds.forEach((b, i) => {
            const s = d.final_sell_bounds[i];
            bc.insertAdjacentHTML(
              "beforeend",
              `<div class="col-md-6 mb-3">
                <div class="card border-0" style="background: linear-gradient(135deg,#ffecd2 0%,#fcb69f 100%);">
                  <div class="card-body">
                    <h6 class="card-title"><i class="bi bi-box"></i> 商品 ${i} 初始價格區間</h6>
                    <p class="card-text">
                      <span class="badge bg-info">買進: ${b[0]} ~ ${b[1]}</span><br/>
                      <span class="badge bg-warning">賣出: ${s[0]} ~ ${s[1]}</span>
                    </p>
                  </div>
                </div>
              </div>`
            );
          });
        }
  
        // 商品統計
        const sc = document.getElementById("perProductStats");
        sc.innerHTML = "";
        d.per_product_stats.forEach((item) => {
          sc.insertAdjacentHTML(
            "beforeend",
            `<div class="col-md-4 mb-3">
              <div class="card border-0" style="background: linear-gradient(135deg,#a18cd1 0%,#fbc2eb 100%);">
                <div class="card-body text-center text-white">
                  <h6 class="card-title"><i class="bi bi-graph-up"></i> 商品 ${item.product_id}</h6>
                  <div class="row">
                    <div class="col-6">
                      <small>買家人數</small><br/>
                      <strong>${item.avg_buy_cnt.toFixed(1)}</strong>
                    </div>
                    <div class="col-6">
                      <small>賣家人數</small><br/>
                      <strong>${item.avg_sell_cnt.toFixed(1)}</strong>
                    </div>
                  </div>
                  <hr class="my-2" style="border-color: rgba(255,255,255,0.3);" />
                  <div class="row">
                    <div class="col-6">
                      <small>買進量</small><br/>
                      <strong>${item.avg_buy_vol.toFixed(1)}</strong>
                    </div>
                    <div class="col-6">
                      <small>賣出量</small><br/>
                      <strong>${item.avg_sell_vol.toFixed(1)}</strong>
                    </div>
                  </div>
                </div>
              </div>
            </div>`
          );
        });
  
        // 下一秒再輪詢
        setTimeout(pollStatus, 1000);
      })
      .catch(console.error);
  }
  
  // 每秒強制刷新 reward.png，避免瀏覽器快取
  setInterval(() => {
    const imgEl = document.getElementById("rewardImg");
    if (!imgEl) return; 
    // 1. 先讀 data-base-src (Django 已解析成 "/static/images/sac_reward.png")
    const baseUrl = imgEl.getAttribute("data-base-src");
    if (!baseUrl) return;
    // 2. 再拼上 ?<timestamp>，讓瀏覽器重新載入
    imgEl.src = baseUrl + "?" + Date.now();
  }, 1000);
  
  /**
   * 4. 動態產生「關聯規則」欄位
   */
  function generateRuleFields() {
    const cnt = parseInt(document.getElementById("rulesCount").value || 0, 10);
    const wrap = document.getElementById("rulesContainer");
    wrap.innerHTML = "";
    for (let i = 0; i < cnt; i++) {
      wrap.insertAdjacentHTML(
        "beforeend",
        `<div class="fieldset-custom">
          <div class="legend-custom"><i class="bi bi-diagram-3"></i> 規則 ${i + 1}</div>
          <div class="row mb-3">
            <div class="col-md-4">
              <label class="form-label"><i class="bi bi-arrow-right"></i> Antecedent（前置項）</label>
              <input type="text" class="form-control" name="antecedent_${i}" placeholder="例如: 0,1" required />
              <small class="text-muted">多品項逗號分隔</small>
            </div>
            <div class="col-md-4">
              <label class="form-label"><i class="bi bi-bullseye"></i> Consequent（結論項）</label>
              <input type="text" class="form-control" name="consequent_${i}" placeholder="例如: 2" required />
              <small class="text-muted">單一索引</small>
            </div>
            <div class="col-md-4">
              <label class="form-label"><i class="bi bi-gear"></i> 規則類型</label>
              <select class="form-select" name="rule_type_${i}" required>
                <option value="complementary">🤝 互補品</option>
                <option value="substitute">🔄 替代品</option>
              </select>
            </div>
          </div>
  
          <div class="row" id="compFields_${i}">
            <div class="col-md-6">
              <label class="form-label"><i class="bi bi-percent"></i> Confidence (0~1)</label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                class="form-control"
                name="confidence_${i}"
                placeholder="0.6"
                required
              />
            </div>
            <div class="col-md-6">
              <label class="form-label"><i class="bi bi-bar-chart"></i> Support (0~1)</label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                class="form-control"
                name="support_${i}"
                placeholder="0.3"
                required
              />
            </div>
          </div>
  
          <div class="row" id="subFields_${i}" style="display: none;">
            <div class="col-md-6">
              <label class="form-label"><i class="bi bi-dice-3"></i> Probability (0~1)</label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                class="form-control"
                name="probability_${i}"
                placeholder="0.5"
              />
            </div>
            <div class="col-md-6">
              <label class="form-label"><i class="bi bi-bar-chart"></i> Support (0~1)</label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                class="form-control"
                name="support_${i}"
                placeholder="0.3"
              />
            </div>
          </div>
        </div>`
      );
  
      // 綁定切換不同規則型態 (互補 or 替代)
      const sel = document.querySelector(`select[name="rule_type_${i}"]`);
      sel.addEventListener("change", () => {
        const comp = document.getElementById(`compFields_${i}`);
        const sub = document.getElementById(`subFields_${i}`);
        if (sel.value === "complementary") {
          comp.style.display = "flex";
          sub.style.display = "none";
          comp.querySelectorAll("input").forEach((el) => {
            el.disabled = false;
            el.required = true;
          });
          sub.querySelectorAll("input").forEach((el) => {
            el.disabled = true;
            el.required = false;
            el.value = "";
          });
        } else {
          comp.style.display = "none";
          sub.style.display = "flex";
          sub.querySelectorAll("input").forEach((el) => {
            el.disabled = false;
            el.required = true;
          });
          comp.querySelectorAll("input").forEach((el) => {
            el.disabled = true;
            el.required = false;
            el.value = "";
          });
        }
      });
      // 預設執行一次切換
      sel.dispatchEvent(new Event("change"));
    }
  }
  
  /**
   * 首次載入 → 先產出一次商品上下限欄位 
   * 再開始輪詢 pollStatus()
   */
  window.onload = () => {
    generateFields();
    pollStatus();
  };
  