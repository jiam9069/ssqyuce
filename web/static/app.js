/* 双色球智能预测分析系统 - 前端逻辑（零构建，直接运行） */
"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);
const toast = (msg, ms = 2600) => {
  const t = $("#toast");
  t.textContent = msg; t.style.display = "block";
  clearTimeout(t._h); t._h = setTimeout(() => (t.style.display = "none"), ms);
};
const fmt = (x, d = 3) => (x == null ? "-" : Number(x).toFixed(d));
const pct = (p) => (p ? (p * 100).toFixed(1) + "%" : "-");
const pad2 = (n) => String(n).padStart(2, "0");
// 奖级名称与奖金（与后端 evaluate.PRIZE_NAME / PRIZE_CASH 一致，回放本地对照用）
const PRIZE_NAME = {1:"一等奖",2:"二等奖",3:"三等奖",4:"四等奖",5:"五等奖",6:"六等奖"};
const PRIZE_CASH = {1:6500000,2:180000,3:3000,4:200,5:10,6:5};
const PRIMES_33 = new Set([2,3,5,7,11,13,17,19,23,29,31]);
function localPrizeLevel(r, b) {
  if (r === 6 && b) return 1;
  if (r === 6) return 2;
  if (r === 5 && b) return 3;
  if (r === 5 || (r === 4 && b)) return 4;
  if (r === 4 || (r === 3 && b)) return 5;
  if (b) return 6;
  return 0;
}
function acValue(reds) {
  const rs = [...reds].sort((a, b) => a - b);
  const diffs = new Set();
  for (let i = 0; i < rs.length; i++) for (let j = i + 1; j < rs.length; j++) diffs.add(rs[j] - rs[i]);
  return diffs.size - (rs.length - 1);
}
function comb(n, k) {
  if (k < 0 || k > n) return 0;
  let r = 1;
  for (let i = 0; i < k; i++) r = r * (n - i) / (i + 1);
  return Math.round(r);
}
// 超几何分布：33 个号码中 K 个属于目标类，抽 6 个命中 k 个的概率
function hyperPMF(K, k) { return comb(K, k) * comb(33 - K, 6 - k) / comb(33, 6); }
function percentile(arr, p) {
  const a = [...arr].sort((x, y) => x - y);
  const idx = (a.length - 1) * p / 100;
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  return a[lo] + (a[hi] - a[lo]) * (idx - lo);
}
let win = "long";
let charts = {};
let allPatterns = [];
let allFeatures = null;
let currentIssue = null;
const TASK_POLL_MS = 1000;
const pendingTasks = {};

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    // 优先展示后端 JSON 里的 error 详情（如“大模型不可用或超时(...)”），
    // 解析失败（如 Cloudflare 524 HTML）时退回 HTTP 状态码。
    let msg = "HTTP " + r.status;
    try {
      const t = await r.text();
      const j = JSON.parse(t);
      if (j && j.error) msg = j.error;
    } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}

// ==================== 任务系统 ====================

async function pollTask(taskId) {
  if (!pendingTasks[taskId]) return;
  try {
    const t = await api("/api/tasks/" + taskId);
    if (t.status === "completed" || t.status === "failed") {
      delete pendingTasks[taskId];
      if (t.status === "completed" && pendingTasks[taskId + "_cb"]) {
        pendingTasks[taskId + "_cb"](t.result);
      }
      if (t.status === "failed") {
        toast("任务失败: " + (t.message || "未知错误"));
      }
    }
  } catch (e) {
    delete pendingTasks[taskId];
  }
}

function runTask(taskId, taskFn) {
  pendingTasks[taskId] = true;
  pendingTasks[taskId + "_cb"] = null;
  const interval = setInterval(() => pollTask(taskId), TASK_POLL_MS);
  taskFn().finally(() => clearInterval(interval));
}

function setBusy(id, text) {
  const b = $(id);
  if (!b) return;
  b.disabled = true; b.innerHTML = "<span class='spin'>⏳</span> " + text;
}
function setFree(id, text) {
  const b = $(id);
  if (!b) return;
  b.disabled = false; b.innerHTML = text;
}

// ==================== 主题 ====================

function toggleTheme() {
  const isDark = document.body.classList.toggle("light");
  localStorage.setItem("theme", isDark ? "dark" : "light");
}
(function initTheme() {
  const t = localStorage.getItem("theme");
  if (t === "light") document.body.classList.add("light");
})();

// ==================== 导航 ====================

function switchTab(name) {
  history.replaceState(null, "", "#" + name);
  $$(".nav-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab-panel").forEach(p => p.classList.toggle("active", p.id === "tab-" + name));
  if (name === "settings") { loadLlmConfig(); loadMethodsConfig(); }
  if (name === "analysis") {
    if (allFeatures) renderWindow(allFeatures);
    ensureDrawHistory();
  }
  if (name === "patterns" && allPatterns.length) renderPatterns(allPatterns, _summary(allPatterns));
  if (name === "replay") populateReplaySelect();
}

let replayData = null;

function populateReplaySelect() {
  const sel = $("#replaySelect");
  if (!sel) return;
  const val = sel.value;
  api("/api/predictions/history?limit=50").then(data => {
    replayData = data;
    sel.innerHTML = "<option value=''>-- 选择期号 --</option>";
    data.forEach(item => {
      const opt = document.createElement("option");
      opt.value = item.issue;
      let label = item.issue + (item.date ? " · " + item.date : "");
      if (item.actual && item.result) {
        const reds = item.result.red_hits || [];
        const maxRed = reds.length ? Math.max(...reds) : 0;
        const blueAny = (item.result.blue_hits || []).some(Boolean);
        label += " · 红中" + maxRed + (blueAny ? "+蓝✓" : "");
        if (item.result.best_level >= 1) label += " · " + (PRIZE_NAME[item.result.best_level] || "");
      } else {
        label += " · 待开奖";
      }
      opt.textContent = label;
      sel.appendChild(opt);
    });
    if (val) sel.value = val;
  }).catch(() => {});
}

function loadReplay() {
  const issue = $("#replaySelect")?.value;
  const box = $("#replayResult");
  if (!box) return;
  if (!issue) { box.innerHTML = ""; return; }
  const item = (replayData || []).find(d => d.issue === issue);
  if (!item) { box.innerHTML = '<div class="note">未找到该期预测记录。</div>'; return; }
  if (!item.actual) { box.innerHTML = '<div class="note">该期尚未开奖，开奖后可对照。</div>'; return; }
  renderReplayItem(item);
}

function renderReplayItem(item) {
  const act = item.actual;
  // 本地逐注对照：命中球金框高亮，不依赖后端 result 字段
  const rows = (item.predictions || []).map(t => {
    const rh = t.reds.filter(r => act.reds.includes(r)).length;
    const bh = t.blue === act.blue ? 1 : 0;
    const lvl = localPrizeLevel(rh, bh);
    return { t, rh, bh, lvl, reward: PRIZE_CASH[lvl] || 0 };
  });
  const sortByHit = $("#replaySort")?.checked;
  const view = sortByHit
    ? rows.slice().sort((a, b) => (b.rh * 2 + b.bh) - (a.rh * 2 + a.bh))
    : rows;
  const nT = rows.length;
  const meanRed = nT ? rows.reduce((a, r) => a + r.rh, 0) / nT : 0;
  const nBlue = rows.filter(r => r.bh).length;
  const bestLvl = nT ? Math.max(...rows.map(r => r.lvl)) : 0;
  const winCount = rows.filter(r => r.lvl > 0).length;
  const totalReward = rows.reduce((a, r) => a + r.reward, 0);

  const actualBalls =
    act.reds.map(r => "<span class='ball red'>" + pad2(r) + "</span>").join("") +
    " <span class='ball blue' style='margin-left:8px'>" + pad2(act.blue) + "</span>";

  const tickets = view.map((r, i) => {
    const balls = r.t.reds.map(n =>
      "<span class='ball red sm" + (act.reds.includes(n) ? " hit" : "") + "' onclick='showNumDetail(" + n + ")'>" + pad2(n) + "</span>").join("");
    const blue = "<span class='ball blue sm" + (r.bh ? " hit" : "") + "' onclick='showNumDetailBlue(" + r.t.blue + ")'>" + pad2(r.t.blue) + "</span>";
    const badge = r.lvl
      ? "<span class='prize-badge lv" + r.lvl + "'>" + (PRIZE_NAME[r.lvl] || "") + " ¥" + r.reward + "</span>"
      : "<span class='prize-badge none'>未中奖</span>";
    const redCls = r.rh >= 4 ? "var(--gold)" : r.rh >= 2 ? "var(--green)" : "var(--muted)";
    const method = r.t.method ? "<span class='badge'>" + escHtml(r.t.method) + "</span>" : "";
    const detail = r.t.reasoning
      ? "<details class='replay-detail'><summary>推理理由</summary><div class='reasoning'>" + escHtml(r.t.reasoning) + "</div></details>"
      : "";
    return '<div class="ticket small replay-ticket">' +
      '<div class="row1">' +
        "<span class='rk'>#" + (i + 1) + "</span>" +
        '<div class="balls">' + balls + " " + blue + "</div>" +
        method +
        '<span class="meta">红中 <b style="color:' + redCls + '">' + r.rh + '</b>/6 · 蓝 ' + (r.bh ? "<b style='color:var(--gold)'>✓</b>" : "—") + "</span>" +
        badge +
      "</div>" + detail + "</div>";
  }).join("");

  $("#replayResult").innerHTML =
    '<div class="replay-actual">' +
      '<span class="label">实际开奖</span>' +
      '<div class="balls" style="font-size:18px">' + actualBalls + "</div>" +
    "</div>" +
    '<div class="metrics">' +
      metricItem("最好奖级", bestLvl ? PRIZE_NAME[bestLvl] : "未中奖") +
      metricItem("平均红球命中", fmt(meanRed, 2), nT + " 注") +
      metricItem("蓝球命中", nBlue + " / " + nT + " 注") +
      metricItem("中奖注数", winCount + " / " + nT) +
      metricItem("总奖金", "¥" + totalReward.toLocaleString(), "投入 ¥" + (nT * 2)) +
    "</div>" +
    (nT ? '<div class="replay-list">' + tickets + "</div>" : '<div class="note">该期无预测记录。</div>');
}

// ==================== 预测 ====================

async function runPredict(regenerate, btnId) {
  if (regenerate === undefined) regenerate = false;
  if (btnId === undefined) btnId = regenerate ? "#btnRegen" : "#btnPredict";
  const n = parseInt($("#cfgTickets")?.value || 10);
  const llm = $("#cfgLlm")?.checked;
  setBusy(btnId, regenerate ? "强制生成中…" : "生成中…");
  $("#predStatus").textContent = "⏳ 正在请求服务器，请稍候…";
  $("#predStatus").style.color = "var(--gold)";
  try {
    const r = await api("/api/predict?n_tickets=" + n + "&regenerate=" + regenerate + "&use_llm=" + llm, {method:"POST"});
    currentIssue = r.issue;
    localStorage.setItem("lastPredictIssue", r.issue);
    renderPredictions(r);
    if (r.from_cache) {
      $("#predStatus").textContent = "✅ 已复用缓存预测（期号 " + r.issue + "）";
      $("#predStatus").style.color = "var(--green)";
      toast("已复用缓存预测");
    } else {
      $("#predStatus").textContent = "✅ 预测已生成（期号 " + r.issue + "）";
      $("#predStatus").style.color = "var(--green)";
      toast("预测已生成");
    }
    if (r.task_id) {
      runTask(r.task_id, () => Promise.resolve(r));
    }
  } catch (e) {
    toast("预测失败: " + e.message);
    $("#predStatus").textContent = "❌ 预测失败: " + e.message;
    $("#predStatus").style.color = "var(--red)";
  }
  setFree(btnId, regenerate ? "🔄 强制重新生成" : "🎯 生成预测");
}

function renderPredictions(res) {
  $("#predIssue").textContent = res.issue ? "目标期号：" + res.issue : "";
  const list = $("#predList");
  const items = res.tickets || [];
  if (!items.length) {
    list.innerHTML = '<div class="note">暂无预测，点击「生成预测」。</div>';
    return;
  }
  window._predTickets = items;
  window._currentProbs = res.red_probs ? {red_probs: res.red_probs, blue_probs: res.blue_probs || []} : {};
  list.innerHTML = items.map((t, i) => {
    const balls = t.reds.map(r => "<span class='ball red sm' onclick='showNumDetail(" + r + ")'>" + String(r).padStart(2,"0") + "</span>").join("");
    const blue = "<span class='ball blue sm' onclick='showNumDetailBlue(" + t.blue + ")'>" + String(t.blue).padStart(2,"0") + "</span>";
    const badge = t.method.startsWith("llm:") ? "<span class='badge llm'>LLM推理</span>"
      : "<span class='badge'>" + escHtml(t.method) + "</span>";
    const rt = t.reasoning ? '<div class="reasoning">💬 ' + escHtml(t.reasoning) + '</div>' : '';
    const used = (t.patterns_used || []).length ? '<div class="reasoning">规律引用：' + escHtml(t.patterns_used.join("、")) + '</div>' : '';
    let evd = "";
    if (t.evidence && typeof t.evidence === "object" && Object.keys(t.evidence).length) {
      const evRows = Object.entries(t.evidence).map(([k,v]) => "<div>· " + escHtml(k) + "：" + escHtml(String(v)) + "</div>").join("");
      evd += '<div class="reasoning">📌 依据：' + evRows + '</div>';
    }
    if (t.counter_evidence && t.counter_evidence.length) {
      evd += '<div class="reasoning">⚠️ 反证：' + escHtml(t.counter_evidence.join("；")) + '</div>';
    }
    if (t.structure_scores && typeof t.structure_scores === "object" && Object.keys(t.structure_scores).length) {
      const sc = Object.entries(t.structure_scores).map(([k,v]) => escHtml(k) + "=" + v).join(" · ");
      evd += '<div class="reasoning">📐 结构分：' + sc + '</div>';
    }
    const confColor = t.confidence > 60 ? "var(--green)" : t.confidence > 40 ? "var(--gold)" : "var(--muted)";
    return '<div class="ticket">' +
      '<div class="row1">' +
        "<span style='color:var(--muted)'>#" + (i+1) + "</span>" +
        '<div class="balls">' + balls + " " + blue + "</div>" +
        badge +
        '<div class="conf"><num style="color:' + confColor + '">置信度 ' + Number(t.confidence).toFixed(1) + '/100</num><div class="bar"><i style="width:' + Math.min(100, t.confidence) + '%"></i></div></div>' +
        "<button class='copy-btn' onclick='copyTicket(" + i + ")' title='复制'>📋</button>" +
        "<button class='fav-btn' onclick='toggleFav(" + i + ")' title='收藏'>☆</button>" +
      '</div>' +
      (rt || used || evd ? '<div class="detail">' + rt + used + evd + '</div>' : '') +
    '</div>';
  }).join("");
  $("#predNote").textContent = res.note || "";
}

function copyTicket(i) {
  const t = window._predTickets?.[i];
  if (!t) return;
  const txt = t.reds.map(r => String(r).padStart(2,"0")).join(" ") + " + " + String(t.blue).padStart(2,"0");
  navigator.clipboard.writeText(txt).then(() => toast("已复制: " + txt)).catch(() => toast("复制失败"));
}

function toggleFav(i) {
  const btns = $$(".fav-btn");
  const btn = btns[i];
  if (!btn) return;
  const isFav = btn.textContent.trim() === "★";
  btn.textContent = isFav ? "☆" : "★";
  const favs = JSON.parse(localStorage.getItem("favTickets") || "[]");
  const key = window._predTickets?.[i] ? JSON.stringify(window._predTickets[i]) : "";
  if (isFav) {
    const idx = favs.indexOf(key);
    if (idx >= 0) favs.splice(idx, 1);
  } else {
    if (key && !favs.includes(key)) favs.push(key);
  }
  localStorage.setItem("favTickets", JSON.stringify(favs));
}

function recalcPredictConfig() {
  localStorage.setItem("cfgTickets", $("#cfgTickets")?.value || "10");
  localStorage.setItem("cfgLlm", $("#cfgLlm")?.checked ? "1" : "0");
}
(function initConfig() {
  const t = localStorage.getItem("cfgTickets");
  const l = localStorage.getItem("cfgLlm");
  if (t && $("#cfgTickets")) $("#cfgTickets").value = t;
  if (l !== null && $("#cfgLlm")) $("#cfgLlm").checked = l === "1";
})();

// ==================== 诊断 ====================

function runDiagnose() {
  const redsStr = ($("#diagReds")?.value || "").trim();
  const blueStr = ($("#diagBlue")?.value || "").trim();
  if (!redsStr || !blueStr) { toast("请输入红球和蓝球"); return; }
  const reds = redsStr.split(",").map(s => parseInt(s.trim())).filter(x => !isNaN(x));
  const blue = parseInt(blueStr);
  if (reds.length !== 6 || new Set(reds).size !== 6 || blue < 1 || blue > 16) {
    toast("红球需6个不重复1-33整数，蓝球需1-16"); return;
  }
  setBusy("#btnDiagnose", "诊断中…");
  $("#diagResult").innerHTML = '<div class="note">⏳ 诊断运行中，请稍候…</div>';
  api("/api/diagnose?reds=" + reds.join(",") + "&blue=" + blue)
    .then(data => {
      if (data.error) { toast(data.error); return; }
      const p = data.profile;
      const meanFreq = p.freq ? p.freq.reduce((a,b) => a+b, 0) / 33 : 0;
      $("#diagResult").innerHTML =
        '<div class="diagnose-card">' +
          '<h4>结构画像</h4>' +
          '<div class="metrics">' +
            metricItem("和值", p.sum, "分位[" + p.sum_pct_low.toFixed(0) + "-" + p.sum_pct_high.toFixed(0) + "] " + (p.sum_in_range ? "✓在区间" : "✗偏离")),
            metricItem("奇偶", p.odd_count + ":" + p.even_count),
            metricItem("三区", p.zone_counts.join("-")),
            metricItem("跨度", p.span),
            metricItem("AC值", p.ac),
            metricItem("连号", p.has_consecutive ? "有" : "无"),
            metricItem("同尾", p.has_same_tail ? "有" : "无"),
            metricItem("质数", p.prime_count),
            metricItem("小号(1-16)", p.size_count_small),
            metricItem("0路(被3整除)", p.route_0_count),
            metricItem("热号数", p.hot_count),
            metricItem("冷号数", p.cold_count),
          '</div>' +
          '<h4>遗漏详情</h4>' +
          '<table><thead><tr><th>号码</th><th>当前遗漏</th><th>平均遗漏</th><th>状态</th></tr></thead><tbody>' +
            (p.omit_detail || []).map(o => {
              const ratio = o.omit_avg > 0 ? (o.omit_cur / o.omit_avg).toFixed(1) : "-";
              const state = o.omit_cur > o.omit_avg * 1.5 ? "<span style='color:var(--red)'>偏冷</span>" :
                            o.omit_cur < o.omit_avg * 0.5 ? "<span style='color:var(--green)'>偏热</span>" : "<span style='color:var(--muted)'>正常</span>";
              return "<tr><td>" + String(o.num).padStart(2,"0") + "</td><td>" + o.omit_cur + "</td><td>" + o.omit_avg + "</td><td>" + state + " (比值" + ratio + ")</td></tr>";
            }).join("") +
          "</tbody></table>" +
          '<h4>历史相似注（' + data.similar_count + " 注）</h4>" +
          '<div class="metrics">' +
            Object.entries(data.similar_red_hits_dist || {}).map(([k,v]) =>
              metricItem("命中" + k + "红", v + "次")
            ).join("") +
            (data.similar_blue_hit_rate > 0 ? metricItem("蓝球命中率", pct(data.similar_blue_hit_rate)) : "") +
          '</div>' +
          (data.note ? '<div class="note">' + escHtml(data.note) + '</div>' : '') +
        '</div>';
    })
    .catch(e => toast("诊断失败: " + e.message))
    .finally(() => setFree("#btnDiagnose", "诊断"));
}

function metricItem(label, value, sub) {
  return '<div class="metric"><div class="k">' + label + '</div><div class="v">' + value + (sub ? '<span class="sub">' + escHtml(sub) + '</span>' : '') + '</div></div>';
}

// ==================== 号码详情弹窗 ====================

function showNumDetail(num) {
  if (!allFeatures) return;
  const w = allFeatures.windows[win] || allFeatures.windows.long;
  const red = w.red;
  const freq = red.freq ? (red.freq[num-1] ?? 0) : 0;
  const omCur = red.omission_current ? (red.omission_current[num-1] ?? 0) : 0;
  const omAvg = red.omission_avg ? (red.omission_avg[num-1] ?? 0) : 0;
  const probArr = window._currentProbs?.red_probs || [];
  const prob = (probArr[num-1] ?? 0).toFixed(4);
  const meanFreq = red.freq ? red.freq.reduce((a,b) => a+b, 0) / 33 : 0;
  const hotCold = freq > meanFreq * 1.2 ? "<span style='color:var(--red)'>热</span>" :
                  freq < meanFreq * 0.8 ? "<span style='color:var(--blue)'>冷</span>" : "<span style='color:var(--muted)'>温</span>";
  $("#numModalTitle").textContent = "红球 " + pad2(num) + " 统计";
  $("#numModalBody").innerHTML =
    '<div class="metrics">' +
      metricItem("出现次数", (freq).toString() + " 次") +
      metricItem("频率", (freq / (w.n_draws || 1)).toFixed(4)) +
      metricItem("当前遗漏", omCur.toString() + " 期") +
      metricItem("平均遗漏", omAvg.toFixed(1) + " 期") +
      metricItem("遗漏比", omAvg > 0 ? (omCur / omAvg).toFixed(2) : "-") +
      metricItem("热冷", hotCold) +
      metricItem("模型概率", prob) +
    '</div>' +
    '<div class="note">窗口: ' + (WIN_LABEL[win] || win) + ' | 最新: ' + allFeatures.issue + "</div>";
  $("#numModal").style.display = "flex";
  renderNumTrendChart(num, false);
}

function showNumDetailBlue(num) {
  if (!allFeatures) return;
  const w = allFeatures.windows[win] || allFeatures.windows.long;
  const blue = w.blue;
  const freq = blue.freq ? (blue.freq[num-1] ?? 0) : 0;
  const omCur = blue.omission_current ? (blue.omission_current[num-1] ?? 0) : 0;
  const omAvg = blue.omission_avg ? (blue.omission_avg[num-1] ?? 0) : 0;
  const probArr = window._currentProbs?.blue_probs || [];
  const prob = (probArr[num-1] ?? 0).toFixed(4);
  $("#numModalTitle").textContent = "蓝球 " + pad2(num) + " 统计";
  $("#numModalBody").innerHTML =
    '<div class="metrics">' +
      metricItem("出现次数", freq.toString() + " 次") +
      metricItem("当前遗漏", omCur.toString() + " 期") +
      metricItem("平均遗漏", omAvg.toFixed(1) + " 期") +
      metricItem("重号率", pct(blue.repeat_rate)) +
      metricItem("模型概率", prob) +
    '</div>' +
    '<div class="note">窗口: ' + (WIN_LABEL[win] || win) + ' | 最新: ' + allFeatures.issue + "</div>";
  $("#numModal").style.display = "flex";
  renderNumTrendChart(num, true);
}

// 号码近 100 期出现轨迹（散点点位 + 号码位置虚线）
function renderNumTrendChart(num, isBlue) {
  const doRender = (issues, arr) => {
    const ch = echartsInit("chNumTrend");
    if (!ch) return;
    const N = Math.min(100, issues.length);
    const iss = issues.slice(-N);
    const pts = [];
    iss.forEach((_, i) => {
      const v = arr[i];
      if (isBlue ? v === num : (v || []).includes(num)) pts.push([i, num]);
    });
    const yMin = isBlue ? 1 : Math.max(1, num - 5);
    const yMax = isBlue ? 16 : Math.min(33, num + 5);
    ch.setOption({
      backgroundColor:"transparent", grid:{left:34, right:10, top:16, bottom:22},
      xAxis:{type:"category", data:iss, axisLabel:{color:"#8b949e", interval: Math.max(0, Math.floor(N / 6) - 1), fontSize:10}},
      yAxis:{type:"value", min:yMin, max:yMax, splitLine:{lineStyle:{color:"#2d333b"}}, axisLabel:{color:"#8b949e"}},
      series:[{
        type:"scatter", data:pts, symbolSize:9,
        itemStyle:{color: isBlue ? "#3b82f6" : "#e5484d"},
        markLine:{silent:true, symbol:"none", lineStyle:{type:"dashed", color:"#8b949e"},
          label:{show:false}, data:[{yAxis: num}]},
        tooltip:{formatter: p => iss[p.value[0]] + " 开出 " + pad2(num)},
      }],
      tooltip:{trigger:"item"},
    });
  };
  if (drawHist && drawHist.issues && drawHist.issues.length) {
    doRender(drawHist.issues, isBlue ? drawHist.blues : drawHist.reds);
  } else {
    api("/api/draws/history?n=100").then(h => {
      if (h && h.issues) doRender(h.issues, isBlue ? h.blues : h.reds);
    }).catch(() => {});
  }
}

function closeNumModal(e) {
  if (e && e.target !== $("#numModal")) return;
  $("#numModal").style.display = "none";
}

// ==================== 数据分析 ====================

let heatMode = "freq";    // 红球热力模式: freq | omit | ratio
let bHeatMode = "freq";   // 蓝球面板模式: freq | omit
let drawHist = null;      // /api/draws/history 缓存（走势 / 分布 / 号码轨迹）
let drawHistN = 0;

const WIN_LABEL = {long: "长期 · 全量", mid: "中期 · 近150期", short: "短期 · 近30期"};

function setWin(name) {
  win = name;
  $$(".tab-panel .tabs .tab").forEach(el => el.classList.toggle("on", el.dataset.w === name));
  const feat = allFeatures;
  if (feat) renderWindow(feat);
}

function setHeatMode(m) {
  heatMode = m;
  $$("#heatModeTabs .mini-tab").forEach(el => el.classList.toggle("on", el.dataset.m === m));
  if (allFeatures) renderHeatMode(allFeatures);
}

function setBHeatMode(m) {
  bHeatMode = m;
  $$("#bHeatModeTabs .mini-tab").forEach(el => el.classList.toggle("on", el.dataset.m === m));
  if (allFeatures) renderBlueHeat(allFeatures);
}

function renderStats(feat) {
  allFeatures = feat;
  $("#winStats").textContent =
    "最新一期 " + feat.issue + " " + feat.date + "：红 " + feat.last_reds.join(" ") + " 蓝 " + feat.last_blue;
  renderWindow(feat);
  window._currentProbs = feat.red_probs ? {red_probs: feat.red_probs, blue_probs: feat.blue_probs || []} : {};
}

function renderWindow(feat) {
  const w = feat.windows[win] || feat.windows.long;
  const tag = $("#anaWinTag");
  if (tag) tag.textContent = WIN_LABEL[win] || win;
  renderAnaOverview(w);
  renderHeatMode(feat);
  renderBlueHeat(feat);
  renderBlueTrend();
  renderOmitScatter(w.red);
  renderOmitBins(w.red.omit_bins);
}

function renderAnaOverview(w) {
  const r = w.red || {}, b = w.blue || {};
  const p5 = r.sum_pct ? (r.sum_pct["5"] ?? r.sum_pct[5]) : null;
  const p95 = r.sum_pct ? (r.sum_pct["95"] ?? r.sum_pct[95]) : null;
  const items = [
    metricItem("样本期数", r.n_draws ?? "-"),
    metricItem("和值均值", fmt(r.sum_mean, 1), "σ " + fmt(r.sum_std, 1)),
    (p5 != null && p95 != null) ? metricItem("和值90%区间", p5.toFixed(0) + " ~ " + p95.toFixed(0)) : "",
    metricItem("奇偶比(均值)", fmt(r.odd_mean, 1) + " : " + fmt(6 - (r.odd_mean || 0), 1)),
    metricItem("跨度均值", fmt(r.span_mean, 1)),
    metricItem("AC值均值", fmt(r.ac_mean, 1)),
    metricItem("连号出现率", pct(r.consecutive_rate)),
    metricItem("同尾出现率", pct(r.same_tail_rate)),
    metricItem("重号均值", fmt(r.repeat_mean, 2)),
    metricItem("热号 TOP6", (r.hot_top6 || []).map(pad2).join(" ")),
    metricItem("冷号 TOP6", (r.cold_top6 || []).map(pad2).join(" ")),
    metricItem("蓝球热号", (b.hot_top3 || []).map(pad2).join(" ")),
    metricItem("蓝球大遗漏", (b.omit_top3 || []).map(pad2).join(" ")),
  ];
  $("#anaOverview").innerHTML = items.join("");
}

// 热力配色：t ∈ [0,1]，热=红 / 冷=蓝
function heatColorFreq(t) {
  t = Math.max(0, Math.min(1, t));
  return "rgb(" + Math.round(20 + t * 205) + "," + Math.round(24 + t * 40) + "," + Math.round(40 + t * 60) + ")";
}
function heatColorCold(t) {
  t = Math.max(0, Math.min(1, t));
  return "rgb(" + Math.round(18 + t * 40) + "," + Math.round(24 + t * 96) + "," + Math.round(46 + t * 209) + ")";
}
function swatchLegend(labels, colors) {
  return labels.map((l, i) => '<span class="sw" style="background:' + colors[i] + '"></span>' + l).join(" ");
}

function renderHeatMode(feat) {
  const w = feat.windows[win] || feat.windows.long;
  const red = w.red || {};
  const freq = red.freq || [], omCur = red.omission_current || [], omAvg = red.omission_avg || [];
  let values, colorFn, tipFn, legendHtml;
  if (heatMode === "freq") {
    const max = Math.max(...freq, 1);
    const mean = freq.reduce((a, x) => a + x, 0) / 33;
    values = freq;
    colorFn = v => heatColorFreq(v / max);
    tipFn = i => pad2(i + 1) + "：出现 " + freq[i] + " 次（33 球平均 " + mean.toFixed(1) + "）" +
      (freq[i] > mean * 1.15 ? " · 热" : freq[i] < mean * 0.85 ? " · 冷" : "");
    legendHtml = swatchLegend(["冷", "中", "热"], [colorFn(0), heatColorFreq(0.5), colorFn(max)]);
  } else if (heatMode === "omit") {
    const max = Math.max(...omCur, 1);
    values = omCur;
    colorFn = v => heatColorCold(v / max);
    tipFn = i => pad2(i + 1) + "：当前遗漏 " + omCur[i].toFixed(0) + " 期 / 平均 " + (omAvg[i] || 0).toFixed(1) + " 期";
    legendHtml = swatchLegend(["低遗漏", "中", "高遗漏"], [colorFn(0), heatColorCold(0.5), colorFn(max)]);
  } else {
    values = omCur.map((v, i) => (omAvg[i] > 0 ? v / omAvg[i] : 0));
    colorFn = v => v >= 1
      ? heatColorCold(Math.min(1, (v - 1) / 1.0))
      : heatColorFreq(Math.min(1, (1 - v) / 0.6));
    tipFn = i => pad2(i + 1) + "：遗漏比 " + values[i].toFixed(2) +
      "（当前 " + omCur[i].toFixed(0) + " / 平均 " + (omAvg[i] || 0).toFixed(1) + "）";
    legendHtml = swatchLegend(["偏热 <0.6", "正常 ≈1", "偏冷 >1.5"], [heatColorFreq(1), "rgb(35,40,48)", heatColorCold(1)]);
  }
  const lastReds = feat.last_reds || [];
  $("#heatFreq").innerHTML = values.map((v, i) => {
    const num = i + 1;
    const cls = "cell" + (lastReds.includes(num) ? " last" : "");
    return '<div class="' + cls + '" style="background:' + colorFn(v) + '" title="' + escHtml(tipFn(i)) + '" onclick="showNumDetail(' + num + ')">' + pad2(num) + "</div>";
  }).join("");
  $("#heatLegend").innerHTML = legendHtml;
}

function renderBlueHeat(feat) {
  const w = feat.windows[win] || feat.windows.long;
  const blue = w.blue || {};
  const freq = blue.freq || [], omCur = blue.omission_current || [], omAvg = blue.omission_avg || [];
  let values, colorFn, tipFn, legend;
  if (bHeatMode === "freq") {
    const max = Math.max(...freq, 1);
    values = freq;
    colorFn = v => heatColorFreq(v / max);
    tipFn = i => pad2(i + 1) + "：出现 " + freq[i] + " 次";
    legend = swatchLegend(["冷", "热"], [colorFn(0), colorFn(max)]);
  } else {
    const max = Math.max(...omCur, 1);
    values = omCur;
    colorFn = v => heatColorCold(v / max);
    tipFn = i => pad2(i + 1) + "：遗漏 " + omCur[i].toFixed(0) + " 期 / 平均 " + (omAvg[i] || 0).toFixed(1) + " 期";
    legend = swatchLegend(["低", "高"], [colorFn(0), colorFn(max)]);
  }
  $("#heatBlue").innerHTML = values.map((v, i) => {
    const num = i + 1;
    const cls = "cell" + (num === feat.last_blue ? " last" : "");
    return '<div class="' + cls + '" style="background:' + colorFn(v) + '" title="' + escHtml(tipFn(i)) + '" onclick="showNumDetailBlue(' + num + ')">' + pad2(num) + "</div>";
  }).join("");
  $("#bHeatLegend").innerHTML = legend;
}

function renderBlueTrend() {
  const ch = echartsInit("chBlueTrend");
  if (!ch) return;
  let issues = null, blues = null;
  if (drawHist && drawHist.blues && drawHist.blues.length) {
    issues = drawHist.issues.slice(-30);
    blues = drawHist.blues.slice(-30);
  } else if (allFeatures && allFeatures.recent && allFeatures.recent.length) {
    issues = allFeatures.recent.map(r => r.issue);
    blues = allFeatures.recent.map(r => r.blue);
  } else return;
  const mean = blues.reduce((a, b) => a + b, 0) / blues.length;
  ch.setOption({
    backgroundColor: "transparent", grid: {left:30, right:10, top:14, bottom:20},
    xAxis: {type:"category", data:issues, axisLabel:{color:"#8b949e", interval: Math.max(0, Math.floor(issues.length / 6) - 1), fontSize:10}},
    yAxis: {type:"value", min:1, max:16, interval:3, splitLine:{lineStyle:{color:"#2d333b"}}, axisLabel:{color:"#8b949e"}},
    series: [{
      type:"line", data:blues, symbol:"circle", symbolSize:7,
      lineStyle:{width:1.5, color:"#3b82f6"}, itemStyle:{color:"#3b82f6"},
      areaStyle:{color:"rgba(59,130,246,.08)"},
      markLine:{silent:true, symbol:"none", lineStyle:{type:"dashed", color:"#d29922"},
        label:{color:"#8b949e", formatter:"均值 " + mean.toFixed(1)}, data:[{yAxis: mean}]},
    }],
    tooltip: {trigger:"axis"},
  });
}

function renderOmitScatter(red) {
  const ch = echartsInit("chOmitScatter");
  if (!ch) return;
  const omCur = red.omission_current || [], omAvg = red.omission_avg || [];
  const data = omCur.map((v, i) => {
    const ratio = omAvg[i] > 0 ? v / omAvg[i] : 0;
    const color = ratio >= 1.5 ? "#58a6ff" : ratio <= 0.5 ? "#f85149" : "#8b949e";
    return {value: [ +(omAvg[i] || 0).toFixed(1), v ], name: pad2(i + 1), itemStyle: {color}};
  });
  const maxV = Math.ceil(Math.max(...omCur, ...omAvg, 5)) + 1;
  ch.setOption({
    backgroundColor:"transparent", grid:{left:38, right:14, top:20, bottom:32},
    xAxis:{type:"value", name:"平均遗漏", nameTextStyle:{color:"#8b949e"}, splitLine:{lineStyle:{color:"#2d333b"}}, axisLabel:{color:"#8b949e"}},
    yAxis:{type:"value", name:"当前遗漏", nameTextStyle:{color:"#8b949e"}, splitLine:{lineStyle:{color:"#2d333b"}}, axisLabel:{color:"#8b949e"}},
    series:[
      {type:"line", data:[[0,0],[maxV, maxV]], symbol:"none", silent:true,
       lineStyle:{type:"dashed", color:"#d29922", width:1}, tooltip:{show:false}},
      {type:"scatter", data, symbolSize:9,
       label:{show:true, fontSize:9, color:"#8b949e", position:"top",
         formatter: p => { const r = p.value[0] > 0 ? p.value[1] / p.value[0] : 0; return (r >= 1.5 || r <= 0.5) ? p.name : ""; }},
       tooltip:{formatter: p => p.name + "：当前遗漏 " + p.value[1] + " / 平均 " + p.value[0]}},
    ],
    tooltip:{trigger:"item"},
  });
}

function renderOmitBins(bins) {
  const ch = echartsInit("chOmitBins");
  if (!ch) return;
  const keys = ["0-5", "6-10", "11-15", "16-20", "21+"];
  const data = keys.map(k => (bins && bins[k]) || 0);
  ch.setOption({
    backgroundColor:"transparent", grid:{left:34, right:10, top:16, bottom:26},
    xAxis:{type:"category", data:keys.map(k => k + " 期"), axisLabel:{color:"#8b949e"}},
    yAxis:{type:"value", name:"号码数", nameTextStyle:{color:"#8b949e"}, splitLine:{lineStyle:{color:"#2d333b"}}, axisLabel:{color:"#8b949e"}},
    series:[{type:"bar", data, itemStyle:{color:"#58a6ff", borderRadius:[3,3,0,0]},
      label:{show:true, position:"top", color:"#e6edf3", fontSize:11}}],
    tooltip:{trigger:"axis"},
  });
}

// ---------- 走势 + 分布（基于 /api/draws/history，客户端计算） ----------

function onTrendRangeChange() { ensureDrawHistory(true); }

function ensureDrawHistory(force) {
  const n = parseInt($("#trendRange")?.value || "300", 10);
  if (drawHist && drawHistN === n && !force) { renderTrendCharts(); renderDistCharts(); return; }
  api("/api/draws/history?n=" + n).then(h => {
    if (!h || !h.issues || !h.issues.length) return;
    drawHist = h; drawHistN = n;
    renderTrendCharts();
    renderDistCharts();
    renderBlueTrend();
  }).catch(e => toast("历史数据加载失败: " + e.message));
}

function renderTrendCharts() {
  if (!drawHist || !drawHist.issues) return;
  const N = drawHist.issues.length;
  const labels = drawHist.issues.map(s => s.slice(4));
  const step = Math.max(1, Math.floor(N / 10));
  const axisCommon = {color:"#8b949e", interval: step - 1};

  // 和值走势：均值 / ±1σ / 90% 分位
  const sums = drawHist.sums;
  const mean = sums.reduce((a, b) => a + b, 0) / sums.length;
  const std = Math.sqrt(sums.reduce((a, b) => a + (b - mean) ** 2, 0) / sums.length);
  const p5 = percentile(sums, 5), p95 = percentile(sums, 95);
  let ch = echartsInit("chSum");
  if (ch) ch.setOption({
    backgroundColor:"transparent", grid:{left:40, right:12, top:20, bottom:24},
    xAxis:{type:"category", data:labels, axisLabel:axisCommon},
    yAxis:{type:"value", scale:true, splitLine:{lineStyle:{color:"#2d333b"}}, axisLabel:{color:"#8b949e"}},
    series:[{
      name:"和值", type:"line", showSymbol:false, data:sums,
      lineStyle:{width:1.6, color:"#3b82f6"}, areaStyle:{color:"rgba(59,130,246,.10)"},
      markLine:{silent:true, symbol:"none", data:[
        {yAxis: mean, lineStyle:{type:"dashed", color:"#d29922"}, label:{color:"#d29922", formatter:"均值 " + mean.toFixed(0)}},
        {yAxis: mean + std, lineStyle:{type:"dotted", color:"#8b949e"}, label:{show:false}},
        {yAxis: mean - std, lineStyle:{type:"dotted", color:"#8b949e"}, label:{show:false}},
        {yAxis: p95, lineStyle:{type:"dotted", color:"#e5484d"}, label:{color:"#e5484d", formatter:"P95 " + p95.toFixed(0), position:"insideEndTop"}},
        {yAxis: p5, lineStyle:{type:"dotted", color:"#e5484d"}, label:{color:"#e5484d", formatter:"P5 " + p5.toFixed(0), position:"insideEndBottom"}},
      ]},
    }],
    tooltip:{trigger:"axis"},
  });

  // 三区比堆积
  const z1 = [], z2 = [], z3 = [];
  drawHist.reds.forEach(r => {
    const a = r.filter(x => x <= 11).length, b = r.filter(x => x >= 12 && x <= 22).length;
    z1.push(a); z2.push(b); z3.push(6 - a - b);
  });
  ch = echartsInit("chZone");
  if (ch) ch.setOption({
    backgroundColor:"transparent", grid:{left:40, right:10, top:24, bottom:24},
    xAxis:{type:"category", data:labels, axisLabel:{show:false}},
    yAxis:{type:"value", max:6, splitLine:{lineStyle:{color:"#2d333b"}}, axisLabel:{color:"#8b949e"}},
    series:[
      {name:"一区 1-11", type:"bar", stack:"z", data:z1, itemStyle:{color:"#e5484d"}, barCategoryGap:"20%"},
      {name:"二区 12-22", type:"bar", stack:"z", data:z2, itemStyle:{color:"#3b82f6"}},
      {name:"三区 23-33", type:"bar", stack:"z", data:z3, itemStyle:{color:"#3fb950"}},
    ],
    tooltip:{trigger:"axis"}, legend:{textStyle:{color:"#8b949e"}, top:0, itemWidth:12, itemHeight:8},
  });

  // 跨度走势
  const spans = drawHist.reds.map(r => Math.max(...r) - Math.min(...r));
  const spanMean = spans.reduce((a, b) => a + b, 0) / spans.length;
  ch = echartsInit("chSpan");
  if (ch) ch.setOption({
    backgroundColor:"transparent", grid:{left:36, right:12, top:20, bottom:24},
    xAxis:{type:"category", data:labels, axisLabel:axisCommon},
    yAxis:{type:"value", min:0, max:32, splitLine:{lineStyle:{color:"#2d333b"}}, axisLabel:{color:"#8b949e"}},
    series:[{
      name:"跨度", type:"line", showSymbol:false, data:spans,
      lineStyle:{width:1.4, color:"#a371f7"},
      markLine:{silent:true, symbol:"none", data:[
        {yAxis: spanMean, lineStyle:{type:"dashed", color:"#d29922"}, label:{color:"#d29922", formatter:"均值 " + spanMean.toFixed(1)}}]},
    }],
    tooltip:{trigger:"axis"},
  });

  // 奇数个数走势
  const odds = drawHist.reds.map(r => r.filter(x => x % 2 === 1).length);
  const oddMean = odds.reduce((a, b) => a + b, 0) / odds.length;
  ch = echartsInit("chOdd");
  if (ch) ch.setOption({
    backgroundColor:"transparent", grid:{left:36, right:12, top:20, bottom:24},
    xAxis:{type:"category", data:labels, axisLabel:axisCommon},
    yAxis:{type:"value", min:0, max:6, interval:1, splitLine:{lineStyle:{color:"#2d333b"}}, axisLabel:{color:"#8b949e"}},
    series:[{
      name:"奇数个数", type:"line", showSymbol:false, data:odds,
      lineStyle:{width:1.4, color:"#3fb950"},
      markLine:{silent:true, symbol:"none", data:[
        {yAxis: oddMean, lineStyle:{type:"dashed", color:"#d29922"}, label:{color:"#d29922", formatter:"均值 " + oddMean.toFixed(2)}},
        {yAxis: 3, lineStyle:{type:"dotted", color:"#8b949e"}, label:{show:false}}]},
    }],
    tooltip:{trigger:"axis"},
  });
}

function distBar(id, cats, data, opt) {
  const ch = echartsInit(id);
  if (!ch) return;
  const base = {
    backgroundColor:"transparent", grid:{left:34, right:8, top:18, bottom:28},
    xAxis:{type:"category", data:cats, axisLabel:{color:"#8b949e", fontSize:10, interval:0}},
    yAxis:{type:"value", splitLine:{lineStyle:{color:"#2d333b"}}, axisLabel:{color:"#8b949e"}},
    series:[{name:"期数", type:"bar", data, itemStyle:{color:"#3b82f6", borderRadius:[3,3,0,0]},
      label:{show:true, position:"top", color:"#8b949e", fontSize:9}}],
    tooltip:{trigger:"axis"},
  };
  ch.setOption(Object.assign(base, opt || {}));
}

// 实际分布 + 超几何理论值（K = 1..33 中目标类号码个数）
function distWithTheory(id, cats, counts, K, N) {
  const theory = cats.map((_, k) => +(hyperPMF(K, k) * N).toFixed(1));
  distBar(id, cats, counts.map(v => ({value: v, itemStyle:{color:"#3b82f6"}})), {
    legend:{data:["实际", "理论(超几何)"], textStyle:{color:"#8b949e"}, top:0, itemWidth:12, itemHeight:8},
    series:[
      {name:"实际", type:"bar", data:counts, itemStyle:{color:"#3b82f6", borderRadius:[3,3,0,0]},
       label:{show:true, position:"top", color:"#8b949e", fontSize:9}},
      {name:"理论(超几何)", type:"line", data:theory, symbol:"circle", symbolSize:5,
       lineStyle:{type:"dashed", color:"#d29922", width:1.5}, itemStyle:{color:"#d29922"}},
    ],
  });
}

function renderDistCharts() {
  if (!drawHist || !drawHist.reds) return;
  const reds = drawHist.reds;
  const N = reds.length;
  const tag = $("#distRangeTag");
  if (tag) tag.textContent = "近 " + N + " 期";

  // 和值分布（每 10 一档，均值所在档高亮金色）
  const sums = reds.map(r => r.reduce((a, b) => a + b, 0));
  const sumMean = sums.reduce((a, b) => a + b, 0) / N;
  const sumBins = {};
  sums.forEach(s => {
    const b = Math.floor((s - 21) / 10);
    const key = (21 + b * 10) + "-" + (30 + b * 10);
    sumBins[key] = (sumBins[key] || 0) + 1;
  });
  const sumKeys = Object.keys(sumBins).sort((a, b) => parseInt(a) - parseInt(b));
  const meanKey = sumKeys.find(k => { const [lo, hi] = k.split("-").map(Number); return sumMean >= lo && sumMean < hi; }) ||
                  sumKeys.find(k => { const [lo, hi] = k.split("-").map(Number); return sumMean >= lo && sumMean <= hi; });
  distBar("chDistSum", sumKeys, sumKeys.map(k => ({
    value: sumBins[k],
    itemStyle: {color: k === meanKey ? "#d29922" : "#3b82f6", borderRadius: [3,3,0,0]},
  })), {xAxis: {axisLabel: {color:"#8b949e", fontSize: 9, interval: 0, rotate: 40}}});

  // 跨度分布（每 4 一档）
  const spans = reds.map(r => Math.max(...r) - Math.min(...r));
  const spanBins = {};
  spans.forEach(s => { const b = Math.floor(s / 4); const key = (b * 4) + "-" + (b * 4 + 3); spanBins[key] = (spanBins[key] || 0) + 1; });
  const spanKeys = Object.keys(spanBins).sort((a, b) => parseInt(a) - parseInt(b));
  distBar("chDistSpan", spanKeys, spanKeys.map(k => spanBins[k]));

  // 奇数个数分布 + 理论（16 个奇数）
  const odds = reds.map(r => r.filter(x => x % 2 === 1).length);
  const oddCats = [0,1,2,3,4,5,6];
  distWithTheory("chDistOdd", oddCats, oddCats.map(k => odds.filter(v => v === k).length), 16, N);

  // 三区比 TOP8
  const zoneCnt = {};
  reds.forEach(r => {
    const a = r.filter(x => x <= 11).length, b = r.filter(x => x >= 12 && x <= 22).length;
    const key = a + "-" + b + "-" + (6 - a - b);
    zoneCnt[key] = (zoneCnt[key] || 0) + 1;
  });
  const zoneTop = Object.entries(zoneCnt).sort((a, b) => b[1] - a[1]).slice(0, 8);
  distBar("chDistZone", zoneTop.map(x => x[0]), zoneTop.map(x => x[1]));

  // AC 值分布
  const acs = reds.map(acValue);
  const acCats = [];
  for (let k = 0; k <= 10; k++) acCats.push(k);
  distBar("chDistAC", acCats, acCats.map(k => acs.filter(v => v === k).length));

  // 小号个数分布 + 理论（1-16 共 16 个）
  const sizes = reds.map(r => r.filter(x => x <= 16).length);
  distWithTheory("chDistSize", oddCats, oddCats.map(k => sizes.filter(v => v === k).length), 16, N);

  // 0 路个数分布 + 理论（被 3 整除共 11 个）
  const routes = reds.map(r => r.filter(x => x % 3 === 0).length);
  distWithTheory("chDistRoute", oddCats, oddCats.map(k => routes.filter(v => v === k).length), 11, N);

  // 质数个数分布 + 理论（11 个质数）
  const primes = reds.map(r => r.filter(x => PRIMES_33.has(x)).length);
  distWithTheory("chDistPrime", oddCats, oddCats.map(k => primes.filter(v => v === k).length), 11, N);
}

function echartsInit(id) {
  if (typeof echarts === "undefined") return null;
  const el = $("#" + id);
  if (!el) return null;
  if (charts[id]) { charts[id].dispose(); }
  charts[id] = echarts.init(el, null, {renderer:"canvas"});
  return charts[id];
}

// ==================== 规律 ====================

function _summary(patterns) {
  const g={A:0,B:0,C:0};
  patterns.forEach(p => { g[p.grade||"C"]++; });
  return g;
}

function renderPatterns(items, summary) {
  allPatterns = items;
  $("#patSummary").textContent = "A:" + summary.A + " B:" + summary.B + " C:" + summary.C;
  const rfCount = allPatterns.filter(p => { const rf = patternRedFlag(p); return rf && rf.flagged; }).length;
  if (rfCount) $("#patSummary").textContent += " · 🔴红牌 " + rfCount;
  $("#patCount").textContent = "共 " + items.length + " 条";
  filterPatterns();
  const ch = echartsInit("chPattern");
  if (ch) {
    const names = items.map(p => p.name_zh);
    const margins = items.map(p => p.margin || 0);
    const cols = margins.map(m => m >= 0 ? "#3fb950" : "#e5484d");
    ch.setOption({
      backgroundColor:"transparent", grid:{left:60,right:16,top:8,bottom:70},
      xAxis:{type:"category", data:names, axisLabel:{color:"#8b949e", rotate:38, fontSize:10}},
      yAxis:{type:"value", splitLine:{lineStyle:{color:"#2d333b"}}, axisLabel:{color:"#8b949e"}},
      series:[{
        type:"bar",
        data: margins.map((m,i) => ({value:m, itemStyle:{color:cols[i], borderRadius:[3,3,0,0]}})),
        label:{show:true, position:"top", fontSize:9, color:"#8b949e"},
      }],
      tooltip:{trigger:"axis"},
    });
  }
}

function filterPatterns() {
  const grade = $("#patGradeFilter")?.value || "";
  const kind = $("#patKindFilter")?.value || "";
  let filtered = allPatterns;
  if (grade) filtered = filtered.filter(p => p.grade === grade);
  if (kind) filtered = filtered.filter(p => p.kind === kind);
  const redN = filtered.filter(p => { const rf = patternRedFlag(p); return rf && rf.flagged; }).length;
  $("#patCount").textContent = "显示 " + filtered.length + "/" + allPatterns.length + " 条" + (redN ? " · 🔴红牌 " + redN : "");
  const tb = $("#patTable tbody");
  tb.innerHTML = filtered.map((p, idx) => {
    const realIdx = allPatterns.findIndex(x => x.key === p.key);
    const bt = p.backtest || {};
    const g = p.grade || "C";
    const hasSeries = bt.series && bt.series.length > 0;
    const rf = patternRedFlag(p);
    return "<tr>" +
      "<td><input type='checkbox' class='pat-cmp' value='" + escHtml(p.key) + "' title='勾选后可多选对比'> " + escHtml(p.name_zh) + (p._mined ? "<span class='badge' style='margin-left:4px'>挖掘</span>" : "") + (rf && rf.flagged ? " <span title='红牌:" + escHtml(rf.reason) + "' style='color:var(--red)'>🔴</span>" : "") + "</td>" +
      "<td>" + p.kind + "</td>" +
      "<td style='color:var(--muted)'>" + escHtml((p.desc||"").slice(0,40)) + "</td>" +
      "<td>" + (bt.n ?? p.sample_size ?? "-") + "</td>" +
      "<td>" + fmt(bt.avg_hits) + "</td><td>" + fmt(bt.expected) + "</td>" +
      "<td style='color:" + ((p.margin||0) >= 0 ? "var(--green)" : "var(--red)") + "'>" + fmt(p.margin,3) + "</td>" +
      "<td class='mono'>" + fmt(p.p_value,4) + "</td>" +
      "<td class='mono'>" + fmt(p.p_adj,4) + "</td>" +
      "<td><span class='badge grade" + g + "'>" + g + (p.refuted ? " ⚠️证伪" : "") + "</span></td>" +
      "<td><button style='font-size:10px;padding:2px 6px;' onclick='showPatDetail(" + idx + ")'>详情</button>" +
      (hasSeries ? "<button style='font-size:10px;padding:2px 6px;margin-left:2px;' onclick='showPatSeries(" + idx + ")'>📈</button>" : "") +
      "</td>" +
    "</tr>";
  }).join("");
}

function showPatDetail(idx) {
  const p = allPatterns[idx];
  if (!p) return;
  const bt = p.backtest || {};
  const rf = patternRedFlag(p);
  $("#patDetailName").textContent = p.name_zh + " [" + p.kind + "]";
  $("#patDetailBody").innerHTML =
    '<div class="note">' + escHtml(p.desc || "") + '</div>' +
    '<div class="metrics" style="margin-top:12px">' +
      metricItem("触发样本", bt.n || p.sample_size || 0) +
      metricItem("平均命中", fmt(bt.avg_hits)) +
      metricItem("期望命中", fmt(bt.expected)) +
      metricItem("边际", fmt(p.margin, 3)) +
      metricItem("p值", fmt(p.p_value, 4)) +
      metricItem("adj_p", fmt(p.p_adj, 4)) +
      metricItem("命中覆盖率", pct(bt.hit_rate_at_least1)) +
      (bt.ci_lower != null ? metricItem("95%CI", "[" + bt.ci_lower + ", " + bt.ci_upper + "]") : "") +
      (bt.avg_fav_size != null ? metricItem("fav大小", bt.avg_fav_size) : "") +
    '</div>' +
    (bt.note ? '<div class="note" style="margin-top:8px;color:var(--red)">' + escHtml(bt.note) + '</div>' : '') +
    (rf ? (rf.flagged
      ? '<div class="note" style="margin-top:8px;color:var(--red)">🔴 红牌：' + escHtml(rf.reason) + '（近' + rf.n + '期边际 ' + fmt(rf.recentMargin,3) + ' vs 总体 ' + fmt(rf.overallMargin,3) + '，连续负边际 ' + rf.missStreak + ' 期）</div>'
      : '<div class="note" style="margin-top:8px">🟢 未触发红牌：近' + rf.n + '期边际 ' + fmt(rf.recentMargin,3) + ' vs 总体 ' + fmt(rf.overallMargin,3) + '，连续负边际 ' + rf.missStreak + ' 期</div>') : '');
  $("#patDetailCard").classList.remove("hidden");
  if (bt.series && bt.series.length > 1) {
    renderPatSeriesChart(bt.series);
  }
  $("#patDetailCard").scrollIntoView({behavior:"smooth"});
}

function showPatSeries(idx) {
  const p = allPatterns[idx];
  if (!p || !p.backtest?.series) return;
  renderPatSeriesChart(p.backtest.series);
  $("#patDetailName").textContent = p.name_zh + " · 边际时间序列";
  $("#patDetailBody").innerHTML = "";
  $("#patDetailCard").classList.remove("hidden");
  $("#patDetailCard").scrollIntoView({behavior:"smooth"});
}

function renderPatSeriesChart(series) {
  const ch = echartsInit("chPatSeries");
  if (!ch || !series.length) return;
  const xs = series.map((s,i) => i);
  const margins = series.map(s => s.margin || 0);
  ch.setOption({
    backgroundColor:"transparent", grid:{left:40,right:8,top:20,bottom:22},
    xAxis:{type:"category", data:xs, axisLabel:{show:false}},
    yAxis:{type:"value", splitLine:{lineStyle:{color:"#2d333b"}}, axisLabel:{color:"#8b949e"}},
    series:[{
      type:"line", showSymbol:true, symbol:"circle", symbolSize:4,
      data:margins, lineStyle:{width:1,color:"#3b82f6"},
      areaStyle:{color:"rgba(59,130,246,.1)"},
    }],
    tooltip:{trigger:"axis"},
  });
}

function closePatDetail() {
  $("#patDetailCard").classList.add("hidden");
}

// ---------- M3.4 规律研究台：红牌预警 + 多选对比 ----------

function patternRedFlag(p) {
  const s = (p.backtest && p.backtest.series) || [];
  if (s.length < 10) return null;
  const n = Math.max(10, Math.min(20, Math.floor(s.length / 3)));
  const recent = s.slice(-n);
  const om = s.reduce((a, b) => a + (b.margin || 0), 0) / s.length;
  const rm = recent.reduce((a, b) => a + (b.margin || 0), 0) / recent.length;
  let miss = 0;
  for (let i = recent.length - 1; i >= 0; i--) { if ((recent[i].margin || 0) < 0) miss++; else break; }
  const flagged = rm < om - 0.02 || miss >= 5;
  return { n, overallMargin: om, recentMargin: rm, missStreak: miss, flagged,
           reason: miss >= 5 ? ("连续 " + miss + " 期负边际")
                             : ("近 " + n + " 期边际 " + fmt(rm, 3) + " 较总体 " + fmt(om, 3) + " 下滑") };
}

function comparePatterns() {
  const checked = Array.from(document.querySelectorAll(".pat-cmp:checked")).map(c => c.value);
  if (checked.length < 2) { toast("请至少勾选 2 条规律（表格首列勾选框）"); return; }
  const ps = allPatterns.filter(p => checked.includes(p.key) && p.backtest && p.backtest.series && p.backtest.series.length > 1);
  if (ps.length < 2) { toast("选中的规律缺少边际序列，无法对比"); return; }
  const ch = echartsInit("chPatternCmp");
  if (!ch) return;
  const cols = ["#3b82f6", "#f59e0b", "#22c55e", "#ef4444"];
  ch.setOption({
    backgroundColor: "transparent", grid: {left: 44, right: 12, top: 28, bottom: 22},
    legend: {textStyle: {color: "#8b949e"}, top: 0},
    tooltip: {trigger: "axis"},
    xAxis: {type: "category", data: Array.from({length: Math.max(...ps.map(p => p.backtest.series.length))}, (_, i) => i), axisLabel: {show: false}},
    yAxis: {type: "value", splitLine: {lineStyle: {color: "#2d333b"}}, axisLabel: {color: "#8b949e"}},
    series: ps.map((p, i) => ({
      name: p.name_zh, type: "line", showSymbol: false,
      lineStyle: {width: 1.2, color: cols[i % 4]},
      data: p.backtest.series.map(s => s.margin || 0),
    })),
  });
  toast("⚖️ 对比 " + ps.length + " 条规律（触发点边际时间序列）");
}

// ==================== 历史 ====================

async function runReplayDiag() {
  const el = $("#replayDiag");
  if (!el) return;
  el.innerHTML = "诊断运行中（纯统计+ML 反事实回放，约 10~30 秒）...";
  try {
    const r = await api("/api/replay/diagnose?n=20&use_ml=true", {method:"POST"});
    if (!r.ok) throw new Error(r.error || "诊断失败");
    const a = r.aggregate || {};
    const rows = (r.rows || []).map(x =>
      "<tr><td class='mono'>" + x.issue + "</td><td>" + x.date + "</td>" +
      "<td>" + fmt(x.red_hits_mean, 2) + "</td><td>" + (x.blue_hit ? "✅" : "—") + "</td>" +
      "<td>" + (x.prize_level >= 5 ? "五等+" : "未中") + "</td><td>¥" + fmt(x.reward, 1) + "</td></tr>"
    ).join("");
    el.innerHTML =
      '<div class="note">最近 ' + r.n + ' 期反事实回放（每期只用此前数据，固定种子）</div>' +
      '<div class="scroll" style="max-height:260px"><table><thead><tr>' +
      "<th>期号</th><th>日期</th><th>红球平均命中</th><th>蓝球命中</th><th>最好等级</th><th>奖金¥</th></tr></thead><tbody>" +
      rows + "</tbody></table></div>" +
      '<div class="note" style="margin-top:8px">汇总：' + a.issues + " 期 × " + a.tickets + " 注 · 红球平均命中 " +
      fmt(a.red_hits_mean, 3) + " · 蓝球命中率 " + pct(a.blue_hit_rate) + " · ≥五等率 " + pct(a.prize_rate_ge5) +
      " · 总奖金 ¥" + fmt(a.reward_total, 1) + " · ROI " + pct(a.roi) +
      ' <span style="color:var(--muted)">（对照：随机基线期望红球 1.09/注、蓝球 6.25%）</span></div>' +
      '<div class="note" style="color:var(--muted)">' + escHtml(r.note || "") + "</div>";
  } catch(e) { el.innerHTML = '<div class="note" style="color:var(--red)">诊断失败: ' + escHtml(e.message) + "</div>"; }
}

async function renderHistory() {
  try {
    const hist = await api("/api/draws/history?n=120");
    const issues = hist.issues.slice(-20);
    const reds = hist.reds.slice(-20);
    const blues = hist.blues.slice(-20);
    const rows = issues.map((iss,i) => {
      const cells = reds[i].map(r => "<span class='ball red sm'>" + String(r).padStart(2,"0") + "</span>").join(" ");
      const b = "<span class='ball blue sm'>" + String(blues[i]).padStart(2,"0") + "</span>";
      return "<tr><td class='mono'>" + iss + "</td><td><div class='balls'>" + cells + " " + b + "</div></td></tr>";
    }).join("");
    $("#histGrid").innerHTML = "<table><tbody>" + rows + "</tbody></table>";
  } catch(e) { console.error(e); }
}

// ==================== 评估 ====================

function setEvalSub(name) {
  $$(".subtab").forEach(el => el.classList.toggle("on", el.dataset.sub === name));
  $$(".sub-panel").forEach(p => p.classList.toggle("hidden", p.id !== "evalSub-" + name));
  // 面板由隐藏变为可见时重算图表尺寸（隐藏容器内初始化的图表宽高为 0）
  requestAnimationFrame(() => Object.values(charts).forEach(c => {
    try { if (c && c.resize) c.resize(); } catch (_) {}
  }));
}

async function runOfflineEval() {
  setBusy("#btnEval", "评估中（30-120s）…");
  try {
    const r = await api("/api/eval/backtest?issues=120&n=10", {method:"POST"});
    renderEval(r);
    toast("离线评估完成");
  } catch(e) { toast("评估失败: " + e.message); }
  setFree("#btnEval", "📊 离线评估");
}

async function runOnline() {
  setBusy("#btnOnline", "对照中…");
  try {
    const r = await api("/api/eval/online", {method:"POST"});
    renderOnline(r.rows || []);
    await loadCumulativeEval();
    toast("已对照 " + (r.newly_checked || 0) + " 期");
  } catch(e) { toast("在线对照失败: " + e.message); }
  setFree("#btnOnline", "✔ 在线对照");
}

function renderEval(r) {
  const s = r.system || {}, b = r.random_baseline || {};
  const defs = [
    {k:"红球平均命中", sys:s.red_hits_mean, rnd:b.red_hits_mean, d:2},
    {k:"蓝球命中率", sys:s.blue_hit_rate, rnd:b.blue_hit_rate, d:3, isPct:true},
    {k:"≥五等奖率", sys:s.prize_rate_ge5, rnd:b.prize_rate_ge5, d:3, isPct:true},
    {k:"总奖金", sys:s.reward_total, rnd:b.reward_total, d:0, isMoney:true},
    {k:"ROI", sys:s.roi, rnd:b.roi, d:1, isPct:true},
  ];
  const cards = defs.map(x => {
    const fmtV = v => x.isMoney ? "¥" + Number(v || 0).toLocaleString() : (x.isPct ? pct(v) : fmt(v, x.d));
    const delta = (x.sys ?? 0) - (x.rnd ?? 0);
    const dTxt = x.isPct
      ? (delta >= 0 ? "+" : "") + (delta * 100).toFixed(1) + "pp"
      : (delta >= 0 ? "+" : "") + delta.toFixed(x.d);
    return '<div class="metric"><div class="k">' + x.k + "（系统）</div>" +
      '<div class="v">' + fmtV(x.sys) + "</div>" +
      '<span class="sub">随机基线 ' + fmtV(x.rnd) + ' · Δ<span style="color:' +
      (delta >= 0 ? "var(--green)" : "var(--red)") + '">' + dTxt + "</span></span></div>";
  }).join("");
  $("#evalArea").innerHTML =
    '<div class="note" style="margin-bottom:8px">离线 walk-forward 回测：' + (r.n_issues ?? "-") + " 期 × " +
    (r.n_tickets_per_issue ?? "-") + " 注/期，系统对照同注数均匀随机基线。</div>" +
    '<div class="metrics">' + cards + "</div>" +
    '<div class="grid2">' +
      '<div><div class="note">红球命中数分布（系统 vs 随机）</div><div class="chart" id="chRedDist" style="height:200px"></div></div>' +
      '<div><div class="note">奖级分布（系统 vs 随机）</div><div class="chart" id="chPrizeDist" style="height:200px"></div></div>' +
    "</div>" +
    '<div class="note callout">' + escHtml(r.note || "") + "</div>";
  // 红球命中数分布
  let ch = echartsInit("chRedDist");
  if (ch) {
    const keys = [...new Set([...Object.keys(s.red_hits_dist || {}), ...Object.keys(b.red_hits_dist || {})])].map(Number).sort((x, y) => x - y);
    ch.setOption({
      backgroundColor:"transparent", grid:{left:38, right:8, top:26, bottom:24},
      xAxis:{type:"category", data:keys.map(k => k + " 红"), axisLabel:{color:"#8b949e"}},
      yAxis:{type:"value", splitLine:{lineStyle:{color:"#2d333b"}}, axisLabel:{color:"#8b949e"}},
      series:[
        {name:"系统", type:"bar", data:keys.map(k => s.red_hits_dist[k] || 0), itemStyle:{color:"#3b82f6", borderRadius:[3,3,0,0]}},
        {name:"随机", type:"bar", data:keys.map(k => b.red_hits_dist[k] || 0), itemStyle:{color:"#8b949e", borderRadius:[3,3,0,0]}},
      ],
      tooltip:{trigger:"axis"}, legend:{textStyle:{color:"#8b949e"}, top:0, itemWidth:12, itemHeight:8},
    });
  }
  // 奖级分布
  ch = echartsInit("chPrizeDist");
  if (ch) {
    const keys = [...new Set([...Object.keys(s.levels_dist || {}), ...Object.keys(b.levels_dist || {})])].map(Number).sort((x, y) => y - x);
    const lvlName = l => l === 0 ? "未中奖" : (PRIZE_NAME[l] || ("等" + l));
    ch.setOption({
      backgroundColor:"transparent", grid:{left:38, right:8, top:26, bottom:24},
      xAxis:{type:"category", data:keys.map(lvlName), axisLabel:{color:"#8b949e", fontSize:10}},
      yAxis:{type:"value", splitLine:{lineStyle:{color:"#2d333b"}}, axisLabel:{color:"#8b949e"}},
      series:[
        {name:"系统", type:"bar", data:keys.map(k => s.levels_dist[k] || 0), itemStyle:{color:"#a371f7", borderRadius:[3,3,0,0]}},
        {name:"随机", type:"bar", data:keys.map(k => b.levels_dist[k] || 0), itemStyle:{color:"#8b949e", borderRadius:[3,3,0,0]}},
      ],
      tooltip:{trigger:"axis"}, legend:{textStyle:{color:"#8b949e"}, top:0, itemWidth:12, itemHeight:8},
    });
  }
}

async function loadCumulativeEval() {
  try {
    const r = await api("/api/eval/cumulative?limit=120");
    renderCumulativeEval(r, false);
  } catch(e) { toast("累计报表加载失败: " + e.message); }
}

function exportCumulativeEval() {
  window.open("/api/eval/export.csv?limit=1000", "_blank");
}

async function loadMethodRecommendations() {
  const area = $("#methodRecommendationArea");
  if (!area) return;
  area.innerHTML = "<div class='note'>🧭 正在计算方法与 uniform 的同期 paired 筛查…</div>";
  try {
    const r = await api("/api/eval/recommendations?limit=120&min_sample=60");
    const items = r.recommendations || [];
    if (!items.length) {
      area.innerHTML = "<div class='note'>暂无可比较方法：需要同一期同时存在方法结果与 uniform 基线。</div>";
      return;
    }
    const meta = {
      insufficient_sample: {label: "样本不足", cls: "gradeC"},
      monitor: {label: "继续观察", cls: "gradeB"},
      keep_or_research: {label: "保留/研究", cls: "gradeA"},
      disable_candidate: {label: "可考虑关闭", cls: "gradeC"},
    };
    const rows = items.map(x => {
      const m = meta[x.status] || {label: x.status, cls: "gradeC"};
      return "<tr><td><code>" + escHtml(x.method) + "</code></td><td>" + x.paired_issues + "</td><td class='mono'>" +
        (x.paired_sign_p == null ? "-" : fmt(x.paired_sign_p, 4)) + "</td><td class='mono'>" +
        (x.mean_reward_delta == null ? "-" : fmt(x.mean_reward_delta, 2)) + "</td><td><span class='badge " + m.cls + "'>" + m.label +
        "</span></td><td>" + escHtml(x.action || "") + "</td></tr>";
    }).join("");
    area.innerHTML =
      "<div class='note' style='margin-bottom:6px'>🧭 方法保留筛查（paired sign-test，对照 uniform；只提示、不自动修改开关）</div>" +
      "<div class='scroll'><table><thead><tr><th>方法</th><th>同期</th><th>p 值</th><th>奖金差(均)</th><th>状态</th><th>建议</th></tr></thead><tbody>" +
      rows + "</tbody></table></div><div class='note callout'>" + escHtml(r.note || "") + "</div>";
  } catch(e) { area.innerHTML = "<div class='note'>方法建议加载失败：" + escHtml(e.message) + "</div>"; }
}

// 取滚动指标最近一点的 95% CI（优先 30 期窗口）
function latestRollingCI(g) {
  const roll = g.rolling || [];
  if (!roll.length) return {};
  const last = roll[roll.length - 1];
  const w = last.w30 || last.w10 || null;
  if (!w) return {};
  const out = {};
  if (w.red_hits && w.red_hits.low != null)
    out.red = w.red_hits.low.toFixed(2) + " ~ " + w.red_hits.high.toFixed(2);
  if (w.blue_hit_rate && w.blue_hit_rate.low != null)
    out.blue = (w.blue_hit_rate.low * 100).toFixed(1) + "~" + (w.blue_hit_rate.high * 100).toFixed(1) + "%";
  return out;
}

function renderCumulativeEval(r, keepMethod) {
  const area = $("#cumulativeEvalArea");
  if (!area) return;
  window._cumEval = r;
  const trendArea = $("#cumTrendArea");
  const groups = (r && r.methods) || [];
  const sel = $("#cumMethodSel");
  if (!groups.length) {
    area.innerHTML = '<div class="note">暂无逐注评估事实：开奖后点击「在线对照」生成累计数据。</div>';
    if (trendArea) trendArea.innerHTML = "";
    if (sel) sel.innerHTML = "";
    return;
  }
  if (sel) {
    const prev = (keepMethod && sel.value && groups.some(g => g.method === sel.value))
      ? sel.value
      : (sel.value && groups.some(g => g.method === sel.value) ? sel.value : groups[0].method);
    sel.innerHTML = groups.map(g => "<option value='" + escHtml(g.method) + "'>方法： " + escHtml(g.method) + "</option>").join("");
    sel.value = prev;
  }
  const method = sel && sel.value ? sel.value : groups[0].method;
  const rows = groups.map(g => {
    const ci = latestRollingCI(g);
    const selMark = g.method === method ? " style='background:var(--blue-soft)'" : "";
    return "<tr" + selMark + "><td><code>" + escHtml(g.method) + "</code></td><td>" + g.issues + "</td><td>" + g.tickets + "</td>" +
      "<td>" + fmt(g.red_hits_mean) + (ci.red ? " <span class='ci'>[" + ci.red + "]</span>" : "") + "</td>" +
      "<td>" + pct(g.blue_hit_rate) + (ci.blue ? " <span class='ci'>[" + ci.blue + "]</span>" : "") + "</td>" +
      "<td>" + pct(g.prize_rate_ge5) + "</td><td>¥" + Number(g.reward_total || 0).toFixed(0) + "</td>" +
      "<td style='color:" + ((g.roi || 0) >= 0 ? "var(--green)" : "var(--red)") + "'>" + pct(g.roi) + "</td></tr>";
  }).join("");
  area.innerHTML =
    '<div class="note" style="margin-bottom:6px">📈 在线累计评估（逐注事实，最多 ' + (r.sample_limit || 120) +
    " 期 · 95% CI 取最近 30 期滚动窗口 · 切换下拉查看趋势）</div>" +
    '<div class="scroll"><table><thead><tr><th>方法</th><th>期数</th><th>注数</th><th>红球均值 [CI]</th><th>蓝球命中率 [CI]</th><th>≥五等奖率</th><th>奖金</th><th>ROI</th></tr></thead><tbody>' +
    rows + "</tbody></table></div>";
  const g = groups.find(x => x.method === method) || groups[0];
  renderCumTrend(g);
}

function renderCumTrend(g) {
  const trendArea = $("#cumTrendArea");
  if (!trendArea) return;
  const roll = g.rolling || [];
  if (roll.length < 2) {
    trendArea.innerHTML = '<div class="note">滚动趋势需要 ≥2 期评估数据（开奖并在线对照后累积）。</div>';
    return;
  }
  trendArea.innerHTML =
    '<div class="grid2" style="margin-top:10px">' +
      '<div><div class="note">红球平均命中 · 10 期滚动（阴影 = 95% CI）· 方法 <code>' + escHtml(g.method) + "</code></div>" +
      '<div class="chart" id="chCumRed" style="height:210px"></div></div>' +
      '<div><div class="note">蓝球命中率 · 10 期滚动（%）</div><div class="chart" id="chCumBlue" style="height:210px"></div></div>' +
    "</div>";
  const iss = roll.map(p => p.issue);
  const step = Math.max(0, Math.floor(iss.length / 8) - 1);
  const pick = key => roll.map(p => (p.w10 || p.w30 || {})[key] || null);
  // 红球命中滚动 + CI 带
  let ch = echartsInit("chCumRed");
  if (ch) {
    const reds = pick("red_hits");
    const mean = reds.map(m => (m && m.mean != null) ? +m.mean.toFixed(3) : null);
    const low = reds.map(m => (m && m.low != null) ? +m.low.toFixed(3) : null);
    const band = reds.map((m, i) => (m && m.high != null && low[i] != null) ? +(m.high - m.low).toFixed(3) : null);
    ch.setOption({
      backgroundColor:"transparent", grid:{left:36, right:10, top:18, bottom:24},
      xAxis:{type:"category", data:iss, axisLabel:{color:"#8b949e", interval: step}},
      yAxis:{type:"value", scale:true, splitLine:{lineStyle:{color:"#2d333b"}}, axisLabel:{color:"#8b949e"}},
      series:[
        {type:"line", data:low, stack:"ci", symbol:"none", lineStyle:{opacity:0}, silent:true, areaStyle:{opacity:0}},
        {type:"line", data:band, stack:"ci", symbol:"none", lineStyle:{opacity:0}, silent:true, areaStyle:{color:"rgba(59,130,246,.14)"}},
        {type:"line", name:"红球命中", data:mean, symbol:"circle", symbolSize:4, lineStyle:{width:1.6, color:"#3b82f6"}},
      ],
      tooltip:{trigger:"axis"},
    });
  }
  // 蓝球命中率滚动
  ch = echartsInit("chCumBlue");
  if (ch) {
    const blues = pick("blue_hit_rate");
    const mean = blues.map(m => (m && m.mean != null) ? +(m.mean * 100).toFixed(2) : null);
    ch.setOption({
      backgroundColor:"transparent", grid:{left:40, right:10, top:18, bottom:24},
      xAxis:{type:"category", data:iss, axisLabel:{color:"#8b949e", interval: step}},
      yAxis:{type:"value", axisLabel:{color:"#8b949e", formatter:"{value}%"}, splitLine:{lineStyle:{color:"#2d333b"}}},
      series:[
        {type:"line", name:"蓝球命中率", data:mean, symbol:"circle", symbolSize:4, lineStyle:{width:1.6, color:"#3fb950"},
         areaStyle:{color:"rgba(63,185,80,.08)"}},
      ],
      tooltip:{trigger:"axis", valueFormatter: v => v + "%"},
    });
  }
}

function renderOnline(rows) {
  const area = $("#onlineEvalArea");
  if (!area) return;
  rows = rows || [];
  const view = rows.slice(-30).reverse();
  area.innerHTML = view.length
    ? '<div class="note" style="margin-bottom:6px">在线对照记录（最近 ' + view.length + " 期，新→旧）</div>" +
      '<div class="scroll"><table>' +
      '<thead><tr><th>期号</th><th>红球命中(均值)</th><th>蓝球命中</th><th>奖金</th><th>注数</th></tr></thead>' +
      '<tbody>' + view.map(r => "<tr><td class='mono'>" + r.issue + "</td><td>" + r.red_hits +
        "</td><td>" + (r.blue_hit ? "✓" : "—") + "</td><td>¥" + Number(r.reward || 0).toFixed(0) +
        "</td><td>" + (r.ticket_count ?? "-") + "</td></tr>").join("") +
      '</tbody></table></div>'
    : '<div class="note">暂无在线对照记录（开奖后可点「在线对照」）。</div>';
}

// ==================== LLM 离线评估（M3.1） ====================

async function runLlmEval() {
  setBusy("#btnLlmEval", "提交评估…");
  try {
    const r = await api("/api/eval/llm/run?issues=60&n=5", {method:"POST"});
    if (!r.ok) throw new Error(r.error || "提交失败");
    const taskId = r.task_id;
    $("#llmEvalArea").innerHTML = "<div class='note'>⏳ 后台评估中（60期×3通道，LLM 通道较慢，约 5~30 分钟）。页面可继续使用，完成后自动刷新本区域。</div>";
    const iv = setInterval(async () => {
      try {
        const t = await api("/api/tasks/" + taskId);
        if (t.status === "completed") {
          clearInterval(iv);
          const rep = await api("/api/eval/llm/latest");
          renderLlmEval(rep);
          toast("LLM 离线评估完成");
        } else if (t.status === "failed") {
          clearInterval(iv);
          $("#llmEvalArea").innerHTML = "<div class='note' style='color:#e63946'>评估失败：" + escHtml(t.message || "未知错误") + "</div>";
        } else {
          const pctv = Math.round((t.progress || 0) * 100);
          $("#llmEvalArea").innerHTML = "<div class='note'>⏳ 评估中：" + escHtml(t.message || "") + "（" + pctv + "%）</div>";
        }
      } catch(e) {}
    }, 4000);
    toast("已提交 LLM 离线评估（后台运行）");
  } catch(e) {
    $("#llmEvalArea").innerHTML = "<div class='note' style='color:#e63946'>提交失败：" + escHtml(e.message) + "</div>";
  }
  setFree("#btnLlmEval", "🧠 LLM 离线评估");
}

function renderLlmEval(rep) {
  const area = $("#llmEvalArea");
  if (!area) return;
  if (!rep || rep.status !== "ready" || !rep.report) {
    area.innerHTML = "<div class='note'>暂无 LLM 离线评估结果（点击「🧠 LLM 离线评估」运行）。</div>";
    return;
  }
  const r = rep.report;
  const sum = r.summary || {};
  const order = ["stat", "stat_llm", "random"];
  const name = {stat:"纯统计", stat_llm:"统计+LLM", random:"随机基线"};
  const rows = order.map(ch => {
    const s = (sum[ch]||{}).metrics || {};
    const u = sum[ch] || {};
    return "<tr><td>" + name[ch] + "</td>" +
      "<td>" + fmt(s.red_hits_mean) + "</td>" +
      "<td>" + pct(s.blue_hit_rate) + "</td>" +
      "<td>" + pct(s.prize_rate_ge5) + "</td>" +
      "<td>" + pct(s.roi) + "</td>" +
      "<td>" + (u.calls||0) + "</td>" +
      "<td>$" + fmt((u.cost_usd||0), 4) + "</td>" +
      "<td>" + Math.round((u.duration_ms||0)/1000) + "s</td></tr>";
  }).join("");
  const comparisons = ((sum.stat_llm||{}).p_values) || [];
  const cmp = comparisons.filter(c => !String(c.metric).endsWith("_adj")).map(c => {
    const label = {stat_llm_vs_stat:"统计+LLM vs 纯统计", stat_llm_vs_random:"统计+LLM vs 随机", stat_vs_random:"纯统计 vs 随机"}[c.pair] || c.pair;
    const mlabel = {red_hits_mean:"红球平均命中", blue_hit_rate:"蓝球命中率", prize_rate_ge5:"≥五等率", roi:"ROI"}[c.metric] || c.metric;
    const sig = c.p < 0.05 ? "<span class='badge gradeA'>显著</span>" : "<span class='badge gradeC'>不显著</span>";
    return "<tr><td>" + label + "</td><td>" + mlabel + "</td><td>" + fmt(c.mean_delta) + "</td><td>" + c.p + " (" + c.method + ")</td><td>" + sig + "</td></tr>";
  }).join("");
  area.innerHTML =
    "<div class='note' style='margin-bottom:6px'>🧠 LLM 离线评估（M3.1）：run " + r.run_id +
    " · " + r.window_issues + " 期 × " + r.tickets + " 注/期 · seed=" + r.seed +
    " · " + (r.created_at||"") + "</div>" +
    "<div class='scroll'><table><thead><tr><th>通道</th><th>红球平均命中</th><th>蓝球命中率</th><th>≥五等率</th><th>ROI</th><th>LLM调用</th><th>估算成本$</th><th>LLM耗时</th></tr></thead><tbody>" + rows + "</tbody></table></div>" +
    "<div class='chart' id='chLlmCmp' style='height:220px;margin-top:8px'></div>" +
    "<div class='scroll' style='margin-top:8px'><table><thead><tr><th>对比</th><th>指标</th><th>Δ均值</th><th>p 值</th><th>结论</th></tr></thead><tbody>" + cmp + "</tbody></table></div>" +
    "<div class='note'>结论：paired 检验（BH 校正后 p<0.05 才视为显著）；cost 为按 token 估算，仅作展示。LLM 通道存在模型噪声，stat/random 通道固定种子可复现。</div>";
  const ch = echartsInit("chLlmCmp");
  if (ch) {
    const cats = ["红球平均命中", "蓝球命中率", "≥五等率", "ROI"];
    const keys = ["red_hits_mean", "blue_hit_rate", "prize_rate_ge5", "roi"];
    ch.setOption({
      backgroundColor:"transparent", grid:{left:40,right:8,top:26,bottom:24},
      tooltip:{trigger:"axis"},
      legend:{textStyle:{color:"#8b949e"}, top:0},
      xAxis:{type:"category", data:cats, axisLabel:{color:"#8b949e"}},
      yAxis:{type:"value", splitLine:{lineStyle:{color:"#2d333b"}}, axisLabel:{color:"#8b949e"}},
      series: order.map(chName => ({
        name: name[chName], type:"bar",
        data: keys.map(k => { const v = ((sum[chName]||{}).metrics||{})[k]; return v == null ? null : +(+v).toFixed(4); }),
      })),
    });
  }
  const trendId = "chLlmTrend";
  const trend = document.createElement("div");
  trend.className = "chart";
  trend.id = trendId;
  trend.style.cssText = "height:180px;margin-top:8px";
  area.appendChild(trend);
  const ch2 = echartsInit(trendId);
  if (ch2 && r.per_issue) {
    const iss = (r.per_issue.stat||[]).map(x => x.issue);
    const win = 10;
    ch2.setOption({
      backgroundColor:"transparent", grid:{left:36,right:8,top:26,bottom:24},
      tooltip:{trigger:"axis"},
      legend:{textStyle:{color:"#8b949e"}, top:0},
      xAxis:{type:"category", data:iss, axisLabel:{color:"#8b949e", interval: Math.max(1, Math.ceil(iss.length/8))}},
      yAxis:{type:"value", splitLine:{lineStyle:{color:"#2d333b"}}, axisLabel:{color:"#8b949e"}},
      series: order.map(chName => {
        const arr = (r.per_issue[chName]||[]).map(x => x.red_hits);
        const smooth = arr.map((_,i) => {
          const s0 = Math.max(0, i - win + 1);
          const seg = arr.slice(s0, i+1);
          return seg.length ? +(seg.reduce((a,b)=>a+b,0)/seg.length).toFixed(3) : null;
        });
        return {name:name[chName], type:"line", smooth:true, showSymbol:false, data:smooth, lineStyle:{width:2}};
      }),
    });
  }
}

// ==================== 数据管理 ====================

async function refreshData() {
  setBusy("#btnRefresh", "抓取中…");
  try {
    const r = await api("/api/refresh", {method:"POST"});
    if (!r.ok) throw new Error(r.error || "刷新失败");
    toast("新增 " + r.inserted_new + " 期，最大期号 " + r.local_max);
    await loadAll();
  } catch(e) { toast("刷新失败: " + e.message); }
  setFree("#btnRefresh", "⟳ 刷新开奖数据");
}

async function showTasks() {
  const area = $("#taskList");
  area.classList.toggle("hidden");
  if (area.classList.contains("hidden")) return;
  try {
    const tasks = await api("/api/tasks?limit=20");
    area.innerHTML = tasks.length
      ? '<table><thead><tr><th>ID</th><th>类型</th><th>状态</th><th>进度</th><th>消息</th><th>时间</th></tr></thead><tbody>' +
        tasks.map(t => "<tr>" +
          "<td class='mono'>" + t.id.slice(0,8) + "</td>" +
          "<td>" + t.kind + "</td>" +
          "<td>" + t.status + "</td>" +
          "<td><div class='bar' style='width:60px'><i style='width:" + (t.progress*100) + "%'></i></div></td>" +
          "<td style='color:var(--muted)'>" + escHtml(t.message || "") + "</td>" +
          "<td class='mono'>" + new Date(t.created_at*1000).toLocaleTimeString() + "</td>" +
        "</tr>").join("") +
      "</tbody></table>"
      : '<div class="note">暂无任务</div>';
  } catch(e) { area.innerHTML = '<div class="note">加载失败</div>'; }
}

async function showStats() {
  const area = $("#statsArea");
  area.classList.toggle("hidden");
  if (area.classList.contains("hidden")) return;
  try {
    const feat = await api("/api/features");
    const w = feat.windows.long.red;
    area.innerHTML =
      '<div class="note">最新期号: ' + feat.issue + " (" + feat.date + ") | 红" + feat.last_reds.join(" ") + " 蓝" + feat.last_blue + '</div>' +
      '<div class="metrics" style="margin-top:8px">' +
        metricItem("历史期数", w.n_draws) +
        metricItem("和值均值", w.sum_mean.toFixed(1)) +
        metricItem("奇偶均值", w.odd_mean.toFixed(1)) +
        metricItem("连号率", pct(w.consecutive_rate)) +
        metricItem("同尾率", pct(w.same_tail_rate)) +
        metricItem("重号均值", w.repeat_mean.toFixed(2)) +
        metricItem("AC均值", w.ac_mean.toFixed(2)) +
        metricItem("红球热号", (w.hot_top6 || []).join(",")) +
        metricItem("红球冷号", (w.cold_top6 || []).join(",")) +
        metricItem("红球遗漏TOP", (w.omit_top6 || []).join(",")) +
      '</div>';
  } catch(e) { area.innerHTML = '<div class="note">加载失败</div>'; }
}

// ==================== 挖掘 & 回测 ====================

async function runBacktest() {
  setBusy("#btnBacktest", "回测中…");
  try {
    const r = await api("/api/patterns/backtest", {method:"POST"});
    renderPatterns(r.items, r.summary);
    toast("回测完成");
  } catch(e) { toast("回测失败: " + e.message); }
  setFree("#btnBacktest", "🧪 重新回测");
}

async function runMining() {
  setBusy("#btnMine", "挖掘中…");
  try {
    const r = await api("/api/mining/run?min_start=300&engine=rf", {method:"POST"});
    if (!r.ok) throw new Error(r.error || "挖掘失败");
    if (r.result) {
      toast("挖掘完成: engine=" + (r.result.engine || "rf") + "，候选 " + r.result.n_candidates +
            "，B级+ " + r.result.accepted + "，通过率 " + pct(r.result.pass_rate));
      const patR = await api("/api/patterns");
      renderPatterns(patR.items, patR.summary);
      loadMiningReports();
    }
    if (r.task_id) toast("挖掘任务 ID: " + r.task_id);
  } catch(e) { toast("挖掘失败: " + e.message); }
  setFree("#btnMine", "⛏️ 自动挖掘");
}

async function loadMiningReports() {
  try {
    const rr = await api("/api/mining/reports?limit=20");
    renderMiningReports(rr.runs || []);
  } catch(e) {}
}

function renderMiningReports(runs) {
  const el = $("#miningRuns");
  if (!el) return;
  el.innerHTML = runs.length
    ? "<b>⛏️ 挖掘运行记录（M3.3）</b><br>" +
      runs.map(r => "· " + (r.created_at || "") + " <code>" + r.engine + "</code> · 特征 " +
        ((r.params && r.params.n_features) || "?") + " 维 · 候选 " + r.candidates +
        " · B级+ " + r.accepted + " · 通过率 " + pct(r.pass_rate) +
        " · avg_lift " + fmt(r.avg_lift, 4)).join("<br>")
    : "";
}

// ==================== 导出 ====================

function exportData() {
  const data = {
    features: allFeatures,
    patterns: allPatterns,
    exportedAt: new Date().toISOString(),
  };
  const blob = new Blob([JSON.stringify(data, null, 2)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "ssq_data_" + new Date().toISOString().slice(0,10) + ".json";
  a.click();
}

// ==================== M2 ML 模型评估 ====================

async function initMlStatus() {
  try {
    const r = await api("/api/ml/status");
    const area = $("#mlEvalArea");
    if (!area) return;
    if (!r.enabled) {
      area.innerHTML = '<div class="note">🤖 ML 概率模型未启用：' + escHtml(r.reason || "reason unknown") + '</div>';
      return;
    }
    if (!r.ready) {
      area.innerHTML = '<div class="note">🤖 ML 概率模型：' + escHtml(r.reason || "后台训练中…") + '</div>';
      return;
    }
    area.innerHTML = '<div class="note">🤖 ML 概率模型已就绪：红球 Brier ' + fmt(r.red_brier, 4) +
      ' · 蓝球 Brier ' + fmt(r.blue_brier, 4) +
      ' · ECE 红 ' + fmt(r.red_calibration_ece, 4) + ' / 蓝 ' + fmt(r.blue_calibration_ece, 4) +
      ' · 训练于 ' + escHtml(r.trained_at || "") + '</div>';
  } catch(e) { /* 静默：评估页未打开也无需报错 */ }
}

async function runMlEval() {
  const btn = $("#btnMlEval");
  const st = $("#mlEvalStatus");
  if (st) st.innerHTML = '<span class="errtxt">⏳ 滚动评估中（约 2~5 分钟）…</span>';
  if (btn) btn.disabled = true;
  try {
    const r = await api("/api/ml/eval?window=30&refit_every=15", {method:"POST"});
    if (!r.ok) { if (st) st.innerHTML = '<span class="errtxt">✗ ' + escHtml(r.error || "评估失败") + '</span>'; return; }
    renderMlEval(r);
    if (st) st.innerHTML = '<span class="oktxt">✓ 完成（' + r.seconds + 's）</span>';
    toast("ML 模型评估完成");
  } catch(e) {
    if (st) st.innerHTML = '<span class="errtxt">✗ ' + escHtml(e.message) + '</span>';
  } finally {
    if (btn) btn.disabled = false;
  }
}

function calTable(bins) {
  if (!bins || !bins.length) return "";
  const rows = bins.filter(b => b.n > 0).map(b =>
    "<tr><td class='mono'>" + (b.bin || "") + "</td><td>" + b.n + "</td><td>" + fmt(b.mean_pred, 4) +
    "</td><td>" + fmt(b.freq, 4) + "</td><td>" + Math.abs((b.mean_pred||0) - (b.freq||0)).toFixed(4) + "</td></tr>").join("");
  return '<div class="scroll" style="max-height:220px"><table><thead><tr><th>预测区间</th><th>n</th><th>平均预测</th><th>实际频率</th><th>偏差</th></tr></thead><tbody>' +
    rows + '</tbody></table></div>';
}

function renderMlEval(r) {
  const red = r.red || {}, blue = r.blue || {};
  const rp = red.paired || {}, bp = blue.paired || {};
  const sec = (title, x, p) =>
    '<h4 class="ana-sec">' + title + "</h4>" +
    '<div class="metrics">' +
      metricItem("Brier(ML)", fmt(x.brier_ml, 4), "均匀基线 " + fmt(x.brier_uniform, 4)) +
      metricItem("log-loss(ML)", fmt(x.logloss_ml, 4), "均匀基线 " + fmt(x.logloss_uniform, 4)) +
      metricItem("paired p", fmt(p.p, 4), p.method === "wilcoxon" ? "Wilcoxon（vs 均匀）" : (p.method || "")) +
    "</div>";
  $("#mlEvalArea").innerHTML =
    '<div class="note" style="margin-bottom:6px">🤖 ML walk-forward 滚动评估：' + (r.n_issues ?? "-") + " 期 · 重训 " +
    (r.refits || 0) + " 次 · 对照均匀随机基线。</div>" +
    sec("🔴 红球模型（33 维）", red, rp) +
    '<div class="note" style="margin:8px 0 4px">红球校准曲线（可靠性图）：预测概率 vs 实际命中频率，越贴对角线越准</div>' +
    calTable(red.calibration) +
    sec("🔵 蓝球模型（16 维）", blue, bp) +
    '<div class="note" style="margin:8px 0 4px">蓝球校准曲线：</div>' +
    calTable(blue.calibration) +
    (r.conclusion ? '<div class="note callout">' + escHtml(r.conclusion) + "</div>" : "");
}

// ==================== 工具 ====================

function escHtml(s) {
  return String(s || "")
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

// ==================== 加载 ====================


// ==================== LLM 配置 ====================

function setLlmStatus(text, cls) {
  const el = $("#llmStatus");
  if (!el) return;
  el.textContent = text;
  el.classList.remove("ok", "bad");
  if (cls) el.classList.add(cls);
  else el.style.color = "";
}

async function loadLlmConfig() {
  try {
    const config = await api("/api/config/llm");
    setLlmStatus(config.configured ? "✅ 已配置" : "❌ 未配置", config.configured ? "ok" : "bad");
    $("#cfgBaseUrl").value = config.base_url || "";
    $("#cfgApiKey").value = config.disabled || !config.api_key ? "" : "******";
    $("#cfgModel").value = config.model || "";
    $("#cfgSamples").value = config.samples || 3;
    $("#cfgLlmEnabled").checked = !config.disabled;
    toggleLlmConfig();
  } catch(e) {
    console.error("loadLlmConfig failed:", e);
    setLlmStatus("❌ 加载失败", "bad");
  }
}

function toggleLlmConfig() {
  const enabled = $("#cfgLlmEnabled")?.checked;
  ["cfgBaseUrl", "cfgApiKey", "cfgModel", "cfgSamples"].forEach(id => {
    const el = $("#" + id);
    if (el) el.disabled = !enabled;
  });
}

async function saveLlmConfig() {
  if (!$("#cfgBaseUrl") || !$("#cfgApiKey") || !$("#cfgModel")) {
    toast("设置表单未加载完成，请刷新页面重试");
    return;
  }
  const payload = {
    base_url: $("#cfgBaseUrl").value.trim(),
    api_key: $("#cfgApiKey").value.trim(),
    model: $("#cfgModel").value.trim(),
    samples: parseInt($("#cfgSamples")?.value || "3") || 3,
    disabled: !$("#cfgLlmEnabled")?.checked,
  };

  if (!payload.disabled && (!payload.base_url || !payload.model || !payload.api_key)) {
    toast("启用 LLM 需填写 API 地址 / Key / 模型名称");
    return;
  }

  const status = $("#llmConfigStatus");
  if (status) status.innerHTML = '<span class="oktxt">⏳ 正在保存…</span>';
  try {
    const res = await api("/api/config/llm", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    if (res && res.ok) {
      toast(payload.disabled ? "LLM 已停用，配置已保存" : "LLM 配置已保存");
      if (status) status.innerHTML = '<span class="oktxt">✓ 已保存，立即生效（无需重启）</span>';
      setLlmStatus(payload.disabled || !res.configured ? "❌ 未配置" : "✅ 已配置", res.configured && !payload.disabled ? "ok" : "bad");
      if ($("#cfgLlm")) $("#cfgLlm").checked = !payload.disabled;
    } else {
      throw new Error((res && res.error) || "服务器无响应");
    }
  } catch(e) {
    console.error("saveLlmConfig failed:", e);
    toast("保存失败: " + e.message);
    if (status) status.innerHTML = '<span class="errtxt">✗ 保存失败：' + e.message + '</span>';
  }
}

async function testLlmConnection() {
  const btn = event.target;
  const resultEl = $("#llmTestResult");
  btn.disabled = true;
  btn.textContent = "测试中…";
  if (resultEl) resultEl.textContent = "";
  try {
    const result = await api("/api/llm/test", {method: "POST"});
    if (result.ok) {
      if (resultEl) resultEl.innerHTML = '<span class="oktxt">✓ 连接成功（' + result.time_ms + 'ms）</span>';
      toast("连接成功");
    } else {
      if (resultEl) resultEl.innerHTML = '<span class="errtxt">✗ ' + (result.error || "连接失败") + '</span>';
    }
  } catch(e) {
    console.error("testLlmConnection failed:", e);
    if (resultEl) resultEl.innerHTML = '<span class="errtxt">✗ ' + e.message + '</span>';
  } finally {
    btn.disabled = false;
    btn.textContent = "🔗 测试连接";
  }
}

// ==================== 方法 A/B 开关（M4.2） ====================

function setMethodsStatus(text, cls) {
  const el = $("#methodsStatus");
  if (!el) return;
  el.textContent = text;
  el.classList.remove("ok", "bad");
  if (cls) el.classList.add(cls);
  else el.style.color = "";
}

function toggleMethodModeHint(mode) {
  const hint = $("#methodModeHint");
  if (!hint) return;
  hint.textContent = mode === "research"
    ? "研究模式：忽略下方开关，全部方法通道启用 —— 用于方法对比实验（决策规则：未经 120 期 paired 验证的方法仅以研究模式存在）。"
    : "生产模式：严格按下方开关过滤方法通道（stat 基线 / ML / LLM / uniform）。";
}

function renderMethodsRegistry(st) {
  const reg = $("#methodsRegistry");
  if (!reg) return;
  const famNames = {stat:"统计基线", blend:"融合", llm:"LLM 推理", ml:"ML 模型", uniform:"均匀对照"};
  const fams = st.families || {};
  const rows = Object.keys(fams).map(f =>
    "<tr><td>" + (famNames[f] || f) + "</td><td><code>" + f + "</code></td><td>" +
    (fams[f] ? "<span class='badge gradeA'>启用</span>" : "<span class='badge gradeC'>关闭</span>") +
    "</td></tr>").join("");
  const regHtml = (st.registry || []).map(r =>
    "<tr><td><code>" + escHtml(r.method) + "</code></td><td>" + escHtml(r.desc) + "</td></tr>").join("");
  reg.innerHTML =
    "<div style='margin:6px 0'><b>方法族开关（按生效模式）：</b>" +
    "<div class='scroll' style='max-height:150px'><table><thead><tr><th>族</th><th>键</th><th>状态</th></tr></thead><tbody>" + rows + "</tbody></table></div></div>" +
    "<div style='margin:6px 0'><b>方法注册表：</b>" +
    "<div class='scroll' style='max-height:150px'><table><thead><tr><th>方法</th><th>说明</th></tr></thead><tbody>" + regHtml + "</tbody></table></div></div>";
}

async function loadMethodsConfig() {
  try {
    const st = await api("/api/methods/status");
    setMethodsStatus(st.mode === "research" ? "🔬 研究模式" : "🏭 生产模式", "ok");
    const modeSel = $("#cfgMethodMode");
    if (modeSel) { modeSel.value = st.mode; toggleMethodModeHint(st.mode); }
    const rawInput = $("#cfgMethods");
    if (rawInput) rawInput.value = st.raw || "";
    renderMethodsRegistry(st);
    // 评估页方法开关指示
    const line = $("#methodEvalLine");
    if (line) {
      const famNames = {stat:"统计", blend:"融合", llm:"LLM", ml:"ML", uniform:"均匀"};
      const fams = st.families || {};
      const parts = Object.keys(fams).map(f => (fams[f] ? "✅" : "⛔") + " " + (famNames[f] || f));
      line.innerHTML = "🎛️ 方法开关：" + (st.mode === "research" ? "研究模式（全部启用）" : "生产模式") +
        " · " + parts.join(" · ") + " &nbsp; <a href='#settings'>前往设置</a>";
    }
  } catch(e) {
    console.error("loadMethodsConfig failed:", e);
    setMethodsStatus("❌ 加载失败", "bad");
  }
}

async function saveMethodsConfig() {
  const status = $("#methodsSaveStatus");
  const payload = {
    methods: ($("#cfgMethods")?.value || "").trim(),
    mode: $("#cfgMethodMode")?.value || "production",
  };
  if (status) status.innerHTML = '<span class="oktxt">⏳ 正在保存…</span>';
  try {
    const res = await api("/api/methods/config", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    if (res && res.ok) {
      toast("方法配置已保存，立即生效（影响之后的预测）");
      if (status) status.innerHTML = '<span class="oktxt">✓ 已保存，模式 ' + res.mode + '</span>';
      setMethodsStatus(res.mode === "research" ? "🔬 研究模式" : "🏭 生产模式", "ok");
      await loadMethodsConfig();
    } else {
      throw new Error((res && res.error) || "服务器无响应");
    }
  } catch(e) {
    console.error("saveMethodsConfig failed:", e);
    toast("保存失败: " + e.message);
    if (status) status.innerHTML = '<span class="errtxt">✗ 保存失败：' + e.message + '</span>';
  }
}

async function loadAll() {
  try {
    const [feat, pats, preds, health] = await Promise.all([
      api("/api/features"),
      api("/api/patterns"),
      api("/api/predictions/last"),
      api("/api/health"),
    ]);
    renderStats(feat);
    renderPatterns(pats.items, pats.summary);
    allPatterns = pats.items;
    if (preds.tickets && preds.tickets.length) {
      renderPredictions({issue: preds.issue, tickets: preds.tickets, note: ""});
    }
    await renderHistory();
    if (health) {
      $("#dataStatus").textContent = health.issues + " 期 | " + (health.max_issue || "");
      if (health.version) {
        const ver = "v" + health.version;
        const vb = $("#verBadge"), vt = $("#verText");
        if (vb) { vb.textContent = ver; vb.title = "系统版本 " + ver + " · M1-M4 已上线 · M5 交互升级中"; }
        if (vt) vt.textContent = ver;
      }
    }
    try {
      const ev = await api("/api/eval");
      renderOnline(ev);
      const cumulative = await api("/api/eval/cumulative?limit=120");
      renderCumulativeEval(cumulative);
      loadMethodRecommendations();
    } catch(e) {}
    initMlStatus();
    try {
      const llmRep = await api("/api/eval/llm/latest");
      renderLlmEval(llmRep);
    } catch(e) {}
    loadMethodsConfig();
    loadMiningReports();
  } catch(e) {
    toast("加载失败: " + e.message);
  }
}

// ==================== 初始化 ====================

window.addEventListener("resize", () => Object.values(charts).forEach(c => c.resize()));

(function handleHash() {
  const hash = location.hash.slice(1) || "predict";
  switchTab(hash);
})();

window.addEventListener("hashchange", () => {
  switchTab(location.hash.slice(1) || "predict");
});

loadAll();
