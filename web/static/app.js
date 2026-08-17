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
let win = "long";
let charts = {};
let allPatterns = [];
let allFeatures = null;
let currentIssue = null;
const TASK_POLL_MS = 1000;
const pendingTasks = {};

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error("HTTP " + r.status);
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
  if (name === "settings") loadLlmConfig();
  if (name === "analysis" && allFeatures) renderWindow(allFeatures);
  if (name === "patterns" && allPatterns.length) renderPatterns(allPatterns, _summary(allPatterns));
  if (name === "replay") populateReplaySelect();
}

function populateReplaySelect() {
  const sel = $("#replaySelect");
  if (!sel) return;
  const val = sel.value;
  api("/api/predictions/history?limit=50").then(data => {
    sel.innerHTML = "<option value=''>-- 选择期号 --</option>";
    data.forEach(item => {
      const opt = document.createElement("option");
      opt.value = item.issue;
      opt.textContent = item.issue + (item.date ? " (" + item.date + ")" : "");
      sel.appendChild(opt);
    });
    if (val) sel.value = val;
  }).catch(() => {});
}

function loadReplay() {
  const issue = $("#replaySelect")?.value;
  if (!issue) { $("#replayResult").innerHTML = ""; return; }
  api("/api/predictions/history?limit=200").then(data => {
    const item = data.find(d => d.issue === issue);
    if (!item) return;
    const act = item.actual;
    const preds = item.predictions || [];
    const res = item.result || {};
    const names = ["","一","二","三","四","五","六"];
    $("#replayResult").innerHTML =
      '<div class="replay-row">' +
        '<div class="replay-col">' +
          '<div class="note">实际开奖</div>' +
          '<div class="balls" style="font-size:18px">' +
            act.reds.map(r => "<span class='ball red'>" + String(r).padStart(2,"0") + "</span>").join("") +
            " <span class='ball blue'>" + String(act.blue).padStart(2,"0") + "</span>" +
          "</div>" +
        "</div>" +
        '<div class="replay-col">' +
          '<div class="note">系统预测 (命中)</div>' +
          preds.map((t,i) => {
            const rh = res.red_hits?.[i] ?? "-";
            const bh = res.blue_hits?.[i] ?? "-";
            const lvl = res.levels?.[i] ?? 0;
            return '<div class="ticket small">' +
              '<div class="balls">' +
                t.reds.map(r => "<span class='ball red sm'>" + String(r).padStart(2,"0") + "</span>").join("") +
                " <span class='ball blue sm'>" + String(t.blue).padStart(2,"0") + "</span>" +
              "</div>" +
              '<span class="meta">红命中' + rh + " 蓝" + (bh ? "✓" : "—") + " → " + (names[lvl] || "-") + "</span>" +
            "</div>";
          }).join("") +
        "</div>" +
      "</div>";
  }).catch(e => toast("加载失败: " + e.message));
}

// ==================== 预测 ====================

async function runPredict(regenerate) {
  if (regenerate === undefined) regenerate = false;
  const n = parseInt($("#cfgTickets")?.value || 10);
  const llm = $("#cfgLlm")?.checked;
  setBusy("#btnPredict", "生成中…");
  try {
    const r = await api("/api/predict?n_tickets=" + n + "&regenerate=" + regenerate + "&use_llm=" + llm, {method:"POST"});
    currentIssue = r.issue;
    localStorage.setItem("lastPredictIssue", r.issue);
    renderPredictions(r);
    toast(r.from_cache ? "已复用缓存预测" : "预测已生成");
    if (r.task_id) {
      runTask(r.task_id, () => Promise.resolve(r));
    }
  } catch (e) { toast("预测失败: " + e.message); }
  setFree("#btnPredict", "🎯 生成预测");
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
    .catch(e => toast("诊断失败: " + e.message));
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
  $("#numModalTitle").textContent = "号码 " + String(num).padStart(2,"0") + " 统计";
  $("#numModalBody").innerHTML =
    '<div class="metrics">' +
      metricItem("出现次数", (freq).toString() + " 次") +
      metricItem("频率", (freq / (w.n_draws || 1)).toFixed(4)) +
      metricItem("当前遗漏", omCur.toString() + " 期") +
      metricItem("平均遗漏", omAvg.toFixed(1) + " 期") +
      metricItem("热冷", hotCold) +
      metricItem("模型概率", prob) +
    '</div>' +
    '<div class="note">窗口: ' + win + ' | 最新: ' + allFeatures.issue + "</div>";
  $("#numModal").style.display = "flex";
}

function showNumDetailBlue(num) {
  if (!allFeatures) return;
  const w = allFeatures.windows[win] || allFeatures.windows.long;
  const blue = w.blue;
  const freq = blue.freq ? (blue.freq[num-1] ?? 0) : 0;
  const omCur = blue.omission_current ? (blue.omission_current[num-1] ?? 0) : 0;
  const probArr = window._currentProbs?.blue_probs || [];
  const prob = (probArr[num-1] ?? 0).toFixed(4);
  $("#numModalTitle").textContent = "蓝球 " + String(num).padStart(2,"0") + " 统计";
  $("#numModalBody").innerHTML =
    '<div class="metrics">' +
      metricItem("出现次数", freq.toString() + " 次") +
      metricItem("当前遗漏", omCur.toString() + " 期") +
      metricItem("重号率", pct(blue.repeat_rate)) +
      metricItem("模型概率", prob) +
    '</div>';
  $("#numModal").style.display = "flex";
}

function closeNumModal(e) {
  if (e && e.target !== $("#numModal")) return;
  $("#numModal").style.display = "none";
}

// ==================== 数据分析 ====================

function setWin(name) {
  win = name;
  $$(".tab-panel .tabs .tab").forEach(el => el.classList.toggle("on", el.dataset.w === name));
  const feat = allFeatures;
  if (feat) renderWindow(feat);
}

function renderStats(feat) {
  allFeatures = feat;
  $("#winStats").textContent =
    "最新一期 " + feat.issue + " " + feat.date + "：红 " + feat.last_reds.join(" ") + " 蓝 " + feat.last_blue;
  renderWindow(feat);
  window._currentProbs = feat.red_probs ? {red_probs: feat.red_probs, blue_probs: feat.blue_probs || []} : {};
}

function renderWindow(feat) {
  const w = feat.windows[win];
  renderHeat(feat, w.red.freq);
  renderOmit(w.red.omission_current);
  renderSumChart(feat.recent);
  renderZoneChart(feat.recent);
}

function heatColor(v, max) {
  const t = Math.max(0, Math.min(1, v / max));
  const r = Math.round(20 + t * 205), g = Math.round(24 + t * 40), b = Math.round(40 + t * 60);
  return "rgb(" + r + "," + g + "," + b + ")";
}

function renderHeat(feat, freq) {
  const max = Math.max(...freq);
  const mean = freq.reduce((a, x) => a + x, 0) / 33;
  const cells = freq.map((v, i) => {
    const num = i + 1;
    const isBlue = num === feat.last_blue;
    const cls = isBlue ? "cell b" : "cell";
    const title = num + ": " + v + "次" + (v > mean * 1.15 ? " 热" : v < mean * 0.85 ? " 冷" : "");
    return '<div class="' + cls + '" style="background:' + (isBlue ? "" : heatColor(v, max)) + '" title="' + title + '" onclick="showNumDetail(' + num + ')">' + String(num).padStart(2,"0") + "</div>";
  }).join("");
  $("#heatFreq").innerHTML = cells;
  $("#heatLegend").innerHTML =
    '<span class="sw" style="background:' + heatColor(0, max) + '"></span>冷 ' +
    '<span class="sw" style="background:' + heatColor(max*0.5, max) + '"></span>中 ' +
    '<span class="sw" style="background:' + heatColor(max, max) + '"></span>热';
}

function renderOmit(om) {
  const top = om.map((v, i) => [i+1, v]).sort((a, b) => b[1] - a[1]).slice(0, 10);
  const ch = echartsInit("chOmit");
  if (!ch) return;
  ch.setOption({
    backgroundColor: "transparent", grid: {left:36, right:8, top:8, bottom:22},
    xAxis: {type:"category", data: top.map(x=>x[0]), axisLabel:{color:"#8b949e"}},
    yAxis: {type:"value", splitLine:{lineStyle:{color:"#2d333b"}}, axisLabel:{color:"#8b949e"}},
    series: [{
      type:"bar", data: top.map(x=>x[1]),
      itemStyle:{color:"#e5484d", borderRadius:[3,3,0,0]},
      label:{show:true, position:"top", color:"#e6edf3", fontSize:10},
    }],
    tooltip: {trigger:"axis"},
  });
}

function renderSumChart(recent) {
  const ch = echartsInit("chSum");
  if (!ch) return;
  const issues = recent.map(r => r.issue.slice(4));
  const sums = recent.map(r => r.sum);
  const mean = sums.reduce((a,b)=>a+b,0) / sums.length;
  ch.setOption({
    backgroundColor:"transparent", grid:{left:40,right:8,top:20,bottom:22},
    xAxis:{type:"category", data:issues, axisLabel:{color:"#8b949e", interval:9}},
    yAxis:{type:"value", splitLine:{lineStyle:{color:"#2d333b"}}, axisLabel:{color:"#8b949e"}},
    series:[
      {name:"和值", type:"line", showSymbol:false, data:sums, lineStyle:{width:2,color:"#3b82f6"}, areaStyle:{color:"rgba(59,130,246,.15)"}},
      {name:"均值", type:"line", showSymbol:false, data:sums.map(()=>mean), lineStyle:{type:"dashed",color:"#d29922"}},
    ],
    tooltip:{trigger:"axis"},
  });
}

function renderZoneChart(recent) {
  const ch = echartsInit("chZone");
  if (!ch) return;
  const z1=[],z2=[],z3=[];
  recent.forEach(r => {
    const a=r.reds.filter(x=>x<=11).length, b=r.reds.filter(x=>x>=12&&x<=22).length;
    z1.push(a); z2.push(b); z3.push(6-a-b);
  });
  ch.setOption({
    backgroundColor:"transparent", grid:{left:40,right:8,top:20,bottom:22},
    xAxis:{type:"category", data:recent.map((_,i)=>i), axisLabel:{show:false}},
    yAxis:{type:"value",max:6,splitLine:{lineStyle:{color:"#2d333b"}}, axisLabel:{color:"#8b949e"}},
    series:[
      {name:"一区1-11",type:"bar",stack:"z",data:z1,itemStyle:{color:"#e5484d"}},
      {name:"二区12-22",type:"bar",stack:"z",data:z2,itemStyle:{color:"#3b82f6"}},
      {name:"三区23-33",type:"bar",stack:"z",data:z3,itemStyle:{color:"#3fb950"}},
    ],
    tooltip:{trigger:"axis"}, legend:{textStyle:{color:"#8b949e"},top:0},
  });
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

async function runOfflineEval() {
  setBusy("#btnEval", "评估中（60-120s）…");
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
    toast("已对照 " + (r.newly_checked || 0) + " 期");
  } catch(e) { toast("在线对照失败: " + e.message); }
  setFree("#btnOnline", "✔ 在线对照");
}

function renderEval(r) {
  const s = r.system, b = r.random_baseline;
  $("#evalArea").innerHTML =
    '<div class="metrics">' +
      metricItem("红球平均命中(系统)", fmt(s.red_hits_mean)) +
      metricItem("红球平均命中(随机)", fmt(b.red_hits_mean)) +
      metricItem("蓝球命中率(系统)", pct(s.blue_hit_rate)) +
      metricItem("蓝球命中率(随机)", pct(b.blue_hit_rate)) +
      metricItem("≥五等奖率(系统)", pct(s.prize_rate_ge5)) +
      metricItem("≥五等奖率(随机)", pct(b.prize_rate_ge5)) +
      metricItem("总奖金(系统/随机)", "¥" + s.reward_total.toFixed(0) + " / ¥" + b.reward_total.toFixed(0)) +
      metricItem("ROI(系统)", pct(s.roi)) +
    '</div>' +
    '<div class="chart" id="chRedDist" style="height:180px"></div>' +
    '<div class="note">' + escHtml(r.note) + '</div>';
  const ch = echartsInit("chRedDist");
  if (ch) {
    const keys = [...new Set([...Object.keys(s.red_hits_dist || {}), ...Object.keys(b.red_hits_dist || {})])].sort();
    ch.setOption({
      backgroundColor:"transparent", grid:{left:34,right:8,top:10,bottom:22},
      xAxis:{type:"category", data:keys.map(k=>k+"红"), axisLabel:{color:"#8b949e"}},
      yAxis:{type:"value", splitLine:{lineStyle:{color:"#2d333b"}}, axisLabel:{color:"#8b949e"}},
      series:[
        {name:"系统", type:"bar", data:keys.map(k=>s.red_hits_dist[k]||0), itemStyle:{color:"#3b82f6"}},
        {name:"随机", type:"bar", data:keys.map(k=>b.red_hits_dist[k]||0), itemStyle:{color:"#8b949e"}},
      ],
      tooltip:{trigger:"axis"}, legend:{textStyle:{color:"#8b949e"},top:0},
    });
  }
}

function renderOnline(rows) {
  const old = document.getElementById("onlineBlock");
  if (old) old.remove();
  const html = rows.length
    ? '<div class="note" style="margin-bottom:6px">在线对照记录：最近 ' + rows.length + ' 期</div>' +
      '<div class="scroll"><table>' +
      '<thead><tr><th>期号</th><th>红球命中</th><th>蓝球命中</th><th>奖金</th></tr></thead>' +
      '<tbody>' + rows.map(r => "<tr><td class='mono'>" + r.issue + "</td><td>" + r.red_hits +
        "</td><td>" + (r.blue_hit ? "✓" : "—") + "</td><td>¥" + Number(r.reward||0).toFixed(0) + "</td></tr>").join("") +
      '</tbody></table></div>'
    : '<div class="note">暂无在线对照记录（开奖后可点「在线对照」）。</div>';
  const ev = $("#evalArea");
  const add = document.createElement("div");
  add.id = "onlineBlock";
  add.style.cssText = "border-top:1px solid var(--border);margin-top:12px;padding-top:10px";
  add.innerHTML = html;
  ev.appendChild(add);
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
  $("#mlEvalArea").innerHTML =
    '<div class="metrics">' +
      metricItem("红球 Brier(ML)", fmt(red.brier_ml, 4)) +
      metricItem("红球 Brier(均匀)", fmt(red.brier_uniform, 4)) +
      metricItem("红球 logloss(ML)", fmt(red.logloss_ml, 4)) +
      metricItem("红球 logloss(均匀)", fmt(red.logloss_uniform, 4)) +
      metricItem("红球 paired p", fmt(rp.p, 4), rp.method === "wilcoxon" ? "Wilcoxon" : rp.method) +
      metricItem("蓝球 Brier(ML)", fmt(blue.brier_ml, 4)) +
      metricItem("蓝球 Brier(均匀)", fmt(blue.brier_uniform, 4)) +
      metricItem("蓝球 logloss(ML)", fmt(blue.logloss_ml, 4)) +
      metricItem("蓝球 logloss(均匀)", fmt(blue.logloss_uniform, 4)) +
      metricItem("蓝球 paired p", fmt(bp.p, 4), bp.method === "wilcoxon" ? "Wilcoxon" : bp.method) +
      metricItem("评估期数", r.n_issues, "重训 " + (r.refits || 0) + " 次") +
    '</div>' +
    '<div style="margin-top:10px" class="note">红球校准曲线（可靠性图）：预测概率 vs 实际命中频率，越贴对角线越准</div>' +
    calTable(red.calibration) +
    '<div style="margin-top:10px" class="note">蓝球校准曲线：</div>' +
    calTable(blue.calibration) +
    '<div style="margin-top:10px" class="note">' + escHtml(r.conclusion || "") + '</div>';
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
        if (vb) { vb.textContent = ver; vb.title = "系统版本 " + ver + "（M1-M3 已上线 · v0.7.0，M4 规划中）"; }
        if (vt) vt.textContent = ver;
      }
    }
    try {
      const ev = await api("/api/eval");
      renderOnline(ev);
    } catch(e) {}
    initMlStatus();
    try {
      const llmRep = await api("/api/eval/llm/latest");
      renderLlmEval(llmRep);
    } catch(e) {}
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
