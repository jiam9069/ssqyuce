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

// ==================== 身份层（W08） ====================
//
// 服务端会话 + CSRF 双提交。前端只需持有 **CSRF token**（内存里，不落 localStorage：
// XSS 能读到的东西不该包括它），Cookie 由浏览器自动携带（HttpOnly + Secure）。
let AUTH = {
  loaded: false,
  authenticated: false,
  enforced: false,
  user: null,
  csrf: "",        // 页面刷新后必须能从 /api/auth/me 恢复；该字段下发属契约变更（李工侧）
};

/** 是否需要在启动时弹登录层。灰度期（未 enforce）一律不弹。 */
function needsLogin() {
  return AUTH.enforced && !AUTH.authenticated;
}

function showAuthOverlay(show, hint) {
  const el = $("#authOverlay");
  if (!el) return;
  el.style.display = show ? "flex" : "none";
  if (show) {
    const h = $("#authHint");
    if (h && hint) h.textContent = hint;
    setTimeout(() => { try { $("#authUser").focus(); } catch (_) {} }, 50);
  }
}

function authFail(elId, msg) {
  const el = $(elId);
  if (!el) { toast(msg); return; }
  el.textContent = msg;
  el.classList.remove("hidden");
}
function authClear(elId) {
  const el = $(elId);
  if (!el) return;
  el.textContent = "";
  el.classList.add("hidden");
}

/** 探测登录态。**任何失败都按"未登录"处理但保留 enforced 原值**，
 *  以免后端临时抖动时把用户踢出并把界面改成"未启用鉴权"。 */
async function loadAuthState() {
  try {
    const j = await api("/api/auth/me", { authProbe: true });
    AUTH.loaded = true;
    AUTH.authenticated = !!j.authenticated;
    AUTH.enforced = !!j.auth_enforced;
    AUTH.user = j.user || null;
    // 契约 API-1 §3.1（v1.1 硬约束）：/api/auth/me 下发 csrf_token。三分支缺一不可：
    //   ① 未登录            → 不赋值（正常空态，值为 null）
    //   ② 已登录 + 非空值    → 赋值（刷新后恢复写能力，无需重新登录）
    //   ③ 已登录 + 空/缺失   → **契约违背**，不是合法空态：记日志 + 按 §1.5 兜底，
    //                         不得静默带空值继续（否则要等写操作 403 才暴露）
    if (j.authenticated) {
      if (typeof j.csrf_token === "string" && j.csrf_token) {
        AUTH.csrf = j.csrf_token;
      } else {
        console.error("[contract] 违背 API-1 §3.1 v1.1：authenticated=true 但 csrf_token 为空/缺失", j);
        recoverCsrfAndHint("/api/auth/me");
      }
    }
    renderUserChip();
    return AUTH;
  } catch (e) {
    console.error("loadAuthState failed:", e);
    AUTH.loaded = true;
    return AUTH;
  }
}

function renderUserChip() {
  const box = $("#userChip");
  if (!box) return;
  if (!AUTH.authenticated || !AUTH.user) {
    box.innerHTML = "";
    box.style.display = "none";
    const nav = $("#navUserAdmin");
    if (nav) nav.style.display = "none";
    return;
  }
  const isAdmin = AUTH.user.role === "admin";
  const role = isAdmin ? " · 管理员" : "";
  box.style.display = "inline-flex";
  box.innerHTML = '<span>👤 <b>' + escHtml(AUTH.user.username) + "</b>" + role + "</span>" +
    '<button class="gear-btn" onclick="doLogout()" title="退出登录">退出</button>';
  // 「更多 → 用户管理」只对管理员出现；普通用户即便猜到 URL 也会被中间件 403
  const nav = $("#navUserAdmin");
  if (nav) nav.style.display = isAdmin ? "" : "none";
}

async function doLogin(ev) {
  if (ev) ev.preventDefault();
  authClear("#authError");
  const btn = $("#authLoginBtn");
  if (btn) { btn.disabled = true; btn.textContent = "登录中…"; }
  try {
    const r = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: ($("#authUser")?.value || "").trim(),
        password: $("#authPass")?.value || "",
      }),
    });
    const j = await r.json().catch(() => ({}));
    if (!r.ok || !j.ok) {
      // 失败原因统一为后端返回文案（人不存在与口令错**必须**是同一句）
      authFail("#authError", j.error || ("登录失败（HTTP " + r.status + "）"));
      return;
    }
    AUTH.authenticated = true;
    AUTH.user = j.user;
    AUTH.csrf = j.csrf_token || "";
    if ($("#authPass")) $("#authPass").value = "";
    showAuthOverlay(false);
    renderUserChip();
    toast("欢迎回来，" + (j.user?.username || ""));
    await loadAll();
  } catch (e) {
    authFail("#authError", "登录请求失败：" + e.message);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = "登录"; }
  }
}

async function doBootstrap(ev) {
  if (ev) ev.preventDefault();
  authClear("#bootError");
  const btn = $("#authBootBtn");
  if (btn) { btn.disabled = true; btn.textContent = "初始化中…"; }
  try {
    const r = await fetch("/api/auth/bootstrap", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        token: $("#bootToken")?.value || "",
        password: $("#bootPass")?.value || "",
      }),
    });
    const j = await r.json().catch(() => ({}));
    if (!r.ok || !j.ok) {
      authFail("#bootError", j.error || ("初始化失败（HTTP " + r.status + "）"));
      return;
    }
    toast("管理员已初始化，请用新口令登录");
    if ($("#authUser")) $("#authUser").value = j.username || "admin";
    if ($("#bootToken")) $("#bootToken").value = "";
    if ($("#bootPass")) $("#bootPass").value = "";
  } catch (e) {
    authFail("#bootError", "初始化请求失败：" + e.message);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = "完成初始化"; }
  }
}

async function doLogout() {
  try {
    await fetch("/api/auth/logout", {
      method: "POST",
      headers: { "X-CSRF-Token": AUTH.csrf || "" },
    });
  } catch (_) {}
  AUTH.authenticated = false;
  AUTH.user = null;
  AUTH.csrf = "";
  renderUserChip();
  if (AUTH.enforced) showAuthOverlay(true, "已退出登录。");
  else location.reload();
}

// ==================== 用户管理（管理员） ====================

async function openUserAdmin() {
  const el = $("#userAdminOverlay");
  if (!el) return;
  if (!AUTH.authenticated || AUTH.user?.role !== "admin") {
    toast("需要管理员权限");
    return;
  }
  closeMore();
  el.style.display = "flex";
  await refreshUserAdmin();
}

function closeUserAdmin() {
  const el = $("#userAdminOverlay");
  if (el) el.style.display = "none";
}

async function refreshUserAdmin() {
  const box = $("#userAdminList");
  if (!box) return;
  box.innerHTML = '<div class="note">加载中…</div>';
  try {
    const j = await api("/api/admin/users");
    const rows = (j.users || []).map(u => {
      const me = AUTH.user && u.user_id === AUTH.user.user_id;
      const inactive = u.status !== "active";
      const del = u.deleted_at ? ' <span class="u-meta">（已删除）</span>' : "";
      const tags =
        '<span class="auth-role-tag ' + (u.role === "admin" ? "is-admin" : "") + '">' + escHtml(u.role) + "</span>" +
        '<span class="auth-role-tag ' + (inactive ? "is-inactive" : "") + '">' + escHtml(u.status) + "</span>";
      // 自保护：不给自己渲染停用/降级/删除按钮（后端也会拒，但界面上不该出现）
      const actions = (me || u.deleted_at) ? "" : [
        '<button class="ghost" onclick="adminToggleUser(' + u.user_id + ', \'' +
          (inactive ? "active" : "disabled") + '\')">' + (inactive ? "启用" : "停用") + '</button>',
        '<button class="ghost" onclick="adminToggleRole(' + u.user_id + ', \'' +
          (u.role === "admin" ? "user" : "admin") + '\')">' +
          (u.role === "admin" ? "降为普通" : "升为管理员") + '</button>',
        '<button class="ghost" onclick="adminResetPassword(' + u.user_id + ')">重置口令</button>',
        '<button class="ghost" onclick="adminDeleteUser(' + u.user_id + ')">删除</button>',
      ].join("");
      return '<div class="auth-user-row">' +
        '<span class="u-name">' + escHtml(u.username) + "</span>" + tags + del +
        '<span class="u-meta">' + (me ? "（我）" : "") +
          (u.last_login_at ? " 上次登录 " + new Date(u.last_login_at * 1000).toLocaleString() : " 从未登录") +
        "</span>" +
        '<span class="u-spacer"></span>' + actions + "</div>";
    }).join("");
    box.innerHTML = rows || '<div class="note">暂无用户</div>';
  } catch (e) {
    box.innerHTML = '<div class="note">加载失败：' + escHtml(e.message) + "</div>";
  }
}

async function adminPatch(userId, payload) {
  return api("/api/admin/users/" + userId, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", "X-CSRF-Token": AUTH.csrf },
    body: JSON.stringify(payload),
  });
}

async function adminToggleUser(userId, status) {
  try {
    await adminPatch(userId, { status });
    toast(status === "active" ? "已启用" : "已停用");
    await refreshUserAdmin();
  } catch (e) { toast("操作失败：" + e.message); }
}

async function adminToggleRole(userId, role) {
  try {
    await adminPatch(userId, { role });
    toast("角色已改为 " + role);
    await refreshUserAdmin();
  } catch (e) { toast("操作失败：" + e.message); }
}

async function adminResetPassword(userId) {
  const pw = prompt("输入新的初始口令（至少 8 位）。留空取消。");
  if (!pw) return;
  try {
    await api("/api/admin/users/" + userId + "/reset-password", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": AUTH.csrf },
      body: JSON.stringify({ password: pw }),
    });
    toast("口令已重置（该用户所有旧会话已失效）");
  } catch (e) { toast("重置失败：" + e.message); }
}

async function adminDeleteUser(userId) {
  if (!confirm("确认删除该用户？删除后该账号无法登录（数据保留）。")) return;
  try {
    await api("/api/admin/users/" + userId, {
      method: "DELETE",
      headers: { "X-CSRF-Token": AUTH.csrf },
    });
    toast("已删除");
    await refreshUserAdmin();
  } catch (e) { toast("删除失败：" + e.message); }
}

async function doCreateUser() {
  const st = $("#nuStatus");
  const name = ($("#nuName")?.value || "").trim();
  const pass = $("#nuPass")?.value || "";
  const role = $("#nuRole")?.value || "user";
  if (!name || !pass) { toast("用户名与初始口令均为必填"); return; }
  if (st) st.innerHTML = "⏳ 创建中…";
  try {
    await api("/api/admin/users", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": AUTH.csrf },
      body: JSON.stringify({ username: name, password: pass, role }),
    });
    if (st) st.innerHTML = '<span class="oktxt">✓ 已创建</span>';
    $("#nuName").value = ""; $("#nuPass").value = "";
    await refreshUserAdmin();
  } catch (e) {
    if (st) st.innerHTML = '<span class="errtxt">✗ ' + escHtml(e.message) + "</span>";
  }
}

/**
 * 统一 fetch 封装。
 *
 * 自动补 `X-CSRF-Token`（写操作）与鉴权失败处理（口径见
 * `docs/team/engineering/BACKEND.md` §2.1 / §2.7，本项目是 Cookie 会话制，
 * **没有 refresh token**，401 靠 `WWW-Authenticate` 头区分三种形态）：
 *
 *  * 401 + `WWW-Authenticate: Cookie` → 会话失效/被踢/过期 → 跳登录层（**并发只跳一次**）；
 *  * 401 + `WWW-Authenticate: Bearer` → 部署配了 LOTT_TOKEN 但没带 → 提示联系管理员，**不跳登录**；
 *  * 401 无 `WWW-Authenticate` → 登录接口本身凭据错 → 就地显示，**不跳转**；
 *  * 403 `CSRF 校验失败` → 页面刷新后 csrf 丢失（不是会话失效）→ 重新拉 `/api/auth/me`
 *    拿新 csrf 后**重试一次**；仍失败则引导重新登录；
 *  * 403 `需要管理员权限` → 明确提示，不静默失败；
 *  * 423 账号锁定 → 提示倒计时，**不自动重试**（重试会续锁）。
 */
async function api(path, opts = {}) {
  const isProbe = !!opts.authProbe;
  const o = Object.assign({}, opts);
  delete o.authProbe;
  // 写操作自动带 CSRF（GET/HEAD 不需要，避免无谓的跨域预检）
  const method = (o.method || "GET").toUpperCase();
  if (!["GET", "HEAD", "OPTIONS"].includes(method) && AUTH.csrf) {
    o.headers = Object.assign({}, o.headers || {}, { "X-CSRF-Token": AUTH.csrf });
  }
  const r = await fetch(path, o);
  if (!r.ok) {
    // 优先展示后端 JSON 里的 error 详情（如“大模型不可用或超时(...)”、
    // 严格模式“无可用规律/LLM 通道不可用”等）；解析失败时退回 HTTP 状态码。
    let msg = "HTTP " + r.status;
    let payload = null;
    try {
      const t = await r.text();
      const j = JSON.parse(t);
      if (j && j.error) msg = j.error;
      payload = j;
    } catch (_) {}
    const err = new Error(msg);
    err.status = r.status;
    err.payload = payload || null;
    if (!isProbe) {
      handleAuthFailure(r, payload, msg, path);
    }
    throw err;
  }
  return r.json();
}

/** 401/403/423 的统一失败分支。抽成函数是为了让「并发只跳一次」的判定只有一处实现，
 *  避免同一逻辑在多个调用点各写一份、越改越不一致。 */
function handleAuthFailure(r, payload, msg, path) {
  const status = r.status;
  if (status === 401) {
    // 按响应头区分，不按路径枚举（分路径枚举迟早漏一个）
    const wwa = (r.headers.get("www-authenticate") || "").toLowerCase();
    if (wwa.indexOf("cookie") >= 0) {
      // 会话失效：只有这一支才跳登录
      AUTH.authenticated = false;
      AUTH.enforced = true;
      AUTH.csrf = "";
      showAuthOverlay(true, "登录状态已失效，请重新登录。");
    } else if (wwa.indexOf("bearer") >= 0) {
      // 部署配置问题：跳登录只会让人白输一次口令
      toast("服务端要求 Bearer Token（部署配置问题），请联系管理员");
    }
    // 无 WWW-Authenticate → 登录接口凭据错，调用方就地显示表单错误，这里不处理
    return;
  }
  if (status === 403) {
    if (msg.indexOf("CSRF") >= 0) {
      // 契约 API-1 §1.5：CSRF 失败是 403 不是 401（会话有效、只是防伪令牌缺失/不匹配）。
      // 契约落地后正常不该出现；仍出现则按 §1.5 兜底：恢复一次 → 仍失败就引导重新登录。
      recoverCsrfAndHint(path);
      return;
    }
    if (msg.indexOf("管理员") >= 0) {
      toast("需要管理员权限：" + path);
      return;
    }
    if (msg.indexOf("账户不可用") >= 0) {
      // 契约 §1.4：status 不可用（bootstrap/disabled/locked/软删）。
      // 语义 = 这个账号不能再用了，而不是口令错；清本地态并跳登录，且**不自动重登**。
      AUTH.authenticated = false;
      AUTH.user = null;
      AUTH.csrf = "";
      showAuthOverlay(true, "该账号已不可用（被停用或锁定），请联系管理员。");
      return;
    }
    return;
  }
  // 契约 §3.2：默认配置下「第 6 次失败」先撞限流返回 429，而不是 423 —— 两者都要表现成「稍后再试」
  if (status === 423 || status === 429) {
    // 423 不要自动重试（会续锁）；429 也不要（会继续消耗限流桶）
    toast(msg || "尝试次数过多，请稍后再试");
    return;
  }
}

let csrfRecovering = false;

/** CSRF 失效兜底：拉新 csrf → 有则静默重试一次；仍失败才引导重新登录。
 *  `csrfRecovering` 保证并发多个 403 时只发起一次恢复、只提示一次（对齐「并发只跳一次」）。 */
async function recoverCsrfAndHint(path) {
  if (csrfRecovering) return;
  csrfRecovering = true;
  try {
    // 注意：必须重新**发一次**探测，不能读 loadAuthState() 的返回值 ——
    // 它返回的是「共享的 AUTH 对象」，读它等于读原值，等于「什么都没验」。
    // （2026-10-21 浏览器实测踩到：因为读的是返回值，恢复永远判成功、永不跳登录。）
    const j = await api("/api/auth/me", { authProbe: true });
    if (j && j.authenticated && typeof j.csrf_token === "string" && j.csrf_token) {
      AUTH.csrf = j.csrf_token;
      toast("凭据已刷新，请重试刚才的操作");
      return;
    }
    AUTH.authenticated = false;
    AUTH.enforced = true;
    showAuthOverlay(true, "页面凭据已失效，请重新登录。");
  } catch (_) {
    AUTH.authenticated = false;
    AUTH.enforced = true;
    showAuthOverlay(true, "页面凭据已失效，请重新登录。");
  } finally {
    csrfRecovering = false;
  }
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

/** 从首屏「诚实收益边界」卡跳到达标卡片（蓝球覆盖 / 大众偏好度）。
 *
 * 两张卡都在「评估报告 → 综合概览」子页里，所以要先切主标签、再切子标签、
 * 最后滚动并高亮。三步都做了失败兜底：任意一步找不到容器就静默跳过，
 * 只保证「不会点一下什么也没发生还报错」。
 */
function jumpToCard(cardId, tabName) {
  const finish = () => {
    const el = document.getElementById(cardId);
    if (!el) { toast("该卡片暂不可用（接口未返回）"); return; }
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    el.classList.add("hb-flash");
    setTimeout(() => el.classList.remove("hb-flash"), 1600);
  };
  // `tabName` 缺省为 "eval"：首屏「诚实收益边界」卡的两个入口（蓝球覆盖 / 大众偏好度）
  // 都在「评估报告 → 综合概览」里。footer 的「随机性诚实声明」在首屏（predict），
  // 显式传 "predict" 即可复用同一个函数，避免再写一个只差两行的孪生函数。
  const tab = tabName || "eval";
  try {
    switchTab(tab);
    if (tab === "eval" && typeof setEvalSub === "function") setEvalSub("overview");
    // 面板刚切换，可能还在 display 变换中：等一帧再滚，否则 scrollIntoView 目标高度为 0
    requestAnimationFrame(() => setTimeout(finish, 60));
  } catch (e) {
    finish();
  }
}

//: V29：现存 tab 白名单。删除页面后必须用它兜底 —— 否则历史书签里的
//: `#analysis` / `#patterns` / `#replay` 会命中一个不存在的 panel，
//: `switchTab` 把所有 `.active` 都摘掉却不加回任何一个 ⇒ **整页白屏**。
//: 未知 tab 一律回落 `predict`，并把 hash 纠正回来（否则用户刷新一次又白屏）。
const VALID_TABS = ["predict", "eval", "data", "settings"];

function switchTab(name) {
  if (!VALID_TABS.includes(name)) {
    name = "predict";
  }
  history.replaceState(null, "", "#" + name);
  $$(".nav-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab-panel").forEach(p => p.classList.toggle("active", p.id === "tab-" + name));
  // 切到研究页时收起「更多」抽屉
  if (name !== "predict") closeMore();
  if (name === "predict") { hideGearPop(); renderCountdown(); }
  if (name === "settings") { loadLlmConfig(); loadMethodsConfig(); }
  // ===== V29：analysis / patterns / replay 三页已移除 =====
  //
  // 用户 2026-10-24 指示「更多」下四个页面功能都不要。删掉页面容器后，
  // 这里的分支也必须一并删除 —— 否则 hashchange 仍能通过 `#analysis` 等
  // URL 把页面切到一个不存在的 tab（`switchTab` 会 replaceState 保留该 hash，
  // 无匹配 panel ⇒ 白屏）。
  //
  // 保留数据加载函数本体（`renderWindow` / `renderPatterns` / `populateReplaySelect`
  // 等）：它们与 `allFeatures` / `allPatterns` 共享同一条数据管线，
  // 而「评估报告」页仍在消费这些数据。贸然删函数会牵动整条链路。
  // V5：切到「评估报告 → 实战战绩」时按需拉进化状态（快接口）；
  // 重操作回放一律留给用户显式点击，绝不在这里自动触发。
  if (name === "eval" && !_evoStatus) loadEvolutionStatus(false);
}

// ==================== 「更多」抽屉 ====================
function openMore() {
  $("#moreOverlay")?.classList.add("open");
  $("#moreDrawer")?.classList.add("open");
  hideGearPop();
}
function closeMore(e) {
  // 点击遮罩或关闭按钮时收起；点击抽屉内部不收起
  if (e && e.target !== $("#moreOverlay")) return;
  $("#moreOverlay")?.classList.remove("open");
  $("#moreDrawer")?.classList.remove("open");
}

// ==================== 齿轮弹层（常用设置） ====================
function toggleGearPop(e) {
  if (e) e.stopPropagation();
  const pop = $("#gearPop");
  if (!pop) return;
  pop.classList.toggle("hidden");
  if (!pop.classList.contains("hidden")) closeMore();
}
function hideGearPop() {
  const pop = $("#gearPop");
  if (pop) pop.classList.add("hidden");
}
document.addEventListener("click", (e) => {
  // 点击弹层外部时收起齿轮弹层
  const pop = $("#gearPop");
  if (pop && !pop.classList.contains("hidden") &&
      !pop.contains(e.target) && !(e.target.closest && e.target.closest(".gear-btn"))) {
    pop.classList.add("hidden");
  }
});

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

async function runPredict(regenerate, btnId, allowFallback) {
  if (regenerate === undefined) regenerate = false;
  if (btnId === undefined) btnId = regenerate ? "#btnRegen" : "#btnPredict";
  const n = parseInt($("#cfgTickets")?.value || 10);
  const llm = $("#cfgLlm")?.checked;
  setBusy(btnId, regenerate ? "强制生成中…" : "生成中…");
  $("#predStatus").textContent = "⏳ 正在请求服务器，请稍候…";
  $("#predStatus").style.color = "var(--gold)";
  try {
    // V16：`allow_stat_fallback=1` 表示用户已在前端明确同意「严格模式不出号时改用统计模式」。
    // 没有这个标记时后端绝不会自动降级 —— 严格模式的语义是"没有真凭实据就不出号"，
    // 悄悄给一份统计票正是它要禁止的事。
    const _fb = allowFallback ? "&allow_stat_fallback=1" : "";
    const r = await api("/api/predict?n_tickets=" + n + "&regenerate=" + regenerate + "&use_llm=" + llm + _fb, {method:"POST"});
    currentIssue = r.issue;
    localStorage.setItem("lastPredictIssue", r.issue);
    renderPredictions(r);
    // 注数以服务端实际返回为准（LLM 推理模式下不补齐到请求注数）
    const nActual = (r.tickets || []).length;
    const nReq = r.requested_tickets || n;
    const cntTxt = nActual < nReq ? nActual + " 注（按大模型实际产出，未补齐）" : nActual + " 注";
    if (r.from_cache) {
      $("#predStatus").textContent = "✅ 已复用缓存预测（期号 " + r.issue + "，" + cntTxt + "）";
      $("#predStatus").style.color = "var(--green)";
      toast("已复用缓存预测");
    } else {
      $("#predStatus").textContent = "✅ 预测已生成（期号 " + r.issue + "，" + cntTxt + "）";
      $("#predStatus").style.color = "var(--green)";
      toast("预测已生成 " + nActual + " 注");
    }
    if (r.task_id) {
      runTask(r.task_id, () => Promise.resolve(r));
    }
  } catch (e) {
    toast("预测失败: " + e.message);
    $("#predStatus").textContent = "❌ " + e.message;
    $("#predStatus").style.color = "var(--red)";
    // V11 严格模式：422 且 strict_mode 时展示专门横幅（无规律/LLM 关）
    const b = $("#strictBanner");
    const p = e.payload || {};
    if (b) {
      if (e.status === 422 && p.strict_mode) {
        b.classList.remove("hidden");
        b.innerHTML = '<span class="strict-ico">⛔</span> <b>严格模式 · 未出号</b><br>' +
          escHtml(p.error || e.message) +
          (p.strict_gates && p.strict_gates.length
            ? '<div class="strict-gates">未通过：' +
              p.strict_gates.map((g) =>
                g === "strategy" ? "规律为 0 条" : g === "llm" ? "LLM 通道不可用" : g
              ).join(" / ") + "</div>"
            : "");
        b.style.color = "var(--red)";
        // V16：让用户**自己选**是否改用统计模式（不替用户决定，也不悄悄降级）。
        if (p.can_fallback) {
          b.innerHTML += '<div class="strict-fallback" style="margin-top:8px">' +
            '<button class="btn" id="btnStatFallback" style="font-size:13px">' +
            '改用统计模式生成</button>' +
            '<div class="note" style="margin-top:6px">' +
            escHtml(p.fallback_hint || "统计模式不含 LLM 判断；生成结果会被明确标注为统计口径。") +
            "</div></div>";
          const fbBtn = $("#btnStatFallback");
          if (fbBtn) {
            fbBtn.onclick = () => {
              fbBtn.disabled = true;
              fbBtn.textContent = "统计模式生成中…";
              runPredict(regenerate, btnId, true);
            };
          }
        }
      } else {
        b.classList.add("hidden");
      }
    }
  }
  setFree(btnId, regenerate ? "🔄 强制重新生成" : "🎯 生成候选注单");
}

function renderPredictions(res) {
  // V7：目标期号 + 数据基准。期号由后端按真实开奖日历（每周二/四/日）推算，
  // 数据落后时会明确提示「数据只到哪一期」，不再让人对着一个凑不出来的期号猜。
  const basis = res.data_basis || {};
  let issueTxt = res.issue ? "目标期号 " + res.issue : "";
  if (basis.basis_issue) issueTxt += "（基于上期 " + basis.basis_issue + " · " + (basis.basis_date || "") + "）";
  $("#predIssue").textContent = issueTxt;
  renderPredMeta(res);
  renderRuleUsage(res);
  const list = $("#predList");
  const items = res.tickets || [];
  if (!items.length) {
    list.innerHTML = '<div class="note">暂无候选注单，点击「生成候选注单」。</div>';
    return;
  }
  window._predTickets = items;
  window._predIssue = res.issue;
  window._currentProbs = res.red_probs ? {red_probs: res.red_probs, blue_probs: res.blue_probs || []} : {};
  list.innerHTML = items.map((t, i) => {
    const balls = t.reds.map(r => "<span class='ball red sm' onclick='event.stopPropagation();showNumDetail(" + r + ")'>" + String(r).padStart(2,"0") + "</span>").join("");
    const blue = "<span class='ball blue sm' onclick='event.stopPropagation();showNumDetailBlue(" + t.blue + ")'>" + String(t.blue).padStart(2,"0") + "</span>";
    const badge = t.method.startsWith("llm:")
      ? "<span class='t-badge llm'>LLM推理</span>"
      : "<span class='t-badge'>" + escHtml(t.method) + "</span>";
    // 结构小结：一句话（和值 · 三区 · 奇偶），描述性、不含评价
    // —— 这里**故意不再放任何分级图形**（★☆ / ●●○ / 进度条 / 颜色深浅）。
    //    2026-10-21 产品审计否决了"改成『形态合理度 ★★☆』+ 标签注明非中奖概率"的方案：
    //    用户先看到星、未必读到标签；任何带级图形都会被读成"星多 = 更易中"。
    //    而 confidence 的真实语义是「混合分布上的概率质量」，即**内部排序权重**，
    //    给它做五级分级等于凭空制造信息。故原位只留下这三项恒等约束的实际取值。
    const sum = ticketSummary(t);
    // 详情（点击展开后才显示）——保留全部证据/规律/反证/结构分，但默认折叠
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
      // 原标签「📐 结构分」会被读成"分数越高越该买"。它的真实语义是
      // ensemble_mass ——「混合分布上的概率质量」，即**系统内部排序权重**，
      // 既不是中奖概率，也不能跨期比较（分母随当期混合分布变化）。
      evd += '<div class="reasoning">系统内部排序权重（非中奖概率，不可跨期比较）：' +
        sc + '</div>';
    }
    const body = (rt || used || evd)
      ? '<div class="t-body">' + rt + used + evd + '</div>'
      : '';
    return '<div class="ticket mini" id="tick-' + i + '" onclick="toggleTicket(' + i + ')">' +
      '<div class="t-head">' +
        '<span class="t-idx">' + (i + 1) + '</span>' +
        '<div class="balls">' + balls + " " + blue + "</div>" +
        badge +
        '<span class="t-sum">' + sum + '</span>' +
        "<button class='t-copy' onclick='event.stopPropagation();copyTicket(" + i + ")'>📋</button>" +
        '<span class="chev">▾</span>' +
      '</div>' + body + '</div>';
  }).join("");
  $("#predNote").textContent = res.note || "";
  renderWinReceipt();
  renderCountdown();
}

// V7：预测元信息（生成时间 / 数据基准 / 数据新鲜度 / 注数口径）
function fmtTs(ts) {
  if (!ts) return "—";
  const n = Number(ts);
  const d = new Date((n > 1e12 ? n : n * 1000));
  if (isNaN(d.getTime())) return "—";
  const p = x => String(x).padStart(2, "0");
  return d.getFullYear() + "-" + p(d.getMonth() + 1) + "-" + p(d.getDate()) + " " +
    p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
}

function renderPredMeta(res) {
  const el = $("#predMeta");
  if (!el) return;
  const basis = res.data_basis || {};
  const items = [];
  // 生成时间：优先用票上的 created_at（真实落库时间），其次 res.generated_at
  let ts = res.generated_at_ts;
  if (!ts) {
    const list = (res.tickets || []).filter(t => t.created_at);
    if (list.length) ts = Math.max.apply(null, list.map(t => Number(t.created_at)));
  }
  let genTxt = ts ? fmtTs(ts) : (res.generated_at || null);
  if (!genTxt) genTxt = res.from_cache ? "未知（历史缓存预测，生成时未记录时间）" : "—";
  items.push('<span class="pm-item">🕒 生成时间 <b class="pm-key">' + escHtml(genTxt) + '</b>' +
    (res.from_cache ? '<span class="pm-warn">（缓存）</span>' : "") + '</span>');
  // V16：口径来源必须可见。stat_fallback = 严格模式未出号后改用统计模式生成的，
  // 不标出来用户会以为这批票过了严格模式（2026-10-22 的 5 注票就是这样被误读的）。
  if (res.source_mode === "stat_fallback") {
    const _g = (res.strict_gates || []).map(g =>
      g === "replayable_rules" ? "无可回放规律" : g === "llm" ? "LLM 不可用" :
      g === "strategy" ? "规律为 0 条" : g).join(" / ");
    items.push('<span class="pm-item" style="color:var(--gold);font-weight:600">' +
      '📊 口径：统计模式（严格模式未出号' + (_g ? "：" + escHtml(_g) : "") + '）</span>');
  }
  // V22 / L2：AI 角色必须显示 —— 「AI 解释了结果」与「AI 选了号」是两件事。
  if (res.ai_role === "advisor") {
    items.push('<span class="pm-item" style="color:var(--muted)">🤖 AI 角色：<b>仅解释</b>' +
      '（未参与选号；模式 advisory）</span>');
  } else if (res.ai_role === "picker") {
    items.push('<span class="pm-item">🤖 AI 角色：<b>参与选号</b>（模式 ' +
      escHtml(res.llm_output_mode || "llm") + '）</span>');
  }
  if (res.source_mode === "strict") {
    // V18：必须交代**凭据类型**——L1（统计规律，更强）与 L2（结构约束，可复核）
    // 是两种不同强度的证据，只写"严格模式"会让用户以为有统计规律支撑。
    const _basis = res.evidence_basis;
    const _bTxt = _basis === "L1_statistical"
      ? "统计规律（可回放）"
      : _basis === "L2_structural" ? "结构约束（可逐注复核）"
      : _basis === "none" ? "无可复核凭据" : "未知";
    items.push('<span class="pm-item" style="color:var(--green)">✅ 口径：严格模式 · 依据：' +
      escHtml(_bTxt) + '</span>');
    if (res.evidence_note_zh) {
      items.push('<span class="pm-item" style="color:var(--muted);font-size:12px">' +
        escHtml(res.evidence_note_zh) + '</span>');
    }
  }
  if (basis.basis_issue) {
    items.push('<span class="pm-item">📥 数据基准 <b>' + escHtml(basis.basis_issue) + '</b>（' +
      escHtml(basis.basis_date || "") + '）</span>');
  }
  if (basis.days_since !== undefined && basis.days_since !== null) {
    const stale = basis.expected_draws_missing;
    items.push('<span class="pm-item' + (stale ? ' pm-warn' : '') + '">' +
      (stale ? "⚠️ 数据落后 " + basis.days_since + " 天（缺 " + basis.missed_draws + " 期，目标期号已按日历顺推）"
             : "✅ 数据与开奖日历一致（距今 " + basis.days_since + " 天）") + '</span>');
  }
  if (basis.next_draw_date) {
    items.push('<span class="pm-item">📅 预计开奖 <b>' + escHtml(basis.next_draw_date) + '</b></span>');
  }
  const nReq = Number(res.requested_tickets || 0);
  const nAct = (res.tickets || []).length;
  if (nReq || nAct) {
    items.push('<span class="pm-item">🎫 注数 <b>' + nAct + '</b>' + (nReq && nReq !== nAct ? " / 请求 " + nReq : "") + '</span>');
  }
  const r = res.rules || {};
  if (r.mode) {
    const modeTxt = r.mode === "active" ? "已生效" : (r.mode === "shadow" ? "影子模式（只记录不改号）" : "未使用规律");
    items.push('<span class="pm-item">🧩 规律 <b>' + escHtml(modeTxt) + '</b>' +
      (r.usable ? '（可用 ' + r.usable + ' 条）' : "") + '</span>');
  }
  el.innerHTML = items.join("");
}

// V7：本期规律使用情况（可用 / 只监控 / 不可用 + 影子模式的实际影响）
function renderRuleUsage(res) {
  const el = $("#ruleBody");
  const tag = $("#ruleTag");
  if (!el) return;
  const r = res.rules || {};
  if (!r.mode) { if (tag) tag.textContent = "未加载"; return; }
  const snap = r.snapshot || {};
  const counts = snap.counts || {};
  if (tag) tag.textContent = r.mode === "active" ? "规律已生效" : (r.mode === "shadow" ? "影子模式" : "未使用规律");
  let h = '<div class="track-metrics">' +
    // V14-A4：口径改名。这里统计的是「生产快照里可回放且可用的规律」，
    // 与「闲时挖掘的证据项(rule_evidence)」是两件不同的事，不能都叫"可用规律"。
    // 后端 routes_predict.py 明令：不得把 L2 结构约束叫成「可用规律」。
    // 口径（V14-A4 + 2026-10-21 文案审计）：这个名字必须自己说清"是什么门槛筛出来的"，
    // 而不是让用户自己去猜"可用"和"可用规律"的区别。
    // 副标题**固定写死**「0 条 = 统计否定结论，不是功能故障」——因为 0 条是本项目的
    // **预期结果**，但用户看到 0 的第一反应是"坏了"。
    // 兜底顺序：优先 usable_replayable（权威）；usable 是已弃用别名，仅兼容旧快照；
    // 两者都缺才回落 0 —— 回落 0 时不假装有数据。
    metricItem("可回放且通过前置注册的规律",
               Number(r.usable_replayable || r.usable || 0),
               "0 条 = 统计否定结论，不是功能故障") +
    metricItem("本轮受规律影响的候选", Number(r.n_resorted || 0), r.mode === "active" ? "已按规律重排" : "影子模式：仅记录") +
    // V12-04（裁决 R-2）：**禁止跨层相加**。usable 只显示长期可用，
    // 阶段可用单独列出并标注「未获支持」，避免出现「N 条可用其中大部分没预测力」。
    metricItem("长期可用 / 监控 / 否决",
      (counts.usable_long_term || 0) + " / " + (counts.monitor || 0) + " / " + (counts.rejected || 0),
      counts.usable_phase
        ? "另：阶段可用 " + counts.usable_phase + " 条（3515 期未获支持，仅观察）"
        : (snap.generated_at ? "快照 " + snap.generated_at : "未生成快照")) +
    // S2（V15）：长期效应档单独一行——语义是「长期平均显著，当前不一定生效」，
    // 绝不能并进「长期可用」里冒充当前有效。
    metricItem("长期效应（长期平均显著·当前不一定生效）",
      Number(counts.usable_persistent_effect || 0),
      "近 K 段方向一致 + 长期 Bonferroni 显著；与「长期可用（持续有效）」不是一回事") +
    '</div>';
  h += '<div class="rule-shadow">' + escHtml(r.note_zh || "") + '</div>';
  const rules = r.usable_rules || [];
  if (rules.length) {
    h += '<div class="scroll" style="margin-top:8px"><table class="battle-table"><thead><tr>' +
      '<th>通过前置注册的规律</th><th>维度</th><th>阶段状态</th><th>效应</th>' +
      '<th>校正p</th></tr></thead><tbody>';
    for (const x of rules) {
      const rg = x.regime || {};
      h += '<tr><td style="text-align:left">' + escHtml((x.statement || "").slice(0, 46)) + '</td>' +
        '<td style="text-align:left">' + escHtml((x.dim && x.dim.key) || "") + ' ' +
        escHtml(JSON.stringify((x.dim && x.dim.params) || {})) + '</td>' +
        '<td>' + escHtml(rg.label || rg.status || "") + '</td>' +
        '<td>' + (rg.effect === null || rg.effect === undefined ? "—" : Number(rg.effect).toFixed(4)) + '</td>' +
        '<td>' + (rg.p_adjusted === null || rg.p_adjusted === undefined ? "—" : Number(rg.p_adjusted).toFixed(4)) + '</td></tr>';
    }
    h += '</tbody></table></div>';
  } else {
    h += '<div class="track-note">当前 0 条规律满足「阶段持续有效 + A/B 验证 + 通过前置注册」。' +
      '这是<b>统计否定结论</b>，不是功能故障——真实彩票数据上本就该是 0 条。' +
      '可到「评估报告 → ⏳ 规律时效」看全部规律的阶段画像，或点「⟳ 规律闭环」重新挖掘验证。</div>';
  }
  h += '<div class="eval-actions" style="margin-top:8px">' +
    '<button onclick="refreshRuleSnapshot()">⟳ 刷新规律快照</button>' +
    '<button onclick="runClosedLoop()">🤖 跑一次 AI 规律闭环（挖掘→验证→使用）</button>' +
    '<span class="hint-inline">闭环约 1~8 分钟（含大模型三轮）；完成后本期规律状况会随之更新</span></div>';
  el.innerHTML = h;
}

async function refreshRuleSnapshot() {
  if (_rgBusy) { toast("正在计算中，请稍候"); return; }
  _rgBusy = true;
  try {
    const r = await api("/api/rules/refresh", { method: "POST" });
    toast("规律快照已刷新：" + JSON.stringify(r.counts || {}));
    await loadLastPrediction();
  } catch (e) { toast("刷新失败: " + e.message); }
  _rgBusy = false;
}

/**
 * 提交后台长任务并轮询进度。
 * 修复用户反馈「点按钮 15 分钟没有任何反应」：同步请求期间页面无任何反馈，
 * 现改为立即拿到 task_id，每 3 秒刷新「阶段 + 进度 + 已用时间」，并支持离开页面。
 */
async function runBackgroundTask(opts) {
  const {submitPath, renderArea, onDone, btnId, busyText, freeText} = opts;
  const area = $(renderArea);
  if (btnId) setBusy(btnId, busyText || "后台运行中…");
  const t0 = Date.now();
  const paint = (msg, pct) => {
    if (!area) return;
    const sec = Math.round((Date.now() - t0) / 1000);
    area.innerHTML = '<div class="track-note warn"><span class="spin">⏳</span> ' + escHtml(msg) +
      (pct !== null && pct !== undefined ? '（' + Math.round(pct * 100) + '%）' : '') +
      ' · 已用 ' + Math.floor(sec / 60) + '分' + (sec % 60) + '秒' +
      '<div class="rg-cell-sub">后台运行中，可以离开本页；完成后会自动刷新结果。</div></div>';
  };
  paint("提交任务…", 0);
  let tid = null;
  try {
    const r = await api(submitPath, {method: "POST"});
    if (!r.ok || !r.task_id) throw new Error(r.error || "提交失败");
    tid = r.task_id;
  } catch (e) {
    if (area) area.innerHTML = '<div class="track-note warn">⚠️ 提交失败：' + escHtml(e.message) + '</div>';
    if (btnId) setFree(btnId, freeText || "重试");
    return null;
  }
  toast("已提交，后台运行中…");
  return await new Promise(resolve => {
    const iv = setInterval(async () => {
      try {
        const t = await api("/api/tasks/" + tid);
        if (t.status === "completed") {
          clearInterval(iv);
          const res = t.result || {};
          if (btnId) setFree(btnId, freeText || "完成");
          if (area) area.innerHTML = "";
          if (onDone) onDone(res);
          resolve(res);
        } else if (t.status === "failed") {
          clearInterval(iv);
          if (btnId) setFree(btnId, freeText || "重试");
          if (area) area.innerHTML = '<div class="track-note warn">❌ 失败：' + escHtml(t.message || "未知错误") + '</div>';
          toast("后台任务失败：" + (t.message || ""));
          resolve(null);
        } else {
          paint(t.message || "运行中…", t.progress);
        }
      } catch (e) { /* 网络抖动：继续轮询 */ }
    }, 3000);
  });
}

async function runClosedLoop() {
  if (window._closedLoopRunning) { toast("闭环已在运行中，请等待完成"); return; }
  if (!window.confirm("AI 规律闭环：维度扫描 → 大模型三轮提规律/自我质疑/修订 → 逐条 as-of 回放验证 → 注册 → 刷新规律快照。\n\n预计 1~8 分钟，将在后台运行并显示每个阶段的进度（可离开本页）。\n\n确定开始？")) return;
  window._closedLoopRunning = true;
  const area = document.querySelector("#evalSub-regime:not(.hidden)") ? "#rgLabBody" : "#ruleBody";
  const res = await runBackgroundTask({
    submitPath: "/api/rules/closed-loop/async?rounds=3&use_llm=1",
    renderArea: area,
    btnId: $("#btnClosedLoop") ? "#btnClosedLoop" : null,
    busyText: "后台运行中…",
    freeText: "🤖 AI 规律闭环（挖掘→验证→使用）",
    onDone: async (r) => {
      const rt = r.runtime || {};
      {
        // V12-04：只用长期可用数，不把阶段可用混进来。
        // V14 文案：把"完成了一个闭环"改成"这次检验通过了多少条" —— 前者听起来像
        // "系统又变强了"，后者才是真实信息。0 条必须明说"为预期结果"，
        // 否则用户会以为闭环跑失败了。
        const _c = rt.counts || {};
        const _ph = Number(_c.usable_phase || 0);
        const _n = Number(_c.usable_long_term || 0);
        toast("检验完成：" + _n + " 条通过前置注册" +
              (_n === 0 ? "（0 条为预期结果）" : "") +
              (_ph ? "；另阶段可用 " + _ph + " 条，仅观察" : ""));
      }
      if (typeof renderLab === "function" && r.claims) renderLab(r);
      await loadLastPrediction();
      if (typeof loadRegimeOverview === "function") loadRegimeOverview(false);
    },
  });
  window._closedLoopRunning = false;
  return res;
}

// 载入最近一期预测（进首页时用，带生成时间与规律使用情况）
async function loadLastPrediction() {
  try {
    const r = await api("/api/predictions/last?_ts=" + Date.now());
    if (!r || !r.issue) return;
    currentIssue = r.issue;
    renderPredictions(r);
    const nActual = (r.tickets || []).length;
    const gen = r.generated_at || "未知时间";
    $("#predStatus").textContent = "📄 已加载上期保存的预测（" + r.issue + "，" + nActual +
      " 注，生成于 " + gen + "）";
    $("#predStatus").style.color = "var(--muted)";
  } catch (e) { /* 首页静默失败，不打扰 */ }
}

// ==================== 号码格式（全站唯一口径） ====================
// 用户反馈：复制的号码是 "01 06 12 21 24 31 + 12"，但「我的号码诊断」等处格式不一致。
// 现在所有「展示/复制/比对」的号码串都只从下面两个函数产出，避免再出现第二种写法
// （补零工具是文件顶部已有的 pad2 常量，不重复定义）。

/** 单注号码串：`01 06 12 21 24 31 + 12`（红球升序补零，蓝球补零）。 */
function formatTicket(reds, blue) {
  const r = (reds || []).map(Number).filter(x => !isNaN(x)).sort((a, b) => a - b);
  return r.map(pad2).join(" ") + " + " + pad2(Number(blue));
}

/** 只展示红球：`01 06 12 21 24 31`。 */
function formatReds(reds) {
  return (reds || []).map(Number).filter(x => !isNaN(x)).sort((a, b) => a - b).map(pad2).join(" ");
}

/** 用户输入解析：兼容「01 06 12」「01,06,12」「1 6 12」与全角空格/逗号。 */
function parseRedsInput(str) {
  return String(str || "")
    .replace(/[，、]/g, ",").replace(/\u3000/g, " ")
    .split(/[\s,]+/).map(s => parseInt(s, 10)).filter(x => !isNaN(x));
}

// 一句话结构小结：和值 · 三区 · 奇偶
function ticketSummary(t) {
  const reds = (t.reds || []).slice().sort((a, b) => a - b);
  const sum = reds.reduce((a, b) => a + b, 0);
  const z1 = reds.filter(x => x <= 11).length;
  const z2 = reds.filter(x => x >= 12 && x <= 22).length;
  const z3 = reds.length - z1 - z2;
  const odd = reds.filter(x => x % 2 === 1).length;
  return "和值" + sum + " · 三区" + z1 + "-" + z2 + "-" + z3 + " · 奇偶" + odd + ":" + (reds.length - odd);
}

// 星级渲染已于 2026-10-21 产品审计后**移除**（`starsFor()` 整体删除）。
// 原因见上面对 `ticketSummary()` 调用点的注释：任何带级图形都会被读成"更易中"。
// 若将来要恢复，必须先解决"如何让用户不把分级读成中奖概率"——目前无解，故不加。

// 折叠/展开单注
function toggleTicket(i) {
  const el = $("#tick-" + i);
  if (el) el.classList.toggle("open");
}

function copyTicket(i) {
  const t = window._predTickets?.[i];
  if (!t) return;
  const txt = formatTicket(t.reds, t.blue);
  navigator.clipboard.writeText(txt).then(() => toast("已复制第 " + (i + 1) + " 注")).catch(() => toast("复制失败"));
}

// 复制全部：把全部推荐注复制并记录到本机（供开奖回执）
function copyAllTickets() {
  const items = window._predTickets || [];
  if (!items.length) { toast("暂无预测"); return; }
  const lines = items.map(t => formatTicket(t.reds, t.blue));
  navigator.clipboard.writeText(lines.join("\n")).then(() => {
    toast("已复制全部 " + items.length + " 注");
    saveMyCopy(items);
  }).catch(() => toast("复制失败"));
}

// 跟我买：挑选「蓝球覆盖最优 + 组合最均衡」的子集（默认 5 注），一键复制
function followMyPick() {
  const items = window._predTickets || [];
  if (!items.length) { toast("暂无预测"); return; }
  const want = Math.min(5, items.length);
  // 贪心：优先选不重复蓝球，其次按匹配度（置信度）排序，保证蓝球覆盖面
  const chosen = [];
  const usedBlue = new Set();
  const rest = items.slice().sort((a, b) => (Number(b.confidence) || 0) - (Number(a.confidence) || 0));
  for (const t of rest) {
    if (chosen.length >= want) break;
    if (!usedBlue.has(t.blue)) { chosen.push(t); usedBlue.add(t.blue); }
  }
  // 蓝球不够去重时，补充剩余
  for (const t of rest) {
    if (chosen.length >= want) break;
    if (!chosen.includes(t)) chosen.push(t);
  }
  const lines = chosen.map(t => formatTicket(t.reds, t.blue));
  navigator.clipboard.writeText(lines.join("\n")).then(() => {
    toast("跟我买：已选 " + chosen.length + " 注（蓝球覆盖 " + usedBlue.size + " 个）并复制");
    saveMyCopy(chosen);
  }).catch(() => toast("复制失败"));
}

// 记录「我复制的注」，供开奖后回执
function saveMyCopy(tickets) {
  const issue = window._predIssue;
  const slim = (tickets || []).map(t => ({reds: t.reds, blue: t.blue}));
  try { localStorage.setItem("myCopy", JSON.stringify({issue: issue, tickets: slim, at: Date.now()})); } catch (_) {}
}

// 距离下一期开奖倒计时（双色球周二/周四/周日 21:15）
function nextDrawDate() {
  const now = new Date();
  const d = new Date(now);
  d.setHours(21, 15, 0, 0);
  for (let i = 0; i < 8; i++) {
    const dow = d.getDay();
    if (dow === 2 || dow === 4 || dow === 0) {
      if (d > now) return d;
    }
    d.setDate(d.getDate() + 1);
  }
  return d;
}
function renderCountdown() {
  const el = $("#predCountdown");
  if (!el) return;
  const target = nextDrawDate();
  const diff = target - new Date();
  if (diff <= 0) { el.textContent = "距开奖 已开奖/开奖中"; return; }
  const days = Math.floor(diff / 86400000);
  const hours = Math.floor((diff % 86400000) / 3600000);
  const mins = Math.floor((diff % 3600000) / 60000);
  const wd = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"][target.getDay()];
  const dateStr = (target.getMonth() + 1) + "月" + target.getDate() + "日 " + wd;
  el.textContent = "⏰ 距 " + dateStr + " 21:15 开奖 " + days + " 天 " + hours + " 时 " + mins + " 分";
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

// V10-04b：前端设置项默认值的版本号。后端默认值变了就必须升它，
// 否则老浏览器里持久化的旧值会"吃掉"新默认值（R1 P1-6 / R3 P0-4 的坑）。
const CFG_LLM_VERSION = "v10";

function recalcPredictConfig() {
  localStorage.setItem("cfgTickets", $("#cfgTickets")?.value || "10");
  localStorage.setItem("cfgLlm", $("#cfgLlm")?.checked ? "1" : "0");
  // 用户显式选择后即视为"已迁移"，不再被默认值回填覆盖
  try { localStorage.setItem("cfgLlmVersion", CFG_LLM_VERSION); } catch(e) {}
}

/**
 * 一次性默认值迁移（V10-04b）。
 * 背景：V10 起后端默认 ``llm_output_mode=off``（不再让 LLM 出号），但旧版本把
 * ``localStorage.cfgLlm`` 持久化成 "1"，页面一开就勾着「启用 LLM 推理」，
 * 于是 ``/api/predict?use_llm=true`` → ``llm_required=True`` → 503 或仍旧只出 LLM 注。
 * 做法：版本不符即删掉旧值 + 写新版本，让后端的 channels 回填真正生效。
 */
function migrateCfgDefaults() {
  let ver = null;
  try { ver = localStorage.getItem("cfgLlmVersion"); } catch(e) { return; }
  if (ver === CFG_LLM_VERSION) return;
  try {
    localStorage.removeItem("cfgLlm");
    localStorage.setItem("cfgLlmVersion", CFG_LLM_VERSION);
  } catch(e) {}
}

/**
 * 用后端通道快照回填「启用 LLM 推理」的默认勾选状态。
 * 判据唯一来自 ``/api/info`` 的 ``channels.llm_can_pick``（= resolve_channels()），
 * 前端不写死、也不自己推。用户显式选过（localStorage 有值）则尊重用户选择。
 */
async function applyChannelDefaults(info) {
  const box = $("#cfgLlm");
  if (!box) return;
  let stored = null;
  try { stored = localStorage.getItem("cfgLlm"); } catch(e) {}
  if (stored === "0" || stored === "1") return;      // 用户显式选择优先
  let ch = (info && info.channels) || null;
  if (!ch) {
    try { const r = await api("/api/info"); ch = (r && r.channels) || null; } catch(e) { ch = null; }
  }
  if (ch && typeof ch.llm_can_pick === "boolean") {
    box.checked = ch.llm_can_pick;
  }
  // 取不到快照时保持 HTML 默认（未勾选）= 后端默认 off，绝不静默打开 LLM
}

(function initConfig() {
  migrateCfgDefaults();
  const t = localStorage.getItem("cfgTickets");
  const l = localStorage.getItem("cfgLlm");
  if (t && $("#cfgTickets")) $("#cfgTickets").value = t;
  if (l !== null && $("#cfgLlm")) $("#cfgLlm").checked = l === "1";
})();

// ==================== 诊断 ====================

function runDiagnose() {
  // 用户反馈：红球、蓝球分成两个输入框不方便 —— 现在直接粘贴整注（与复制的格式一致）。
  // 兼容两种输入：`01 06 12 21 24 31 + 12`（带 +）与 `01 06 12 21 24 31 12`（纯空格）。
  const raw = ($("#diagTicket")?.value || "").trim();
  if (!raw) { toast("请输入整注号码，如 01 06 12 21 24 31 + 12"); return; }
  // 先按 + 切分：左侧是红球、右侧是蓝球（最贴合复制的 `01 06 ... + 12` 格式）
  const plus = raw.split("+");
  let blue = null;
  let red_raw = raw;
  if (plus.length >= 2) {
    red_raw = plus.slice(0, -1).join("+");
    blue = parseInt(plus[plus.length - 1].trim(), 10);
  }
  const reds = parseRedsInput(red_raw);
  if (blue === null && reds.length > 6) {
    blue = reds.pop();                                       // 无 + 时末位视为蓝球
  }
  if (reds.length !== 6 || new Set(reds).size !== 6 || blue < 1 || blue > 16) {
    toast("请按 01 06 12 21 24 31 + 12 格式输入（红球 6 个不重复 1-33，蓝球 1-16）"); return;
  }
  setBusy("#btnDiagnose", "诊断中…");
  $("#diagResult").innerHTML = '<div class="note">⏳ 诊断运行中，请稍候…</div>';
  api("/api/diagnose?reds=" + reds.join(",") + "&blue=" + blue)
    .then(data => {
      if (data.error) { toast(data.error); return; }
      const p = data.profile;
      window._diagTicket = {reds: p.reds || reds, blue: p.blue || blue};
      const meanFreq = p.freq ? p.freq.reduce((a,b) => a+b, 0) / 33 : 0;
      $("#diagResult").innerHTML =
        '<div class="diagnose-card">' +
          '<div class="diag-ticket">' + escHtml(formatTicket(p.reds || reds, p.blue || blue)) +
            '<button class="diag-copy" onclick="copyDiagnoseTicket()">📋 复制</button></div>' +
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
            (data.similar_blue_hit_rate > 0
              ? metricItem("蓝球命中率", pct(data.similar_blue_hit_rate),
                           "样本 " + (data.similar_count || 0) + " 注 · 非能力指标")
              : "") +
          '</div>' +
          (data.note ? '<div class="note">' + escHtml(data.note) + '</div>' : '') +
        '</div>';
    })
    .catch(e => toast("诊断失败: " + e.message))
    .finally(() => setFree("#btnDiagnose", "诊断"));
}

// 复制上一次诊断的号码（与票卡同一格式）
function copyDiagnoseTicket() {
  const t = window._diagTicket;
  if (!t) { toast("暂无诊断号码"); return; }
  navigator.clipboard.writeText(formatTicket(t.reds, t.blue))
    .then(() => toast("已复制 " + formatTicket(t.reds, t.blue)))
    .catch(() => toast("复制失败"));
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
  // V29：analysis 页已移除，`#winStats` 不再存在。
  // **关键**：`window._currentProbs` 必须**无条件**赋值 —— 它被出号/诊断链路消费
  // （`window._currentProbs` 是统计概率分布的全局缓存），
  // 若因为页面消失就提前 return，会把一个真实功能悄悄弄坏，而症状是
  // 「某处概率为空」—— 极难与本次改动联系起来。
  window._currentProbs = feat.red_probs ? {red_probs: feat.red_probs, blue_probs: feat.blue_probs || []} : {};
  const _ws = $("#winStats");
  if (!_ws) return;          // 页面已移除：只跳过渲染，不跳过上面的赋值
  _ws.textContent =
    "最新一期 " + feat.issue + " " + feat.date + "：红 " + feat.last_reds.join(" ") + " 蓝 " + feat.last_blue;
  renderWindow(feat);
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
  // V29：patterns 页已移除，`#patSummary` 不再存在。
  // 保留函数本体（`allPatterns` 仍被评估报告页消费），但必须提前返回 ——
  // 否则 `$("#patSummary").textContent` 会在 null 上抛 TypeError，
  // 把整条数据加载链（loadAll → 后续卡片）一起打断。
  const _ps = $("#patSummary");
  if (!_ps) return;
  _ps.textContent = "A:" + summary.A + " B:" + summary.B + " C:" + summary.C;
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
  // V29：patterns 页已移除，`#patCount` / `#patSummary` 不再存在。
  // 该函数仍可能被旧引用触发（如全局刷新），提前返回避免 null 上取属性。
  if (!$("#patCount")) return;
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
      ' <span style="color:var(--muted)">（样本 ' + a.issues + ' 期；对照随机基线期望红球 1.09/注、蓝球 6.25%。' +
      '样本不足 ' + METHOD_MIN_ISSUES + ' 期时，上述比率与基线之差属随机波动，不构成能力结论）</span></div>' +
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
  // V5：进入「实战战绩」子页时拉一次进化状态（快接口），重操作回放留给显式点击
  if (name === "evolution" && !_evoStatus) loadEvolutionStatus(false);
  // V6：进入「规律时效」子页时自动加载阶段画像（只读快接口，重算仍需显式点击）
  if (name === "regime" && !_rgOverview) loadRegimeOverview(false);
  // V10-10：进入「综合概览」时加载三卡 + 样本并排（首次才拉，之后靠用户点刷新）
  if (name === "overview" && !_v10CardsLoaded) { _v10CardsLoaded = true; loadV10Cards(); }
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
    if (r.by_method) renderMethodBreakdown(r.by_method);
    else await loadMethodBreakdown();
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
    // V4：Δ 不构成能力证据——样本期内与随机基线的差值属波动范围
    return '<div class="metric"><div class="k">' + x.k + "（系统）</div>" +
      '<div class="v">' + fmtV(x.sys) + "</div>" +
      '<span class="sub">随机基线 ' + fmtV(x.rnd) + ' · Δ<span style="color:var(--muted)">' + dTxt + "</span></span></div>";
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
  // V10-04b：eval_results 是**组合口径**（该期所有方法所有注），且命中数已保留浮点
  // （旧实现 int() 截断把 1.5 抹成 1）。方法对比请看上方"方法战绩（分列）"。
  area.innerHTML = view.length
    ? '<div class="note" style="margin-bottom:6px">在线对照记录（组合口径：该期全部方法所有注；最近 ' +
      view.length + " 期，新→旧）</div>" +
      '<div class="scroll"><table>' +
      '<thead><tr><th>期号</th><th>红球命中(均值)</th><th>蓝球命中率</th><th>奖金</th><th>注数</th></tr></thead>' +
      '<tbody>' + view.map(r => "<tr><td class='mono'>" + r.issue + "</td><td>" + fmt(r.red_hits, 2) +
        "</td><td>" + (r.blue_hit == null ? "-" : (Number(r.blue_hit) * 100).toFixed(1) + "%") +
        "</td><td>¥" + Number(r.reward || 0).toFixed(0) +
        "</td><td>" + (r.ticket_count ?? "-") + "</td></tr>").join("") +
      '</tbody></table></div>'
    : '<div class="note">暂无在线对照记录（开奖后可点「在线对照」）。</div>';
}

// V10-04b（R1 P1-4）：方法战绩**按族分列**渲染。
// 各族注数与口径不同，绝不显示"合计/加权平均"——那正是此前把 LLM 差表现
// 平均掉的根因。数据源：/api/eval/by-method（= evaluate.method_breakdown）。
// 族名走后端 family_names（evaluate.FAMILY_NAME_ZH 的唯一出处），前端只留兜底顺序
const METHOD_FAMILY_FALLBACK = {llm:"LLM 推理", stat:"统计基线", blend:"融合 blend",
  ml:"ML 模型", uniform:"均匀随机", other:"其它"};
const METHOD_FAMILY_ORDER = ["llm", "stat", "blend", "ml", "uniform", "other"];

// 0 是有效值（0% 命中率与"缺数据"必须区分），故不复用 pct()
const _rate = (v) => (v == null || isNaN(Number(v)) ? "-" : (Number(v) * 100).toFixed(1) + "%");

function renderMethodBreakdown(report) {
  const area = $("#methodBreakdownArea");
  if (!area) return;
  const fams = (report && report.families) || {};
  const keys = METHOD_FAMILY_ORDER.filter(k => fams[k]);
  if (!keys.length) {
    area.innerHTML = '<div class="note">方法战绩（分列）：暂无 eval_details 数据（开奖并对照后生成）。</div>';
    return;
  }
  area.innerHTML =
    '<div class="note" style="margin-bottom:6px">方法战绩（按族分列，共 ' +
      (report.n_issues || 0) + " 期 / " + (report.n_tickets || 0) +
      ' 注）——各族各算各的，<b>不得相加或取加权平均</b></div>' +
    '<div class="scroll"><table>' +
    '<thead><tr><th>通道</th><th>期数</th><th>注数</th><th>红球平均命中</th>' +
    '<th>蓝球命中率</th><th>≥五等奖率</th><th>ROI</th></tr></thead><tbody>' +
    keys.map(k => {
      const f = fams[k] || {};
      const famName = ((report && report.family_names) || {})[k] || METHOD_FAMILY_FALLBACK[k] || k;
      return "<tr><td>" + escHtml(famName) +
        "</td><td>" + (f.n_issues ?? "-") + "</td><td>" + (f.n_tickets ?? "-") +
        "</td><td>" + fmt(f.red_hits_mean, 2) + "</td><td>" + _rate(f.blue_hit_rate) +
        "</td><td>" + _rate(f.prize_rate_ge5) + "</td><td>" + _rate(f.roi) + "</td></tr>";
    }).join("") +
    "</tbody></table></div>" +
    '<div class="note">' + escHtml((report && report.note) || "") + "</div>";
}

async function loadMethodBreakdown() {
  try { renderMethodBreakdown(await api("/api/eval/by-method?limit_issues=200")); }
  catch(e) {
    const area = $("#methodBreakdownArea");
    if (area) area.innerHTML = '<div class="note">方法战绩加载失败：' + escHtml(e.message) + "</div>";
  }
}

// ==================== V10-10：三卡 + 通道徽标 + 样本并排 ====================
// 三卡是否已加载（首次进入「综合概览」才拉取，避免开局打满 4 个重请求）
let _v10CardsLoaded = false;

// 数据来源（每个卡只有唯一出口，前后端不各写一份口径）：
//   🧾 票单分布  → GET /api/portfolio/distribution → engine.ticket_distribution()
//   📋 前置注册  → GET /api/registry/status        → hypothesis.load_registry() + 快照 prereg
//   🤖 ML 诊断   → GET /api/ml/diagnosis           → ml_model.diagnose_ml_channels()
//   📐 样本并排  → GET /api/samples/side-by-side   → replay.samples_side_by_side()
//   通道口径    → GET /api/info                    → methods.resolve_channels()

const FAMILY_LABEL = {stat:"统计基线", blend:"融合 blend", llm:"LLM 推理",
  ml:"ML 模型", uniform:"均匀随机"};

/** 小数格式化：null/NaN → "-"，其余保留 d 位。 */
function _f(v, d) {
  if (v === null || v === undefined || v === "" || isNaN(Number(v))) return "-";
  return Number(v).toFixed(d === undefined ? 4 : d);
}
/** 概率 → 百分比（0 必须显示成 0.0%，不能像 pct() 那样被当假值吞掉）。 */
function _p2(v, d) {
  if (v === null || v === undefined || isNaN(Number(v))) return "-";
  return (Number(v) * 100).toFixed(d === undefined ? 2 : d) + "%";
}

// ----- 通道口径徽标（本期推荐页） -----
function renderChannelBadges(ch) {
  const el = $("#channelBadges");
  if (!el) return;
  if (!ch) { el.innerHTML = '<span class="ch-badge muted">通道口径不可用</span>'; return; }
  const fams = ch.families || {};
  const order = ["stat", "blend", "ml", "llm", "uniform"];
  let h = order.map(k => {
    const on = !!fams[k];
    return '<span class="ch-badge ' + (on ? "on" : "off") + '" title="方法族 ' + k +
      (on ? "：已启用" : "：已关闭") + '">' + (on ? "✓ " : "✕ ") +
      escHtml(FAMILY_LABEL[k] || k) + "</span>";
  }).join("");
  const canPick = !!ch.llm_can_pick;
  h += '<span class="ch-badge ' + (canPick ? "pick" : "nopick") + '" title="' +
    (canPick ? "LLM 可以出号（llm_output_mode=" + escHtml(ch.llm_output_mode || "") + "）"
             : "LLM 不参与出号（llm_output_mode=" + escHtml(ch.llm_output_mode || "off") + "）") +
    '">' + (canPick ? "🤖 LLM 出号：是" : "🤖 LLM 出号：否") + "</span>";
  h += '<span class="ch-badge muted" title="生效开关来源 / 运行模式 / 快照指纹">' +
    escHtml((ch.source || "?") + " · " + (ch.mode || "?") + " · " + (ch.llm_output_mode || "?")) +
    "</span>";
  el.innerHTML = h;
}

// ----- 卡 1：票单分布 -----
function renderDistribution(data) {
  const body = $("#distBody"), tag = $("#distTag");
  if (!body) return;
  if (!data || data.ok === false) {
    if (tag) tag.textContent = "不可用";
    body.innerHTML = '<div class="note">票单分布不可用：' +
      escHtml((data && data.error) || "接口未返回") + "</div>";
    return;
  }
  const d = data.distribution || {};
  const par = d.p_at_least_r || {};
  const honesty = data.honesty || {};
  if (tag) tag.textContent = (data.source === "prediction"
      ? "当前推荐 " + (d.n ?? 0) + " 注" + (data.issue ? " · " + data.issue : "")
      : "示例票 " + (d.n ?? 0) + " 注") +
      (data.exact ? " · 精确枚举" : " · iid 近似");
  const metric = (k, v, sub) =>
    '<div class="v10-metric"><div class="k">' + k + '</div><div class="v">' + v +
    "</div>" + (sub ? '<div class="sub">' + sub + "</div>" : "") + "</div>";
  body.innerHTML =
    // 诚实边界提示条（规格要求：不可省略）
    '<div class="v10-honesty"><b>改变分布不改变 EV：</b>' +
      escHtml(honesty.note_zh || "只改变波动结构，不改变期望值。") + "</div>" +
    '<div class="v10-metrics">' +
      metric("期望命中 E", _f(d.expected_hits, 3), "= n·12/11，与选号无关") +
      metric("P(≥4 红)", _p2(par["4"]), d.p_at_least_r_method === "exact_enum"
        ? "精确枚举" : "近似：" + _p2((d.p_at_least_r_iid || {})["4"])) +
      metric("P(≥5 红)", _p2(par["5"]), d.p_at_least_r_method === "exact_enum"
        ? "精确枚举" : "近似：" + _p2((d.p_at_least_r_iid || {})["5"])) +
      metric("冗余度 Ā", _f(d.redundancy, 4),
        "中性值 " + _f(d.redundancy_neutral, 4) + "（12/11）· 上限 " + _f(d.redundancy_max, 3)) +
      metric("方差比", _f(d.var_ratio, 4), "var_T/σ² 下界 0") +
      metric("P(≥1 蓝)", _p2(d.p_any_blue), "覆盖 " + (d.blue_coverage_count ?? "-") + "/16") +
    "</div>" +
    '<div class="v10-metrics">' +
      metric("仅六等奖返奖比", _f(d.ev_ratio_sixth, 5), "= 0.15625，与注数无关") +
      metric("整注返奖比（全奖级）", _f(d.ev_ratio_full, 4),
        "严格口径 " + _f(d.ev_ratio_full_strict, 4)) +
    "</div>" +
    // 奖金假设必须与 ev_ratio_full 一起展示
    '<div class="v10-honesty"><b>奖金假设（浮动奖只能给假设常量）：</b>' +
      escHtml(d.prize_assumption || "未提供") +
      (d.ev_ratio_full_note ? "；" + escHtml(d.ev_ratio_full_note) : "") + "</div>" +
    '<div class="note">' + escHtml(d.var_ratio_note || data.note_zh || "") + "</div>";
}

async function loadDistribution() {
  try { renderDistribution(await api("/api/portfolio/distribution")); }
  catch(e) {
    const body = $("#distBody"), tag = $("#distTag");
    if (tag) tag.textContent = "不可用";
    if (body) body.innerHTML = '<div class="note">票单分布加载失败：' + escHtml(e.message) + "</div>";
  }
}

// ============ V14-B2/B3：蓝球去重覆盖 + 大众偏好度 ============
//
// 数据来源（每个数字只有一个出口，前后端不各写一份口径）：
//   🔵 蓝球覆盖  → GET /api/portfolio/blue-coverage → blue_coverage.optimize()
//   👥 大众偏好  → GET /api/portfolio/popularity    → popularity.describe()
//
// 两个端点的纪律都在后端强制、前端只负责**完整呈现**：
//
//   1) 披露不可拆。`disclosure_lines` 是后端一次性给出的真相全集，前端**逐行全渲染**，
//      不做截断、不做"精简版"。尤其禁止只展示「随机重复 47.55% → 覆盖 62.50%」——
//      少了「成本 / 净亏 / 返奖比恒定」这三句，读者会把它理解成"收益率变高了"，
//      而那正是这套模型最容易被误读的地方。
//
//   2) 偏好度只给档位。`tier` 是三档标称（TIER_LABELS），**不返回也不显示任何注数**。
//      实测效应量 R² ≈ 0.008，只能排序不能定值；渲染层不得自行做注数换算——
//      哪怕只是"顺手算一下"也不许，那会把一个只够排序的信号包装成可执行数字。
//
//   3) 降级容错。两卡各自独立 try/catch，任一端点 401/503/超时只让**本卡**显示
//      "不可用"，绝不往外抛——`loadV10Cards()` 是 `Promise.all` 并发入口，
//      一次未捕获的 reject 会连累整页首屏。

/** 每注价格（与后端 blue_coverage.TICKET_PRICE 同值，仅用于文案，不参与计算）。 */
const TICKET_PRICE_YUAN = 2;

/** 百分数：保留 d 位。0 必须显示成 0.00%，不能被当假值吞掉。 */
function _pct(v, d) {
  if (v === null || v === undefined || isNaN(Number(v))) return "-";
  return (Number(v) * 100).toFixed(d === undefined ? 2 : d) + "%";
}

/** 元：保留 d 位，带千分位-free 的固定写法（金额不大，避免本地化差异）。 */
function _yuan(v, d) {
  if (v === null || v === undefined || isNaN(Number(v))) return "-";
  return Number(v).toFixed(d === undefined ? 2 : d);
}

/** 从当前推荐里取「第一个注单」的红球，作为偏好度查询对象。
 *  取不到时返回 null，由调用方走默认形态——**不猜、不编**。 */
function _firstTicketReds() {
  const src = (typeof STATE !== "undefined" && STATE && STATE.lastPrediction) || null;
  const tickets = (src && (src.tickets || src.bets)) || null;
  if (!Array.isArray(tickets) || !tickets.length) return null;
  const t = tickets[0] || {};
  const reds = t.reds || t.red || t.front || null;
  if (!Array.isArray(reds)) return null;
  const nums = reds.map(Number).filter(x => Number.isInteger(x) && x >= 1 && x <= 33);
  return nums.length === 6 && new Set(nums).size === 6 ? nums.slice().sort((a, b) => a - b) : null;
}

/** 降级文案：把状态码翻译成用户能懂的一句话，同时保留原始 message 供排查。 */
function _degradeNote(what, e) {
  const st = e && e.status;
  const pl = (e && e.payload) || {};          // 后端 JSON 体（api() 已挂在 err.payload 上）
  // ┌─ 通用规则（2026-10-21 Product 定，适用于全站）────────────────────────┐
  // │ 凡「粗/细两级字段**并存**」，不许用 `||` 读，必须 filter(Boolean) 后全查。│
  // └──────────────────────────────────────────────────────────────────┘
  //
  // 为什么这条是**通用**的，而不只是这一处的特例：
  //   `a || b` 的语义是「a 缺失时用 b」。但当后端把 `reason`（粗分类）与
  //   `unfitted_reason`（细分原因）**同时**给出时，两者是层级关系、不是备选关系。
  //   用 `||` 读，只要 a 有值（哪怕是粗的），更精确的 b 就永远读不到 ——
  //   这不是「写错了一个值」，是**回退链在语义上不成立**。
  //
  // 实测（2026-10-21）：预热关闭时后端返回
  //   reason=model_not_ready + unfitted_reason=warming_disabled
  // 若写 `pl.reason || pl.unfitted_reason`，前端会把「部署配置关闭」
  // （永久、该找运维）误报成「临时未就绪」（该刷新）。
  //
  // 守卫：tests/frontend/v14_portfolio_render_probe.py 的 [11] 一段，
  // 按**形态**拦（正则匹配 `pl.reason || pl.unfitted_reason`），不只拦这一行。
  const reasons = [pl.reason, pl.unfitted_reason].filter(Boolean);
  const has = (...names) => names.some(n => reasons.indexOf(n) >= 0);
  const retryAfter = pl.retry_after;

  // 503 必须再细分，否则用户不知道该等还是该找运维。
  // 判据用后端回的 `reason`，不用 HTTP 状态码 —— 三种 503 的"用户该做什么"完全不同：
  //   warming           → 刷新就好（一次性，几秒）
  //   insufficient_data → 等更多期开奖（短期无解，但功能没坏）
  //   model_not_ready   → 看日志/找运维
  // 只有当 `retry_after` 是**具体秒数**时才鼓励"稍后重试"；
  // `retry_after === null` 表示后端自己也认为"等不到"，此时任何"稍后重试"
  // 都是在让用户等一件不会发生的事。
  if (st === 503) {
    let hint;
    if (has("model_warming", "warming")) {
      const n = Number(retryAfter);
      hint = Number.isFinite(n) && n > 0
        ? "模型正在拟合（约 " + n + " 秒），稍后刷新即可"
        : "模型正在拟合，稍后刷新即可";
    } else if (has("insufficient_data")) {
      hint = "历史样本不足（需 ≥300 期含销量与头奖注数的开奖），短期无法给出档位";
    } else if (has("warmup_disabled", "warming_disabled")) {
      hint = "当前部署关闭了后台预热，本卡在此部署下不可用（无重试）";
    } else if (has("retry_exhausted")) {
      // 预热重试循环已跑完（后端 `for attempt in range(1, 11)` 退出）。
      // 与 warming_disabled 同属「**本进程内不会再就绪**」，但原因不同：
      // 前者是配置关闭、后者是重试失败。文案必须同样明确「别再等」，
      // 否则用户会一直刷新一件永远不回来的事 —— 这是后端注释点名的
      // 「第三种假承诺」。
      hint = "模型多次重试仍失败，本进程内不会再自动恢复（需管理员查看日志或重启服务）";
    } else if (has("fit_failed")) {
      hint = "模型拟合失败（非数据不足），需管理员查看日志";
    } else {
      // 兜底：后端只给了泛化的 model_not_ready，且**带**了具体 retry_after 才提重试
      const n = Number(retryAfter);
      hint = (retryAfter !== null && retryAfter !== undefined &&
              Number.isFinite(n) && n > 0)
        ? "服务暂不可用（模型未就绪），约 " + n + " 秒后可再试"
        : "服务暂不可用（模型未就绪），且该状态下不会自动恢复";
    }
    return "【" + what + "不可用】" + hint + "：" + String((e && e.message) || e);
  }

  let hint = "接口未返回";
  if (st === 401) hint = "需要登录后查看（本卡不阻塞其余内容）";
  else if (st === 403) hint = "无权访问本卡";
  else if (st === 400) hint = "参数不合法";
  else if (st === 404) hint = "接口不存在（后端版本可能较旧）";
  else if (!st) hint = "网络超时或连接失败";
  return "【" + what + "不可用】" + hint + "：" + String((e && e.message) || e);
}

/** 披露行的最小富文本渲染：后端的 disclosure 文本用 `**...**` 标出必须被看见的限定
 *  （如「与注数无关」「长期仍为负期望」）。先整体转义再只把 `**x**` 转成 <strong>，
 *  这样既保住了 XSS 防线，又不会在页面上漏出裸星号。
 *  **不**实现完整 Markdown —— 只认这一种标记，多一种就多一条被滥用的路径。 */
function _renderDisclosureLine(text) {
  const safe = escHtml(String(text === null || text === undefined ? "" : text));
  return safe.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

// ----- 卡 7：蓝球去重覆盖 -----
function renderBlueCoverage(d) {
  const body = $("#blueCovBody"), tag = $("#blueCovTag");
  if (!body) return;
  if (!d || d.ok === false) {
    if (tag) tag.textContent = "不可用";
    body.innerHTML = '<div class="note">蓝球覆盖计划不可用：' +
      escHtml((d && d.error) || "接口未返回") + "</div>";
    return;
  }
  const n = d.n_tickets ?? 0;
  const plan = Array.isArray(d.plan) ? d.plan : [];
  const covered = d.blue_coverage_count ?? 0;
  if (tag) tag.textContent = n + " 注 · 覆盖 " + covered + "/16 个蓝球";

  const metric = (k, v, sub) =>
    '<div class="v10-metric"><div class="k">' + k + '</div><div class="v">' + v +
    "</div>" + (sub ? '<div class="sub">' + sub + "</div>" : "") + "</div>";

  // ⚠ 下面四项**必须同屏**出现在同一个 .v10-metrics 容器里。
  // 只看前两项（概率对比）是本卡最容易造成的误读，所以概率、成本、净亏、返奖比
  // 放在同一行，物理上不给出「只截一半」的结构。
  const cost = d.cost_yuan ?? (n * TICKET_PRICE_YUAN);
  const evSixth = d.ev_sixth_yuan;
  const netSixth = (d.net_sixth_yuan !== undefined && d.net_sixth_yuan !== null)
    ? d.net_sixth_yuan
    : (Number(evSixth) - Number(cost));

  const lines = Array.isArray(d.disclosure_lines) ? d.disclosure_lines : [];
  // 部分口径降级项（D-17）。正常路径下后端返回空数组 ⇒ 这里什么都不渲染，
  // 满足「没异常时不误报」。
  //
  // ⚠ `full_ev=false` 时后端**也**返回空数组 —— 因为那是**用户主动选择**
  // 不要整注口径，不是降级。所以前端不需要另外判断 `full_ev`：
  // 「有没有降级」完全由 `degraded_items` 是否为空决定，单一判据。
  // （若哪天前端加了 `full_ev=false` 开关，也不得凭空造出降级提示 ——
  //   用户的主动选择不该被当成系统故障。）
  const degraded = Array.isArray(d.degraded_items) ? d.degraded_items : [];
  const disclosureHtml = lines.length
    ? lines.map((ln, i) =>
        '<div class="v14-line" data-line="' + (i + 1) + '">' +
          _renderDisclosureLine(ln) + "</div>").join("")
    // 后端契约保证 disclosure_lines 非空；万一缺失，宁可显示占位也不能静默省略
    : '<div class="note">披露文本缺失（后端未返回 disclosure_lines），本卡结论不可单独使用。</div>';

  const flags = d.honesty_flags || {};
  const flagRow = (key, label, val) => {
    if (val === undefined) return "";
    const yes = val === true;
    return '<span class="v14-flag ' + (yes ? "yes" : "no") + '" data-flag="' + key + '">' +
      escHtml(label) + "：" + (yes ? "是" : "否") + "</span>";
  };

  body.innerHTML =
    '<div class="v10-metrics">' +
      metric("P(≥1 次中蓝) · 去重覆盖", _pct(d.p_any_blue_covered),
        "计划：" + escHtml(plan.join(" ")) ) +
      metric("P(≥1 次中蓝) · 随机重复", _pct(d.p_any_blue_random_repeat),
        "同注数、不去重的对照值") +
      metric("成本", "¥" + _yuan(cost, 0), n + " 注 × ¥" + TICKET_PRICE_YUAN) +
      metric("六等奖期望回收", "¥" + _yuan(evSixth, 2),
        "净亏 ¥" + _yuan(netSixth, 3)) +
      // 六等奖返奖比恒为 15.625%（= (n/16 × ¥5) / (2n) = 5/32），**与注数无关**。
      // 2026-10-21 Lead 复现裁定：恒等式在所有 n 都成立；n > 16 时变的是
      // 「至少中一次」的概率（封顶 100%），不是返奖比。故此处注解**固定写恒定**，
      // 饱和信息由下面独立的饱和提示块承载，两者不混在一行里。
      metric("六等奖返奖比", _pct(d.ev_ratio_sixth, 3),
        "恒定 = 0.15625，与注数无关") +
    "</div>" +
    (d.ev_ratio_full !== undefined
      ? '<div class="v10-metrics">' +
          metric("整注返奖比（一~六等全计）", _pct(d.ev_ratio_full, 2),
            "含浮动奖估算 · 长期仍为负期望") +
          metric("整注期望回收", "¥" + _yuan(d.ev_full_yuan, 2), "按上项返奖比 × 成本") +
        "</div>"
      : "") +
    // ---- 部分口径降级（D-17，2026-10-21）----
    //
    // 后端在「整注返奖比算不出来」时不会静默：它把这些写进 `degraded_items`
    // （id + reason + how_to_restore），六等奖口径仍然照常返回。
    //
    // 前端不渲染它 = **后端已经不再静默，前端仍在静默**，
    // 用户会以为「整注返奖比」这个数本来就不存在，而不是「本次算不出来」。
    // 这与 D-17 要修的「静默 except」是同一类病，只是发生在传递链的另一层。
    //
    // 措辞分层（与后端一致）：
    //   · 明说「部分口径不可用」，**不是**「整卡失败」—— 六等奖那组指标是好的；
    //   · `how_to_restore` 标注为「给运维的」，避免让普通用户去读函数名。
    (degraded.length
      ? '<div class="v14-degraded" id="blueCovDegraded">' +
        '<div class="v14-degraded-head"><b>⚠ 部分口径降级</b>' +
        '<span class="v14-degraded-sub">下列数据本次不可用，其余指标不受影响</span></div>' +
        '<ul class="v14-degraded-list">' +
          degraded.map(it =>
            '<li data-degraded-id="' + escHtml(String(it.id || "")) + '">' +
            escHtml(String(it.reason || it.id || "未提供原因")) + "</li>").join("") +
        "</ul>" +
        (degraded.some(it => it.how_to_restore)
          ? '<div class="v14-degraded-fix"><b>恢复方法（给运维）：</b>' +
            degraded.filter(it => it.how_to_restore)
              .map(it => escHtml(String(it.how_to_restore))).join("；") +
            "</div>"
          : "") +
        "</div>"
      : "") +
    // n > 16：蓝球池（16 个）被占满。**概率**已封顶，「至少中一次」不会再提高；
    // 再加注只增加"重复中奖"的次数，成本继续线性增加。
    // 注意别把这条写成"返奖比被稀释"—— 返奖比恒为 15.625%，恒等式不随 n 变。
    (d.coverage_saturated
      ? '<div class="v10-honesty" id="blueCovSaturated"><b>概率已饱和：</b>' +
        "蓝球只有 16 个，买超过 16 注时，「至少中一次蓝球」的概率已达 100%，" +
        "再加注不会提高它，只会增加「重复中奖」的次数，而成本继续线性增加。" +
        "返奖比仍恒为 15.625%（六等奖口径）。" +
        "</div>"
      : "") +
    '<div class="v10-honesty" id="blueCovDisclosure">' +
      "<b>披露（全文，不可拆开读）：</b>" +
      '<div class="v14-disclosure-lines">' + disclosureHtml + "</div>" +
      '<div class="v14-flags">' +
        flagRow("increases_hit_probability", "提高中奖概率", flags.increases_hit_probability) +
        flagRow("increases_expected_return", "提高期望收益", flags.increases_expected_return) +
        flagRow("increases_prize_frequency", "提高中奖频率", flags.increases_prize_frequency) +
      "</div>" +
    "</div>";
}


async function loadBlueCoverage() {
  const body = $("#blueCovBody"), tag = $("#blueCovTag");
  const raw = $("#blueCovN")?.value;
  let n = parseInt(raw, 10);
  if (!Number.isInteger(n)) n = 10;
  n = Math.min(100, Math.max(1, n));
  const inp = $("#blueCovN");
  if (inp && String(inp.value) !== String(n)) inp.value = String(n);
  if (tag) tag.textContent = "计算中…";
  try {
    const d = await api("/api/portfolio/blue-coverage?n_tickets=" + n);
    renderBlueCoverage(d);
  } catch (e) {
    // 降级：只让本卡失效，不往外抛（见本段开头第 3 条）
    if (tag) tag.textContent = "不可用";
    if (body) {
      body.innerHTML = '<div class="note">' + escHtml(_degradeNote("蓝球覆盖计划", e)) + "</div>";
    }
  }
}

// ----- 卡 8：大众偏好度（只给档位，不给注数） -----
// 档位 → 样式。取值必须与后端 `popularity.tier_of()` 严格对齐：low / mid / high。
// 早期误把 mid 写成 medium，会静默落到默认样式 —— 档位看着"显示出来了"，语义却丢了。
const POP_TIER_CLASS = { low: "cool", mid: "common", high: "hot" };

function renderPopularity(d) {
  const body = $("#popBody"), tag = $("#popTag");
  if (!body) return;
  if (!d || d.ok === false) {
    if (tag) tag.textContent = "不可用";
    body.innerHTML = '<div class="note">大众偏好度不可用：' +
      escHtml((d && d.error) || "接口未返回（模型可能尚未拟合）") + "</div>";
    return;
  }
  const tier = String(d.tier || "");
  const label = d.tier_label || "未标注";
  if (tag) tag.textContent = label;
  body.innerHTML =
    '<div class="v14-tier-row">' +
      '<span class="v14-tier ' + escHtml(POP_TIER_CLASS[tier] || "common") +
        '" data-tier="' + escHtml(tier) + '">' + escHtml(label) + "</span>" +
      (Array.isArray(d.reds) && d.reds.length
        ? '<span class="v14-reds mono">形态：' + escHtml(d.reds.join(" ")) + "</span>"
        : "") +
    "</div>" +
    (Array.isArray(d.reasons) && d.reasons.length
      ? '<ul class="v14-reasons">' +
          d.reasons.map(r => "<li>" + escHtml(String(r)) + "</li>").join("") +
        "</ul>"
      : "") +
    (d.caveat
      ? '<div class="v10-honesty"><b>档位含义：</b>' + escHtml(String(d.caveat)) + "</div>"
      : "") +
    '<div class="v10-honesty"><b>纪律说明：</b>' + escHtml(String(d.honesty || "未提供")) +
      (d.r2_increment !== undefined
        ? "<div class='sub v14-r2'>形态对头奖注数的增量解释力 R² ≈ " +
          escHtml(String(d.r2_increment)) + "（只够排序，不足以定值）</div>"
        : "") +
    "</div>";
}

async function loadPopularity() {
  const body = $("#popBody"), tag = $("#popTag");
  const subj = $("#popSubject");
  const reds = _firstTicketReds();
  const query = reds || [1, 2, 3, 4, 5, 6];   // 无推荐时用默认形态，仅作演示，不冒充当前推荐
  if (subj) {
    subj.textContent = reds
      ? "取当前推荐首注：" + reds.join(" ")
      : "暂无当前推荐，使用示例形态 1 2 3 4 5 6";
  }
  if (tag) tag.textContent = "查询中…";
  try {
    renderPopularity(await api(
      "/api/portfolio/popularity?reds=" + encodeURIComponent(query.join(","))));
  } catch (e) {
    // 降级：只让本卡失效，不往外抛
    if (tag) tag.textContent = "不可用";
    if (body) {
      body.innerHTML = '<div class="note">' + escHtml(_degradeNote("大众偏好度", e)) + "</div>";
    }
  }
}

// ----- 卡 2：前置注册 -----
function renderRegistry(rep) {
  const body = $("#preregBody"), tag = $("#preregTag");
  if (!body) return;
  if (!rep || rep.ok === false) {
    if (tag) tag.textContent = "不可用";
    body.innerHTML = '<div class="note">前置注册状态不可用：' +
      escHtml((rep && rep.error) || "接口未返回") + "</div>";
    return;
  }
  if (tag) tag.textContent = "已登记 " + (rep.n_total || 0) + " 条 · 累计检验 " +
    (rep.n_tested_cumulative || 0) + " 次";
  const cnt = (k, label, cls) =>
    '<div class="v10-metric"><div class="k">' + label + '</div><div class="v">' +
    (rep[k] ?? 0) + "</div>" + (cls ? '<div class="sub">' + cls + "</div>" : "") + "</div>";
  const items = rep.items || [];
  const rows = items.length
    ? items.map(it => {
        const st = it.prereg_status || "-";
        const cls = st === "passed" ? "gradeA" : (st === "failed" || st === "invalidated")
          ? "gradeC" : "";
        return "<tr><td class='mono'>" + escHtml(String(it.exp_id || "-")) + "</td>" +
          "<td>" + escHtml(String(it.source || "-")) + "</td>" +
          "<td><span class='badge " + cls + "'>" + escHtml(st) + "</span></td>" +
          "<td>" + _f(it.p_cumulative, 5) + "</td>" +
          "<td>" + (it.n_tested_cumulative ?? "-") + "</td>" +
          "<td>" + (it.divergence ? "⚠ 分叉" : "—") + "</td>" +
          "<td class='v10-reason'>" + escHtml(it.reason || "-") + "</td></tr>";
      }).join("")
    : "<tr><td colspan='7' class='v10-reason'>暂无已登记实验（前置注册由每期调度器写入）。</td></tr>";
  const snap = rep.snapshot || {};
  body.innerHTML =
    '<div class="v10-metrics">' +
      cnt("registered", "已登记") + cnt("passed", "通过 passed") +
      cnt("provisional", "暂定 provisional", "批内显著但累计不显著") +
      cnt("failed", "失败 failed") + cnt("invalidated", "失效") +
    "</div>" +
    '<div class="v10-metrics">' +
      '<div class="v10-metric"><div class="k">累计检验次数 n_tested_cumulative</div>' +
        '<div class="v">' + (rep.n_tested_cumulative ?? 0) + '</div>' +
        '<div class="sub">Bonferroni 校正分母（R1）· 上期判定 ' +
        escHtml(String(rep.last_issue || "-")) + '</div></div>' +
      '<div class="v10-metric"><div class="k">批规模约束</div><div class="v">' +
        (rep.min_batch_size ?? "-") + " ~ " + (rep.max_batch_size ?? "-") + '</div>' +
        '<div class="sub">R2 每批最少条数 / R5 上限</div></div>' +
    "</div>" +
    '<div class="scroll"><table class="v10-table"><thead><tr>' +
      "<th>exp_id</th><th>来源</th><th>prereg 状态</th><th>p_cumulative</th>" +
      "<th>n_tested_cum</th><th>口径分叉</th><th>理由</th>" +
    "</tr></thead><tbody>" + rows + "</tbody></table></div>" +
    '<div class="note">' + escHtml(rep.note_zh || "") + "</div>" +
    (snap.available
      ? '<div class="note">规律快照基准 ' + escHtml(String(snap.base_issue || "-")) +
        " · 快照计数 = " +
        JSON.stringify((snap.counts || {}).by_layer || {
          usable_long_term: (snap.counts || {}).usable_long_term,
          usable_phase: (snap.counts || {}).usable_phase,
          monitor: (snap.counts || {}).monitor,
          rejected: (snap.counts || {}).rejected,
        }) + (snap.stale ? " · ⚠ 快照已过期" : "") + "</div>"
      : '<div class="note">规律快照尚未生成（前端只看已登记实验）。</div>');
}

async function loadRegistryStatus() {
  try { renderRegistry(await api("/api/registry/status")); }
  catch(e) {
    const body = $("#preregBody"), tag = $("#preregTag");
    if (tag) tag.textContent = "不可用";
    if (body) body.innerHTML = '<div class="note">前置注册状态加载失败：' + escHtml(e.message) + "</div>";
  }
}

// ----- 卡 3：ML 诊断 -----
function renderMlDiagnosis(rep) {
  const body = $("#mlDiagBody"), tag = $("#mlDiagTag");
  if (!body) return;
  if (!rep) { if (tag) tag.textContent = "未运行"; return; }
  if (rep.ok === false && !rep.channels) {
    if (tag) tag.textContent = "不可用";
    body.innerHTML = '<div class="note">ML 诊断不可用：' + escHtml(rep.error || "未知错误") +
      "</div>";
    return;
  }
  const trusted = !!rep.trusted;
  if (tag) {
    tag.textContent = trusted ? "trusted=True" : "trusted=False"
      + (rep.cached ? " · 缓存" : "") ;
  }
  const chans = rep.channels || {};
  const chRows = Object.keys(chans).map(k => {
    const c = chans[k] || {};
    return "<tr><td class='mono'>" + escHtml(k) + "</td><td>" +
      (c.trusted ? "<span class='badge gradeA'>可信</span>"
                 : "<span class='badge gradeC'>不可信</span>") +
      "</td><td>" + escHtml(String(c.tier || "-")) + "</td><td>" + _f(c.p_value, 4) +
      "</td><td>" + (c.n_numbers ?? "-") + "</td><td>" +
      (c.n_samples_per_number ?? "-") + "</td><td>" +
      (c.importance_significant ? "是" : "否") + "</td></tr>";
  }).join("");
  body.innerHTML =
    '<div class="v10-metrics">' +
      '<div class="v10-metric"><div class="k">trusted</div><div class="v">' +
        (trusted ? "True" : "False") + "</div><div class='sub'>两通道同时可信才算 True</div></div>" +
      '<div class="v10-metric"><div class="k">tier</div><div class="v">' +
        escHtml(String(rep.tier || "-")) + '</div><div class="sub">signal_found / monitor</div></div>' +
      '<div class="v10-metric"><div class="k">样本期数</div><div class="v">' +
        (rep.n_draws ?? "-") + '</div><div class="sub">max_issues=' +
        ((rep.params || {}).max_issues ?? "-") + "</div></div>" +
    "</div>" +
    // 诚实边界提示条（规格要求：不可省略）
    '<div class="v10-honesty"><b>trusted=False 是随机彩票的诚实结论，不是故障。</b> ' +
      escHtml(rep.honesty_zh || "") + "</div>" +
    (chRows ? '<div class="scroll"><table class="v10-table"><thead><tr>' +
      "<th>通道</th><th>可信</th><th>tier</th><th>联合 p 值</th><th>号码数</th>" +
      "<th>每号码样本</th><th>重要性显著</th></tr></thead><tbody>" + chRows +
      "</tbody></table></div>" : "") +
    '<div class="note">' + escHtml(rep.note_zh || "") + "</div>";
}

async function runMlDiagnosis(force) {
  const body = $("#mlDiagBody"), tag = $("#mlDiagTag"), btn = $("#btnMlDiag");
  if (body) body.innerHTML = '<div class="note">🤖 ML 诊断运行中（置换检验，约 10~30s）…</div>';
  if (tag) tag.textContent = "运行中…";
  if (btn) { btn.disabled = true; }
  try {
    const url = "/api/ml/diagnosis" + (force ? "?force=true&max_numbers=3&n_null=20" : "");
    let rep = await api(url);
    // D-3：重参数（默认档）后端改成了「立即返回 202 + task_id，后台算」，
    // 因为一次诊断要重训红/蓝两通道 × 多号码 × n_null 次零分布（实测 39~90s），
    // 同步请求会先被网关超时掐断。这里轮询任务直到完成。
    if (rep && rep.pending && rep.task_id) {
      if (body) body.innerHTML = '<div class="note">🤖 ML 诊断后台计算中（首次约 40~90 秒，'
        + '之后走缓存秒回）…</div>';
      const deadline = Date.now() + 300000;
      while (Date.now() < deadline) {
        await new Promise((res) => setTimeout(res, 3000));
        const t = await api("/api/tasks/" + encodeURIComponent(rep.task_id));
        if (t && (t.status === "completed" || t.status === "failed")) {
          if (t.status !== "completed") throw new Error(t.message || "后台诊断失败");
          // 任务完成后结果已写盘缓存，再请求一次端点即可拿到结构化结果
          rep = await api(url);
          break;
        }
        const pct = t && typeof t.progress === "number" ? Math.round(t.progress * 100) : null;
        if (body && pct !== null) {
          body.innerHTML = '<div class="note">🤖 ML 诊断后台计算中… ' + pct + '%</div>';
        }
      }
    }
    renderMlDiagnosis(rep);
  } catch(e) {
    if (tag) tag.textContent = "失败";
    if (body) body.innerHTML = '<div class="note">ML 诊断失败：' + escHtml(e.message) + "</div>";
  } finally {
    if (btn) { btn.disabled = false; }
  }
}

// ----- 卡 6：形态规律（组合恒等约束，V12）-----
// 这些不是"预测下一期"的规律，而是组合数学恒等约束：
// 和值/跨度/三区/奇偶/连号几乎必然落在这些区间内（实测 vs C(33,6) 理论一致）。
// 作为严格模式的「结构门控条件」，LLM 在此约束下生成注单，形态必然合理。
// （旧注释把这里写成"可用规律"，是 V12 之前的说法；后端 routes_predict.py 已明令禁用该叫法。）
async function loadShapeRules(force) {
  const body = document.querySelector("#shapeRuleBody");
  const tag = document.querySelector("#shapeRuleTag");
  if (body && !force) {
    const dl = body.getAttribute("data-loaded");
    if (dl === "1") return;
  }
  if (!body) return;
  if (tag) tag.textContent = "加载中…";
  try {
    const rep = await api("/api/shape/rules?_ts=" + Date.now());
    body.setAttribute("data-loaded", "1");
    if (!rep || rep.ok === false || !rep.rules) {
      body.innerHTML = '<div class="note">形态规律加载失败：' + escHtml((rep && rep.error) || "未知") + "</div>";
      if (tag) tag.textContent = "失败";
      return;
    }
    if (tag) tag.textContent = rep.n_draws + " 期实测";
    const rows = (rep.rules || []).map((r) => {
      const am = (r.actual_mean !== null && r.actual_mean !== undefined)
        ? "实测均值 " + Number(r.actual_mean).toFixed(2) : "";
      const ar = (r.actual_rate !== undefined && r.actual_rate !== null)
        ? "实测在带 " + (r.actual_rate * 100).toFixed(1) + "%" : "";
      return '<tr><td><b>' + escHtml(r.name_zh) + "</b><br>" +
        '<span class="hint-inline">' + escHtml(r.desc_zh || "") + "</span></td>" +
        "<td>" + (r.theoretical_rate * 100).toFixed(0) + "%</td>" +
        "<td>" + escHtml(am) + " " + escHtml(ar) + "</td></tr>";
    }).join("");
    body.innerHTML = '<div class="table-wrap"><table class="data-table">' +
      "<thead><tr><th>形态规律</th><th>理论覆盖率</th><th>实测</th></tr></thead>" +
      "<tbody>" + rows + "</tbody></table></div>" +
      '<div class="note" style="margin-top:8px">' + escHtml(rep.note_zh || "") + "</div>";
  } catch (e) {
    body.innerHTML = '<div class="note">形态规律加载失败：' + escHtml(e.message) + "</div>";
    if (tag) tag.textContent = "失败";
  }
}

// ----- 卡 5：多尺度规律扫描（V11）-----
// 这张卡的存在意义：把每档的「检出下限 MDE」摆在明面上。
// 用户直觉是"窗口越短越容易找规律"，数学上却是反的 —— 40 期需 +35% 才算显著。
// 不标 MDE，用户就会被短窗口的"显著"骗到。
let _multiScaleMdeRows = null;

async function loadMultiScaleMde() {
  if (_multiScaleMdeRows) return _multiScaleMdeRows;
  try {
    const d = await api("/api/multiscale/mde-table");
    _multiScaleMdeRows = d && d.rows ? d.rows : [];
  } catch (_) {
    _multiScaleMdeRows = [];
  }
  return _multiScaleMdeRows;
}

function _verdictMeta(v) {
  if (v === "credible") return { txt: "可信", cls: "ok", icon: "★" };
  if (v === "underpowered") return { txt: "功效不足", cls: "warn", icon: "⚠" };
  if (v === "insufficient") return { txt: "样本不足", cls: "mute", icon: "–" };
  return { txt: "不显著", cls: "mute", icon: "·" };
}

function renderMultiScale(rep, mdeRows) {
  const body = $("#multiScaleBody"), mdeBox = $("#multiScaleMde"), tag = $("#multiScaleTag");
  if (!body) return;
  if (!rep || rep.ok === false) {
    if (tag) tag.textContent = "失败";
    body.innerHTML = '<div class="note">多尺度扫描失败：' + escHtml((rep && rep.error) || "未知错误") + "</div>";
    return;
  }
  // MDE 对照条：让"窗口越短下限越高"一眼可见
  if (mdeBox && mdeRows && mdeRows.length) {
    mdeBox.innerHTML = '<div class="note" style="margin-bottom:8px">' +
      "<b>各档检出下限（MDE）</b>：" +
      mdeRows.filter((r) => r.in_use || r.n === 3515 || r.n === 100)
        .map((r) => '<span class="tag">' + r.n + " 期 " + r.mde_rel_pct + "</span>").join(" ") +
      "</div>";
  }
  const scales = (rep.scales || []).slice().reverse();   // 从短到长，便于对照
  const rows = rep.results || [];
  if (tag) tag.textContent = "扫描 " + (rep.n_scanned || 0) + " 个维度";
  if (!rows.length) {
    body.innerHTML = '<div class="note">没有需要关注的维度（全部不显著）——' +
      "这在随机开奖数据上是<b>正常且预期</b>的结果。</div>";
    return;
  }
  const head = "<tr><th>维度</th>" + scales.map((sc) =>
    "<th>" + escHtml(sc.label_zh) + "<br><span class=\"hint-inline\">MDE " +
    (sc.window ? (sc.window + "期") : "全期") + "</span></th>").join("") +
    "<th>建议</th></tr>";
  const bodyRows = rows.map((r) => {
    const name = escHtml(r.key + " " + JSON.stringify(r.params || {}));
    const cells = scales.map((sc) => {
      const v = (r.scales || {})[sc.key];
      if (!v || v.verdict === "insufficient") return "<td class=\"mute\">–</td>";
      const m = _verdictMeta(v.verdict);
      const dv = (v.delta >= 0 ? "+" : "") + Number(v.delta).toFixed(3);
      return "<td class=\"" + m.cls + "\" title=\"" + escHtml(v.reason_zh || "") + "\">" +
        m.icon + " " + dv + "<br><span class=\"hint-inline\">" + m.txt + "</span></td>";
    }).join("");
    const recTxt = { watch: "值得观察", "short-noise": "短档噪声", "regime-shift": "疑似结构变化",
                     weak: "弱", none: "无" }[r.recommendation] || r.recommendation;
    return "<tr><td>" + name + "</td>" + cells + "<td>" + escHtml(recTxt) + "</td></tr>";
  }).join("");
  body.innerHTML = '<div class="table-wrap"><table class="data-table"><thead>' + head +
    "</thead><tbody>" + bodyRows + "</tbody></table></div>" +
    '<div class="note" style="margin-top:8px">' + escHtml(rep.note_zh || "") + "</div>";
}

async function runMultiScale(force) {
  const tag = $("#multiScaleTag"), btn = $("#btnMultiScale");
  if (tag) tag.textContent = "扫描中…";
  if (btn) btn.disabled = true;
  try {
    const mdeRows = await loadMultiScaleMde();
    const rep = await api("/api/multiscale/scan" + (force ? "?only_interesting=false" : ""));
    renderMultiScale(rep, mdeRows);
    const msBody = $("#multiScaleBody");
    if (msBody) msBody.setAttribute("data-loaded", "1");
  } catch (e) {
    if (tag) tag.textContent = "失败";
    const body = $("#multiScaleBody");
    if (body) body.innerHTML = '<div class="note">多尺度扫描失败：' + escHtml(e.message) + "</div>";
  } finally {
    if (btn) btn.disabled = false;
  }
}

// ----- 卡 4：两套样本并排（禁止相加） -----
function renderSamplesSideBySide(rep) {
  const body = $("#samplesBody"), tag = $("#samplesTag");
  if (!body) return;
  if (!rep || rep.ok === false) {
    if (tag) tag.textContent = "不可用";
    body.innerHTML = '<div class="note">样本并排不可用：' +
      escHtml((rep && rep.error) || "接口未返回") + "</div>";
    return;
  }
  const live = rep.live || {}, replayBlk = rep.replay || {};
  if (tag) tag.textContent = "mixable=" + (rep.mixable ? "True" : "False") + " · 禁止相加";
  const side = (title, blk, cls) => {
    const sig = blk.significance || {};
    return '<div class="v10-side ' + cls + '">' +
      "<div><b>" + escHtml(title) + "</b></div>" +
      '<div class="note">source = <span class="mono">' + escHtml(String(blk.source || "-")) +
        "</span></div>" +
      "<table class='v10-table'><tbody>" +
      "<tr><th>样本期数 n_issues</th><td>" + (blk.n_issues ?? 0) + "</td></tr>" +
      "<tr><th>注数 n_tickets</th><td>" + (blk.n_tickets ?? 0) + "</td></tr>" +
      "<tr><th>可检出 MDE</th><td>" + _f(blk.mde, 4) + "</td></tr>" +
      "<tr><th>红球平均命中</th><td>" + _f(blk.red_hits_mean, 4) + "</td></tr>" +
      "<tr><th>蓝球命中率</th><td>" + _p2(blk.blue_hit_rate) + "</td></tr>" +
      (sig && sig.p_value !== undefined
        ? "<tr><th>显著性 p</th><td>" + _f(sig.p_value, 4) + "</td></tr>" : "") +
      "</tbody></table>" +
      '<div class="note">' + escHtml(blk.note_zh || blk.source_zh || "") + "</div>" +
      "</div>";
  };
  body.innerHTML =
    '<div class="v10-honesty"><b>两套样本禁止相加：</b>' +
      escHtml(rep.note_zh || "来源与口径都不同，只允许并排对照。") + "</div>" +
    '<div class="v10-sides">' +
      side("真实交战样本", live, "live") +
      side("离线回放样本", replayBlk, "replay") +
    "</div>" +
    '<div class="v10-nomix">mixable = False：本页刻意不显示任何合计/加权平均。' +
      (replayBlk.modes_zh ? " " + escHtml(replayBlk.modes_zh) : "") + "</div>";
}

async function loadSamplesSideBySide() {
  try { renderSamplesSideBySide(await api("/api/samples/side-by-side")); }
  catch(e) {
    const body = $("#samplesBody"), tag = $("#samplesTag");
    if (tag) tag.textContent = "不可用";
    if (body) body.innerHTML = '<div class="note">样本并排加载失败：' + escHtml(e.message) + "</div>";
  }
}

/** V10-10：三卡 + 样本并排的统一加载入口（页面初始化 / 切到评估页时调用）。
 *
 * V14-B2/B3 起并入蓝球覆盖 + 大众偏好度两卡。它们各自内部已经 try/catch，
 * 这里再加 `.catch` 是第二道保险：`Promise.all` 里任何一个未捕获的 reject
 * 都会让整个首屏渲染挂掉，而这两张卡是**附加信息**，不该有这种权力。
 */
async function loadV10Cards() {
  await Promise.all([
    loadDistribution(),
    loadRegistryStatus(),
    loadSamplesSideBySide(),
    loadShapeRules(false),      // V12 形态规律卡
    loadMultiScaleMde(),        // V11 多尺度 MDE（供卡 5 表头）
    loadBlueCoverage(),         // V14-B3 蓝球去重覆盖（披露不可拆）
    loadPopularity(),           // V14-B2 大众偏好度（只给档位）
  ].map(p => Promise.resolve(p).catch(() => null)));
  // V11 多尺度扫描卡：首次进 overview 时自动渲染（force=true 拉全部维度）
  const msBody = document.querySelector("#multiScaleBody");
  if (msBody && !msBody.getAttribute("data-loaded")) runMultiScale(true);
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
    // P2#8：`≥五等率` / `ROI` 是比率，必须与样本量绑在一起读 —— 3 期里中 1 次是 33%，
    // 与 100 期里 33 次完全不是一回事。表头写死样本框（期数 × 注数），表脚再复述一次，
    // 与本项目 `runCounterfactual` 里已有的写法保持同一口径。
    "<div class='scroll'><table><thead><tr><th>通道</th><th>红球平均命中</th>" +
      "<th>蓝球命中率</th>" +
      "<th>≥五等率<br><span class='th-sub'>（样本 " + (r.window_issues || 0) + " 期 × " +
        (r.tickets || 0) + " 注）</span></th>" +
      "<th>ROI<br><span class='th-sub'>（同上样本框）</span></th>" +
      "<th>LLM调用</th><th>估算成本$</th><th>LLM耗时</th></tr></thead><tbody>" + rows +
      "</tbody></table></div>" +
    "<div class='chart' id='chLlmCmp' style='height:220px;margin-top:8px'></div>" +
    "<div class='scroll' style='margin-top:8px'><table><thead><tr><th>对比</th><th>指标</th><th>Δ均值</th><th>p 值</th><th>结论</th></tr></thead><tbody>" + cmp + "</tbody></table></div>" +
    "<div class='note'>结论：paired 检验（BH 校正后 p<0.05 才视为显著）；cost 为按 token 估算，仅作展示。" +
      "LLM 通道存在模型噪声，stat/random 通道固定种子可复现。" +
      "样本 " + (r.window_issues || 0) + " 期 × " + (r.tickets || 0) +
      " 注 —— ≥五等率与 ROI 的抽样误差随样本量平方根缩小，样本不足 " + METHOD_MIN_ISSUES +
      " 期时，与随机基线之差属随机波动，不构成能力结论。</div>";
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

// ==================== V5：实战战绩 + 进化过程 ====================
//
// 口径（用户已明确）：准确性 = 用历史数据找规律特点 → 用于实战 → **通过实战结果对比**。
// 本区块的唯一职责是**呈现实战对比事实**：哪个方法在多少期实战里如何、差异是否显著、
// 找出来的规律有没有通过验证。不再重复「彩票随机」这类提醒。
//
// 但工程上的诚实要求照旧，且是本区块的硬约束：
//   · 一切比率必须带样本量（走 metricWithSample / sampleEnough）
//   · ROI 必须双口径并排：`roi` 与剔除一二等奖的 `roi_ex_top`
//   · 同时比多个方法时，未校正 p<0.05 不可信 —— 必须显示「未校正 N → 校正 M」
//
// 性能：/api/battle/report 是重操作（limit=900 约 20~60 秒，limit=0 全量约 5 分钟），
// **绝不在首屏自动触发**，只能由用户点击。首屏只拉 /api/evolution/status（很快）。

let _battleReport = null;      // 最近一次战绩回放结果（供重绘/切换口径用）
let _evoStatus = null;         // 最近一次进化状态
let _battleBusy = false;       // 防重入：回放期间禁止再次触发

/** ROI 专用格式化：pct() 对 0 与负数会退化成 "-"，ROI 必须能显示负值。 */
function fmtRoi(v) {
  if (v == null || isNaN(Number(v))) return "-";
  const x = Number(v) * 100;
  return (x >= 0 ? "+" : "") + x.toFixed(1) + "%";
}

/** 判定 → 配色类名，保证「显著更优/更差」一眼可辨。 */
function verdictClass(v) {
  if (v === "显著更优") return "pos";
  if (v === "显著更差") return "neg";
  if (v === "样本不足") return "insufficient";
  return "flat";
}

/** 长耗时操作的统一进度反馈：按钮进入忙碌态 + 显式「预计 X 分钟」提示。 */
function setBusy(btnId, busy, text, note) {
  const b = $(btnId);
  if (b) {
    b.disabled = !!busy;
    if (busy) b.innerHTML = "<span class='spin'>⏳</span> " + escHtml(text || "运行中…");
    else if (text) b.innerHTML = escHtml(text);
  }
  const el = $(note);
  if (el) el.innerHTML = busy ? "" : (el.innerHTML || "");
}

/** 进化总览卡：假说库状态 + **只有 validated 才影响出号** 的事实。 */
function renderEvolutionStatus(st) {
  const el = $("#evoStatusBody");
  const tag = $("#evoStatusTag");
  if (!el) return;
  if (!st || st.available === false) {
    // 模块未就绪：优雅降级，不得报错、不得伪装成「无规律」
    if (tag) tag.textContent = "模块未就绪";
    el.innerHTML = '<div class="track-note warn">⚠️ 自进化模块当前不可用' +
      (st && st.reason ? "（" + escHtml(st.reason) + "）" : "") +
      "。<b>此处不代表规律库为空</b>，请稍后重试或检查服务端 lottery/evolution.py 是否就绪。</div>";
    return;
  }
  const by = st.by_status || {};
  const proposed = Number(by.proposed || 0);
  const validated = Number(by.validated || 0);
  const rejected = Number(by.rejected || 0);
  const total = Number(st.total || 0);
  const ov = st.active_overrides || {};
  const appliedN = (ov.applied || []).length;
  const skipped = ov.skipped || [];
  const minIssues = Number(st.min_issues || METHOD_MIN_ISSUES);

  if (tag) tag.textContent = "库内 " + total + " 条 · 已验证 " + validated + " 条";

  el.innerHTML =
    '<div class="track-metrics">' +
      '<div class="track-item"><div class="k">规律总数</div><div class="v">' + total +
        '<span class="sub">库内全部条目</span></div></div>' +
      '<div class="track-item"><div class="k">待验证 proposed</div><div class="v">' + proposed +
        '<span class="sub">尚未通过实战验证</span></div></div>' +
      '<div class="track-item"><div class="k">已验证 validated</div><div class="v ' +
        (validated > 0 ? "pos" : "insufficient") + '">' + validated +
        '<span class="sub">唯一会改变出号的状态</span></div></div>' +
      '<div class="track-item"><div class="k">已否决 rejected</div><div class="v">' + rejected +
        '<span class="sub">留档供复盘</span></div></div>' +
      '<div class="track-item"><div class="k">实际生效覆盖项</div><div class="v">' + appliedN +
        '<span class="sub">applied 条</span></div></div>' +
    '</div>' +
    // 核心事实：只有 validated 影响出号。当前若为 0，明确说「尚未找到通过验证的规律」。
    (validated > 0
      ? '<div class="track-note">✅ 已有 <b>' + validated + '</b> 条规律通过实战 A/B 验证，正参与出号（生效覆盖项 ' +
        appliedN + ' 条）。</div>'
      : '<div class="track-note warn">⚠️ <b>尚未找到通过验证的规律</b>：库内 ' + total +
        ' 条全部处于 <code>proposed</code>（待验证），实际生效覆盖项 <b>0</b> 条 —— ' +
        '也就是说这些规律<b>目前不影响出号</b>，必须先通过配对实战回放验证才会被采用。</div>') +
    (skipped.length
      ? '<div class="track-note">未生效条目（' + skipped.length + ' 条，全部因状态未达 validated）：' +
        skipped.slice(0, 12).map(s => '<code>' + escHtml(s.id) + '</code>').join("、") +
        (skipped.length > 12 ? " 等" : "") + '</div>'
      : '') +
    '<div class="track-note">裁决门槛：单条规律需在 ≥ ' + minIssues +
      ' 期配对回放上通过多重检验校正，才可能从 proposed 晋升 validated。' +
      '状态迁移：显著支持 → validated（影响出号）· 显著反对 → rejected（退场留档）· 样本不足 → 保持 proposed。</div>';

  renderValidatedRules(st);
}

/** 已验证规律明细（latest_validated）。当前为空时如实说明。 */
function renderValidatedRules(st) {
  const host = $("#evoValidatedBody");
  if (!host) return;
  const lv = (st && st.latest_validated) || [];
  if (!lv.length) {
    host.innerHTML = '<div class="track-note">当前无 validated 规律明细（已验证 0 条）。</div>';
    return;
  }
  host.innerHTML = '<div class="scroll"><table><thead><tr><th>规律</th><th>动作</th><th>证据</th></tr></thead><tbody>' +
    lv.map(h => {
      const ev = h.evidence || {};
      const n = ev.n != null ? ev.n : (h.n_samples != null ? h.n_samples : null);
      return '<tr><td>' + escHtml(h.statement || h.id) + '</td><td class="mono">' +
        escHtml(typeof h.action === "object" ? JSON.stringify(h.action) : h.action) + '</td><td>' +
        // 证据必须带样本量：这是「通过实战验证」的可核查依据
        'n=' + (n == null ? "-" : n) + (ev.lift != null ? " · lift " + fmt(ev.lift, 4) : "") +
        (ev.metric ? " · " + escHtml(ev.metric) : "") + '</td></tr>';
    }).join("") + "</tbody></table></div>";
}

async function loadEvolutionStatus(manual) {
  const tag = $("#evoStatusTag");
  if (tag && !_evoStatus) tag.textContent = "加载中…";
  try {
    const st = await api("/api/evolution/status");
    _evoStatus = st;
    renderEvolutionStatus(st);
    if (manual) toast("🧬 进化状态已刷新");
  } catch (e) {
    // 503 / 网络失败：降级提示，不炸页面
    _evoStatus = null;
    if (tag) tag.textContent = "加载失败";
    const body = $("#evoStatusBody");
    if (body) body.innerHTML = '<div class="track-note warn">无法读取进化状态（' +
      escHtml(e.message) + '）。<b>此处不代表规律库为空</b>，请稍后重试。</div>';
  }
}

/** 触发回放。limit=0 为全量（约 5 分钟），必须二次确认。 */
async function runBattleReport(limit) {
  if (_battleBusy) { toast("回放正在进行中，请等待完成"); return; }
  const isFull = !limit;
  const nTickets = Number(($("#cfgTickets") && $("#cfgTickets").value) || 10);

  if (isFull) {
    // 全量回放约 5 分钟：显式确认，避免用户以为页面卡死
    const ok = window.confirm(
      "⚠️ 全量回放将回放全部历史期数，预计耗时约 5 分钟（343~373 秒）。\n\n" +
      "期间页面会持续显示进度，请不要关闭或刷新页面。\n\n确定开始吗？");
    if (!ok) return;
  }

  _battleBusy = true;
  const btn = isFull ? "#btnEvoBattleFull" : "#btnEvoBattle";
  const other = isFull ? "#btnEvoBattle" : "#btnEvoBattleFull";
  setBusy(btn, true, isFull ? "全量回放中…" : "回放中…");
  const ob = $(other); if (ob) ob.disabled = true;

  const prog = $("#evoBattleProgress");
  if (prog) {
    prog.innerHTML = '<div class="track-note warn">⏳ ' +
      (isFull ? "全量回放进行中，预计约 5 分钟" : "小窗口回放进行中，预计 20~60 秒") +
      '（各方法逐期出号并与真实开奖核对）。页面仍在响应，请勿刷新。</div>';
  }
  const body = $("#evoBattleBody");
  if (body) body.innerHTML = '<div class="note">回放运行中…</div>';

  try {
    const url = "/api/battle/report?limit=" + (isFull ? 0 : limit) +
      "&n_tickets=" + nTickets + "&start=300";
    const rep = await api(url);
    if (rep.available === false) {
      renderBattleUnavailable(rep);
    } else if (rep.ok === false) {
      // 例如「样本不足：300 期 ≤ start=300」——这是有效的诚实答复，不是崩溃
      if (body) body.innerHTML = '<div class="track-note warn">⚠️ 无法生成回放报告：' +
        escHtml(rep.error || "未知原因") + '</div>';
      const t2 = $("#evoBattleTag"); if (t2) t2.textContent = "报告不可用";
    } else {
      _battleReport = rep;
      renderBattleReport(rep);
    }
  } catch (e) {
    if (body) body.innerHTML = '<div class="track-note warn">回放失败：' + escHtml(e.message) +
      '。<b>这不代表方法之间无差异</b>，请重试。</div>';
    const t3 = $("#evoBattleTag"); if (t3) t3.textContent = "回放失败";
  } finally {
    _battleBusy = false;
    setBusy(btn, false, isFull ? "🐢 全量回放（约 5 分钟）" : "▶ 加载战绩榜（近 900 期，约 20~60 秒）");
    if (ob) ob.disabled = false;
    if (prog) prog.innerHTML = "";
  }
}

function renderBattleUnavailable(rep) {
  const t = $("#evoBattleTag"); if (t) t.textContent = "模块未就绪";
  const body = $("#evoBattleBody");
  if (body) body.innerHTML = '<div class="track-note warn">⚠️ 战斗回放模块当前不可用' +
    (rep && rep.error ? "（" + escHtml(rep.error) + "）" : "") +
    '。<b>此处不代表方法之间无差异</b>。</div>';
}

/** 战绩榜主渲染：比率一律带样本量；ROI 双口径并排。 */
function renderBattleReport(rep) {
  const tag = $("#evoBattleTag");
  const body = $("#evoBattleBody");
  if (!body) return;
  const rows = rep.results || [];
  const nIssues = Number(rep.n_issues || 0);
  const minIssues = Number(rep.min_issues || METHOD_MIN_ISSUES);
  const enough = sampleEnough(nIssues);

  if (tag) tag.textContent = "实战 " + nIssues + " 期 · " + rows.length + " 个方法";
  if (!rows.length) {
    body.innerHTML = '<div class="note">本次回放没有可对照的方法结果。</div>';
    return;
  }

  // 样本不足时不得给出「最优方法」结论
  const best = rows.reduce((a, b) => (a && a.red_hits_mean >= b.red_hits_mean) ? a : b, null);

  const table =
    '<div class="scroll"><table class="battle-table"><thead><tr>' +
      '<th>方法</th><th>期数</th><th>红球均值</th><th>蓝球命中率</th><th>≥五等奖率</th>' +
      '<th>奖金¥</th><th>ROI</th><th>ROI（剔一二等）</th><th>一二等奖期数</th>' +
    '</tr></thead><tbody>' +
    rows.map(r => {
      const rIssues = Number(r.issues || 0);
      const rEnough = sampleEnough(rIssues);
      // 比率类指标一律带样本量；样本不足时不裸展示百分比
      const cell = (val) => rEnough ? pct(val) : '<span class="insufficient-txt">样本不足</span>';
      const topN = Number(r.n_top_prize_issues || 0);
      const roiDiff = (Number(r.roi || 0) - Number(r.roi_ex_top || 0));
      return '<tr>' +
        '<td class="mono">' + escHtml(r.method) +
          (best && r.method === best.method && enough ? ' <span class="tag">红球最高</span>' : '') + '</td>' +
        '<td>' + rIssues + '</td>' +
        // 红球均值也是「看起来像能力」的读数 → 走样本门槛口径
        '<td>' + (rEnough ? fmt(r.red_hits_mean, 3) : '<span class="insufficient-txt">样本不足</span>') + '</td>' +
        '<td>' + cell(r.blue_hit_rate) + '</td>' +
        '<td>' + cell(r.prize_rate_ge5) + '</td>' +
        '<td>¥' + Number(r.reward_total || 0).toFixed(0) + '</td>' +
        '<td class="' + (Number(r.roi) >= 0 ? "pos" : "neg") + '">' + fmtRoi(r.roi) + '</td>' +
        '<td class="' + (Number(r.roi_ex_top) >= 0 ? "pos" : "neg") + '">' + fmtRoi(r.roi_ex_top) +
          (topN > 0 ? ' <span class="tag" title="含 ' + topN + ' 期一二等奖">↑离群</span>' : '') + '</td>' +
        '<td>' + topN + '</td>' +
      '</tr>';
    }).join("") + "</tbody></table></div>";

  // ROI 双口径说明：这是最容易被误读的地方，必须写清为什么两个数不一样
  const roiNote =
    '<div class="track-note warn bidx">📌 <b>ROI 为什么给两个口径</b>：右列 <code>roi_ex_top</code> 剔除了一、二等奖期数。' +
    '单期命中 6 红（二等奖 ¥180,000）这类离群值会一次性把 ROI 抬到 +200% 以上，' +
    '那是<b>单期偶然</b>而非方法稳定能力；只看左列会严重高估。两列差异越大，说明该方法的 ROI 越依赖个别离群期。' +
    '</div>';

  body.innerHTML =
    // 样本量口径（首要）
    '<div class="track-metrics">' +
      '<div class="track-item"><div class="k">实战回放期数</div><div class="v">' + nIssues +
        '<span class="sub">start=' + Number(rep.start || 0) + ' · 每期 ' + Number(rep.n_tickets || 0) + ' 注</span></div></div>' +
      '<div class="track-item"><div class="k">方法数</div><div class="v">' + rows.length +
        '<span class="sub">同条件逐期出号</span></div></div>' +
      '<div class="track-item"><div class="k">裁决门槛</div><div class="v">' + minIssues +
        '<span class="sub">期（后端 method_min_issues）</span></div></div>' +
      '<div class="track-item"><div class="k">样本是否充分</div><div class="v ' + (enough ? "pos" : "insufficient") + '">' +
        (enough ? "充分" : "不足") + '<span class="sub">' + (enough ? "可做显著性判决" : "不构成结论") + '</span></div></div>' +
    '</div>' +
    (rep.gate_zh ? '<div class="track-note">' + escHtml(rep.gate_zh) + '</div>' : '') +
    table + roiNote +
    (enough ? "" : '<div class="track-note warn">⚠️ ' + INSUFFICIENT_NOTE +
      '：当前仅 ' + nIssues + ' 期回放，低于门槛 ' + minIssues + ' 期，上表比率属波动读数，<b>不能</b>据此判定方法优劣。</div>') +
    // 不可回放方法如实标注，不假装它们参与了对比
    ((rep.non_replayable || []).length
      ? '<div class="track-note">未参与回放的方法（如实标注，非缺失）：' +
        rep.non_replayable.map(x => '<b>' + escHtml(x.method) + '</b>（' + escHtml(x.reason) + '）').join("；") + '</div>'
      : '') +
    (rep.discipline_zh ? '<div class="track-note">回放纪律：' + escHtml(rep.discipline_zh) + '</div>' : '') +
    (rep.note_zh ? '<div class="track-note">' + escHtml(rep.note_zh) + '</div>' : '');

  renderBattleComparison(rep);
}

/** 渲染两两对比：核心是「未校正显著 N 组 → 校正后仅 M 组」。 */
function renderBattleComparison(rep) {
  const el = $("#evoCompareBody");
  const tag = $("#evoCompareTag");
  if (!el) return;
  const c = rep.comparison || {};
  const pairs = c.pairs || [];
  const raw = Number(c.n_significant_uncorrected || 0);
  const adj = Number(c.n_significant_corrected || 0);
  const nIssues = Number(rep.n_issues || 0);
  const enough = sampleEnough(nIssues);

  if (tag) tag.textContent = pairs.length + " 组配对 · alpha=" + (c.alpha != null ? c.alpha : 0.05);

  if (!pairs.length) {
    el.innerHTML = '<div class="note">本次回放无可比较的方法配对。</div>';
    return;
  }

  const vc = c.verdict_counts || {};
  el.innerHTML =
    // 本项目核心方法论：同时比多个方法时 p<0.05 不可信
    '<div class="multi-correct-banner' + (adj === 0 ? " warn" : "") + '">' +
      '<div class="mcb-main">未校正显著 <b>' + raw + '</b> 组 &nbsp;→&nbsp; 校正后仅 <b>' + adj + '</b> 组</div>' +
      '<div class="mcb-sub">同时比 ' + pairs.length + ' 个方法对，每个都按 p&lt;0.05 判会天然产生假阳性（' +
        raw + ' 组里大部分是偶然）。上表按 Bonferroni / BH(FDR) 校正后，只有 <b>' + adj +
        '</b> 组真正站得住 —— 所以「p&lt;0.05」单独出现不能当结论。</div>' +
    '</div>' +
    '<div class="track-metrics">' +
      ['显著更优', '显著更差', '无显著差异', '样本不足'].map(k =>
        '<div class="track-item"><div class="k">' + k + '</div><div class="v ' + verdictClass(k) + '">' +
        Number(vc[k] || 0) + '<span class="sub">' + pairs.length + ' 组中</span></div></div>').join("") +
    '</div>' +
    '<div class="scroll"><table class="battle-table"><thead><tr>' +
      '<th>对比</th><th>期数</th><th>Δ红球</th><th>Δ蓝球率</th><th>未校正 p</th><th>校正后 p</th><th>判定</th>' +
    '</tr></thead><tbody>' +
    pairs.map(p => {
      const pEnough = sampleEnough(Number(p.n_issues || 0));
      return '<tr>' +
        '<td class="mono">' + escHtml(p.method) + ' <span class="vs">vs</span> ' + escHtml(p.vs) + '</td>' +
        '<td>' + Number(p.n_issues || 0) + '</td>' +
        '<td class="' + (Number(p.delta_red_hits) >= 0 ? "pos" : "neg") + '">' +
          (Number(p.delta_red_hits) >= 0 ? "+" : "") + fmt(p.delta_red_hits, 4) + '</td>' +
        '<td>' + (p.delta_blue_rate == null ? "-" :
          ((Number(p.delta_blue_rate) >= 0 ? "+" : "") + fmt(p.delta_blue_rate, 4))) + '</td>' +
        '<td>' + (p.p_uncorrected == null ? "-" : fmt(p.p_uncorrected, 4)) + '</td>' +
        '<td>' + (p.p_adjusted == null ? "-" : fmt(p.p_adjusted, 4)) + '</td>' +
        '<td class="' + verdictClass(p.verdict) + '">' + escHtml(p.verdict || "-") +
          (pEnough ? "" : ' <span class="tag">样本不足</span>') + '</td>' +
      '</tr>';
    }).join("") + "</tbody></table></div>" +
    (c.note_zh ? '<div class="track-note">检验方法：' + escHtml(c.note_zh) + '</div>' : '') +
    (c.summary_zh ? '<div class="track-note">' + escHtml(c.summary_zh) + '</div>' : '') +
    (enough ? "" : '<div class="track-note warn">⚠️ ' + INSUFFICIENT_NOTE + '。</div>');
}

/** 跑一轮进化：含 LLM 调用（1~2 分钟）+ 每条 proposed 假说最多 600 期配对回放，
 *  实测可能超过 10 分钟，且会写 hypotheses.json —— 必须二次确认。 */
async function runEvolutionOnce() {
  const ok = window.confirm(
    "⚠️ 跑一轮进化将：\n" +
    "  1) 挖掘新规律（调用 LLM，约 1~2 分钟）\n" +
    "  2) 写入服务器 data/hypotheses.json\n" +
    "  3) 对全部 proposed 规律各做最多 600 期 A/B 配对回放，再裁决\n\n" +
    "⚠️ 后两步随规律条数累加，整体<b>可能超过 10 分钟</b>。\n" +
    "期间页面会持续显示进度，请勿刷新。确定开始吗？");
  if (!ok) return;

  const prog = $("#evoRunProgress");
  const out = $("#evoRunBody");
  setBusy("#btnEvoRun", true, "进化运行中…");
  const refresh = $("#btnEvoStatusRefresh"); if (refresh) refresh.disabled = true;
  if (prog) {
    prog.innerHTML = '<div class="track-note warn">⏳ 进化运行中（挖掘 → 注册 → A/B 验证 → 裁决），' +
      '整体<b>可能超过 10 分钟</b>。含 LLM 调用与逐条假说回放，请耐心等待，页面未卡死。</div>';
  }
  if (out) out.innerHTML = "";

  try {
    const limit = 900;
    const nTickets = Number(($("#cfgTickets") && $("#cfgTickets").value) || 10);
    const r = await api("/api/evolution/run?limit=" + limit + "&n_tickets=" + nTickets +
      "&start=300&use_llm=1", { method: "POST" });
    if (r.available === false || r.ok === false) {
      if (out) out.innerHTML = '<div class="track-note warn">⚠️ 进化未能完成：' +
        escHtml(r.error || r.reason || "未知原因") + '</div>';
    } else {
      renderEvolutionRun(r);
    }
    // 本轮已返回最新 status，直接消费；失败或缺失时再回退单独请求
    if (r.status && r.status.available !== false) {
      _evoStatus = r.status;
      renderEvolutionStatus(r.status);
    } else {
      await loadEvolutionStatus(false);
    }
  } catch (e) {
    if (out) out.innerHTML = '<div class="track-note warn">进化执行失败：' + escHtml(e.message) +
      '。<b>这不代表没有规律进入验证流程</b>；失败只记录不抛出，请重试。</div>';
  } finally {
    setBusy("#btnEvoRun", false, "🧬 跑一轮进化（可能超过 10 分钟）");
    if (refresh) refresh.disabled = false;
    if (prog) prog.innerHTML = "";
  }
}

/** 本轮进化结果：挖到几条 / 注册几条 / 裁决结果。 */
function renderEvolutionRun(r) {
  const el = $("#evoRunBody");
  if (!el) return;
  const mined = r.mined || {};
  const reg = r.registered || {};
  const res = r.resolution || {};
  const validated = (res.validated || []).length;
  const rejected = (res.rejected || []).length;
  const insufficient = Number(res.insufficient || 0);
  const unchanged = Number(res.unchanged || 0);
  const by = r.by_status || {};

  el.innerHTML =
    '<div class="track-metrics">' +
      '<div class="track-item"><div class="k">本轮挖到</div><div class="v">' +
        (Number(mined.stat || 0) + Number(mined.llm || 0)) +
        '<span class="sub">统计 ' + Number(mined.stat || 0) + " · LLM " + Number(mined.llm || 0) + '</span></div></div>' +
      '<div class="track-item"><div class="k">新注册</div><div class="v">' + Number(reg.registered || 0) +
        '<span class="sub">去重命中 ' + Number(reg.deduped || 0) + '</span></div></div>' +
      '<div class="track-item"><div class="k">参与 A/B 验证</div><div class="v">' + Number(r.verified || 0) +
        '<span class="sub">proposed 条目</span></div></div>' +
      '<div class="track-item"><div class="k">晋升 validated</div><div class="v ' + (validated > 0 ? "pos" : "insufficient") + '">' +
        validated + '<span class="sub">将影响出号</span></div></div>' +
      '<div class="track-item"><div class="k">否决 rejected</div><div class="v">' + rejected +
        '<span class="sub">退场留档</span></div></div>' +
      '<div class="track-item"><div class="k">样本不足 / 未变</div><div class="v">' +
        insufficient + " / " + unchanged + '<span class="sub">保持 proposed</span></div></div>' +
    '</div>' +
    (mined.llm_used === false && mined.llm_error
      ? '<div class="track-note warn">本轮 LLM 未参与（' + escHtml(mined.llm_error) + '），仅用统计挖掘。</div>'
      : '') +
    (validated > 0
      ? '<div class="track-note">✅ 本轮有 <b>' + validated + '</b> 条规律通过验证，已开始影响出号。</div>'
      : '<div class="track-note warn">本轮<b>无规律通过验证</b>（晋升 0 条）：库内仍全部为 proposed，' +
        '即<b>尚未找到通过验证的规律</b>，出号策略未被本轮改变。这是实战裁决的正常结果，不是失败。</div>') +
    '<div class="track-note">本轮结束后库内状态：' +
      Object.keys(by).map(k => escHtml(k) + " " + Number(by[k] || 0)).join(" · ") + '</div>';
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

/**
 * 读 LLM 配置。
 *
 * 走 **`/api/user/llm/config`**（W08 起新增），不再走 `/api/config/llm`：
 * 后者是管理员专用端点，普通用户调用会 403，"我的配置"却报"需要管理员权限"
 * 会把用户彻底搞糊涂 —— 而他的模型配置本来就该由他自己管。
 */

/**
 * 读并渲染「我的 LLM 配置」（MU-2 起真正按账号隔离）。
 *
 * 后端 `/api/user/llm/config` 的 `scope` 有三态，界面必须**说清是哪一种**，
 * 否则用户会分不清"我正在花自己的钱"和"我正在花管理员的额度"：
 *   user   → 用的是我自己的 Key（不受系统配额限制）
 *   system → 我在用管理员的共享通道（受今日配额限制）
 *   none   → 没有可用通道，会走纯统计出号（**照常出号，不是故障**）
 */
let _llmScope = "none";

async function loadLlmConfig() {
  try {
    const cfg = await api("/api/user/llm/config");
    _llmScope = cfg.scope || "none";

    const bad = {user: cfg.configured, system: cfg.configured, none: false}[_llmScope];
    setLlmStatus(
      bad ? (cfg.configured ? "✅ 已配置" : "❌ 未配置") : "⚪ 纯统计出号",
      bad && cfg.configured ? "ok" : "bad");

    $("#cfgBaseUrl").value = cfg.base_url || "";
    // Key 输入框**永远留空**，只用 placeholder 显示脱敏值与"留空=不修改"。
    // 若把脱敏串填进 value，用户一保存就会把它当成真 Key 提交上去。
    $("#cfgApiKey").value = "";
    $("#cfgApiKey").placeholder = cfg.has_api_key
      ? ((cfg.api_key_masked || "******") + "（已保存 · 留空=不修改）")
      : "sk-...";
    $("#cfgModel").value = cfg.model || "";
    $("#cfgSamples").value = cfg.samples || 3;
    $("#cfgLlmEnabled").checked = !cfg.disabled;
    $("#cfgUseCustomKey").checked = !!cfg.use_custom_key;

    renderLlmScope(cfg);
    renderLlmUsage(cfg);
    toggleLlmConfig();
  } catch(e) {
    console.error("loadLlmConfig failed:", e);
    setLlmStatus("❌ 加载失败", "bad");
    const b = $("#llmScopeNote");
    if (b) {
      b.style.display = "block";
      b.innerHTML = "读取配置失败：" + escHtml(e.message) +
        "。这不影响出号 —— 没有可用通道时会自动走纯统计。";
    }
  }
}

/** 顶部来源说明：把"我这会儿在花谁的钱"讲清楚。 */
function renderLlmScope(cfg) {
  const b = $("#llmScopeNote");
  if (!b) return;
  b.style.display = "block";
  const admin = (AUTH.user && AUTH.user.role === "admin");
  const map = {
    user: "✅ 当前生效：<b>你自己的配置</b>（消耗你自己 Key 的额度，不受系统配额限制）。",
    system: "ℹ️ 当前生效：<b>管理员共享通道</b>（受每日配额限制）。"
          + "想不受限制，可在下方勾选「使用我自己的 API Key」并填写你的配置。",
    none: "⚪ 当前<b>没有可用的 LLM 通道</b>，预测会走<b>纯统计模式照常出号</b>（不需要任何 Key）。"
        + "想启用大模型，请在下方填写配置。",
  };
  let html = map[_llmScope] || map.none;
  if (cfg.reason) html += '<br><span class="llm-reason">原因：' + escHtml(cfg.reason) + "</span>";
  if (admin && !cfg.system_channel_configured) {
    html += '<br><span class="llm-reason">提示：你是管理员，可在此配置<b>共享通道</b>'
          + "（对所有未配个人 Key 的用户生效），或到「更多 → 系统默认」单独维护。</span>";
  }
  b.innerHTML = html;
}

/** 今日共享通道用量。自备 Key 不受配额限制，因此不显示这条。 */
function renderLlmUsage(cfg) {
  const box = $("#llmUsage");
  if (!box) return;
  const quota = Number(cfg.daily_token_quota || 0);
  if (_llmScope === "user" || !quota) {
    // 自备 Key（花自己的钱）或配额=0（不限）→ 这条没有意义
    box.style.display = "none";
    return;
  }
  const used = Number(cfg.today_tokens_used || 0);
  const pctUsed = Math.max(0, Math.min(100, Math.round((used / quota) * 100)));
  box.style.display = "flex";
  const fill = $("#llmUsageFill");
  if (fill) {
    fill.style.width = pctUsed + "%";
    fill.className = "llm-usage-fill" + (pctUsed >= 90 ? " danger" : pctUsed >= 70 ? " warn" : "");
  }
  const num = $("#llmUsageNum");
  if (num) num.textContent = used.toLocaleString() + " / " + quota.toLocaleString() + " tokens（" + pctUsed + "%）";
}

function toggleLlmConfig() {
  const enabled = !!$("#cfgLlmEnabled")?.checked;
  // 只有勾了「使用我自己的 Key」才需要填地址/Key/模型；
  // 不勾时这些字段对共享通道无意义，禁用可避免"填了却不生效"的困惑。
  const custom = !!$("#cfgUseCustomKey")?.checked;
  ["cfgLlmEnabled", "cfgUseCustomKey"].forEach(id => {
    const el = $("#" + id);
    if (el) el.disabled = false;                 // 这两个开关永远可点
  });
  ["cfgBaseUrl", "cfgApiKey", "cfgModel", "cfgSamples"].forEach(id => {
    const el = $("#" + id);
    if (el) el.disabled = !(enabled && custom);
  });
  const clearBtn = $("#btnClearKey");
  if (clearBtn) clearBtn.disabled = !custom;
}

/** 删除已保存的 Key（显式操作，与"留空=不修改"区分开）。 */
async function clearLlmKey() {
  if (!confirm("确认删除已保存的 API Key？删除后该账号将回落到共享通道（若管理员配置了）或纯统计。")) {
    return;
  }
  const status = $("#llmConfigStatus");
  try {
    const r = await fetch("/api/user/llm/config", {
      method: "POST",
      headers: {"Content-Type": "application/json", "X-CSRF-Token": AUTH.csrf || ""},
      body: JSON.stringify({clear_api_key: true}),
    });
    const j = await r.json().catch(() => ({}));
    if (!r.ok || !j.ok) throw new Error(j.error || ("HTTP " + r.status));
    toast("API Key 已删除");
    if (status) status.innerHTML = '<span class="oktxt">✓ Key 已删除</span>';
    await loadLlmConfig();
  } catch (e) {
    toast("删除失败：" + e.message);
    if (status) status.innerHTML = '<span class="errtxt">✗ 删除失败：' + escHtml(e.message) + "</span>";
  }
}

async function saveLlmConfig() {
  if (!$("#cfgBaseUrl") || !$("#cfgApiKey") || !$("#cfgModel")) {
    toast("设置表单未加载完成，请刷新页面重试");
    return;
  }
  const custom = !!$("#cfgUseCustomKey")?.checked;
  const payload = {
    use_custom_key: custom,
    samples: parseInt($("#cfgSamples")?.value || "3") || 3,
    disabled: !$("#cfgLlmEnabled")?.checked,
  };
  // 只有勾了"用自己的 Key"才提交连接参数；否则提交空值会把已存的配置抹掉。
  if (custom) {
    payload.base_url = $("#cfgBaseUrl").value.trim();
    payload.model = $("#cfgModel").value.trim();
  }
  // **关键**：只有真的输入了新 Key 才提交 api_key。
  // 输入框是空的（只显示脱敏占位），无条件提交空串会把已保存的 Key 抹掉
  // —— 这是"改个注数就丢了 API Key"的经典写法。
  const typedKey = ($("#cfgApiKey").value || "").trim();
  if (typedKey) payload.api_key = typedKey;

  if (custom && !payload.disabled) {
    const hasSavedKey = ($("#cfgApiKey").placeholder || "").includes("已保存");
    if (!payload.base_url || !payload.model || (!typedKey && !hasSavedKey)) {
      toast("使用自己的 Key 需填写 API 地址 / Key / 模型名称");
      return;
    }
  }

  const status = $("#llmConfigStatus");
  if (status) status.innerHTML = '<span class="oktxt">⏳ 正在保存…</span>';
  try {
    const res = await api("/api/user/llm/config", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    if (res && res.ok) {
      toast(payload.disabled ? "LLM 已停用，配置已保存" : "配置已保存");
      if (status) status.innerHTML = '<span class="oktxt">✓ 已保存，立即生效（无需重启）</span>';
      if ($("#cfgLlm")) $("#cfgLlm").checked = !payload.disabled;
      $("#cfgApiKey").value = "";
      await loadLlmConfig();      // 重新拉脱敏值，确认落库后的真实状态
    } else {
      throw new Error((res && res.error) || "服务器无响应");
    }
  } catch(e) {
    console.error("saveLlmConfig failed:", e);
    toast("保存失败: " + e.message);
    if (status) status.innerHTML = '<span class="errtxt">✗ 保存失败：' + escHtml(e.message) + "</span>";
  }
}

async function testLlmConnection() {
  const btn = event.target;
  const resultEl = $("#llmTestResult");
  btn.disabled = true;
  btn.textContent = "测试中…（推理型模型可能数十秒）";
  if (resultEl) resultEl.textContent = "";
  try {
    const result = await api("/api/llm/test", {method: "POST"});
    if (result.ok) {
      const extra = [];
      if (result.model) extra.push(escHtml(result.model));
      if (result.max_tokens) extra.push("输出预算 " + result.max_tokens);
      if (result.reasoning_model) extra.push("推理型（已自动适配）");
      if (resultEl) resultEl.innerHTML = '<span class="oktxt">✓ 结构化出文正常（' +
        (result.time_ms / 1000).toFixed(1) + "s" + (extra.length ? " · " + extra.join(" · ") : "") +
        "）</span>";
      toast("连接与出文正常（" + (result.time_ms / 1000).toFixed(1) + "s）");
    } else {
      if (resultEl) resultEl.innerHTML = '<span class="errtxt">✗ ' + escHtml(result.error || "连接失败") +
        (result.hint ? '<br><span class="hint-inline">' + escHtml(result.hint) + "</span>" : "") + "</span>";
    }
  } catch(e) {
    console.error("testLlmConnection failed:", e);
    if (resultEl) resultEl.innerHTML = '<span class="errtxt">✗ ' + escHtml(e.message) + "</span>";
  } finally {
    btn.disabled = false;
    btn.textContent = "🔗 测试连接与出文";
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

// ==================== U5：历史战绩 + 中奖回执 ====================

// 首页「历史战绩」卡：真实已对照样本的诚实表现
// 数据源：/api/eval/cumulative -> {methods:[{method, ..., issues, tickets, rows:[{issue,...}]}], sample_limit}
//
// 口径说明（U9 修正）：
//   · `sample_limit` 是**查询上限**（本页请求 120），不等于实际已累积的样本量；
//     卡片标题必须报「实际去重期数」，否则在样本只有 1 期时会谎称已有 120 期。
//   · `g.issues` 是**每个方法各自**的期数，跨方法直接相加会把同一期重复计 N 次
//     （6 个方法、1 期 → 错报 6 期），故用 rows 里的 issue 去重求并集。
const TRACK_METHOD_MIN_ISSUES = 60;   // 与后端 LOTT_METHOD_RECOMMENDATION_MIN_ISSUES 对齐：低于此不做方法优劣结论

// V4 诚实口径：任何「看起来像能力」的比率都必须带样本量。
// 后端阈值 LOTT_METHOD_RECOMMENDATION_MIN_ISSUES（config.METHOD_RECOMMENDATION_MIN_ISSUES=60）
// 通过 /api/info 的 method_min_issues 下发，前端不在第二处硬编码数字。
let METHOD_MIN_ISSUES = TRACK_METHOD_MIN_ISSUES;

/** 样本量是否足够下结论（统一入口，避免各处各写一个门槛）。 */
function sampleEnough(n) { return Number(n || 0) >= METHOD_MIN_ISSUES; }

/** 带样本量的指标：样本不足时**不**单独展示百分比，而是给出样本量与「不构成结论」。 */
function metricWithSample(label, value, n, opts) {
  const o = opts || {};
  const enough = sampleEnough(n);
  const shown = enough ? (o.text != null ? o.text : pct(value)) : "样本不足";
  const sub = "样本 " + Number(n || 0) + " 期" + (enough ? "" : " · 不构成结论");
  return '<div class="track-item"><div class="k">' + escHtml(label) + '</div><div class="v' +
    (enough ? "" : " insufficient") + '">' + shown +
    '<span class="sub">' + escHtml(sub) + '</span></div></div>';
}

/** 样本不足时的统一提示文案（前端唯一出处）。 */
// 样本不足的统一定性文案（V4 诚实口径）。内联进各渲染器时保持逐字一致，
// 便于结构守卫测试直接断言字面量、也避免同一句话出现两种措辞。
const INSUFFICIENT_NOTE = "样本不足，不构成结论";

function renderTrackRecord(cumulative) {
  const el = $("#trackRecord");
  const tag = $("#trackRecordTag");
  if (!el) return;
  const groups = (cumulative && cumulative.methods) || [];
  const cap = Number((cumulative && cumulative.sample_limit) || 0);
  if (!groups.length) {
    if (tag) tag.textContent = "尚无已对照样本 · 诚实口径（双色球独立随机，无超额保证）";
    el.innerHTML = '<div class="track-note">暂无逐注评估事实：开奖后到「更多 → 系统 → 评估报告」点「在线对照」生成累计数据。</div>';
    return;
  }
  // 汇总口径：跨方法聚合（总注数加权），给出一个诚实的系统整体数字
  let tTickets = 0, tRed = 0, tBlue = 0, tGe5 = 0, tReward = 0;
  let bestMethod = null, bestScore = -Infinity;
  let bestLevel = 0;
  const issueSet = new Set();          // 真实去重期数（跨方法并集）
  groups.forEach(g => {
    (g.rows || []).forEach(r => { if (r && r.issue != null) issueSet.add(String(r.issue)); });
    tTickets += (g.tickets || 0);
    tRed += (g.red_hits_mean || 0) * (g.tickets || 0);
    tBlue += (g.blue_hit_rate || 0) * (g.tickets || 0);
    tGe5 += (g.prize_rate_ge5 || 0) * (g.tickets || 0);
    tReward += (g.reward_total || 0);
    // 从逐注 rows 提取历史最好奖级
    (g.rows || []).forEach(r => { if ((r.prize_level || 0) > bestLevel) bestLevel = r.prize_level; });
    // 最好方法：综合红球命中 + 蓝球命中
    const sc = (g.red_hits_mean || 0) + (g.blue_hit_rate || 0) * 6;
    if (sc > bestScore) { bestScore = sc; bestMethod = g; }
  });
  const nIssues = issueSet.size;
  const blueRate = tTickets ? tBlue / tTickets : 0;
  const ge5Rate = tTickets ? tGe5 / tTickets : 0;
  const redMean = tTickets ? tRed / tTickets : 0;
  const bestLvl = bestLevel > 0 ? (PRIZE_NAME[bestLevel] || ("等" + bestLevel)) : null;

  // 标题报实际样本量；查询上限与「未达上限」的实情写进脚注，避免误读成已有 120 期
  if (tag) {
    tag.textContent = "已对照 " + nIssues + " 期 · " + tTickets + " 注 · 诚实口径（双色球独立随机，无超额保证）";
  }
  const enoughSample = sampleEnough(nIssues);
  // V4：期数是「样本量」本身，不能当作能力指标；比率类指标一律经 metricWithSample，
  // 样本不足时显示「样本不足」而非一个看起来像能力的百分比（真实事故：1 期 10 注的
  // 10.0% 蓝球命中率被当作能力展示，并对照理论值 6.25% 暗示系统有水平）。
  el.innerHTML =
    '<div class="track-metrics">' +
      '<div class="track-item"><div class="k">已对照期数</div><div class="v">' + nIssues +
        '<span class="sub">查询上限 ' + (cap || "-") + ' 期</span></div></div>' +
      metricWithSample("红球命中均值", null, nIssues, { text: fmt(redMean, 2) + " /6" }) +
      metricWithSample("蓝球命中率", blueRate, nIssues) +
      metricWithSample("≥五等奖率", ge5Rate, nIssues) +
      '<div class="track-item"><div class="k">累计奖金</div><div class="v">¥' + Number(tReward || 0).toFixed(0) +
        '<span class="sub">' + tTickets + ' 注 · 期望为负</span></div></div>' +
      (bestLvl ? '<div class="track-item"><div class="k">最好奖级</div><div class="v pos">' + bestLvl +
        '<span class="sub">单期波动，非能力</span></div></div>' : '') +
    '</div>' +
    // 样本不足时明确写出来，不让 5 个百分比并排站着暗示「系统有水平」
    (enoughSample ? '' :
      '<div class="track-note warn">⚠️ 样本不足，不构成结论：当前仅 ' + nIssues +
      ' 期已对照样本，远低于方法比较门槛 ' + METHOD_MIN_ISSUES +
      ' 期。上表比率是随机波动的读数，<b>不能</b>作为系统优劣或预测能力的证据。</div>') +
    '<div class="track-note">' +
      (bestMethod
        ? (enoughSample
            ? '最佳方法 <code>' + escHtml(bestMethod.method) + '</code>：红球 ' + fmt(bestMethod.red_hits_mean, 2) +
              ' · 蓝球 ' + pct(bestMethod.blue_hit_rate) + ' · 奖金 ¥' + Number(bestMethod.reward_total || 0).toFixed(0) +
              '。（仍需随机性检验确认非波动，见「随机性检验」卡。）'
            : '样本仅 ' + nIssues + ' 期，<b>远不足以比较方法优劣</b>（后端筛查阈值 ' + TRACK_METHOD_MIN_ISSUES +
              ' 期），故此处不给出「最佳方法」结论。')
        : '') +
      '双色球为独立随机事件，系统不承诺超过随机基线；此处为真实开奖对照的诚实数字。' +
      (nIssues < (cap || 0)
        ? '（当前累计 ' + nIssues + ' 期，尚未达到查询上限 ' + (cap || "-") +
          ' 期——样本会随每期开奖后的「在线对照」自动累积。）'
        : '') +
    '</div>';
}

function renderTrackRecordError(e) {
  const el = $("#trackRecord");
  const tag = $("#trackRecordTag");
  if (tag) tag.textContent = "样本数据加载失败 · 诚实口径（双色球独立随机，无超额保证）";
  if (el) el.innerHTML = '<div class="track-note">无法读取累计评估数据（' +
    escHtml((e && e.message) || "未知错误") + '）。此处不代表样本为 0，请刷新重试。</div>';
}

// ==================== V4：随机性检验卡 ====================
//
// 这张卡回答用户最根本的疑问：「为什么准确率上不去」。
// 结论来自 /api/rigor/randomness（后端 lottery/rigor.py）：
//   · verdict —— 这批开奖数据与「均匀随机」是否可区分
//   · min_detectable_effect —— 在当前 N 期样本量下，能检出的最小提升幅度
// 关键口径：若数据与随机不可区分，那么任何「命中率高于理论值」都只是波动。
// 依赖缺失 / 检验失败时降级为明确提示，绝不伪装成「已通过检验」。

const RIGOR_VERDICT_TEXT = {
  consistent_with_random: { t: "与均匀随机一致", cls: "ok" },
  deviation:              { t: "检出偏离",       cls: "warn" },
  insufficient_data:      { t: "样本不足，未结论", cls: "warn" },
};

function renderRigorCard(report) {
  const el = $("#rigorBody");
  const tag = $("#rigorTag");
  if (!el) return;
  const r = report || {};
  const v = RIGOR_VERDICT_TEXT[r.verdict] || { t: r.verdict || "未知", cls: "" };
  const n = Number(r.n_draws || r.n_valid || 0);
  const mde = r.min_detectable_effect || {};
  if (tag) tag.textContent = "n=" + n + " 期 · " + v.t;

  const lift = (mde.relative_lift != null)
    ? (Number(mde.relative_lift) * 100).toFixed(1) + "%"
    : "-";
  const hitPp = (mde.hit_rate_pp != null) ? Number(mde.hit_rate_pp).toFixed(2) : "-";

  // 唯一的「结论句」：数据与随机不可区分 ⇒ 准确率上不去是数学事实，不是调参问题
  const concl = (r.verdict === "consistent_with_random")
    ? '这批开奖数据与「均匀随机」在统计上<b>不可区分</b>。因此本系统的「命中率」只是随机波动的读数——' +
      '<b>准确率上不去不是调参问题，而是双色球本身没有可利用规律</b>。'
    : (r.verdict === "deviation")
      ? '本批数据检出了统计偏离（见下方检验明细）。偏离可能来自样本量、口径或数据本身，' +
        '需在更多期数据上复核，<b>不足以支撑「系统能预测」的结论</b>。'
      : '当前样本量下检验无法给出结论，请勿据下方比率判断系统能力。';

  el.innerHTML =
    '<div class="rigor-verdict">' +
      '<span class="rigor-badge ' + v.cls + '">' + escHtml(v.t) + '</span>' +
      '<span class="rigor-n">样本量 ' + n + ' 期</span>' +
    '</div>' +
    '<div class="track-note rigor-concl">' + concl + '</div>' +
    '<div class="rigor-mde">在当前 <b>' + n + ' 期样本量下，能检出的最小提升</b>为：相对随机基线 ≥ <b>' + lift +
      '</b>（命中率提升 ≥ ' + hitPp + ' 个百分点）才可能被检出（α=' + (r.alpha != null ? r.alpha : "-") +
      '，功效 ' + (mde.power != null ? mde.power : "-") + '）。<br>' +
      escHtml(mde.note_zh || "低于该幅度的「优势」与随机波动无法区分。") + '</div>' +
    renderRigorTests(r.tests) +
    '<div class="track-note">双色球为独立随机事件，本系统不承诺中奖率。' +
      '检验方法：' + escHtml(r.method || "-") + '</div>';
}

function renderRigorTests(tests) {
  if (!tests || typeof tests !== "object") return "";
  const rows = Object.keys(tests).map(k => {
    const t = tests[k] || {};
    const p = (t.p_value != null) ? Number(t.p_value).toFixed(4) : "-";
    const sig = (t.p_value != null && Number(t.p_value) < 0.05);
    return '<tr><td>' + escHtml(k) + '</td><td>' + (t.statistic != null ? fmt(t.statistic, 3) : "-") +
      '</td><td class="' + (sig ? "warn" : "") + '">' + p + '</td><td>' +
      (sig ? "偏离" : "与随机一致") + '</td></tr>';
  }).join("");
  if (!rows) return "";
  return '<details class="rigor-tests"><summary>检验明细（' + Object.keys(tests).length + ' 项）</summary>' +
    '<div class="scroll"><table><thead><tr><th>检验</th><th>统计量</th><th>p 值</th><th>判定</th></tr></thead>' +
    '<tbody>' + rows + '</tbody></table></div>' +
    '<div class="note">p 值经多重检验校正后判定；单项 p&lt;0.05 在多项检验下属预期波动。</div></details>';
}

function renderRigorUnavailable(reason) {
  const el = $("#rigorBody");
  const tag = $("#rigorTag");
  if (tag) tag.textContent = "检验模块未就绪";
  if (el) el.innerHTML = '<div class="track-note">随机性检验暂不可用（' +
    escHtml(reason || "依赖未就绪") + '）。这不代表「检验通过」，也不代表系统有能力；' +
    '在检验可用前，请勿把任何命中率当作预测能力。</div>';
}

// 策略显著性：把「某方法看起来准」交给多重检验校正来判断
async function renderRigorSignal(strategy, window) {
  const el = $("#rigorSignal");
  if (!el) return;
  el.innerHTML = '<div class="track-note">检验中…</div>';
  try {
    const s = await api("/api/rigor/signal?strategy=" + encodeURIComponent(strategy) +
                        "&window=" + encodeURIComponent(window));
    const survives = !!s.survives_correction;
    el.innerHTML =
      '<div class="track-note">策略 <code>' + escHtml(strategy) + '</code> · 窗口 ' + (s.window || window) +
        ' 期 · 对照 ' + (s.n_trials || 0) + ' 期<br>' +
        '观察均值 ' + fmt(s.observed_mean, 4) + ' vs 随机基线 ' + fmt(s.baseline_mean, 4) +
        '（相对 ' + ((s.relative_lift || 0) * 100).toFixed(2) + '%）<br>' +
        '校正后 p = ' + (s.adjusted_p != null ? Number(s.adjusted_p).toFixed(4) : "-") + ' → ' +
        (survives
          ? '<b class="warn">在校正后仍显著</b>（仍需样本外复核）'
          : '<b>未通过多重检验校正</b>：该「优势」与随机波动无法区分') +
        (s.sample_note ? '<br>' + escHtml(s.sample_note) : '') + '</div>';
  } catch (e) {
    el.innerHTML = '<div class="track-note">显著性检验不可用（' +
      escHtml((e && e.message) || "未知错误") + '）。不构成「通过检验」。</div>';
  }
}

// V23 / L3：前向回测读数——把「我们试过了，数据说没有」变成可复核的事实。
// 它必须显示**结论**，而不是只丢一堆数字让人自己猜。
async function loadWalkForward() {
  const tag = $("#wfTag"), body = $("#wfBody");
  if (!body) return;
  try {
    const r = await api("/api/rigor/walk_forward?method=stat:freq&color=blue");
    const z = (r.z_median === null || r.z_median === undefined) ? "—" : Number(r.z_median).toFixed(2);
    const passed = !!r.passed;
    tag.textContent = passed ? "过线" : "未过线";
    tag.style.color = passed ? "var(--green)" : "var(--muted)";
    const rows = [
      ["命中率", r.hit_rate === null || r.hit_rate === undefined
        ? "—" : (r.color === "red" ? Number(r.hit_rate).toFixed(3) + " 个/期"
                                   : (Number(r.hit_rate) * 100).toFixed(2) + "%")],
      ["随机基线", r.baseline === null || r.baseline === undefined
        ? "—" : (r.color === "red" ? Number(r.baseline).toFixed(3) + " 个/期"
                                   : (Number(r.baseline) * 100).toFixed(2) + "%")],
      ["z（多种子中位数）", z],
      ["门槛", String(r.z_min)],
      ["评估注数", String(r.n || 0)],
      ["各次 z", (r.z_by_seed || []).join(" / ") || "—"],
    ];
    const kv = rows.map(([k, v]) =>
      '<div class="kv"><span class="muted">' + escHtml(k) + '</span><b>' +
      escHtml(v) + '</b></div>').join("");
    const headline = '<div style="margin:8px 0;font-weight:600;color:' +
      (passed ? "var(--green)" : "var(--muted)") + '">' + escHtml(r.headline_zh || "") + '</div>';
    const honesty = r.honesty_zh
      ? '<div class="muted" style="font-size:12px;margin-top:6px">' + escHtml(r.honesty_zh) + '</div>' : "";
    body.innerHTML = headline + kv + honesty;
  } catch (e) {
    tag.textContent = "不可用";
    body.innerHTML = '<div class="muted">' + escHtml(e.message || "加载失败") + '</div>';
  }
}

// V27：方差优化选号卡。
//
// 它展示的是「票单结构好不好」，**不是**「号码准不准」。
// 三个数字必须同时出现（缺一即误导）：
//   ① var_ratio 降了多少（真实增益，任何人可用 ticket_distribution 复核）
//   ② 并集覆盖 / P(≥4红)（副产物，同样可复核）
//   ③ 期望收益没变（EV 恒为 n × ¥1.0458）
async function loadVariancePick(nTickets) {
  const tag = $("#vpTag"), body = $("#vpBody"), honesty = $("#vpHonesty");
  const seedNote = $("#vpSeedNote");
  if (!body) return;
  const n = Math.max(1, Math.min(50, parseInt(nTickets, 10) || 10));
  body.innerHTML = '<div class="muted">计算中…</div>';
  tag.textContent = "计算中…";
  tag.style.color = "var(--muted)";
  try {
    const r = await api("/api/portfolio/variance-pick?n_tickets=" + n +
                        "&seed=42&with_baseline=true&baseline_samples=60");
    if (!r || r.ok === false) {
      tag.textContent = "不可用";
      body.innerHTML = '<div class="muted">' +
        escHtml((r && r.error) || "计算失败") + '</div>';
      return;
    }
    const s = r.summary || {};
    const b = (r.baseline && r.baseline.ok) ? r.baseline : null;
    const imp = b ? b.improvement : null;
    const varTxt = Number(s.var_ratio || 0).toFixed(3);
    const varImp = imp ? (imp.var_ratio_pct > 0 ? "+" : "") + imp.var_ratio_pct + "%" : "—";
    const p4 = s.p_at_least_r ? s.p_at_least_r["4"] : null;
    const p4Imp = imp && imp.p_at_least_r_pct
      ? (imp.p_at_least_r_pct["4"] > 0 ? "+" : "") + imp.p_at_least_r_pct["4"] + "%"
      : "—";
    const rows = [
      ["注数", String(s.n || n)],
      ["红球并集", String(s.union_coverage || 0) + " / 33" +
        (imp ? "（随机均值 " + b.random_baseline.union_coverage_mean.toFixed(1) + "）" : "")],
      ["方差比 var_ratio", varTxt + (imp ? "（比随机 " + varImp + "）" : "")],
      ["P(≥1 注中 ≥4 红)", p4 === null ? "—" : (Number(p4) * 100).toFixed(3) + "%" +
        (imp ? "（" + p4Imp + "）" : "")],
      ["P(≥1 注中 ≥3 红)", s.p_at_least_r ? (Number(s.p_at_least_r["3"]) * 100).toFixed(2) + "%" : "—"],
      ["期望命中数", Number(s.expected_hits || 0).toFixed(3) + "（= n × 12/11，恒等式）"],
    ];
    const kv = rows.map(function (kvrow) {
      return '<div class="kv"><span class="muted">' + escHtml(kvrow[0]) + '</span><b>' +
        escHtml(kvrow[1]) + '</b></div>';
    }).join("");
    const tickets = (r.tickets || []).map(function (t) {
      return '<div class="ticket-line" style="font-family:monospace">' +
        (t.reds || []).map(function (x) { return String(x).padStart(2, "0"); }).join(" ") +
        '</div>';
    }).join("");
    body.innerHTML = kv +
      '<div style="margin-top:10px"><b>红球组合（蓝球请自行选择）</b></div>' +
      '<div style="margin-top:4px">' + tickets + '</div>';
    tag.textContent = b ? "方差 " + varImp : "已生成";
    tag.style.color = "var(--green)";
    if (honesty) {
      const h = r.honesty || {};
      honesty.innerHTML = escHtml(h.ev_unchanged_zh || "") + " " +
        escHtml(h.p_at_least_r_note_zh || "") + " " +
        escHtml(h.no_prediction_zh || "");
    }
    if (seedNote) seedNote.textContent = "种子 " + r.seed + " · 同种子可复现";
  } catch (e) {
    tag.textContent = "不可用";
    body.innerHTML = '<div class="muted">' + escHtml((e && e.message) || "加载失败") + '</div>';
  }
}

async function loadRigor() {
  const card = $("#rigorCard");
  try {
    const rep = await api("/api/rigor/randomness");
    if (rep && rep.available === false) {
      renderRigorUnavailable(rep.error || "依赖未就绪");
      return;
    }
    renderRigorCard(rep);
    renderRigorSignal("hot", 100);
  } catch (e) {
    // 503（模块未就绪）/ 网络错误：明确降级，绝不静默当成「检验通过」
    renderRigorUnavailable((e && e.message) || "未知错误");
  }
}

// 首页「上期开奖回执」：
//   ① 上期系统推荐的每一组号码 + 每注命中情况（红中 X/6 · 蓝 ✓/— · 奖级）
//   ② 你在本机复制的注（localStorage myCopy）中了几块
// 数据源：/api/predictions/history（含 issue/date/actual/predictions）；命中本地对照，
// 与「历史回放」页同一套口径，不依赖后端预计算字段。
async function renderWinReceipt() {
  const card = $("#winReceiptCard");
  const box = $("#winReceipt");
  const tag = $("#winReceiptTag");
  if (!card || !box) return;
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem("myCopy") || "null"); } catch (_) {}
  let hist = [];
  try { hist = (await api("/api/predictions/history?limit=50")) || []; } catch (_) { hist = []; }

  // 「上期」= 最近一期「已开奖 且有系统推荐记录」的期号
  const prev = hist.find(h => h.actual && (h.predictions || []).length);
  const hasCopy = !!(saved && saved.issue && (saved.tickets || []).length);
  if (!prev && !hasCopy) { card.classList.add("hidden"); return; }

  const parts = [];
  let tagText = "";
  if (prev) {
    tagText = prev.issue + (prev.date ? " · " + prev.date : "");
    parts.push(renderReceiptRecommendations(prev));
  }
  if (hasCopy) {
    if (!tagText) tagText = saved.issue;
    parts.push(renderMyCopyReceipt(saved, hist.find(h => h.issue === saved.issue)));
  }
  parts.push('<div class="receipt-note">双色球每期独立随机，未中奖是常态；系统不承诺优于随机基线。请理性投注。</div>');

  card.classList.remove("hidden");
  if (tag) tag.textContent = tagText;
  box.innerHTML = parts.join("");
}

// 上期系统推荐的逐注命中明细
function renderReceiptRecommendations(item) {
  const act = item.actual || {};
  const actReds = act.reds || [];
  const rows = (item.predictions || []).filter(t => t && t.reds && t.reds.length).map(t => {
    const rh = t.reds.filter(r => actReds.includes(r)).length;
    const bh = (t.blue === act.blue) ? 1 : 0;
    const lvl = localPrizeLevel(rh, bh);
    return { t: t, rh: rh, bh: bh, lvl: lvl, reward: PRIZE_CASH[lvl] || 0 };
  });
  if (!rows.length) return "";
  const n = rows.length;
  const meanRed = rows.reduce((a, r) => a + r.rh, 0) / n;
  const nBlue = rows.filter(r => r.bh).length;
  const winCount = rows.filter(r => r.lvl > 0).length;
  const reward = rows.reduce((a, r) => a + r.reward, 0);
  const best = Math.max.apply(null, rows.map(r => r.lvl));

  const actualBalls = actReds.map(r => "<span class='ball red sm'>" + pad2(r) + "</span>").join("") +
    " <span class='ball blue sm' style='margin-left:6px'>" + pad2(act.blue) + "</span>";

  const tickets = rows.map((r, i) => {
    const balls = r.t.reds.map(x =>
      "<span class='ball red sm" + (actReds.includes(x) ? " hit" : "") + "'>" + pad2(x) + "</span>").join("");
    const blue = "<span class='ball blue sm" + (r.bh ? " hit" : "") + "'>" + pad2(r.t.blue) + "</span>";
    const badge = r.lvl
      ? "<span class='prize-badge lv" + r.lvl + "'>" + (PRIZE_NAME[r.lvl] || ("等" + r.lvl)) + " ¥" + r.reward + "</span>"
      : "<span class='prize-badge none'>未中奖</span>";
    const method = r.t.method ? "<span class='badge'>" + escHtml(r.t.method) + "</span>" : "";
    const redCls = r.rh >= 4 ? "var(--gold)" : r.rh >= 2 ? "var(--green)" : "var(--muted)";
    return "<div class='ticket small replay-ticket'><div class='row1'>" +
      "<span class='rk'>#" + (i + 1) + "</span>" +
      "<div class='balls'>" + balls + " " + blue + "</div>" + method +
      "<span class='meta'>红中 <b style='color:" + redCls + "'>" + r.rh + "</b>/6 · 蓝 " +
      (r.bh ? "<b style='color:var(--gold)'>✓</b>" : "—") + "</span>" + badge +
      "</div></div>";
  }).join("");

  return '<div class="receipt-line">上期实际开奖 <span class="balls" style="display:inline-flex">' +
      actualBalls + "</span></div>" +
    '<div class="receipt-line">系统上期推荐 <b>' + n + "</b> 注：红球命中均值 <b>" + fmt(meanRed, 2) +
      "</b>/6 · 蓝球命中 <b>" + nBlue + "</b>/" + n + " · 中奖 <b>" + winCount + "</b>/" + n +
      (reward > 0 ? " · 合计 ¥" + reward.toLocaleString() : "") +
      (best > 0 ? " · 最好 " + (PRIZE_NAME[best] || ("等" + best)) : "") + "</div>" +
    '<div class="replay-list receipt-list">' + tickets + "</div>";
}

// 「我复制的注」回执（本机记录）
function renderMyCopyReceipt(saved, item) {
  const my = saved.tickets || [];
  if (!item || !item.actual) {
    return '<div class="receipt-line receipt-lose">你复制的 ' + my.length + " 注（期号 " +
      escHtml(saved.issue) + "）尚未开奖，开奖后自动回执。</div>" +
      '<div class="receipt-note">已记录在本机，开奖后可在此查看是否中奖。</div>';
  }
  const act = item.actual;
  const actReds = act.reds || [];
  let winCount = 0, reward = 0, best = 0, bestRed = 0;
  my.forEach(t => {
    const rh = (t.reds || []).filter(r => actReds.includes(r)).length;
    const bh = (t.blue === act.blue) ? 1 : 0;
    const lvl = localPrizeLevel(rh, bh);
    if (lvl > 0) { winCount++; reward += (PRIZE_CASH[lvl] || 0); }
    if (lvl > best) best = lvl;
    if (rh > bestRed) bestRed = rh;
  });
  return "<div class='receipt-line receipt-copy-tag'>你的复制注（期号 " + escHtml(saved.issue) + "）：</div>" +
    (winCount > 0
      ? '<div class="receipt-line receipt-win">🎉 你复制的 ' + my.length + " 注中，" + winCount +
        " 注中奖，共 ¥" + reward.toFixed(0) +
        (best ? "，最好 " + (PRIZE_NAME[best] || ("等" + best)) : "") + "。</div>"
      : '<div class="receipt-line receipt-lose">你复制的 ' + my.length +
        " 注本期未中奖（红球最多命中 " + bestRed + " 红）。</div>");
}

// ==================== 加载 ====================

/**
 * D-19：能力清单卡（`/api/capabilities`）。
 *
 * ## 硬约束（不做就是负价值）
 *
 * `can` 与 `cannot` **必须同屏、同视觉权重、不可折叠**。
 * 只展示 `can` 不展示 `cannot`，这张卡就从「让用户看到系统做不到什么」
 * 变成「吹牛工具」—— 比不接更糟，因为它有「官方认证」的外观。
 *
 * ## 为什么两栏用 grid 而不是两个独立卡
 *
 * 独立卡可以由任意一方先渲染完、另一方后渲染，中间出现
 * 「只看到能做到」的窗口期。**同屏 = 同一个容器**，这在结构上消除那个窗口。
 *
 * ## 数值来源
 *
 * `constants` 段全部由后端实时读取（`engine._single_ticket_ev` 等），
 * 前端**不写死任何一个数**，只做格式化展示。
 */
async function loadCapabilities() {
  const body = $("#capBody");
  if (!body) return;
  try {
    const d = await api("/api/capabilities");
    renderCapabilities(d);
  } catch (err) {
    // 降级：不清空、不白屏，明确说「这张卡没读到」而不是「系统没有能力」
    $("#capTag").textContent = "读取失败";
    body.innerHTML = '<div class="note">能力清单暂时读不到（' +
      escHtml(err && err.message ? err.message : "网络错误") +
      '）。这不代表系统不可用，其他卡片仍可正常使用。</div>';
  }
}

function renderCapabilities(d) {
  const can = (d && d.can) || [];
  const cannot = (d && d.cannot) || [];
  const cons = (d && d.constants) || {};
  const degraded = (d && d.degraded_items) || [];

  $("#capTag").textContent = can.length + " 项能做 · " + cannot.length + " 项不能做";

  const canHtml = can.map(function (it) {
    return '<div class="cap-item">' +
      '<div class="cap-title">✅ ' + escHtml(it.title || it.id) + '</div>' +
      (it.honest_bound ? '<div class="cap-bound">边界：' + escHtml(it.honest_bound) + '</div>' : '') +
      (it.evidence_script ? '<div class="cap-evi">证据：<code>' + escHtml(it.evidence_script) + '</code></div>' : '') +
      '</div>';
  }).join("");

  const cannotHtml = cannot.map(function (it) {
    return '<div class="cap-item cap-item-no">' +
      '<div class="cap-title">🚫 ' + escHtml(it.title || it.id) + '</div>' +
      (it.why_not ? '<div class="cap-bound">原因：' + escHtml(it.why_not) + '</div>' : '') +
      '</div>';
  }).join("");

  // 常量段：全部实时值，前端口径与后端一致
  const consRows = [
    ["单注返奖比", cons.ev_single != null ? (cons.ev_single * 100).toFixed(2) + "%" : "—"],
    ["六等奖返奖比", cons.ev_sixth != null ? (cons.ev_sixth * 100).toFixed(3) + "%" : "—"],
    ["显著性水平 α", cons.alpha != null ? String(cons.alpha) : "—"],
    ["单球检验空间", cons.single != null ? String(cons.single) : "—"],
    ["交叉检验空间", cons.cross != null ? String(cons.cross) : "—"],
  ].map(function (r) {
    return '<span class="cap-kv"><b>' + escHtml(r[0]) + '</b> ' + escHtml(r[1]) + '</span>';
  }).join("");

  // 降级项：单一判据 = degraded_items 是否为空（不另外判断 full_ev 等，
  // 避免与后端形成两处判据，见 D-19 纪律「同一事实只允许一处判据」）
  const degHtml = degraded.length
    ? '<div class="cap-degraded"><b>当前有 ' + degraded.length + ' 项能力受限：</b>' +
      degraded.map(function (i) {
        return '<div class="cap-deg-item">· ' + escHtml(i.reason || i.id) +
          (i.how_to_restore ? '<br><span class="cap-deg-fix">恢复方式（给运维）：' +
            escHtml(i.how_to_restore) + '</span>' : '') + '</div>';
      }).join("") + '</div>'
    : '';

  body.innerHTML =
    '<div class="cap-grid">' +
      '<div class="cap-col"><div class="cap-head">能做到</div>' + canHtml + '</div>' +
      '<div class="cap-col"><div class="cap-head cap-head-no">做不到</div>' + cannotHtml + '</div>' +
    '</div>' +
    '<div class="cap-consts"><b>关键口径（后端实时读取）：</b>' + consRows + '</div>' +
    degHtml;
}

async function loadAll() {
  // 强制鉴权且未登录时不再打接口（登录成功后会再次调用本函数）
  if (AUTH.loaded && needsLogin()) return;
  try {
    // 防缓存：预测/健康这类"随时会变"的接口带时间戳，确保刷新页面拿到的是最新数据
    // （用户反馈：重新生成后刷新页面又显示 15 日的旧数据）。
    const bust = "?_ts=" + Date.now();
    const [feat, pats, preds, health] = await Promise.all([
      api("/api/features"),
      api("/api/patterns"),
      api("/api/predictions/last" + bust),
      api("/api/health" + bust),
    ]);
    renderStats(feat);
    renderPatterns(pats.items, pats.summary);
    allPatterns = pats.items;
    if (preds.tickets && preds.tickets.length) {
      // V7：带上后端返回的生成时间 / 数据基准 / 规律使用情况（此前这几项被丢弃，
      // 用户因此看不到「预测是什么时候生成的」）
      renderPredictions(Object.assign({note: ""}, preds));
    } else {
      // 尚无该期预测：仍然展示「上期数据基准 + 数据是否落后 + 规律快照」，避免空页让人误解
      renderPredictions({issue: preds.issue, tickets: [], note: "", data_basis: preds.data_basis,
                         generated_at: null, rules: null});
      $("#predList").innerHTML = '<div class="note">本期（' + escHtml(preds.issue || "") +
        '）还没有生成候选注单，点右上角「🎯 生成候选注单」。</div>';
      renderCountdown();
    }
    await renderHistory();
    // D3：首屏「诚实收益边界」卡在页面加载时就用实时值填充。
    // 不挂在 `loadV10Cards()` 上 —— 那个入口只在用户切到「评估报告 → 综合概览」时才触发，
    // 而首屏卡是**首屏**的内容，不能等用户点进去才有真值。
    // `loadBlueCoverage()` 内部已吞掉一切异常并切「示例值」标注，这里不必再包 try。
    loadBlueCoverage();
    loadCapabilities();   // D-19：能力清单（can/cannot 同屏不可折叠）
    if (health) {
      $("#dataStatus").textContent = health.issues + " 期 | " + (health.max_issue || "");
      if (health.version) {
        const ver = "v" + health.version;
        const vb = $("#verBadge"), vt = $("#verText");
        if (vb) { vb.textContent = ver; vb.title = "系统版本 " + ver + " · v4.12.0 V32"; }
        if (vt) vt.textContent = ver;
      }
    }
    try {
      const ev = await api("/api/eval");
      renderOnline(ev);
      loadMethodBreakdown();   // V10-04b：方法战绩按族分列
      const cumulative = await api("/api/eval/cumulative?limit=120");
      renderCumulativeEval(cumulative);
      renderTrackRecord(cumulative);   // U5/U9 历史战绩
      loadMethodRecommendations();
    } catch(e) {
      // 取数失败时明确告知，别让标题停在「加载中…」、更不要假装样本为 0
      renderTrackRecordError(e);
    }
    loadRigor();   // V4：随机性检验卡（依赖缺失时自行降级，不影响其它区块）
  loadWalkForward();  // V23：前向回测卡（同样自降级；失败不阻塞首屏）
  loadVariancePick(10);  // V27：方差优化选号卡（自降级；不阻塞首屏）
  (function bindVariancePick(){
    const btn = $("#vpRun"), inp = $("#vpN");
    if (btn && inp) btn.addEventListener("click", function(){ loadVariancePick(inp.value); });
  })();
    initMlStatus();
    try {
      const llmRep = await api("/api/eval/llm/latest");
      renderLlmEval(llmRep);
    } catch(e) {}
    loadMethodsConfig();
    loadMiningReports();
    renderWinReceipt();
    // V4：样本量门槛以 /api/info 的后端常量为准（前后端同一个数字）
    try {
      const info = await api("/api/info");
      if (info && Number(info.method_min_issues) > 0) {
        METHOD_MIN_ISSUES = Number(info.method_min_issues);
      }
      // V10-04b：用后端 channels.llm_can_pick 回填「启用 LLM」默认值
      await applyChannelDefaults(info);
      // V10-10：本期推荐页的出号通道口径徽标（families + llm_can_pick）
      renderChannelBadges(info && info.channels);
      const cum = await api("/api/eval/cumulative?limit=120");
      renderTrackRecord(cum);   // 门槛更新后重绘，确保展示与后端一致
    } catch(e) {}
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

/**
 * 启动流程：**先探测身份，再决定是否加载数据**。
 *
 * 顺序不能反过来：`loadAll()` 会并发打 4 个受保护接口，未登录时全部 401，
 * 4 条错误 toast 叠加会把登录层遮住，用户看到的是"一堆报错"而不是"请登录"。
 */
(async function boot() {
  await loadAuthState();
  if (needsLogin()) {
    showAuthOverlay(true, "本系统已启用多用户，请登录后使用。");
    return;                        // 未登录：不加载数据，避免连环 401 弹窗
  }
  // 灰度期（未启用鉴权）或已登录：正常加载
  loadAll();
})();

// ==================== V6：规律时效（阶段有效性）+ 多维度发现 + LLM 规律工坊 ====================

let _rgOverview = null;
let _eraMode = localStorage.getItem("eraMode") || "adaptive";   // "adaptive" | "calendar"
let _rgFamilies = null;
let _rgBusy = false;

/** 「更多 → 规律时效」入口：切到评估页并选中对应子页。 */
function openRegimeTab() {
  switchTab("eval");
  setEvalSub("regime");
  // 显式收起「更多」抽屉：否则遮罩会盖在切换后的页面上，看起来"点了没反应"
  closeMore();
  if (!_rgOverview) loadRegimeOverview(false);
}

const RG_STATUS_ORDER = ["persistent", "persistent_effect", "emerging", "fading", "stale", "flipped",
  "reversed", "noise", "no_dimension", "insufficient", "error", "unannotated"];
const RG_STATUS_COLOR = {
  persistent: "var(--green)",
  // S2：长期效应档用青蓝色，**刻意不像 persistent 的绿色**——它描述长期平均效应，
  // 不代表「当前正在生效」，颜色与文案都必须让用户一眼看出区别。
  persistent_effect: "var(--cyan, #4cc9f0)",
  emerging: "var(--gold)", fading: "var(--yellow, #d29922)",
  stale: "var(--red)", flipped: "var(--red)", reversed: "var(--muted)",
  noise: "var(--muted)", no_dimension: "var(--muted)", insufficient: "var(--muted)",
  error: "var(--red)", unannotated: "var(--muted)",
};

function rgEraCell(era, status) {
  // 单元格：年份段 + 效应 + 校正后 p（显著时才着色）
  const p = era.p_adjusted;
  const sig = (p !== null && p !== undefined && p < 0.05);
  const d = era.delta;
  const cls = sig ? (d >= 0 ? "pos" : "neg") : "flat";
  const dtxt = (d === null || d === undefined) ? "—" : (d >= 0 ? "+" : "") + Number(d).toFixed(3);
  return '<td class="' + cls + '" title="' + escHtml((era.note_zh || "")) + '">' +
    '<div class="rg-cell">' + (sig ? "★ " : "") + dtxt + '</div>' +
    '<div class="rg-cell-sub">' + (p === null || p === undefined ? "—" : "p=" + Number(p).toFixed(3)) + '</div>' +
    '</td>';
}

function renderRegimeOverview(r) {
  const el = $("#rgOverviewBody");
  const tag = $("#rgOverviewTag");
  if (!el) return;
  if (!r || r.ok === false) {
    if (tag) tag.textContent = "加载失败";
    el.innerHTML = '<div class="track-note warn">⚠️ ' + escHtml((r && r.error) || "读取失败") +
      '。<b>此处不代表没有规律</b>。</div>';
    return;
  }
  const counts = r.status_counts || {};
  const items = r.items || [];
  const eras = r.eras || [];
  const legend = r.status_legend || {};
  if (tag) tag.textContent = "共 " + items.length + " 条 · " + eras.length + " 个阶段";

  let metrics = '<div class="track-metrics">';
  metrics += '<div class="track-item"><div class="k">阶段数</div><div class="v">' + eras.length +
    '<span class="sub">' + escHtml((eras[0] && eras[0].name) || "") + " … " +
    escHtml((eras[eras.length - 1] && eras[eras.length - 1].name) || "") + '</div></div>';
  for (const st of RG_STATUS_ORDER) {
    if (!counts[st]) continue;
    metrics += '<div class="track-item"><div class="k">' + escHtml(legend[st] || st) +
      '</div><div class="v" style="color:' + (RG_STATUS_COLOR[st] || "inherit") + '">' +
      counts[st] + '</div></div>';
  }
  metrics += '</div>';

  let banner = '<div class="multi-correct-banner' + ((counts.stale || counts.flipped) ? ' warn' : '') + '">' +
    '<div class="mcb-main">用户问题「近期规律都不生效」的量化答案：' +
    '<b style="color:var(--red)">' + (counts.stale || 0) + ' 条陈旧</b> · ' +
    '<b style="color:var(--gold)">' + (counts.fading || 0) + ' 条衰减</b> · ' +
    '<b style="color:var(--green)">' + (counts.persistent || 0) + ' 条持续</b></div>' +
    '<div class="mcb-sub">判定口径：每阶段单独检验并做 BH 校正；「陈旧」= 最近一次显著距今超过 ' +
    (r.stale_after_draws || 300) + ' 期；「衰减」= 上一阶段显著但最近阶段不再显著。' +
    '阶段样本门槛 ' + (r.min_era_n || 60) + ' 期，近期加权窗口 ' + (r.recent_window || 150) + ' 期。</div>' +
    '</div>';

  let table = '<div class="scroll"><table class="battle-table rg-table"><thead><tr><th>规律</th><th>假说状态</th>';
  for (const e of eras) table += '<th>' + escHtml(e.name) + '<div class="rg-th-sub">n=' + e.n + '</div></th>';
  table += '<th>近期效应</th><th>状态判定</th></tr></thead><tbody>';
  const sorted = items.slice().sort((a, b) => {
    const ia = RG_STATUS_ORDER.indexOf(a.regime_status || "unannotated");
    const ib = RG_STATUS_ORDER.indexOf(b.regime_status || "unannotated");
    return ia - ib;
  });
  for (const it of sorted) {
    const st = it.regime_status || "unannotated";
    const rg = (it.regime || {});
    table += '<tr><td title="' + escHtml(it.statement || "") + '">' + escHtml((it.statement || "").slice(0, 42)) +
      '<div class="rg-cell-sub">' + escHtml(it.action_type || "") + '</div></td>' +
      '<td class="flat">' + escHtml(it.status || "") + '</td>';
    const erows = (it.eras || []);
    if (erows.length) {
      for (const e of erows) table += rgEraCell(e, st);
      for (let i = erows.length; i < eras.length; i++) table += '<td class="flat">—</td>';
    } else {
      for (let i = 0; i < eras.length; i++) table += '<td class="flat"><span class="insufficient-txt">无维度</span></td>';
    }
    const eff = it.recent_effect;
    table += '<td class="' + (eff === null || eff === undefined ? "flat" : (eff >= 0 ? "pos" : "neg")) + '">' +
      (eff === null || eff === undefined ? "—" : (eff >= 0 ? "+" : "") + Number(eff).toFixed(3)) + '</td>' +
      '<td style="text-align:left"><span class="rg-badge" style="border-color:' + (RG_STATUS_COLOR[st] || "var(--border)") +
      ';color:' + (RG_STATUS_COLOR[st] || "inherit") + '">' + escHtml(it.regime_label || st) + '</span>' +
      '<div class="rg-cell-sub">' + escHtml((it.regime_reason || "").slice(0, 90)) + '</div></td></tr>';
  }
  if (!sorted.length) table += '<tr><td colspan="' + (eras.length + 3) + '" class="flat">假说库为空：先跑一次「跑一轮进化」或「LLM 规律工坊」。</td></tr>';
  table += '</tbody></table></div>';

  el.innerHTML = metrics + banner + table +
    '<div class="track-note">📖 单元格 = 该阶段效应（★ 表示 BH 校正后显著）；「无维度」= 该规律不是按维度表述的（结构类/权重类），' +
    '它的效果由「实战战绩」子页的 A/B 验证给出。</div>' + '<div class="track-note">' + escHtml(r.note_zh || "") + '</div>';
}

/** 切换阶段切分口径（自适应按期数 / 自然年 9 段），切换后重算阶段画像。 */
function setEraMode(mode) {
  _eraMode = mode === "calendar" ? "calendar" : "adaptive";
  localStorage.setItem("eraMode", _eraMode);
  document.querySelectorAll("#eraSwitch button").forEach(b =>
    b.classList.toggle("on", b.dataset.era === _eraMode));
  const hint = $("#eraHint");
  if (hint) hint.textContent = _eraMode === "adaptive"
    ? "阶段切分方式：自适应 = 按固定 150 期一段（历史约 23 段、近年更密；单年切法只有 118~151 期，检测力不足）。"
    : "阶段切分方式：按自然年（2003-2008 / … / 2026，共 9 段）。旧口径，单段样本小、检测力低，仅供对照。";
  _rgOverview = null;
  loadRegimeOverview(false);
}

async function loadRegimeOverview(manual) {
  if (_rgBusy) { toast("正在计算中，请稍候"); return; }
  _rgBusy = true;
  setBusy("#btnRgLoad", "加载中…");
  const tag = $("#rgOverviewTag");
  if (tag) tag.textContent = "加载中…";
  try {
    const r = await api("/api/regime/overview?era_mode=" + encodeURIComponent(_eraMode));
    _rgOverview = r;
    renderRegimeOverview(r);
    if (manual) toast("阶段画像已刷新");
  } catch (e) {
    renderRegimeOverview({ ok: false, error: e.message });
  }
  setFree("#btnRgLoad", "▶ 加载阶段画像（约 5~20 秒）");
  _rgBusy = false;
}

async function refreshRegime() {
  if (_rgBusy) { toast("正在计算中，请稍候"); return; }
  if (!window.confirm("重算全部假说的阶段时效，会写回 data/hypotheses.json。\n预计 5~30 秒，确定继续？")) return;
  _rgBusy = true;
  setBusy("#btnRgRefresh", "重算中…");
  try {
    const r = await api("/api/regime/refresh", { method: "POST" });
    toast("已标注 " + (r.annotated || 0) + " 条（无维度 " + (r.no_dimension || 0) + " 条）");
    await loadRegimeOverview(false);
  } catch (e) { toast("重算失败: " + e.message); }
  setFree("#btnRgRefresh", "🔄 重算阶段时效（写库）");
  _rgBusy = false;
}

/* ---------- 多维度发现 ---------- */

function renderDiscovery(res) {
  const el = $("#rgDiscoveryBody");
  const tag = $("#rgDiscoveryTag");
  if (!el) return;
  if (!res || res.ok === false) {
    if (tag) tag.textContent = "失败";
    el.innerHTML = '<div class="track-note warn">⚠️ ' + escHtml((res && res.error) || "扫描失败") + '</div>';
    return;
  }
  if (tag) tag.textContent = res.n_tested + " 组 · 校正后最优 p=" +
    (res.best_quick_p_adjusted === null || res.best_quick_p_adjusted === undefined ? "—" :
      Number(res.best_quick_p_adjusted).toFixed(3));
  const sc = res.scan_summary || res;
  const _sc = res.status_count || {};
  const _pe = (_sc.persistent_effect || 0);
  let h = '<div class="multi-correct-banner' + ((_sc.persistent || _sc.emerging) ? ' warn' : '') + '">' +
    '<div class="mcb-main">扫描 <b>' + (res.n_tested || 0) + '</b> 组维度×参数 · 纯随机预期假阳性约 <b>' +
    Number(res.expected_false_positives || 0).toFixed(1) + '</b> 个 · 结论：' + escHtml(res.verdict || "") +
    (_pe ? ' · <span style="color:' + RG_STATUS_COLOR.persistent_effect + '">长期效应 ' + _pe + ' 条</span>' : "") +
    '</div>' +
    '<div class="mcb-sub">' + escHtml(res.note_zh || "") +
    '『阶段状态』是该维度在全历史上的阶段画像，不是出号建议。' +
    '（n_tested=' + (res.n_tested || 0) + ' 为本轮维度×参数检验总数，S7 新增位置/逐号/和值三维度后分母随之增大；' +
    '『长期效应』= 长期平均显著 + 近 K 段方向一致，<b>当前不一定在生效</b>，与『持续有效』不是一回事。）</div></div>';
  h += '<div class="scroll"><table class="battle-table"><thead><tr>' +
    '<th>维度</th><th>参数</th><th>n</th><th>均值</th><th>期望</th><th>差值</th><th>z</th>' +
    '<th>p</th><th>BH校正p</th><th>阶段状态</th></tr></thead><tbody>';
  for (const it of (res.items || [])) {
    const q = it.quick || {};
    const f = it.filter || {};
    const rg = it.regime || {};
    // S2：长期效应档在「阶段状态」格里并列显示方向一致性（K 段里几段同号）与
    // 长期口 Bonferroni p —— 让用户看到它靠的是什么，而不是一个孤立状态词。
    let peSub = "";
    if (it.status === "persistent_effect") {
      const pe = rg.persistent_effect || {};
      const lt = pe.long_term || {};
      const sa = pe.sign_agreement || rg.sign_agreement || {};
      const parts = [];
      if (sa.n_used !== undefined) parts.push("方向一致 " + (sa.agree || 0) + "/" + (sa.n_used || 0) + (sa.sign === "neg" ? "（负向）" : "（正向）"));
      if (lt.p_adj !== undefined) parts.push("长期p校正=" + Number(lt.p_adj).toFixed(3) + (lt.m ? "（m=" + lt.m + "）" : ""));
      if (parts.length) peSub = '<div class="rg-cell-sub" style="color:' + (RG_STATUS_COLOR.persistent_effect) + '">' + escHtml(parts.join(" · ")) + '</div>';
    }
    h += '<tr><td>' + escHtml(f.name_zh || f.key) + '<div class="rg-cell-sub">' + escHtml(f.target || "") + '</div></td>' +
      '<td class="flat" style="text-align:left">' + escHtml(JSON.stringify(f.params || {})) + '</td>' +
      '<td>' + (q.n || 0) + '</td><td>' + Number(q.mean || 0).toFixed(4) + '</td>' +
      '<td class="flat">' + Number(q.expected || 0).toFixed(4) + '</td>' +
      '<td class="' + (q.delta >= 0 ? "pos" : "neg") + '">' + (q.delta >= 0 ? "+" : "") + Number(q.delta || 0).toFixed(4) + '</td>' +
      '<td>' + Number(q.z || 0).toFixed(2) + '</td>' +
      '<td>' + (q.p === undefined ? "—" : Number(q.p).toFixed(4)) + '</td>' +
      '<td>' + (q.p_adjusted === undefined || q.p_adjusted === null ? "—" : "<b>" + Number(q.p_adjusted).toFixed(3) + "</b>") + '</td>' +
      '<td style="text-align:left"><span class="rg-badge" style="border-color:' + (RG_STATUS_COLOR[it.status] || "var(--border)") +
      ';color:' + (RG_STATUS_COLOR[it.status] || "inherit") + '">' + escHtml(it.status_zh || it.status || "") + '</span>' +
      peSub +
      '<div class="rg-cell-sub">' + escHtml((it.reason_zh || "").slice(0, 80)) + '</div></td></tr>';
  }
  h += '</tbody></table></div>';
  el.innerHTML = h;
}

async function runDiscoveryScan() {
  if (_rgBusy) { toast("正在计算中，请稍候"); return; }
  _rgBusy = true;
  setBusy("#btnDiscScan", "扫描中…");
  const tag = $("#rgDiscoveryTag");
  if (tag) tag.textContent = "扫描中…";
  try {
    const r = await api("/api/discovery/scan?top=15&refresh=1");
    renderDiscovery(r);
    toast("扫描完成：" + (r.n_tested || 0) + " 组");
  } catch (e) {
    renderDiscovery({ ok: false, error: e.message });
  }
  setFree("#btnDiscScan", "🛰️ 扫描维度空间（约 5~30 秒）");
  _rgBusy = false;
}

async function loadDiscoveryFamilies() {
  const el = $("#rgFamiliesBody");
  if (_rgFamilies) {
    el.innerHTML = el.innerHTML ? "" : renderFamilies(_rgFamilies);
    return;
  }
  try {
    const r = await api("/api/discovery/families");
    _rgFamilies = r;
    el.innerHTML = renderFamilies(r);
    fillProbeFamilies(r);
  } catch (e) { toast("维度表加载失败: " + e.message); }
}

function renderFamilies(r) {
  let h = '<div class="track-note">共 ' + (r.n_families || 0) + ' 个维度族、' + (r.n_combos || 0) + ' 组参数（这就是「可搜索的规律空间」）。' +
    '『诚实结论』列说明该维度当前能做到什么、不能做到什么——能搜到维度 ≠ 立刻有规律。</div>' +
    '<div class="scroll"><table class="battle-table"><thead><tr><th>族</th><th>维度</th><th>定义</th><th>参数组数</th><th>诚实结论</th></tr></thead><tbody>';
  for (const f of (r.families || [])) {
    h += '<tr><td>' + escHtml(f.name_zh) + '</td><td class="flat" style="text-align:left">' + escHtml(f.dim_zh) +
      '</td><td class="flat" style="text-align:left">' + escHtml(f.ratio) + '</td><td>' + f.n_param_sets +
      '</td><td class="flat" style="text-align:left">' + escHtml(f.note_zh || "—") + '</td></tr>';
  }
  return h + '</tbody></table></div>';
}

/* ---------- LLM 规律工坊 ---------- */

const LAB_VERDICT_COLOR = {
  support: "var(--green)", reject: "var(--red)", regime_only: "var(--gold)",
  noise: "var(--muted)", unmappable: "var(--muted)", insufficient: "var(--muted)",
};

function renderLab(res) {
  const el = $("#rgLabBody");
  const tag = $("#rgLabTag");
  if (!el) return;
  if (!res || res.ok === false) {
    if (tag) tag.textContent = "失败";
    el.innerHTML = '<div class="track-note warn">⚠️ ' + escHtml((res && (res.detail || res.error)) || "失败") + '</div>';
    return;
  }
  const claims = res.claims || [];
  const counts = res.verdict_counts || res.summary || {};
  if (tag) tag.textContent = claims.length + " 条 · LLM " + (res.llm_used ? "已参与" : "未参与");
  let h = "";
  if (res.error) h += '<div class="track-note warn">⚠️ ' + escHtml(res.error) + '</div>';
  if (res.overall) h += '<div class="multi-correct-banner"><div class="mcb-main">模型总结</div><div class="mcb-sub">' +
    escHtml(res.overall) + '</div></div>';
  h += '<div class="track-metrics">';
  for (const k of Object.keys(counts)) {
    h += '<div class="track-item"><div class="k">' + escHtml(k) + '</div><div class="v" style="color:' +
      (LAB_VERDICT_COLOR[k] || "inherit") + '">' + counts[k] + '</div></div>';
  }
  h += '</div>';
  if (claims.length) {
    h += '<div class="scroll"><table class="battle-table"><thead><tr><th>LLM 提出的规律</th><th>落地维度</th>' +
      '<th>n</th><th>近期效应</th><th>校正p</th><th>结论</th></tr></thead><tbody>';
    for (const c of claims) {
      const st = c.regime_status || (c.regime && c.regime.status) || "";
      h += '<tr><td style="text-align:left">' + escHtml(c.title || "") +
        (c.mechanism ? '<div class="rg-cell-sub">机理：' + escHtml(String(c.mechanism).slice(0, 90)) + '</div>' : "") +
        (c.stage_view ? '<div class="rg-cell-sub">阶段判断：' + escHtml(String(c.stage_view).slice(0, 90)) + '</div>' : "") + '</td>' +
        '<td style="text-align:left">' + escHtml(c.label || ((c.filters || []).map(f => f.key + JSON.stringify(f.params || {})).join(" + "))) + '</td>' +
        '<td>' + (c.n || 0) + '</td>' +
        '<td class="' + (c.effect === null || c.effect === undefined ? "flat" : (c.effect >= 0 ? "pos" : "neg")) + '">' +
        (c.effect === null || c.effect === undefined ? "—" : (c.effect >= 0 ? "+" : "") + Number(c.effect).toFixed(4)) + '</td>' +
        '<td>' + (c.p_adjusted === null || c.p_adjusted === undefined ? "—" : Number(c.p_adjusted).toFixed(4)) + '</td>' +
        '<td style="text-align:left"><span class="rg-badge" style="border-color:' + (LAB_VERDICT_COLOR[c.verdict] || "var(--border)") +
        ';color:' + (LAB_VERDICT_COLOR[c.verdict] || "inherit") + '">' + escHtml(c.verdict_zh || c.verdict || "") + '</span>' +
        '<div class="rg-cell-sub">' + escHtml((c.reason || "").slice(0, 110)) + '</div></td></tr>';
    }
    h += '</tbody></table></div>';
  } else {
    h += '<div class="track-note">本轮没有产生可展示的规律。' +
      (res.llm_used ? "（LLM 参与但未给出可落地的规律）" : "（LLM 未参与，只做了维度扫描）") + '</div>';
  }
  if ((res.next_dimensions || []).length) {
    h += '<div class="track-note">💡 模型建议继续挖的角度：' + res.next_dimensions.map(escHtml).join("、") + '</div>';
  }
  if (res.scan_summary) {
    h += '<div class="track-note">扫描伴随结论：' + escHtml(res.scan_summary.verdict || "") +
      '（已测 ' + (res.scan_summary.n_tested || 0) + ' 组，预期假阳性 ' +
      Number(res.scan_summary.expected_false_positives || 0).toFixed(1) + ' 个）</div>';
  }
  el.innerHTML = h;
}

async function runLlmLab() {
  if (window._labRunning) { toast("工坊已在运行中，请等待完成"); return; }
  if (!window.confirm("LLM 规律工坊会调用大模型三轮（提规律 → 自我质疑 → 修订），\n预计 1~8 分钟，将在后台运行并显示阶段进度。\n\n确定开始吗？")) return;
  window._labRunning = true;
  const tag = $("#rgLabTag");
  if (tag) tag.textContent = "后台运行中…";
  await runBackgroundTask({
    submitPath: "/api/llm_lab/run/async?rounds=3&use_llm=1",
    renderArea: "#rgLabBody",
    btnId: $("#btnLabRun") ? "#btnLabRun" : null,
    busyText: "后台运行中…",
    freeText: "🧠 只跑 LLM 规律工坊",
    onDone: (r) => {
      renderLab(r);
      toast("工坊完成：" + ((r.claims || []).length) + " 条规律");
      loadRegimeOverview(false);
    },
  });
  window._labRunning = false;
}

async function loadLlmLabLatest() {
  try {
    const r = await api("/api/llm_lab/latest");
    renderLab(r);
  } catch (e) { toast("读取失败: " + e.message); }
}

async function registerLabClaims() {
  if (!window.confirm("把工坊里已通过 as-of 回放（support/reject/仅阶段有效）的规律注册进假说库。\nunmappable 与 noise 不会入库。继续？")) return;
  try {
    const r = await api("/api/llm_lab/register", { method: "POST" });
    toast("已注册 " + ((r.registered && r.registered.registered) || 0) + " 条（跳过 " + (r.skipped_n || 0) + " 条）");
    loadRegimeOverview(false);
  } catch (e) { toast("注册失败: " + e.message); }
}

/* ---------- 检索式核验 ---------- */

function fillProbeFamilies(r) {
  const sel = $("#rgProbeFamily");
  if (!sel || sel.dataset.filled) return;
  sel.innerHTML = "";
  for (const f of (r.families || [])) {
    const o = document.createElement("option");
    o.value = f.key;
    o.textContent = f.name_zh + "（" + f.dim_zh + "）";
    sel.appendChild(o);
  }
  sel.dataset.filled = "1";
  syncProbeParams();
}

async function syncProbeParams() {
  if (!_rgFamilies) { await loadDiscoveryFamilies(); return; }
  const key = $("#rgProbeFamily") && $("#rgProbeFamily").value;
  const fam = ((_rgFamilies.families) || []).find(f => f.key === key);
  const sel = $("#rgProbeParams");
  if (!sel || !fam) return;
  sel.innerHTML = "";
  (fam.params || []).forEach((p, i) => {
    const o = document.createElement("option");
    o.value = String(i);
    o.textContent = JSON.stringify(p);
    sel.appendChild(o);
  });
}

async function probeParams() {
  if (!_rgFamilies) { await loadDiscoveryFamilies(); }
  const key = ($("#rgProbeFamily") || {}).value;
  const fam = ((_rgFamilies && _rgFamilies.families) || []).find(f => f.key === key);
  const idx = Number((($("#rgProbeParams") || {}).value) || 0);
  if (!fam) { toast("请先选择维度"); return; }
  const params = (fam.params || [])[idx] || {};
  const el = $("#rgProbeBody");
  if (el) el.innerHTML = '<div class="track-note"><span class="spin">⏳</span> 正在 as-of 回放…</div>';
  try {
    const r = await api("/api/backtest/regime?key=" + encodeURIComponent(key) +
      "&params=" + encodeURIComponent(JSON.stringify(params)));
    const rg = r.regime || {};
    let h = '<div class="multi-correct-banner"><div class="mcb-main">' + escHtml(r.filter && r.filter.label) +
      ' → <b style="color:' + (RG_STATUS_COLOR[rg.status] || "inherit") + '">' + escHtml(rg.label_zh || rg.status) + '</b></div>' +
      '<div class="mcb-sub">' + escHtml(rg.reason_zh || "") + '</div></div>';
    h += '<div class="scroll"><table class="battle-table"><thead><tr><th>阶段</th><th>n</th><th>均值</th><th>期望</th><th>差值</th><th>p</th><th>BH校正p</th></tr></thead><tbody>';
    for (const e of (rg.eras || [])) {
      h += '<tr><td style="text-align:left">' + escHtml(e.era) + '</td><td>' + (e.n || 0) + '</td>' +
        '<td>' + (e.mean === null || e.mean === undefined ? "—" : Number(e.mean).toFixed(4)) + '</td>' +
        '<td class="flat">' + (e.expected === null || e.expected === undefined ? "—" : Number(e.expected).toFixed(4)) + '</td>' +
        '<td class="' + (e.delta >= 0 ? "pos" : "neg") + '">' + (e.delta === null || e.delta === undefined ? "—" :
          (e.delta >= 0 ? "+" : "") + Number(e.delta).toFixed(4)) + '</td>' +
        '<td>' + (e.p_value === null || e.p_value === undefined ? "—" : Number(e.p_value).toFixed(4)) + '</td>' +
        '<td>' + (e.p_adjusted === null || e.p_adjusted === undefined ? "—" : Number(e.p_adjusted).toFixed(4)) + '</td></tr>';
    }
    const rc = rg.recent || {};
    h += '</tbody></table></div><div class="track-note">近期加权窗口：' + (rc.n || 0) + ' 期，效应 ' +
      (rc.delta === null || rc.delta === undefined ? "—" : Number(rc.delta).toFixed(4)) +
      '，校正后 p=' + (rc.p_adjusted === null || rc.p_adjusted === undefined ? "—" : Number(rc.p_adjusted).toFixed(4)) +
      '；可检出下限 ' + (rc.mde === null || rc.mde === undefined ? "—" : Number(rc.mde).toFixed(4)) + '</div>';
    if (el) el.innerHTML = h;
  } catch (e) {
    if (el) el.innerHTML = '<div class="track-note warn">⚠️ ' + escHtml(e.message) + '</div>';
  }
}
