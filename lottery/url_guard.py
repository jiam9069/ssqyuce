"""LLM 上游 URL 的安全校验（V36，2026-10-24 审计 P1-2）。

## 这个模块解决什么

用户可以在 `/api/user/llm/config` 里填写任意 `base_url`，随后服务器会**主动**
向该地址 POST（`llm_client.chat()`）。这构成**认证后 SSRF**：

* 服务端可以访问用户本机访问不到的内网地址（`169.254.169.254` 元数据、
  `127.0.0.1:6379`、`10.x`、`172.16-31.x`、`192.168.x`）；
* 更糟的是错误路径会把响应的前 200 字符回传给用户
  （`llm_client.py` 的 `last_err = f"HTTP {code}: {r.text[:200]}"` 经 503 透出），
  于是"盲打"变成了**可读响应**的半盲 SSRF。

审计实测路径：普通用户配 `base_url` → 触发预测 → 服务器 POST 到指定地址 →
错误文本回到用户手里。`/api/llm/test` 虽然是 admin-only，但 `/api/predict`
会走用户自己那份配置，所以**任何登录用户都能触发**。

## 为什么要在两处校验

1. **保存时**（`user_llm_store.save`）：给出即时反馈，让用户知道自己填了什么
   不被允许，而不是等到出号时才 503；
2. **发请求前**（`llm_client.chat`）：这是**真正的安全边界**。只校验保存侧不够 ——
   历史数据里已存的配置、管理员配的全局通道、以及将来任何绕过 save 的写入路径
   都会直接进到 chat()。安全判定必须落在"即将发出去的那一刻"。

两处共用本模块，避免"两套口径各说各话"（这正是审计里反复出现的失败模式）。

## 为什么不是 DNS 解析后再校验

解析后校验需要处理 DNS rebinding（解析一次、连接另一次），要拿到**实际连接**
用的 IP 才能可靠，而 `requests` 不暴露这个钩子（需要自定义 HTTPAdapter）。
本模块的策略是**默认拒绝字面量与内网域名**、只放行明确公网的目标，
并以"域名必须能解析到公网 IP"作为第二道闸。残余风险（rebinding）在文末注明。
"""
from __future__ import annotations

import ipaddress
import socket
from typing import Optional, Tuple
from urllib.parse import urlsplit

#: 允许的 scheme。只放 https/http —— 挡掉 file:// gopher:// dict:// 等
#: 「用 base_url 当通用协议跳板」的玩法。
ALLOWED_SCHEMES = ("http", "https")

#: 明确禁止的主机名（云元数据与本地别名）。这些即使解析成公网也得拒，
#: 因为它们的语义就是"从内部访问自己"。
BLOCKED_HOSTNAMES = frozenset({
    "metadata", "metadata.google.internal", "metadata.goog",
    "localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback",
    "instance-data", "metadata.azure.com",
})


def _ip_is_public(ip: ipaddress._BaseAddress) -> bool:
    """单个 IP 是否属于"可以安全外连"的范围。

    逐项列出而不是只写 `is_private`：`is_private` 在 Python 3.13 把
    169.254/16（链路本地）算作 private，但 6to4/teredo 等又有自己的边界；
    显式判断更难被误读，也更容易在测试里逐个钉死。
    """
    if isinstance(ip, ipaddress.IPv6Address):
        # IPv4-mapped（::ffff:127.0.0.1）按内层 IPv4 判，否则会绕过所有 v4 规则
        if ip.ipv4_mapped is not None:
            return _ip_is_public(ip.ipv4_mapped)
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_multicast or ip.is_reserved or ip.is_unspecified
        # CGNAT 100.64/10：is_private 在部分版本不覆盖
        or (isinstance(ip, ipaddress.IPv4Address)
            and ip in ipaddress.ip_network("100.64.0.0/10"))
        # 云元数据常见地址，显式点名
        or (isinstance(ip, ipaddress.IPv4Address)
            and ip == ipaddress.ip_address("169.254.169.254"))
    )


def _resolve_all_infos(host: str):
    """解析主机名全部 A/AAAA 记录，返回 IP 列表；解析失败返回 None。

    返回 None 与返回 [] 语义不同：None = "问不到"（不该判不安全），
    [] = "问了但一个都没有"（异常，交给调用方当不确定处理）。
    """
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except Exception:                            # noqa: BLE001
        return None
    out = []
    for info in infos:
        try:
            out.append(ipaddress.ip_address(info[4][0]))
        except ValueError:
            continue
    return out or None


def _resolve_all(host: str) -> Tuple[bool, str]:
    """解析主机名的**全部** A/AAAA 记录。返回 (全为公网, 原因)。

    只要有一条解析到内网就拒 —— 攻击者可以给一个域名同时解析到公网与内网
    （DNS 轮询 / 多记录），若只检查第一条就会被绕过。
    """
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        return False, f"域名无法解析（{e}）。请填写可公网访问的地址。"
    except Exception as e:                       # noqa: BLE001
        return False, f"域名解析异常（{type(e).__name__}）。请确认地址拼写。"
    if not infos:
        return False, "域名没有解析出任何 IP。请确认地址拼写。"
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return False, f"域名解析出无法识别的地址 {addr}。"
        if not _ip_is_public(ip):
            return False, (
                f"该域名解析到内网/保留地址 {addr}，出于安全考虑不允许"
                "（服务器会替你访问该地址）。请改用公网可访问的 LLM 服务端点。")
    return True, ""


def check_llm_base_url(base_url: Optional[str], *,
                       allow_private: bool = False) -> Tuple[bool, str]:
    """校验一个 LLM `base_url` 是否可安全外连。

    返回 ``(ok, reason)``；``reason`` 是**面向用户的可执行提示**（项目 D20 纪律：
    不许以"失败/错误"结尾而无下一步）。

    ``allow_private=True`` 供**自建内网 LLM** 的部署显式开启
    （通过环境变量 ``LOTT_LLM_ALLOW_PRIVATE_URL=1``），默认关闭：
    一个需要连内网模型的部署，运维**明确**开这个开关比"默认全放"安全得多。
    """
    raw = (base_url or "").strip()
    if not raw:
        return False, "请填写 LLM 服务地址（形如 https://api.example.com/v1）。"
    if allow_private:
        # 显式放行内网：仍要求 scheme 合法，否则就成了"绕过一切"的后门
        parts = urlsplit(raw)
        if parts.scheme.lower() not in ALLOWED_SCHEMES:
            return False, ("地址必须以 http:// 或 https:// 开头。"
                           "如需连接自建内网服务，请确认协议头正确。")
        if not parts.hostname:
            return False, "地址里没有可识别的主机名。请检查是否漏写域名或 IP。"
        return True, ""

    parts = urlsplit(raw)
    scheme = (parts.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        return False, (f"不支持 {scheme or '空'} 协议。请填写 http:// 或 https:// 开头的地址"
                       "（例如 https://api.openai.com/v1）。")
    host = (parts.hostname or "").strip().rstrip(".")
    if not host:
        return False, "地址里没有可识别的主机名。请检查是否漏写域名或 IP。"
    if host.lower() in BLOCKED_HOSTNAMES:
        return False, (f"不允许把 {host} 作为 LLM 地址 —— 服务器会替你访问它，"
                       "这会让系统去连本机/云元数据。请改用公网可访问的模型服务。")

    # 1) 字面量 IP：直接判，不必解析
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        if not _ip_is_public(ip):
            return False, (f"{host} 属于内网/保留地址，不允许作为 LLM 地址"
                           "（服务器会替你访问它）。请改用公网可访问的模型服务端点。")
        return True, ""

    # 2) 域名：解析后逐个核对，**解析到内网即拒**。
    #
    # ⚠️ 关键区分（V37 修正）：**"解析不到"≠"不安全"**。
    # 第一版把解析失败也判成拒绝，结果在无 DNS 的环境（CI、离线机器、
    # 内网自建解析）里把 `https://a.example.com` 也拦掉了 —— 那是误伤，
    # 而且会在测试里表现为"保存配置 400"，看起来像产品坏了。
    #
    # 真正的安全语义是：**能确认是内网就拒；无法确认就放行给下一道闸**
    # （发请求前 `chat()` 还会再查一次，且那时若解析到内网同样会拒）。
    # 把"不确定"判成拒绝属于 fail-closed 的过度使用 —— 它挡的是自己的用户。
    infos = _resolve_all_infos(host)
    if infos is None:
        # 解析失败：不否决（可能只是本机没有 DNS/域名尚未生效）
        return True, ""
    for ip in infos:
        if not _ip_is_public(ip):
            return False, (
                f"该域名解析到内网/保留地址 {ip}，出于安全考虑不允许"
                "（服务器会替你访问该地址）。请改用公网可访问的 LLM 服务端点。")
    return True, ""


def allow_private_from_env() -> bool:
    """环境开关：``LOTT_LLM_ALLOW_PRIVATE_URL=1`` 才放行内网地址。"""
    import os
    return (os.environ.get("LOTT_LLM_ALLOW_PRIVATE_URL") or "0").strip() == "1"
