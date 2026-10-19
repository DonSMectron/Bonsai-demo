#!/usr/bin/env python3
"""Conversation compaction proxy for the Bonsai llama-server (adds a /compact command).

The built-in llama-server web UI keeps the whole chat history and resends it on every
turn, so a long chat eventually exceeds the context window and the request fails. This
proxy sits between the UI (or any OpenAI-compatible client) and llama-server and fixes
that without touching the server:

  * Type `/compact` (optionally followed by what to focus on, e.g. `/compact keep the
    API design decisions`) as a chat message. The proxy asks the model to summarize the
    conversation and replies with that summary as an assistant message.
  * From then on, every request that contains that summary message is rewritten before
    it reaches the server: all messages before the summary are dropped and the summary
    is folded into the system prompt. The UI still shows the full chat; the model only
    sees the summary plus what came after it.
  * Automatic compaction: if a request would use more than --threshold of the server's
    context, older turns are summarized on the fly (cached, so it happens once per
    compaction, not once per message) and the last few turns are kept verbatim. The
    threshold can be changed live with the slider this proxy adds to the chat page.

Everything else (static UI files, /props, streaming, tool calls, images) is passed
through unchanged. Standard library only.

Usage: compact_proxy.py [--port 8081] [--upstream http://127.0.0.1:8080]
Then open http://localhost:8081 instead of http://localhost:8080.
"""
import argparse
import gzip
import hashlib
import http.client
import json
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MARKER = "### Conversation summary (compacted)"
FOOTER = "_Older messages are folded into this summary and are no longer sent to the model. Keep chatting as usual._"
HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
               "proxy-authenticate", "proxy-authorization", "proxy-connection"}
CHARS_PER_TOKEN = 3
IMAGE_TOKENS = 1024
MAX_CACHE_ENTRIES = 16
MIN_THRESHOLD, MAX_THRESHOLD = 0.1, 0.95
SETTINGS_PATH = "/_compact/settings"
WIDGET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compact_widget.html")

SUMMARY_SYSTEM = (
    "You are a conversation compactor. You write a faithful, dense summary of a conversation "
    "between a user and an AI assistant so the conversation can continue after the older "
    "messages are dropped. You never answer the conversation yourself."
)
SUMMARY_INSTRUCTIONS = (
    "Write the updated summary now. Keep: the user's goals and requirements, key facts and "
    "names, decisions made and why, code, commands, file paths and identifiers that matter "
    "(verbatim, short snippets only), important tool results, open problems, and next steps. "
    "Drop pleasantries and redundant detail. Use short Markdown sections or bullets, under "
    "about 700 words. Output only the summary."
)


class Config:
    port = 8081
    host = "127.0.0.1"
    upstream_host = "127.0.0.1"
    upstream_port = 8080
    ctx = 0
    threshold = 0.75
    auto = True


cfg = Config()
_ctx_cache = {"at": 0.0, "value": 0}
_summary_cache = []
_cache_lock = threading.Lock()


class UpstreamError(RuntimeError):
    pass


def log(msg):
    sys.stderr.write("[compact-proxy %s] %s\n" % (time.strftime("%H:%M:%S"), msg))
    sys.stderr.flush()


def upstream_json(method, path, payload=None, timeout=3600):
    conn = http.client.HTTPConnection(cfg.upstream_host, cfg.upstream_port, timeout=timeout)
    try:
        body = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"} if body else {}
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        if resp.status >= 400:
            raise UpstreamError("upstream %s %s -> %d: %s" % (
                method, path, resp.status, data.decode("utf-8", "replace")[:500]))
        return json.loads(data)
    finally:
        conn.close()


def context_size():
    if cfg.ctx:
        return cfg.ctx
    now = time.time()
    if _ctx_cache["value"] and now - _ctx_cache["at"] < 60:
        return _ctx_cache["value"]
    try:
        props = upstream_json("GET", "/props", timeout=5)
        n = int(props["default_generation_settings"]["n_ctx"])
    except Exception:
        return _ctx_cache["value"] or 8192
    _ctx_cache.update(at=now, value=n)
    return n


def text_of(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content
                         if isinstance(p, dict) and p.get("type") == "text")
    return str(content)


def image_count(content):
    if not isinstance(content, list):
        return 0
    return sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image_url")


def rough_tokens(messages):
    chars = sum(len(text_of(m.get("content"))) + len(json.dumps(m.get("tool_calls") or ""))
                for m in messages)
    return chars // CHARS_PER_TOKEN


def count_tokens(messages, tools=None):
    parts, extra = [], 0
    for m in messages:
        parts.append(text_of(m.get("content")))
        for tc in m.get("tool_calls") or []:
            parts.append(json.dumps(tc))
        extra += 8 + IMAGE_TOKENS * image_count(m.get("content"))
    if tools:
        parts.append(json.dumps(tools))
    text = "\n".join(parts)
    try:
        n = len(upstream_json("POST", "/tokenize", {"content": text}, timeout=30)["tokens"])
    except Exception:
        n = len(text) // CHARS_PER_TOKEN + 1
    return n + extra


def parse_summary(text):
    if not text.startswith(MARKER):
        return None
    body = text[len(MARKER):]
    return body.split(FOOTER)[0].strip()


def fold(messages):
    """Split into (leading system messages, summary text or "", messages after the summary)."""
    i = 0
    while i < len(messages) and messages[i].get("role") in ("system", "developer"):
        i += 1
    systems, convo, summary = messages[:i], messages[i:], ""
    for j in range(len(convo) - 1, -1, -1):
        if convo[j].get("role") != "assistant":
            continue
        found = parse_summary(text_of(convo[j].get("content")))
        if found is not None:
            summary, convo = found, convo[j + 1:]
            break
    return systems, summary, convo


def build(systems, summary, convo):
    systems = [dict(m) for m in systems]
    if summary:
        note = ("## Summary of the earlier conversation\n"
                "Older messages were compacted into this summary. Treat it as established context.\n\n"
                + summary)
        if systems:
            content = systems[0].get("content")
            if isinstance(content, list):
                systems[0]["content"] = content + [{"type": "text", "text": "\n\n" + note}]
            else:
                systems[0]["content"] = (content or "") + "\n\n" + note
        else:
            systems = [{"role": "system", "content": note}]
    return systems + convo


def digest(summary, convo):
    raw = json.dumps([summary, convo], sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def cached_prefix(summary, convo):
    with _cache_lock:
        entries = sorted(_summary_cache, key=lambda e: -e[0])
    for k, key, cached in entries:
        if 0 < k < len(convo) and digest(summary, convo[:k]) == key:
            return cached, k
    return None


def store_prefix(summary, convo, k, result):
    with _cache_lock:
        _summary_cache.append((k, digest(summary, convo[:k]), result))
        del _summary_cache[:-MAX_CACHE_ENTRIES]


def clip(s, n):
    if len(s) <= n:
        return s
    return s[:n * 2 // 3] + "\n...[truncated]...\n" + s[-(n // 3):]


def render(messages):
    pieces = []
    for m in messages:
        body = text_of(m.get("content")).strip()
        imgs = image_count(m.get("content"))
        if imgs:
            body += " [%d image(s) omitted]" % imgs
        if m.get("role") == "tool":
            body = clip(body, 1500)
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            body += "\n[tool call: %s(%s)]" % (fn.get("name"), clip(str(fn.get("arguments")), 400))
        pieces.append("%s: %s" % (str(m.get("role", "?")).upper(), body))
    return pieces


def chunk_pieces(pieces, budget):
    chunks, current, size = [], [], 0
    for piece in pieces:
        while len(piece) > budget:
            if current:
                chunks.append("\n\n".join(current))
                current, size = [], 0
            chunks.append(piece[:budget])
            piece = piece[budget:]
        if size + len(piece) > budget and current:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        current.append(piece)
        size += len(piece) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def llm(user_prompt, max_tokens):
    out = upstream_json("POST", "/v1/chat/completions", {
        "messages": [{"role": "system", "content": SUMMARY_SYSTEM},
                     {"role": "user", "content": user_prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    text = text_of(((out.get("choices") or [{}])[0].get("message") or {}).get("content")).strip()
    if not text:
        raise UpstreamError("the model returned an empty summary")
    return text


def summarize(prior, messages, focus=""):
    """Fold `messages` into `prior`, one context-sized chunk at a time so the summarizer
    itself never overflows even when the chat being compacted is nearly full."""
    ctx = context_size()
    out_tokens = min(3072, max(512, ctx // 8))
    budget = max(2000, (ctx - out_tokens - 1500) * 2 * CHARS_PER_TOKEN // 3 - len(prior))
    focus_line = "\nPay special attention to: %s\n" % focus if focus else ""
    summary = prior
    for chunk in chunk_pieces(render(messages), budget):
        summary = llm(
            "<previous_summary>\n%s\n</previous_summary>\n\n<new_messages>\n%s\n</new_messages>\n%s\n%s"
            % (summary or "(none)", chunk, focus_line, SUMMARY_INSTRUCTIONS), out_tokens)
    return summary


def pick_tail_start(convo, keep_tokens):
    """Index of the oldest user message such that everything from it on fits in keep_tokens;
    falls back to the last user message. -1 if there is no user message to keep from."""
    user_idx = [i for i, m in enumerate(convo) if m.get("role") == "user"]
    if not user_idx:
        return -1
    for i in user_idx:
        if rough_tokens(convo[i:]) <= keep_tokens:
            return i
    return user_idx[-1]


def auto_compact(messages, tools):
    systems, base_summary, base_convo = fold(messages)
    summary, convo, consumed = base_summary, base_convo, 0
    hit = cached_prefix(base_summary, base_convo)
    if hit:
        summary, consumed = hit
        convo = base_convo[consumed:]
    ctx = context_size()
    limit = int(ctx * cfg.threshold)
    rebuilt = build(systems, summary, convo)
    if count_tokens(rebuilt, tools) <= limit:
        return rebuilt
    tail = pick_tail_start(convo, limit // 3)
    if tail <= 0:
        log("context is over the threshold but there is nothing older to compact")
        return rebuilt
    started = time.time()
    new_summary = summarize(summary, convo[:tail])
    store_prefix(base_summary, base_convo, consumed + tail, new_summary)
    rebuilt = build(systems, new_summary, convo[tail:])
    log("auto-compacted %d messages in %.1fs (limit %d of %d tokens)" % (
        tail, time.time() - started, limit, ctx))
    return rebuilt


def parse_command(text):
    t = text.strip()
    if t.lower().startswith("/compact") and (len(t) == 8 or t[8].isspace()):
        return t[8:].strip()
    return None


def inject_widget(page):
    """Add the settings slider to the chat UI's HTML (falls back to appending)."""
    try:
        with open(WIDGET_PATH, encoding="utf-8") as f:
            widget = f.read()
    except OSError as e:
        log("settings widget not injected: %s" % e)
        return page
    html = page.decode("utf-8", "replace")
    at = html.rfind("</body>")
    return (html[:at] + widget + html[at:] if at >= 0 else html + widget).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "BonsaiCompactProxy/1.0"

    def log_message(self, *args):
        pass

    def send_json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def reply_chat(self, request, text):
        created = int(time.time())
        base = {"id": "chatcmpl-compact-%d" % created, "created": created,
                "model": request.get("model") or "bonsai"}
        if request.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            chunk = dict(base, object="chat.completion.chunk")
            for delta, finish in (({"role": "assistant", "content": ""}, None),
                                  ({"content": text}, None), ({}, "stop")):
                event = dict(chunk, choices=[{"index": 0, "delta": delta, "finish_reason": finish}])
                self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            self.send_json(200, dict(base, object="chat.completion", choices=[{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": text}}]))

    def forward(self, method, body, extra_headers=None):
        # The UI shell is served gzip-only, so Accept-Encoding must pass through. Only the
        # index page is decoded here, so the settings widget can be added to it.
        inject = method == "GET" and self.path.split("?", 1)[0] in ("/", "/index.html")
        conn = http.client.HTTPConnection(cfg.upstream_host, cfg.upstream_port, timeout=3600)
        responding = False
        try:
            skip = HOP_HEADERS | {"host", "content-length"}
            if inject:
                skip |= {"accept-encoding", "if-none-match", "if-modified-since"}
            headers = {k: v for k, v in self.headers.items() if k.lower() not in skip}
            headers["Host"] = "%s:%d" % (cfg.upstream_host, cfg.upstream_port)
            if inject:
                headers["Accept-Encoding"] = "gzip"
            if body is not None:
                headers["Content-Length"] = str(len(body))
            conn.request(method, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            responding = True
            if inject and resp.status == 200 and resp.getheader("Content-Type", "").startswith("text/html"):
                return self.send_index(resp)
            self.send_response(resp.status, resp.reason)
            for k, v in resp.getheaders():
                if k.lower() not in HOP_HEADERS:
                    self.send_header(k, v)
            for k, v in (extra_headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except OSError as e:
            if not responding:
                self.send_json(502, {"error": {"message": "upstream unreachable: %s" % e,
                                               "type": "proxy_error", "code": 502}})
        finally:
            conn.close()

    def send_index(self, resp):
        page = resp.read()
        if resp.getheader("Content-Encoding", "").lower() == "gzip":
            page = gzip.decompress(page)
        page = inject_widget(page)
        self.send_response(200)
        for k, v in resp.getheaders():
            if k.lower() not in HOP_HEADERS | {"content-length", "content-encoding", "etag", "last-modified"}:
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def settings(self, body):
        if body:
            try:
                patch = json.loads(body)
                threshold = patch.get("threshold", cfg.threshold)
                auto = patch.get("auto", cfg.auto)
                if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) \
                        or not MIN_THRESHOLD <= threshold <= MAX_THRESHOLD or not isinstance(auto, bool):
                    raise ValueError
            except (ValueError, AttributeError):
                return self.send_json(400, {"error": {
                    "message": "expected {\"threshold\": %s-%s, \"auto\": true|false}" % (
                        MIN_THRESHOLD, MAX_THRESHOLD), "type": "bad_request", "code": 400}})
            cfg.threshold, cfg.auto = float(threshold), auto
            log("settings: auto-compaction %s at %d%%" % ("on" if auto else "off", round(cfg.threshold * 100)))
        try:
            ctx = context_size()
        except Exception:
            ctx = 0
        self.send_json(200, {"threshold": cfg.threshold, "auto": cfg.auto, "ctx": ctx})

    def handle_chat(self, body):
        try:
            request = json.loads(body)
            messages = request["messages"]
            assert isinstance(messages, list) and messages
        except Exception:
            return self.forward("POST", body)
        try:
            last = messages[-1]
            command = parse_command(text_of(last.get("content"))) if last.get("role") == "user" else None
            if command is not None:
                return self.compact_command(request, messages[:-1], command)
            systems, summary, convo = fold(messages)
            if cfg.auto:
                request["messages"] = auto_compact(messages, request.get("tools"))
            elif summary:
                request["messages"] = build(systems, summary, convo)
            else:
                return self.forward("POST", body)
        except (UpstreamError, OSError) as e:
            log("compaction failed, forwarding the request unchanged: %s" % e)
            return self.forward("POST", body)
        if request["messages"] == messages:
            return self.forward("POST", body)
        self.forward("POST", json.dumps(request).encode(), {"X-Bonsai-Compacted": "1"})

    def compact_command(self, request, history, focus):
        _, summary, convo = fold(history)
        if not convo:
            return self.reply_chat(request, "Nothing new to compact yet.")
        started = time.time()
        try:
            new_summary = summarize(summary, convo, focus)
        except (UpstreamError, OSError) as e:
            log("/compact failed: %s" % e)
            return self.send_json(500, {"error": {"message": "Compaction failed: %s" % e,
                                                  "type": "compact_error", "code": 500}})
        log("/compact folded %d messages in %.1fs" % (len(convo), time.time() - started))
        self.reply_chat(request, "%s\n\n%s\n\n%s" % (MARKER, new_summary, FOOTER))

    def handle_any(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        path = self.path.split("?", 1)[0]
        if path == SETTINGS_PATH and self.command in ("GET", "POST"):
            return self.settings(body if self.command == "POST" else None)
        if self.command == "POST" and path in ("/v1/chat/completions", "/chat/completions") and body:
            return self.handle_chat(body)
        self.forward(self.command, body)

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = handle_any


def make_server(host, port):
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def parse_upstream(url):
    parts = urllib.parse.urlsplit(url if "//" in url else "http://" + url)
    if parts.scheme != "http" or not parts.hostname:
        raise SystemExit("--upstream must be an http://host:port URL, got %r" % url)
    return parts.hostname, parts.port or 80


def main():
    env = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default=env("BONSAI_PROXY_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(env("BONSAI_PROXY_PORT", "8081")))
    ap.add_argument("--upstream", default=env("BONSAI_UPSTREAM", "http://127.0.0.1:8080"))
    ap.add_argument("--ctx", type=int, default=int(env("BONSAI_COMPACT_CTX", "0")),
                    help="context size in tokens (default: ask the server via /props)")
    ap.add_argument("--threshold", type=float, default=float(env("BONSAI_COMPACT_AT", "0.75")),
                    help="auto-compact when a request exceeds this fraction of the context")
    ap.add_argument("--no-auto", action="store_true", default=env("BONSAI_COMPACT_AUTO", "1") == "0",
                    help="only compact on /compact; never automatically")
    args = ap.parse_args()
    if not MIN_THRESHOLD <= args.threshold <= MAX_THRESHOLD:
        raise SystemExit("--threshold must be between %s and %s" % (MIN_THRESHOLD, MAX_THRESHOLD))
    cfg.host, cfg.port, cfg.ctx = args.host, args.port, args.ctx
    cfg.threshold, cfg.auto = args.threshold, not args.no_auto
    cfg.upstream_host, cfg.upstream_port = parse_upstream(args.upstream)
    server = make_server(cfg.host, cfg.port)
    try:
        upstream_json("GET", "/health", timeout=3)
        state = "reachable, context %d tokens" % context_size()
    except Exception:
        state = "not reachable yet"
    log("listening on http://%s:%d -> http://%s:%d (%s)" % (
        cfg.host, cfg.port, cfg.upstream_host, cfg.upstream_port, state))
    log("type /compact in the chat to summarize it; auto-compaction %s" % (
        "at %d%% of context" % round(cfg.threshold * 100) if cfg.auto else "off"))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
