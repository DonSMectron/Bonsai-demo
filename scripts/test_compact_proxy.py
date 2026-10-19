#!/usr/bin/env python3
"""Tests for compact_proxy.py using a mock llama-server. Run: python scripts/test_compact_proxy.py"""
import gzip
import http.client
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compact_proxy as cp  # noqa: E402

N_CTX = 4000


class MockLlama(BaseHTTPRequestHandler):
    seen = []
    summaries = 0

    def log_message(self, *a):
        pass

    def reply(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/props":
            return self.reply({"default_generation_settings": {"n_ctx": N_CTX}})
        # like the real server, static files are only served gzip-compressed
        if "gzip" not in self.headers.get("Accept-Encoding", ""):
            self.send_response(415)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        data = gzip.compress(b"<html><body>ui</body></html>" if self.path == "/" else b"console.log(1)")
        self.send_response(200)
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Type", "text/html; charset=utf-8" if self.path == "/" else "text/javascript")
        self.send_header("ETag", "abc")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            return self.reply({"tokens": [0] * (len(body["content"]) // 4)})
        MockLlama.seen.append(body)
        if "conversation compactor" in body["messages"][0]["content"]:
            MockLlama.summaries += 1
            return self.reply({"choices": [{"message": {"content": "SUMMARY-%d" % MockLlama.summaries}}]})
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in ("hel", "lo"):
                evt = {"choices": [{"delta": {"content": piece}}]}
                self.wfile.write(b"data: " + json.dumps(evt).encode() + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self.reply({"choices": [{"message": {"role": "assistant", "content": "hello"}}]})


def serve(handler, port=0):
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def post(port, payload):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request("POST", "/v1/chat/completions", json.dumps(payload),
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    return resp.status, dict(resp.getheaders()), resp.read().decode()


def turn(i, size=10):
    return [{"role": "user", "content": "question %d " % i + "x" * size},
            {"role": "assistant", "content": "answer %d " % i + "y" * size}]


class CompactProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = serve(MockLlama)
        cls.proxy = cp.make_server("127.0.0.1", 0)
        threading.Thread(target=cls.proxy.serve_forever, daemon=True).start()
        cp.cfg.upstream_host, cp.cfg.upstream_port = "127.0.0.1", cls.upstream.server_address[1]
        cp.cfg.ctx, cp.cfg.threshold, cp.cfg.auto = 0, 0.75, True
        cls.port = cls.proxy.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.proxy.shutdown()
        cls.upstream.shutdown()

    def setUp(self):
        MockLlama.seen.clear()
        MockLlama.summaries = 0
        cp.cfg.auto, cp.cfg.threshold = True, 0.75
        with cp._cache_lock:
            cp._summary_cache.clear()

    def test_passthrough_static_and_unchanged_chat(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port)
        conn.request("GET", "/app.js", headers={"Accept-Encoding": "gzip"})
        resp = conn.getresponse()
        self.assertEqual(resp.getheader("Content-Encoding"), "gzip")
        self.assertEqual(gzip.decompress(resp.read()), b"console.log(1)")
        msgs = turn(1) + [{"role": "user", "content": "hi"}]
        status, headers, body = post(self.port, {"messages": msgs})
        self.assertEqual(status, 200)
        self.assertNotIn("X-Bonsai-Compacted", headers)
        self.assertEqual(MockLlama.seen[-1]["messages"], msgs)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "hello")

    def test_index_gets_widget_even_without_gzip_support(self):
        for headers in ({"Accept-Encoding": "gzip"}, {}):
            conn = http.client.HTTPConnection("127.0.0.1", self.port)
            conn.request("GET", "/", headers=headers)
            resp = conn.getresponse()
            page = resp.read().decode()
            self.assertEqual(resp.status, 200)
            self.assertIsNone(resp.getheader("Content-Encoding"))
            self.assertEqual(int(resp.getheader("Content-Length")), len(page.encode()))
            self.assertIn("ui", page)
            self.assertIn("/_compact/settings", page)
            self.assertTrue(page.rstrip().endswith("</body></html>"))
            self.assertLess(page.index("__bonsaiCompactWidget"), page.rindex("</body>"))

    def test_settings_endpoint(self):
        def settings(body=None):
            conn = http.client.HTTPConnection("127.0.0.1", self.port)
            conn.request("POST" if body is not None else "GET", cp.SETTINGS_PATH,
                         json.dumps(body) if body is not None else None)
            resp = conn.getresponse()
            return resp.status, json.loads(resp.read())

        self.assertEqual(settings(), (200, {"threshold": 0.75, "auto": True, "ctx": N_CTX}))
        self.assertEqual(settings({"threshold": 0.5}), (200, {"threshold": 0.5, "auto": True, "ctx": N_CTX}))
        self.assertEqual(settings({"auto": False})[1]["auto"], False)
        self.assertEqual(settings()[1]["threshold"], 0.5)
        for bad in ({"threshold": 0.01}, {"threshold": 1.5}, {"threshold": "x"},
                    {"threshold": True}, {"auto": "yes"}, []):
            self.assertEqual(settings(bad)[0], 400, bad)
        self.assertEqual(settings()[1]["threshold"], 0.5)

    def test_slider_changes_when_compaction_triggers(self):
        history = []
        for i in range(1, 4):
            history += turn(i, size=900)
        msgs = history + [{"role": "user", "content": "q"}]  # roughly 40% of N_CTX
        post(self.port, {"messages": msgs})
        self.assertEqual(MockLlama.summaries, 0)
        cp.cfg.threshold = 0.2
        post(self.port, {"messages": msgs})
        self.assertGreater(MockLlama.summaries, 0)

    def test_streaming_passthrough(self):
        _, _, body = post(self.port, {"stream": True, "messages": [{"role": "user", "content": "hi"}]})
        self.assertIn('"hel"', body)
        self.assertTrue(body.rstrip().endswith("data: [DONE]"))

    def test_compact_command_non_streaming(self):
        msgs = [{"role": "system", "content": "be nice"}] + turn(1) + turn(2) + \
               [{"role": "user", "content": "/compact keep the names"}]
        status, _, body = post(self.port, {"messages": msgs})
        text = json.loads(body)["choices"][0]["message"]["content"]
        self.assertEqual(status, 200)
        self.assertTrue(text.startswith(cp.MARKER))
        self.assertIn("SUMMARY-1", text)
        sent = MockLlama.seen[-1]["messages"][1]["content"]
        self.assertIn("question 1", sent)
        self.assertIn("Pay special attention to: keep the names", sent)
        self.assertEqual(MockLlama.seen[-1]["chat_template_kwargs"], {"enable_thinking": False})

    def test_compact_command_streaming(self):
        msgs = turn(1) + [{"role": "user", "content": "/compact"}]
        _, _, body = post(self.port, {"stream": True, "messages": msgs})
        events = [json.loads(line[6:]) for line in body.splitlines()
                  if line.startswith("data: {")]
        text = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
        self.assertIn(cp.MARKER, text)
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")
        self.assertTrue(body.rstrip().endswith("data: [DONE]"))

    def test_nothing_to_compact(self):
        _, _, body = post(self.port, {"messages": [{"role": "user", "content": "/compact"}]})
        self.assertIn("Nothing new to compact", body)
        self.assertEqual(MockLlama.summaries, 0)

    def test_summary_message_drops_older_history(self):
        summary_msg = {"role": "assistant", "content": "%s\n\nSUMMARY-X\n\n%s" % (cp.MARKER, cp.FOOTER)}
        msgs = [{"role": "system", "content": "be nice"}] + turn(1) + \
               [{"role": "user", "content": "/compact"}, summary_msg] + turn(3) + \
               [{"role": "user", "content": "next"}]
        status, headers, _ = post(self.port, {"messages": msgs})
        sent = MockLlama.seen[-1]["messages"]
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("X-Bonsai-Compacted"), "1")
        self.assertEqual([m["role"] for m in sent], ["system", "user", "assistant", "user"])
        self.assertIn("be nice", sent[0]["content"])
        self.assertIn("SUMMARY-X", sent[0]["content"])
        self.assertNotIn(cp.FOOTER, sent[0]["content"])
        self.assertNotIn("question 1", json.dumps(sent))

    def test_summary_without_auto_still_folds(self):
        cp.cfg.auto = False
        summary_msg = {"role": "assistant", "content": cp.MARKER + "\n\nSUMMARY-Y"}
        msgs = turn(1) + [summary_msg, {"role": "user", "content": "next"}]
        post(self.port, {"messages": msgs})
        sent = MockLlama.seen[-1]["messages"]
        self.assertEqual([m["role"] for m in sent], ["system", "user"])
        self.assertIn("SUMMARY-Y", sent[0]["content"])

    def test_auto_compact_and_cache(self):
        history = []
        for i in range(1, 9):
            history += turn(i, size=900)
        msgs = history + [{"role": "user", "content": "latest question"}]
        post(self.port, {"messages": msgs})
        sent = MockLlama.seen[-1]["messages"]
        first_calls = MockLlama.summaries
        self.assertGreaterEqual(first_calls, 1)
        self.assertLess(len(sent), len(msgs))
        self.assertEqual(sent[-1]["content"], "latest question")
        self.assertIn("SUMMARY-%d" % first_calls, sent[0]["content"])
        self.assertEqual(sent[1]["role"], "user")
        # same chat plus one more message reuses the cached summary instead of re-summarizing
        msgs2 = msgs + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "again"}]
        post(self.port, {"messages": msgs2})
        self.assertEqual(MockLlama.summaries, first_calls)
        self.assertIn("SUMMARY-%d" % first_calls, MockLlama.seen[-1]["messages"][0]["content"])
        self.assertEqual(MockLlama.seen[-1]["messages"][-1]["content"], "again")

    def test_oversized_transcript_is_summarized_in_chunks(self):
        history = []
        for i in range(1, 40):
            history += turn(i, size=900)
        post(self.port, {"messages": history + [{"role": "user", "content": "go"}]})
        self.assertGreater(MockLlama.summaries, 1)
        for body in MockLlama.seen:
            if "conversation compactor" in body["messages"][0]["content"]:
                self.assertLess(len(body["messages"][1]["content"]), N_CTX * 3)

    def test_compaction_failure_falls_back_to_unchanged_request(self):
        original = cp.summarize
        cp.summarize = lambda *a, **k: (_ for _ in ()).throw(cp.UpstreamError("boom"))
        try:
            history = []
            for i in range(1, 9):
                history += turn(i, size=900)
            msgs = history + [{"role": "user", "content": "q"}]
            status, _, _ = post(self.port, {"messages": msgs})
        finally:
            cp.summarize = original
        self.assertEqual(status, 200)
        self.assertEqual(MockLlama.seen[-1]["messages"], msgs)

    def test_parse_command(self):
        self.assertEqual(cp.parse_command("/compact"), "")
        self.assertEqual(cp.parse_command("  /Compact focus on x "), "focus on x")
        self.assertIsNone(cp.parse_command("/compactor"))
        self.assertIsNone(cp.parse_command("please /compact"))


if __name__ == "__main__":
    unittest.main()
