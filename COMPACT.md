# Compacting long chats (`/compact`)

The built-in llama-server chat UI resends the whole conversation on every turn. Once it
no longer fits in the context window (65,536 tokens by default on a 64 GB machine, see
[Context Size](README.md#context-size)) requests start to fail. `compact_proxy.py` is a
small proxy that adds an LM Studio style `/compact` command and automatic compaction. It
needs no changes to the server and uses only the Python standard library.

## Use it

Start the llama-server as usual, then in a second terminal:

```powershell
.\scripts\start_compact_proxy.ps1      # macOS/Linux: ./scripts/start_compact_proxy.sh
```

Open **http://localhost:8081** (not 8080) and chat as normal.

- **`/compact`** as a chat message summarizes the conversation. Add focus text if you
  like: `/compact keep the database schema decisions`. The summary appears as an
  assistant message headed "Conversation summary (compacted)".
- From then on the model only sees that summary plus what you say after it. The UI still
  shows the old messages; they just stop counting against the context.
- **Automatic compaction**: if a request would use more than 75% of the server's context,
  the oldest turns are summarized on the fly and the most recent turns are kept as they
  are. The summary is cached, so this happens once per compaction rather than every
  message. A reply to such a request carries an `X-Bonsai-Compacted: 1` header and the
  proxy window logs it.
- **Slider**: a small "Compact at 75%" button sits in the bottom-right corner of the chat
  page. Click it to open the panel, drag the slider (10% to 95%) to choose when automatic
  compaction starts, or untick "on" to turn automatic compaction off. The panel shows
  the equivalent token count. Changes apply to the next message, and the browser
  remembers your choice (it is re-applied if the proxy restarts). Lower values compact
  earlier and keep more headroom for long replies; higher values keep more of the chat
  verbatim.

Summarizing runs one extra, non-thinking model request (typically a few seconds to a
minute on a long chat). If a compaction fails, the proxy logs it and forwards your request
unchanged.

## Settings

| Env var | Flag | Default | Meaning |
|---|---|---|---|
| `BONSAI_PROXY_PORT` | `--port` | `8081` | Port the proxy listens on |
| `BONSAI_PROXY_HOST` | `--host` | `127.0.0.1` | Bind address |
| `BONSAI_UPSTREAM` | `--upstream` | `http://127.0.0.1:8080` | The llama-server to forward to |
| `BONSAI_COMPACT_AT` | `--threshold` | `0.75` | Starting value for the slider: auto-compact above this fraction of the context (0.1 to 0.95) |
| `BONSAI_COMPACT_AUTO` | `--no-auto` | on | `BONSAI_COMPACT_AUTO=0` / `--no-auto`: only compact on `/compact` |
| `BONSAI_COMPACT_CTX` | `--ctx` | from `/props` | Context size in tokens, if the server's value should be overridden |

Any OpenAI-compatible client can point at the proxy too (`http://localhost:8081/v1`).
Everything other than chat completions, including streaming, tool calls and images, passes
through unchanged.

## Notes

- Compaction is lossy by nature. For work where exact wording matters, copy it out first.
- The auto-compaction cache lives in the proxy process; restarting it just means the next
  oversized request summarizes again.
- Tests (mock server, no model needed): `python scripts/test_compact_proxy.py`.
