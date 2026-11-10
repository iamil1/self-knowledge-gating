/* Local chat server: static UI + SSH-tunneled proxy to the remote RAG service.
 *
 * Config priority: ./config.local.json  >  environment variables.
 *   config.local.json: {"RAG_SSH_PASSWORD": "..."}  (required; git-ignored)
 *   RAG_SSH_HOST      default connect.bjb2.seetacloud.com
 *   RAG_SSH_PORT      default 17284
 *   RAG_REMOTE_PORT   default 18231
 *
 * Usage: npm run dev -- --host 127.0.0.1 --port 7100
 */
const fs = require("fs");

function loadLocalConfig() {
  try {
    const cfg = JSON.parse(fs.readFileSync(path.join(__dirname, "config.local.json"), "utf8"));
    for (const [k, v] of Object.entries(cfg)) {
      if (process.env[k] === undefined) process.env[k] = String(v);
    }
  } catch { /* no local config file */ }
}

const express = require("express");
const { Client } = require("ssh2");
const net = require("net");
const http = require("http");
const path = require("path");

loadLocalConfig();

function argValue(name, dflt) {
  const i = process.argv.indexOf(`--${name}`);
  return i > -1 && process.argv[i + 1] ? process.argv[i + 1] : dflt;
}
const HOST = argValue("host", "127.0.0.1");
const PORT = parseInt(argValue("port", "7100"), 10);

const SSH_HOST = process.env.RAG_SSH_HOST || "connect.bjb2.seetacloud.com";
const SSH_PORT = parseInt(process.env.RAG_SSH_PORT || "17284", 10);
const SSH_USER = "root";
const SSH_PASSWORD = process.env.RAG_SSH_PASSWORD;
const REMOTE_PORT = parseInt(process.env.RAG_REMOTE_PORT || "18231", 10);

if (!SSH_PASSWORD) {
  console.error("Missing RAG_SSH_PASSWORD env var");
  process.exit(1);
}

const app = express();
app.use(express.json({ limit: "1mb" }));
app.use(express.static(path.join(__dirname, "public")));

let tunnelLocalPort = null;
let tunnelReady = false;

function setupTunnel() {
  return new Promise((resolve, reject) => {
    const conn = new Client();
    conn
      .on("ready", () => {
        const server = net.createServer((sock) => {
          conn.forwardOut(sock.remoteAddress || "127.0.0.1",
            sock.remotePort || 0, "127.0.0.1", REMOTE_PORT,
            (err, stream) => {
              if (err) { sock.destroy(); return; }
              sock.pipe(stream).pipe(sock);
            });
        });
        server.listen(0, "127.0.0.1", () => {
          tunnelLocalPort = server.address().port;
          tunnelReady = true;
          console.log(`SSH tunnel up: 127.0.0.1:${tunnelLocalPort} -> remote:${REMOTE_PORT}`);
          resolve();
        });
      })
      .on("error", (err) => {
        tunnelReady = false;
        reject(err);
      })
      .connect({ host: SSH_HOST, port: SSH_PORT, username: SSH_USER,
                 password: SSH_PASSWORD, readyTimeout: 20000 });
  });
}

function proxyToRemote(reqBody) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(reqBody);
    const req = http.request({
      host: "127.0.0.1", port: tunnelLocalPort, path: "/chat",
      method: "POST",
      headers: { "Content-Type": "application/json",
                 "Content-Length": Buffer.byteLength(data) },
      timeout: 300000,
    }, (res) => {
      let body = "";
      res.on("data", (c) => (body += c));
      res.on("end", () => {
        try { resolve(JSON.parse(body)); }
        catch (e) { reject(new Error(`bad upstream json: ${body.slice(0, 200)}`)); }
      });
    });
    req.on("timeout", () => { req.destroy(new Error("upstream timeout")); });
    req.on("error", reject);
    req.write(data);
    req.end();
  });
}

app.get("/api/status", (req, res) => {
  res.json({ tunnel: tunnelReady, ssh: `${SSH_USER}@${SSH_HOST}:${SSH_PORT}` });
});

app.post("/api/chat", async (req, res) => {
  if (!tunnelReady) return res.status(503).json({ error: "tunnel not ready" });
  const question = (req.body && req.body.question || "").trim();
  if (!question) return res.status(400).json({ error: "empty question" });
  try {
    const out = await proxyToRemote({ question });
    res.json(out);
  } catch (e) {
    res.status(502).json({ error: String(e.message || e) });
  }
});

setupTunnel()
  .then(() => {
    app.listen(PORT, HOST, () => {
      console.log(`RAG chat UI: http://${HOST}:${PORT}`);
    });
  })
  .catch((err) => {
    console.error("SSH tunnel failed:", err.message);
    process.exit(1);
  });
