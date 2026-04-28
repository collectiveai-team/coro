import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("./src/", import.meta.url));
const requestedPort = Number(process.env.PORT || 5177);
const hasExplicitPort = Boolean(process.env.PORT);

const contentTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
};

function resolvePath(urlPath) {
  const cleanPath = urlPath === "/" ? "/index.html" : decodeURIComponent(urlPath);
  const resolved = normalize(join(root, cleanPath));
  if (!resolved.startsWith(root)) return null;
  return resolved;
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
  const filePath = resolvePath(url.pathname);
  if (!filePath) {
    res.writeHead(403, { "content-type": "text/plain; charset=utf-8" });
    res.end("Forbidden");
    return;
  }

  try {
    const body = await readFile(filePath);
    res.writeHead(200, { "content-type": contentTypes[extname(filePath)] || "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
    res.end("Not found");
  }
});

function listen(port) {
  server.listen(port, "127.0.0.1");
}

server.on("listening", () => {
  const address = server.address();
  console.log(`Transcription compare app: http://localhost:${address.port}`);
});

server.on("error", (error) => {
  if (error.code === "EADDRINUSE" && !hasExplicitPort) {
    listen(requestedPort + 1);
    return;
  }
  throw error;
});

listen(requestedPort);
