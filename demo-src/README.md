# GitHub Pages demo

This is the standalone React/Vite demo embedded in the GitHub Pages project website. It checks the demo rule in the browser and needs no Python backend, Blender, API key, or Docker container.

## Preview the published website locally

From the ZendoWorld repository root on your Mac:

```bash
python -m http.server 8080 --bind 127.0.0.1 --directory docs
```

Open http://localhost:8080/ for the project website, or http://localhost:8080/demo/ for the demo directly. Stop the server with Ctrl+C.

## Develop the demo

```bash
cd demo-src
npm ci
npm run dev
```

Open the address Vite prints (usually http://localhost:5173/).

## Build

```bash
npm run build
```

The Vite config writes the compiled demo into `../docs/demo/`. The project landing page is `../docs/index.html`. Building locally updates these files; publishing requires committing and deploying them to GitHub Pages.
