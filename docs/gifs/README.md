# Product Showcase GIFs

This folder holds GIFs used in the root [README.md](../../README.md) to showcase **Chat**, **Compare**, and **A2A**.

## Product experience

**Chat**

![Chat](chat.gif)

**Compare**

![Compare](compare.gif)

**A2A**

![A2A](a2a.gif)

---

## Required files

- `chat.gif` — Chat feature
- `compare.gif` — Compare feature  
- `a2a.gif` — A2A feature

## How to capture

Use a screen recorder that can export GIF (e.g. [LICEcap](https://www.cockos.com/licecap/), [Kap](https://getkap.co/), [ScreenToGif](https://www.screentogif.com/), or `ffmpeg`).

1. **Start the app**
   - Backend: `./start_backend.sh` (or run openint-backend + openint-agents)
   - UI: `cd openint-ui && npm run dev`
   - Open http://localhost:5173 (or your dev URL)

2. **Capture each GIF**

### Chat (`chat.gif`)

- Go to **Chat** (home).
- Type a natural-language question (e.g. *Show me transactions in California over $1000*).
- Click send or press Enter.
- Show the answer and source citations.
- Keep the clip short (e.g. 5–15 seconds).

### Compare (`compare.gif`)

- Go to **Compare**.
- Enter a sentence (e.g. from “I'm feeling lucky” or type one).
- Click **Compare**.
- Show the 3 model columns with highlighted spans (green = agreement, amber = disagreement).
- Optional: show “I'm feeling lucky” once.
- Keep the clip short (e.g. 5–15 seconds).

### A2A (`a2a.gif`)

- Go to **A2A**.
- Choose sentence count (e.g. 3).
- Click **Run A2A**.
- Show the wedge (sentence → 3 model outputs) and the result cards (sentence + 3 model annotations with tags).
- Keep the clip short (e.g. 10–20 seconds).

3. **Export as GIF**
   - Resolution: 800–1200 px width is enough for readability.
   - Frame rate: 5–10 fps to keep file size reasonable.
   - Save as `chat.gif`, `compare.gif`, `a2a.gif` in this folder (`docs/gifs/`).

4. **Commit**
   ```bash
   git add docs/gifs/*.gif
   git commit -m "Add product showcase GIFs (Chat, Compare, A2A)"
   ```

## Placeholder

If the GIF files are missing, the README will show broken image links until you add `chat.gif`, `compare.gif`, and `a2a.gif` here.
