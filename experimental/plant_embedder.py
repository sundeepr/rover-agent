"""
experimental/plant_embedder.py

Captures frames from left/right wheel cameras, extracts green plant blobs,
computes DINOv2 embeddings for each blob, and stores them in a SQLite
database for later labeling (weed vs crop classification).

Workflow
--------
1. Capture frames from left/right cameras (local USB or WebSocket)
2. Detect green blobs via NGRDI vegetation index
3. For each blob, crop the bounding box, resize to 224×224
4. Compute DINOv2 (ViT-S/14) embedding — 384-dim float32 vector
5. Store embedding + JPEG crop + metadata in SQLite

Database schema
---------------
  plants(id, ts, session, camera, frame_idx, blob_idx,
         bbox_x, bbox_y, bbox_w, bbox_h, area,
         embedding BLOB, image_jpeg BLOB,
         label TEXT,   -- 'crop' | 'weed' | NULL = unlabeled
         notes TEXT)

Usage
-----
  # Capture + embed
  python experimental/plant_embedder.py \\
      --left-cam /dev/cam-left \\
      --right-cam /dev/cam-right \\
      --db plant_embeddings.db \\
      --session field_run_01

  # WebSocket cameras
  python experimental/plant_embedder.py \\
      --left-cam ws://192.168.1.10:5010 \\
      --right-cam ws://192.168.1.10:5011 \\
      --db plant_embeddings.db

  # Label existing entries interactively
  python experimental/plant_embedder.py \\
      --db plant_embeddings.db \\
      --label

Controls (live capture window)
------------------------------
  s     — save blobs from current frame to DB
  SPACE — skip frame without saving
  q     — quit

Controls (label mode window)
-----------------------------
  c     — label as CROP
  w     — label as WEED
  d     — delete this entry
  n     — skip / next
  q     — quit

Requirements
------------
  pip install torch torchvision transformers pillow opencv-python-headless
"""

import argparse
import sqlite3
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── Add parent dir so we can import frame_source ─────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frame_source import open_frame_source

# ── Constants ─────────────────────────────────────────────────────────────────
_EMBED_DIM      = 384          # DINOv2 ViT-S/14 CLS token dimension
_CROP_SIZE      = 224          # DINOv2 input size
_MIN_BLOB_AREA  = 800          # px² — ignore tiny blobs
_BLOB_PAD       = 20           # px — padding around bounding box
_VEG_THRESHOLD  = 20           # NGRDI score threshold (0–255 after scaling)
_DISPLAY_W      = 640
_DISPLAY_H      = 480


# ── Vegetation detection ──────────────────────────────────────────────────────

def _ngrdi_mask(frame_bgr: np.ndarray, threshold: int = _VEG_THRESHOLD) -> np.ndarray:
    """NGRDI = (G-R)/(G+R+ε), scaled to [0,255], thresholded."""
    b = frame_bgr[:, :, 0].astype(np.float32)
    g = frame_bgr[:, :, 1].astype(np.float32)
    r = frame_bgr[:, :, 2].astype(np.float32)
    vi  = (g - r) / (g + r + 1e-6)
    vi8 = (vi * 255).clip(0, 255).astype(np.uint8)
    _, mask = cv2.threshold(vi8, threshold, 255, cv2.THRESH_BINARY)
    return mask


def extract_blobs(frame_bgr: np.ndarray,
                  min_area: int = _MIN_BLOB_AREA,
                  pad: int = _BLOB_PAD):
    """
    Return list of (contour, bbox, crop_bgr) for each detected plant blob.
    bbox = (x, y, w, h) in frame pixel coordinates.
    """
    mask = _ngrdi_mask(frame_bgr)
    # Morphological clean-up: close small holes, remove tiny speckles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    h, w  = frame_bgr.shape[:2]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        # Pad bounding box, clamp to frame
        x1 = max(0, x - pad);       y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad);  y2 = min(h, y + bh + pad)
        crop = frame_bgr[y1:y2, x1:x2]
        blobs.append((cnt, (x1, y1, x2 - x1, y2 - y1), crop, int(area)))
    return blobs


# ── DINOv2 embedding ──────────────────────────────────────────────────────────

class DINOv2Embedder:
    """
    Wraps facebook/dinov2-small (ViT-S/14) to produce 384-dim CLS embeddings.
    Lazy-loaded on first call.  Falls back gracefully if torch is unavailable.
    """

    def __init__(self, device: str = "cpu"):
        self._device = device
        self._model  = None
        self._proc   = None

    def _load(self):
        try:
            from transformers import AutoImageProcessor, AutoModel
            import torch
            print("Loading DINOv2 (facebook/dinov2-small) …", flush=True)
            self._proc  = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
            self._model = AutoModel.from_pretrained("facebook/dinov2-small")
            self._model.eval()
            self._model.to(self._device)
            self._torch = torch
            print("DINOv2 ready.", flush=True)
        except ImportError as e:
            print(f"[WARN] DINOv2 unavailable ({e}) — embeddings will be zeros.", flush=True)
            self._model = "unavailable"

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Return 384-dim float32 embedding for a BGR crop image."""
        if self._model is None:
            self._load()
        if self._model == "unavailable":
            return np.zeros(_EMBED_DIM, dtype=np.float32)

        from PIL import Image
        img_rgb = cv2.cvtColor(
            cv2.resize(crop_bgr, (_CROP_SIZE, _CROP_SIZE)), cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        inputs  = self._proc(images=pil_img, return_tensors="pt")
        inputs  = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            out = self._model(**inputs)
        # CLS token: shape [1, 384]
        return out.last_hidden_state[:, 0, :].squeeze().cpu().numpy().astype(np.float32)


# ── Database ──────────────────────────────────────────────────────────────────

def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS plants (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          REAL    NOT NULL,
            session     TEXT,
            camera      TEXT    NOT NULL,
            frame_idx   INTEGER,
            blob_idx    INTEGER,
            bbox_x      INTEGER, bbox_y INTEGER,
            bbox_w      INTEGER, bbox_h INTEGER,
            area        INTEGER,
            embedding   BLOB    NOT NULL,
            image_jpeg  BLOB    NOT NULL,
            label       TEXT    DEFAULT NULL,
            notes       TEXT    DEFAULT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON plants(label)")
    conn.commit()
    return conn


def save_plant(conn: sqlite3.Connection,
               session: str,
               camera: str,
               frame_idx: int,
               blob_idx: int,
               bbox: tuple,
               area: int,
               embedding: np.ndarray,
               crop_bgr: np.ndarray) -> int:
    _, jpeg = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    emb_bytes = embedding.tobytes()
    x, y, w, h = bbox
    cur = conn.execute("""
        INSERT INTO plants
          (ts, session, camera, frame_idx, blob_idx,
           bbox_x, bbox_y, bbox_w, bbox_h, area,
           embedding, image_jpeg)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (time.time(), session, camera, frame_idx, blob_idx,
          x, y, w, h, area, emb_bytes, jpeg.tobytes()))
    conn.commit()
    return cur.lastrowid


def load_embedding(row) -> np.ndarray:
    return np.frombuffer(row, dtype=np.float32).copy()


# ── Drawing helpers ───────────────────────────────────────────────────────────

def annotate_frame(frame: np.ndarray, blobs, saved: bool = False) -> np.ndarray:
    out = frame.copy()
    for i, (cnt, (x, y, w, h), _, area) in enumerate(blobs):
        col = (0, 200, 50) if saved else (0, 140, 255)
        cv2.drawContours(out, [cnt], -1, col, 2)
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 1)
        cv2.putText(out, f"#{i} {area}px", (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    status = "SAVED" if saved else f"{len(blobs)} blobs  [s]=save  [SPACE]=skip  [q]=quit"
    cv2.putText(out, status, (8, out.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 0), 1)
    return out


# ── Capture mode ──────────────────────────────────────────────────────────────

def run_capture(args):
    embedder  = DINOv2Embedder(device=args.device)
    conn      = init_db(args.db)
    session   = args.session or datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_idx = 0

    # Open cameras
    sources = {}
    if args.left_cam:
        src = open_frame_source(args.left_cam, "left", fps=10)
        if src:
            sources["left"] = src
    if args.right_cam:
        src = open_frame_source(args.right_cam, "right", fps=10)
        if src:
            sources["right"] = src

    if not sources:
        print("No cameras available — check --left-cam / --right-cam", flush=True)
        return

    # Wait for WebSocket sources to connect
    print("Waiting for cameras to connect…", flush=True)
    deadline = time.time() + 15
    while time.time() < deadline:
        if all(s.is_open() for s in sources.values()):
            break
        time.sleep(0.3)

    print(f"Session: {session}")
    print(f"DB: {args.db}")
    print(f"Cameras: {list(sources.keys())}")
    print("Controls: [s]=save blobs  [SPACE]=skip  [q]=quit")

    total_saved = 0

    while True:
        frame_idx += 1
        saved_this_frame = False

        for cam_name, src in sources.items():
            frame = src.read()
            if frame is None:
                continue

            blobs = extract_blobs(frame, min_area=args.min_area, pad=args.blob_pad)
            vis   = annotate_frame(frame, blobs, saved=False)
            vis   = cv2.resize(vis, (_DISPLAY_W, _DISPLAY_H))
            cv2.putText(vis, f"{cam_name.upper()} | frame {frame_idx} | session={session}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)
            cv2.imshow(f"Plant Embedder — {cam_name}", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"\nDone. Total saved: {total_saved}")
                for s in sources.values():
                    s.release()
                conn.close()
                cv2.destroyAllWindows()
                return

            if key == ord('s') and blobs:
                print(f"  Embedding {len(blobs)} blob(s) from {cam_name}…", end=" ", flush=True)
                for bi, (cnt, bbox, crop, area) in enumerate(blobs):
                    emb = embedder.embed(crop)
                    rid = save_plant(conn, session, cam_name, frame_idx, bi,
                                     bbox, area, emb, crop)
                    print(f"id={rid}", end=" ", flush=True)
                print(flush=True)
                total_saved += len(blobs)
                # Show saved confirmation
                vis2 = annotate_frame(frame, blobs, saved=True)
                vis2 = cv2.resize(vis2, (_DISPLAY_W, _DISPLAY_H))
                cv2.imshow(f"Plant Embedder — {cam_name}", vis2)
                cv2.waitKey(400)
                saved_this_frame = True

        # Pace the loop when no key pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    for s in sources.values():
        s.release()
    conn.close()
    cv2.destroyAllWindows()
    print(f"\nDone. Total saved: {total_saved}")


# ── Label mode ────────────────────────────────────────────────────────────────

def run_label(args):
    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        "SELECT id, camera, session, area, image_jpeg, label "
        "FROM plants ORDER BY id"
    ).fetchall()

    if not rows:
        print("No entries in database.")
        conn.close()
        return

    print(f"{len(rows)} entries. Controls: [c]=crop  [w]=weed  [d]=delete  [n]=skip  [q]=quit")
    labeled = 0

    for row in rows:
        pid, camera, session, area, jpeg_bytes, current_label = row
        buf   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        crop  = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if crop is None:
            continue

        # Display at a readable size
        display = cv2.resize(crop, (448, 448))
        status  = f"id={pid}  cam={camera}  session={session}  area={area}px  label={current_label or 'unlabeled'}"
        canvas  = np.zeros((500, 448, 3), dtype=np.uint8)
        canvas[:448, :] = display
        cv2.putText(canvas, status[:60], (4, 462),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.putText(canvas, "[c]=crop  [w]=weed  [d]=del  [n]=skip  [q]=quit",
                    (4, 482), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)
        cv2.imshow("Label Plants", canvas)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('c'):
                conn.execute("UPDATE plants SET label='crop' WHERE id=?", (pid,))
                conn.commit()
                labeled += 1
                print(f"  id={pid} → crop")
                break
            elif key == ord('w'):
                conn.execute("UPDATE plants SET label='weed' WHERE id=?", (pid,))
                conn.commit()
                labeled += 1
                print(f"  id={pid} → weed")
                break
            elif key == ord('d'):
                conn.execute("DELETE FROM plants WHERE id=?", (pid,))
                conn.commit()
                print(f"  id={pid} → deleted")
                break
            elif key == ord('n'):
                break
            elif key == ord('q'):
                print(f"\nLabeled {labeled} entries this session.")
                conn.close()
                cv2.destroyAllWindows()
                return

    conn.close()
    cv2.destroyAllWindows()
    print(f"\nAll entries reviewed. Labeled {labeled} this session.")
    _print_stats(args.db)


# ── Stats ─────────────────────────────────────────────────────────────────────

def _print_stats(db_path: str):
    conn = sqlite3.connect(db_path)
    total  = conn.execute("SELECT COUNT(*) FROM plants").fetchone()[0]
    crop   = conn.execute("SELECT COUNT(*) FROM plants WHERE label='crop'").fetchone()[0]
    weed   = conn.execute("SELECT COUNT(*) FROM plants WHERE label='weed'").fetchone()[0]
    unlbl  = conn.execute("SELECT COUNT(*) FROM plants WHERE label IS NULL").fetchone()[0]
    print(f"\nDB stats: total={total}  crop={crop}  weed={weed}  unlabeled={unlbl}")
    conn.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Plant blob embedder (DINOv2)")
    p.add_argument("--left-cam",   default=None,
                   help="Left camera device or ws:// URL")
    p.add_argument("--right-cam",  default=None,
                   help="Right camera device or ws:// URL")
    p.add_argument("--db",         default="plant_embeddings.db",
                   help="SQLite database path (default: plant_embeddings.db)")
    p.add_argument("--session",    default=None,
                   help="Session name for grouping captures (default: timestamp)")
    p.add_argument("--label",      action="store_true",
                   help="Run interactive labeling mode instead of capture")
    p.add_argument("--stats",      action="store_true",
                   help="Print database statistics and exit")
    p.add_argument("--device",     default="cpu",
                   help="PyTorch device for DINOv2 (cpu / cuda / mps, default: cpu)")
    p.add_argument("--min-area",   type=int, default=_MIN_BLOB_AREA,
                   help=f"Minimum blob area in px² (default {_MIN_BLOB_AREA})")
    p.add_argument("--blob-pad",   type=int, default=_BLOB_PAD,
                   help=f"Padding around each blob bounding box (default {_BLOB_PAD})")
    p.add_argument("--veg-threshold", type=int, default=_VEG_THRESHOLD,
                   help=f"NGRDI threshold 0-255 (default {_VEG_THRESHOLD})")
    args = p.parse_args()

    if args.stats:
        _print_stats(args.db)
        return

    if args.label:
        run_label(args)
    else:
        run_capture(args)


if __name__ == "__main__":
    main()
