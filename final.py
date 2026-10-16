import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from PIL import Image, ImageTk
import math, struct

try:
    import cv2 
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# colors
BG = "#1E1E2E"
SURFACE = "#2A2A3E"
ACCENT = "#7C6AF7"
ACCENT2 = "#A78BFA"
TEXT = "#E2E8F0"
TEXT_DIM = "#94A3B8"
SUCCESS = "#34D399"
WARNING = "#FBBF24"
DANGER = "#F87171"
BORDER= "#3A3A5C"

BTN_STYLE = dict(bg=ACCENT, fg="white", font=("Segoe UI", 13, "bold"),
                 relief="flat", cursor="hand2", padx=20, pady=10,
                 activebackground=ACCENT2, activeforeground="white")
BTN_SM = dict(bg=SURFACE, fg=TEXT, font=("Segoe UI", 10),
                 relief="flat", cursor="hand2", padx=12, pady=6,
                 activebackground=BORDER, activeforeground=TEXT)


# custom alert for informations, alerts and errors
_ALERT_CFG = {
    "info":    {"icon": "i", "color": ACCENT2, "btn_bg": ACCENT},
    "success": {"icon": "OK", "color": SUCCESS, "btn_bg": "#059669"},
    "warning": {"icon": "!", "color": WARNING, "btn_bg": "#D97706"},
    "error":   {"icon": "X", "color": DANGER, "btn_bg": "#DC2626"},
}

def custom_alert(parent, title: str, message: str, kind: str = "info"):
    cfg = _ALERT_CFG.get(kind, _ALERT_CFG["info"])
    dlg = tk.Toplevel(parent)
    dlg.title("")
    dlg.configure(bg=SURFACE)
    dlg.resizable(False, False)
    dlg.grab_set()
    dlg.focus_set()
    parent.update_idletasks()
    pw, ph = parent.winfo_width(), parent.winfo_height()
    px, py = parent.winfo_rootx(), parent.winfo_rooty()
    w, h = 420, 200
    dlg.geometry(f"{w}x{h}+{px+(pw-w)//2}+{py+(ph-h)//2}")
    tk.Frame(dlg, bg=cfg["color"], height=4).pack(fill="x")
    body = tk.Frame(dlg, bg=SURFACE, padx=24, pady=16)
    body.pack(fill="both", expand=True)
    hdr = tk.Frame(body, bg=SURFACE)
    hdr.pack(fill="x", pady=(0, 8))
    tk.Label(hdr, text=cfg["icon"], bg=cfg["color"], fg="white",
             font=("Segoe UI", 10, "bold"), width=3, pady=2).pack(side="left", padx=(0, 10))
    tk.Label(hdr, text=title, bg=SURFACE, fg=cfg["color"],
             font=("Segoe UI", 12, "bold")).pack(side="left")
    tk.Label(body, text=message, bg=SURFACE, fg=TEXT,
             font=("Segoe UI", 10), wraplength=370, justify="left").pack(anchor="w", pady=(0, 16))
    tk.Button(body, text="  OK  ", bg=cfg["btn_bg"], fg="white",
              font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2",
              padx=16, pady=6, activebackground=cfg["color"],
              activeforeground="white", command=dlg.destroy).pack(anchor="e")
    dlg.bind("<Return>", lambda e: dlg.destroy())
    dlg.bind("<Escape>", lambda e: dlg.destroy())
    dlg.wait_window()

class _MessageBox:
    def showinfo(self, title, message, parent=None):
        custom_alert(parent, title, message, "success")
    def showwarning(self, title, message, parent=None):
        custom_alert(parent, title, message, "warning")
    def showerror(self, title, message, parent=None):
        custom_alert(parent, title, message, "error")

messagebox = _MessageBox()

# vowels
VOWELS = set("aeiouăâîAEIOUĂÂÎ")

def split_layers(text: str):
    b_parts, c_parts = [], []
    i = 0
    while i < len(text):
        if text[i] in VOWELS:
            group = []
            while i < len(text) and text[i] in VOWELS:
                group.append(text[i]); i += 1
            b_parts.append(f"|{len(group)}")
            c_parts.extend(group)
        else:
            b_parts.append(text[i]); i += 1
    return "".join(b_parts), "".join(c_parts)

def reconstruct(b: str, c: str) -> str:
    result, ci, bi = [], 0, 0
    while bi < len(b):
        if b[bi] == '|':
            bi += 1
            digits = []
            while bi < len(b) and b[bi].isdigit():
                digits.append(b[bi]); bi += 1
            k = int("".join(digits)) if digits else 0
            for _ in range(k):
                if ci < len(c):
                    result.append(c[ci]); ci += 1
        else:
            result.append(b[bi]); bi += 1
    return "".join(result)

# HAAR DWT
def haar_dwt2(arr):
    a = arr.astype(np.float64)
    L = (a[:, 0::2] + a[:, 1::2]) / np.sqrt(2)
    H = (a[:, 0::2] - a[:, 1::2]) / np.sqrt(2)
    LL = (L[0::2] + L[1::2]) / np.sqrt(2)
    LH = (L[0::2] - L[1::2]) / np.sqrt(2)
    HL = (H[0::2] + H[1::2]) / np.sqrt(2)
    HH = (H[0::2] - H[1::2]) / np.sqrt(2)
    return LL, LH, HL, HH

def haar_idwt2(LL, LH, HL, HH):
    r, c = LL.shape
    L = np.zeros((r*2, c)); H = np.zeros((r*2, c))
    L[0::2] = (LL + LH) / np.sqrt(2); L[1::2] = (LL - LH) / np.sqrt(2)
    H[0::2] = (HL + HH) / np.sqrt(2); H[1::2] = (HL - HH) / np.sqrt(2)
    out = np.zeros((r*2, c*2))
    out[:, 0::2] = (L + H) / np.sqrt(2); out[:, 1::2] = (L - H) / np.sqrt(2)
    return out

DELIM_A = b'\xDE\xAD\xBE\xEF'
DELIM_B = b'\xCA\xFE\xBA\xBE'
DELIM_C = b'\xF0\x0D\xF0\x0D'

def bytes_to_bits(data):
    return [int(b) for byte in data for b in format(byte, '08b')]

def bits_to_bytes(bits):
    return bytes(int("".join(str(x) for x in bits[i:i+8]), 2)
                 for i in range(0, len(bits)-7, 8))

def encode_payload(msg: str, delim: bytes) -> bytes:
    p = msg.encode('utf-8')
    return delim + struct.pack('>I', len(p)) + p

def decode_payload(data: bytes, delim: bytes):
    idx = data.find(delim)
    if idx == -1: return None
    s = idx + len(delim)
    if s + 4 > len(data): return None
    n = struct.unpack('>I', data[s:s+4])[0]
    t = s + 4
    if t + n > len(data): return None
    try: return data[t:t+n].decode('utf-8')
    except: return None


def encode_payload_c(vowels: str, blocks: list) -> bytes:
    meta = struct.pack('>H', len(blocks))
    for bi, bj in blocks:
        meta += struct.pack('>HH', bi, bj)
    body = meta + vowels.encode('utf-8')
    return DELIM_C + struct.pack('>I', len(body)) + body

def decode_payload_c(data: bytes):
    idx = data.find(DELIM_C)
    if idx == -1: return "", []
    s = idx + len(DELIM_C)
    if s + 4 > len(data): return "", []
    n = struct.unpack('>I', data[s:s+4])[0]
    cs = s + 4
    if cs + n > len(data): return "", []
    content = data[cs:cs+n]
    if len(content) < 2: return "", []
    nb = struct.unpack('>H', content[:2])[0]
    me = 2 + nb * 4
    if len(content) < me: return "", []
    blocks = [struct.unpack('>HH', content[2+i*4:6+i*4]) for i in range(nb)]
    try: vowels = content[me:].decode('utf-8')
    except: vowels = ""
    return vowels, blocks

# layer a

def embed_layer_a(img_arr, message: str):
    arr = img_arr.copy()
    h, w = arr.shape[:2]
    usable = int(h * w * 0.03)
    payload = encode_payload(message, DELIM_A)
    bits = bytes_to_bits(payload)
    if len(bits) > usable:
        raise ValueError(f"Layer A: mesaj prea lung ({len(bits)} biti > {usable} disponibili)")
    for idx in range(len(bits)):
        row, col = divmod(idx, w)
        arr[row, col, 2] = (arr[row, col, 2] & 0xFE) | bits[idx]
    return arr.astype(np.uint8)

def extract_layer_a(img_arr):
    h, w = img_arr.shape[:2]
    usable = int(h * w * 0.03)
    bits = [img_arr[divmod(p, w)[0], divmod(p, w)[1], 2] & 1 for p in range(usable)]
    return decode_payload(bits_to_bytes(bits), DELIM_A) or ""

def capacity_a(img_arr):
    h, w = img_arr.shape[:2]
    return max(0, int(h*w*0.03)//8 - len(DELIM_A) - 4)

# layer b

BLOCK = 8
ALPHA_B = 1.5

def _orb_blocks(channel: np.ndarray):
    h, w = channel.shape
    bh, bw = h // BLOCK, w // BLOCK
    orb_list = []

    if CV2_AVAILABLE:
        orb = cv2.ORB_create(nfeatures=2000, fastThreshold=10)
        kps = orb.detect(channel.astype(np.uint8), None)
        kps = sorted(kps, key=lambda k: k.response, reverse=True)
        seen = set()
        for kp in kps:
            bj = int(kp.pt[0]) // BLOCK
            bi = int(kp.pt[1]) // BLOCK
            if 0 <= bi < bh and 0 <= bj < bw and (bi, bj) not in seen:
                seen.add((bi, bj))
                orb_list.append((bi, bj))
        raster_rest = [(bi, bj) for bi in range(bh) for bj in range(bw)
                       if (bi, bj) not in seen]
    else:
        raster_rest = [(bi, bj) for bi in range(bh) for bj in range(bw)]

    return orb_list + raster_rest

def embed_layer_b(img_arr, message: str):
    arr = img_arr.copy().astype(np.float64)
    payload = encode_payload(message, DELIM_B)
    bits = bytes_to_bits(payload)
    n_bits = len(bits)

    channel = arr[:, :, 1].copy()
    blocks = _orb_blocks(channel)

    blocks_used = []
    bit_idx = 0

    for bi, bj in blocks:
        if bit_idx >= n_bits:
            break

        blk = channel[bi*BLOCK:(bi+1)*BLOCK, bj*BLOCK:(bj+1)*BLOCK]
        LL, LH, HL, HH = haar_dwt2(blk)
        flat = LH.flatten()

        bits_in_block = min(16, n_bits - bit_idx)
        for m in range(bits_in_block):
            flat[m] = ALPHA_B if bits[bit_idx + m] == 1 else -ALPHA_B

        blk_new = haar_idwt2(LL, flat.reshape(LH.shape), HL, HH)
        blk_clipped = np.clip(blk_new, 0, 255)
        _, LH_check, _, _ = haar_dwt2(blk_clipped)
        flat_check  = LH_check.flatten()

        signs_ok = all(
            (flat_check[m] > 0) == (flat[m] > 0)
            for m in range(bits_in_block)
        )

        if not signs_ok:
            continue

        channel[bi*BLOCK:(bi+1)*BLOCK, bj*BLOCK:(bj+1)*BLOCK] = blk_clipped
        blocks_used.append((bi, bj))
        bit_idx += bits_in_block

    if bit_idx < n_bits:
        raise ValueError(
            f"Layer B: nu s-au gasit suficiente blocuri stabile "
            f"({len(blocks_used)} gasite, {math.ceil(n_bits/16)} necesare)")

    arr[:, :, 1] = np.clip(channel, 0, 255)
    return np.clip(arr, 0, 255).astype(np.uint8), blocks_used

def extract_layer_b(img_arr, blocks_used: list):
    if not blocks_used:
        return ""
    channel  = img_arr[:, :, 1].astype(np.float64)
    all_bits = []
    for bi, bj in blocks_used:
        blk = channel[bi*BLOCK:(bi+1)*BLOCK, bj*BLOCK:(bj+1)*BLOCK]
        _, LH, _, _ = haar_dwt2(blk)
        flat = LH.flatten()
        for m in range(16):
            all_bits.append(1 if flat[m] > 0 else 0)
    return decode_payload(bits_to_bytes(all_bits), DELIM_B) or ""

def capacity_b(img_arr):
    h, w = img_arr.shape[:2]
    n = (h//BLOCK) * (w//BLOCK)
    return max(0, (n*16 - (len(DELIM_B)+4)*8) // 8)

# layer c

SS_KEY = 42
ALPHA_C = 10.0
SPREAD = 8 

def _pn_seq(bit_idx: int) -> list:
    rng = np.random.default_rng(SS_KEY * 100003 + bit_idx)
    return [1.0 if v == 1 else -1.0 for v in rng.integers(0, 2, size=SPREAD)]

def _raster_blocks(channel):
    h, w = channel.shape
    return [(bi, bj) for bi in range(h//BLOCK) for bj in range(w//BLOCK)]

def embed_layer_c(img_arr, vowels: str, blocks_used: list):
    arr = img_arr.copy().astype(np.float64)
    payload = encode_payload_c(vowels, blocks_used)
    bits = bytes_to_bits(payload)
    n_bits = len(bits)

    channel = arr[:, :, 0].copy()
    blocks = _raster_blocks(channel)
    bpb = 16 // SPREAD 
    needed = math.ceil(n_bits / bpb)

    if needed > len(blocks):
        raise ValueError(f"Layer C: payload prea lung ({needed} blocuri nec., {len(blocks)} disponibil)")

    bit_idx = 0
    for k in range(needed):
        bi, bj = blocks[k]
        blk = channel[bi*BLOCK:(bi+1)*BLOCK, bj*BLOCK:(bj+1)*BLOCK]
        LL, LH, HL, HH = haar_dwt2(blk)
        flat = HL.flatten()
        for b in range(bpb):
            if bit_idx >= n_bits: break
            sym = 1.0 if bits[bit_idx] == 1 else -1.0
            pn = _pn_seq(bit_idx)
            base = b * SPREAD
            for c in range(SPREAD):
                flat[base + c] = ALPHA_C * sym * pn[c]
            bit_idx += 1
        blk_new = haar_idwt2(LL, LH, flat.reshape(HL.shape), HH)
        channel[bi*BLOCK:(bi+1)*BLOCK, bj*BLOCK:(bj+1)*BLOCK] = np.clip(blk_new, 0, 255)

    arr[:, :, 0] = np.clip(channel, 0, 255)
    return np.clip(arr, 0, 255).astype(np.uint8)

def extract_layer_c(img_arr):
    channel = img_arr[:, :, 0].astype(np.float64)
    blocks = _raster_blocks(channel)
    bpb = 16 // SPREAD
    all_bits = []
    bit_idx = 0
    for bi, bj in blocks:
        blk = channel[bi*BLOCK:(bi+1)*BLOCK, bj*BLOCK:(bj+1)*BLOCK]
        _, _, HL, _ = haar_dwt2(blk)
        flat = HL.flatten()
        for b in range(bpb):
            pn = _pn_seq(bit_idx)
            base = b * SPREAD
            corr = sum(flat[base+c] * pn[c] for c in range(SPREAD))
            all_bits.append(1 if corr > 0 else 0)
            bit_idx += 1
    return decode_payload_c(bits_to_bytes(all_bits))

def capacity_c(img_arr):
    h, w = img_arr.shape[:2]
    n = (h//BLOCK) * (w//BLOCK)
    bpb = 16 // SPREAD
    bits_total = n * bpb
    overhead = (10 + 400) * 8
    return max(0, (bits_total - overhead) // 8)


# metrics
def calc_psnr_mse(orig, stego):
    o = orig.astype(np.float64); s = stego.astype(np.float64)
    mse = np.mean((o - s)**2)
    if mse == 0: return float('inf'), 0.0
    return 10 * math.log10(255**2 / mse), mse

# for images
def load_image_as_rgb(path):
    img = Image.open(path)
    if img.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        if 'A' in img.mode:
            background.paste(img, mask=img.split()[-1])
        else:
            background.paste(img)
        return np.array(background, dtype=np.uint8)
    return np.array(img.convert('RGB'), dtype=np.uint8)

def arr_to_pil(arr):
    return Image.fromarray(arr.astype(np.uint8), 'RGB')

def make_thumb(pil_img, max_w=280, max_h=280):
    img = pil_img.copy()
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(img)

# for gui
def card(parent, **kw):
    return tk.Frame(parent, bg=SURFACE, bd=0, relief="flat", **kw)

def scroll_text(parent, height=4, **kw):
    frame = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
    txt = tk.Text(frame, height=height, bg="#252535", fg=TEXT,
                    font=("Consolas", 10), insertbackground=TEXT,
                    relief="flat", bd=4, wrap="word", **kw)
    sb = ttk.Scrollbar(frame, command=txt.yview)
    txt.configure(yscrollcommand=sb.set)
    txt.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")
    return frame, txt

# main window 
class MainWindow:
    def __init__(self, root):
        self.root = root
        root.title("Steganography")
        root.configure(bg=BG)
        root.geometry("400x320")
        root.resizable(False, False)

        tk.Label(root, text="Steganography", bg=BG, fg=TEXT,
                 font=("Segoe UI", 20, "bold")).pack(pady=(46, 90))        


        bf = tk.Frame(root, bg=BG); bf.pack()
        tk.Button(bf, text="Hide",   width=12, command=self.open_hide, **BTN_STYLE).grid(row=0, column=0, padx=12)
        tk.Button(bf, text="Decode", width=12, command=self.open_decode, **BTN_STYLE).grid(row=0, column=1, padx=12)

    def open_hide(self):   HideWindow(tk.Toplevel(self.root), self.root)
    def open_decode(self): DecodeWindow(tk.Toplevel(self.root), self.root)


# hide window
class HideWindow:
    def __init__(self, win, parent):
        self.win = win; self.parent = parent
        self.orig_arr = None; self.stego_arr = None
        win.title("Hide")
        win.configure(bg=BG)
        win.geometry("980x720")
        win.grab_set()
        self._build_ui()

    def _build_ui(self):
        win = self.win
        top = tk.Frame(win, bg=SURFACE, pady=10)
        top.pack(fill="x")
        tk.Label(top, text="Hide Message", bg=SURFACE, fg=ACCENT2,
                 font=("Segoe UI", 14, "bold")).pack(side="left", padx=20)
        tk.Button(top, text="<- Back", command=win.destroy, **BTN_SM).pack(side="right", padx=16)

        body = tk.Frame(win, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=10)

        # left preview
        left = card(body, width=310)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)
        tk.Label(left, text="Image Preview", bg=SURFACE, fg=TEXT_DIM,
                 font=("Segoe UI", 9, "bold")).pack(pady=(10, 4))
        self.preview_lbl = tk.Label(left, bg="#1A1A2E", text="No image loaded",
                                    fg=TEXT_DIM, font=("Segoe UI", 10))
        self.preview_lbl.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        tk.Button(left, text="Load Image", command=self._load_image,
                  **BTN_STYLE).pack(fill="x", padx=12, pady=(0, 10))

        # right
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        # capacity card
        cc = card(right); cc.pack(fill="x", pady=(0, 8))
        tk.Label(cc, text="Image Capacity", bg=SURFACE, fg=ACCENT2,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(8, 2))
        self.cap_labels = {}
        for key, name in [("size","Dimensiune"), ("A","Layer A (LSB 3%)"),
                          ("B","Layer B (DWT+ORB)"), ("C","Layer C (DSSS)"), ("total","Total")]:
            row = tk.Frame(cc, bg=SURFACE); row.pack(fill="x", padx=12, pady=1)
            tk.Label(row, text=name+":", bg=SURFACE, fg=TEXT_DIM,
                     font=("Segoe UI", 9)).pack(side="left")
            lbl = tk.Label(row, text="—", bg=SURFACE, fg=SUCCESS, font=("Consolas", 9))
            lbl.pack(side="right")
            self.cap_labels[key] = lbl
        tk.Frame(cc, bg=BG, height=6).pack()

        mc = card(right); mc.pack(fill="both", expand=True, pady=(0, 8))

        def fwc(parent, label_text, height=3, readonly=False):
            hdr = tk.Frame(parent, bg=SURFACE); hdr.pack(fill="x", padx=12, pady=(8, 0))
            tk.Label(hdr, text=label_text, bg=SURFACE, fg=ACCENT2,
                     font=("Segoe UI", 9, "bold")).pack(side="left")
            ctr = tk.Label(hdr, text="0 chars", bg=SURFACE, fg=TEXT_DIM, font=("Segoe UI", 8))
            ctr.pack(side="right")
            frm, txt = scroll_text(parent, height=height)
            frm.pack(fill="x", padx=12, pady=(2, 4))
            if readonly: txt.config(state="disabled", bg="#1C1C2E")
            return txt, ctr

        self.fake_txt, self.fake_ctr = fwc(mc, "Mesaj Fake (Layer A)", height=2)
        self.real_txt, self.real_ctr = fwc(mc, "Mesaj Real (complet)", height=3)
        self.b_txt,    self.b_ctr    = fwc(mc, "Layer B — schelet generat", height=2, readonly=True)
        self.c_txt,    self.c_ctr    = fwc(mc, "Layer C — vocale generate", height=2, readonly=True)

        self.fake_txt.bind("<KeyRelease>", lambda e: self._upd(self.fake_txt, self.fake_ctr))
        self.real_txt.bind("<KeyRelease>", lambda e: self._upd(self.real_txt, self.real_ctr))

        # metrics card
        self.metrics_card = card(right)
        tk.Label(self.metrics_card, text="Metrici Calitate", bg=SURFACE, fg=ACCENT2,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12, pady=(6, 2))
        self.metrics_lbl = tk.Label(self.metrics_card, text="—", bg=SURFACE, fg=SUCCESS,
                                    font=("Consolas", 9))
        self.metrics_lbl.pack(anchor="w", padx=12, pady=(0, 6))

        # buttons
        br = tk.Frame(win, bg=BG); br.pack(fill="x", padx=16, pady=(0, 12))
        for txt, cmd in [("Generate Layers", self._generate_layers),
                         ("Hide", self._hide),
                         ("Save Stego Image", self._save)]:
            tk.Button(br, text=txt, command=cmd, **BTN_STYLE).pack(side="left", padx=6)

    def _upd(self, txt, ctr):
        ctr.config(text=f"{len(txt.get('1.0','end-1c'))} chars")

    def _set_ro(self, txt, ctr, val):
        txt.config(state="normal"); txt.delete("1.0","end"); txt.insert("1.0", val)
        txt.config(state="disabled"); ctr.config(text=f"{len(val)} chars")

    @staticmethod
    def _format_c(vowels: str, blocks: list) -> str:
        blocks_str = "".join(f"({bi},{bj})" for bi, bj in blocks)
        return vowels + blocks_str

    def _load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Imagini","*.png *.bmp *.tiff *.tif *.jpg *.jpeg"),("Toate","*.*")])
        if not path: return
        try:
            self.orig_arr = load_image_as_rgb(path)
            h, w = self.orig_arr.shape[:2]
            thumb = make_thumb(arr_to_pil(self.orig_arr))
            self.preview_lbl.config(image=thumb, text=""); self.preview_lbl.image = thumb
            cA, cB, cC = capacity_a(self.orig_arr), capacity_b(self.orig_arr), capacity_c(self.orig_arr)
            self.cap_labels["size"].config(text=f"{w}x{h} px")
            self.cap_labels["A"].config(text=f"aprox. {cA} caractere")
            self.cap_labels["B"].config(text=f"aprox. {cB} caractere")
            self.cap_labels["C"].config(text=f"aprox. {cC} caractere (incl. metadata)")
            self.cap_labels["total"].config(text=f"aprox. {cA+cB+cC} caractere")
        except Exception as e:
            messagebox.showerror("Eroare", str(e), parent=self.win)

    def _generate_layers(self):
        real = self.real_txt.get("1.0", "end-1c").strip()
        if not real:
            messagebox.showwarning("Atentie", "Introdu mesajul real mai intai.", parent=self.win); return
        b, c = split_layers(real)
        self._set_ro(self.b_txt, self.b_ctr, b)
        self._set_ro(self.c_txt, self.c_ctr, c)

    def _hide(self):
        if self.orig_arr is None:
            messagebox.showwarning("Atentie", "Incarca o imagine mai intai.", parent=self.win); return
        fake = self.fake_txt.get("1.0","end-1c").strip()
        real = self.real_txt.get("1.0","end-1c").strip()
        b_val = self.b_txt.get("1.0","end-1c")
        c_val = self.c_txt.get("1.0","end-1c")
        if not fake or not real:
            messagebox.showwarning("Atentie", "Completeaza ambele campuri de mesaj.", parent=self.win); return
        if not b_val.strip():
            messagebox.showwarning("Atentie", "Apasa 'Generate Layers' mai intai.", parent=self.win); return
        try:
            arr = embed_layer_a(self.orig_arr, fake)          # layer A
            arr, blocks_used = embed_layer_b(arr, b_val)      # layer B
            arr = embed_layer_c(arr, c_val, blocks_used)      # layer C
            self.stego_arr = arr

            # C with vowels and metadata
            self._set_ro(self.c_txt, self.c_ctr,
                         self._format_c(c_val, blocks_used))

            psnr, mse = calc_psnr_mse(self.orig_arr, arr)
            psnr_str = f"{psnr:.2f} dB" if psnr != float('inf') else "inf"
            self.metrics_lbl.config(
                text=f"PSNR: {psnr_str}   MSE: {mse:.4f}   Blocuri ORB Layer B: {len(blocks_used)}",
                fg=SUCCESS if psnr > 35 else WARNING)
            self.metrics_card.pack(fill="x", pady=(0, 8))

            thumb = make_thumb(arr_to_pil(arr))
            self.preview_lbl.config(image=thumb); self.preview_lbl.image = thumb

            messagebox.showinfo("Succes",
                f"Mesajele au fost ascunse!\nPSNR: {psnr_str}   MSE: {mse:.4f}\n"
                f"Blocuri Layer B (ORB): {len(blocks_used)}",
                parent=self.win)
        except Exception as e:
            messagebox.showerror("Eroare embedding", str(e), parent=self.win)

    def _save(self):
        if self.stego_arr is None:
            messagebox.showwarning("Atentie", "Apasa 'Hide' mai intai.", parent=self.win); return
        path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG","*.png")], initialfile="stego_image.png")
        if path:
            arr_to_pil(self.stego_arr).save(path, format="PNG")
            messagebox.showinfo("Salvat", f"Imaginea stego salvata:\n{path}", parent=self.win)


#decode window
class DecodeWindow:
    def __init__(self, win, parent):
        self.win = win; self.parent = parent
        self.stego_arr = None
        win.title("Decode")
        win.configure(bg=BG)
        win.geometry("940x700")
        win.grab_set()
        self._build_ui()

    def _build_ui(self):
        win = self.win
        top = tk.Frame(win, bg=SURFACE, pady=10); top.pack(fill="x")
        tk.Label(top, text="Decode Message", bg=SURFACE, fg=ACCENT2,
                 font=("Segoe UI", 14, "bold")).pack(side="left", padx=20)
        tk.Button(top, text="<- Back", command=win.destroy, **BTN_SM).pack(side="right", padx=16)

        body = tk.Frame(win, bg=BG); body.pack(fill="both", expand=True, padx=16, pady=10)

        # preview
        left = card(body, width=300); left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)
        tk.Label(left, text="Stego Image", bg=SURFACE, fg=TEXT_DIM,
                 font=("Segoe UI", 9, "bold")).pack(pady=(10, 4))
        self.preview_lbl = tk.Label(left, bg="#1A1A2E", text="No image loaded",
                                    fg=TEXT_DIM, font=("Segoe UI", 10))
        self.preview_lbl.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        tk.Button(left, text="Load Stego Image", command=self._load,
                  **BTN_STYLE).pack(fill="x", padx=12, pady=(0, 10))

        # results
        right = card(body); right.pack(side="left", fill="both", expand=True)

        def ro_field(parent, lbl_text, height=2):
            tk.Label(parent, text=lbl_text, bg=SURFACE, fg=ACCENT2,
                     font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12, pady=(8, 0))
            frm, txt = scroll_text(parent, height=height)
            frm.pack(fill="x", padx=12, pady=(2, 0))
            txt.config(state="disabled", bg="#1C1C2E")
            return txt

        self.a_txt = ro_field(right, "Layer A — Mesaj Fake (LSB)")
        self.b_txt = ro_field(right, "Layer B — Schelet (DWT+ORB)")
        self.c_txt = ro_field(right, "Layer C — Vocale (DSSS) + metadata blocuri")
        self.out_txt = ro_field(right, "Mesaj Reconstruit Final", height=3)

        self.meta_lbl = tk.Label(right, text="", bg=SURFACE, fg=TEXT_DIM,
                                 font=("Segoe UI", 8), anchor="w")
        self.meta_lbl.pack(fill="x", padx=12, pady=(2, 4))

        tk.Frame(right, bg=BG, height=4).pack()

        br = tk.Frame(win, bg=BG); br.pack(fill="x", padx=16, pady=(0, 12))
        tk.Button(br, text="Decode", command=self._decode, **BTN_STYLE).pack(side="left", padx=6)

    def _load(self):
        path = filedialog.askopenfilename(
            filetypes=[("PNG","*.png"),("Toate","*.*")])
        if not path: return
        try:
            self.stego_arr = load_image_as_rgb(path)
            thumb = make_thumb(arr_to_pil(self.stego_arr))
            self.preview_lbl.config(image=thumb, text=""); self.preview_lbl.image = thumb
        except Exception as e:
            messagebox.showerror("Eroare", str(e), parent=self.win)

    def _set_ro(self, widget, text):
        widget.config(state="normal"); widget.delete("1.0","end")
        widget.insert("1.0", text); widget.config(state="disabled")

    def _decode(self):
        if self.stego_arr is None:
            messagebox.showwarning("Atentie", "Incarca imaginea stego mai intai.", parent=self.win); return
        try:
            vowels, blocks_used = extract_layer_c(self.stego_arr)
            b = extract_layer_b(self.stego_arr, blocks_used)
            a = extract_layer_a(self.stego_arr)
            reconstructed = reconstruct(b, vowels) if b and vowels else (b or "")

            self._set_ro(self.a_txt, a or "[nimic extras]")
            self._set_ro(self.b_txt, b or "[nimic extras]")
            # layer C raw
            c_display = (vowels + "".join(f"({bi},{bj})" for bi, bj in blocks_used)
                         ) if vowels or blocks_used else "[nimic extras]"
            self._set_ro(self.c_txt,   c_display)
            self._set_ro(self.out_txt, reconstructed or "[nu s-a putut reconstitui]")
            self.meta_lbl.config(
                text=f"Metadata din Layer C: {len(blocks_used)} locatii blocuri Layer B recuperate")

            messagebox.showinfo("Decodare completa",
                f"Toate straturile au fost extrase cu succes!\n"
                f"Blocuri Layer B recuperate din Layer C: {len(blocks_used)}",
                parent=self.win)
        except Exception as e:
            messagebox.showerror("Eroare decodare", str(e), parent=self.win)

# entry point
if __name__ == "__main__":
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()
