"""Leitura de mão: landmarks (MediaPipe) + rede ONNX de linhas. Porte do quiz (qA4.html)."""
import math, sys
import numpy as np, cv2
import onnxruntime as ort
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision as mpv
from PIL import Image, ImageDraw, ImageOps

HERE = __file__.rsplit('/', 1)[0]
TC = [0.517969,0.906342,0.395638,0.811939,0.323677,0.679026,0.265929,0.571673,0.210353,0.509843,0.434432,0.511703,0.402061,0.365756,0.386467,0.271350,0.380352,0.192511,0.507119,0.498259,0.510014,0.321379,0.510534,0.212832,0.516650,0.129003,0.574118,0.518092,0.596654,0.358200,0.606185,0.261688,0.613913,0.177517,0.636313,0.564216,0.664468,0.447373,0.679090,0.374957,0.687863,0.302700]
S = 256
_sess = None
_hl = None


def sess():
    global _sess
    if _sess is None:
        _sess = ort.InferenceSession(HERE + '/palm.onnx', providers=['CPUExecutionProvider'])
    return _sess


def hl():
    global _hl
    if _hl is None:
        o = mpv.HandLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=HERE + '/hand.task'), num_hands=1,
                                      min_hand_detection_confidence=.3, min_hand_presence_confidence=.3)
        _hl = mpv.HandLandmarker.create_from_options(o)
    return _hl


def ai_lines(L):
    p = 1 / (1 + np.exp(-L.astype(np.float64).reshape(S, S)))
    PG = np.array([[80,176],[96,146],[108,126],[130,120],[148,125],[166,136],[186,150],[186,200],[168,236],[134,246],[106,232],[88,204]], np.int32)
    m = np.zeros((S, S), np.uint8); cv2.fillPoly(m, [PG], 1)
    p = p * m
    def at(x, y):
        return p[y, x] if 0 <= x < S and 0 <= y < S else 0.0
    def pk(x, y):
        x = int(x); y = int(y)
        return max(at(x, y), .85*at(x, y-1), .85*at(x, y+1), .85*at(x-1, y), .85*at(x+1, y))
    lerp = lambda a, b, t: a + (b-a)*t

    def dp(t0, t1, lo, hi, pr, f, wp, lam, maxd):
        T = t1 - t0 + 1; V = S
        sc = np.full((T, V), -1e9); bk = np.full((T, V), -1, np.int32)
        for k in range(T):
            t = t0 + k
            a = max(0, math.ceil(lo(t))); b = min(V-1, math.floor(hi(t)))
            for v in range(a, b+1):
                d = (v - pr(t)) / wp
                u = f(t, v) - .08*d*d
                if k == 0:
                    sc[0, v] = u; continue
                best = -1e9; bi = -1
                for dv in range(-maxd, maxd+1):
                    q = v + dv
                    if q < 0 or q >= V: continue
                    s = sc[k-1, q] - lam*abs(dv)
                    if s > best: best = s; bi = q
                sc[k, v] = best + u; bk[k, v] = bi
        bv = int(np.argmax(sc[T-1])); bs = sc[T-1, bv]
        if bs <= -1e9: return None
        out = [0]*T
        for k in range(T-1, -1, -1):
            out[k] = bv; bv = int(bk[k, bv])
            if bv < 0 and k > 0: return None
        return out

    def trim(pts, thr=.3, gmax=9):
        v = [pk(q[0], q[1]) for q in pts]; n = len(v)
        sm = [np.mean(v[max(0, i-2):min(n-1, i+2)+1]) for i in range(n)]
        best = None; bs = 0; st = -1; gap = 0; sm_sum = 0; last = -1
        for i in range(n+1):
            on = i < n and sm[i] >= thr
            if on:
                if st < 0: st = i; sm_sum = 0
                sm_sum += sm[i]; last = i; gap = 0
            elif st >= 0:
                gap += 1
                if gap > gmax or i == n:
                    if sm_sum > bs: bs = sm_sum; best = (st, last)
                    st = -1
        if not best: return None
        seg = pts[best[0]:best[1]+1]; sv = v[best[0]:best[1]+1]
        ln = sum(math.hypot(seg[i][0]-seg[i-1][0], seg[i][1]-seg[i-1][1]) for i in range(1, len(seg)))
        return {'pts': seg, 'len': ln, 'mean': float(np.mean(sv))}

    res = {}
    CX, CY, A0, A1 = 90, 195, -104, 56
    def pol(t, r):
        a = t*math.pi/180; return (CX + r*math.cos(a), CY + r*math.sin(a))
    lr = dp(A0, A1, lambda t: 20, lambda t: 66, lambda t: 39, lambda t, r: pk(round(pol(t, r)[0]), round(pol(t, r)[1])), 12, .1, 1)
    if lr:
        pts = []; lastq = None
        for k, r in enumerate(lr):
            q = tuple(int(round(c)) for c in pol(A0+k, r))
            if lastq is None or q != lastq: pts.append(q)
            lastq = q
        k0 = 0
        while k0 < len(pts) and pts[k0][0] < 97: k0 += 1
        res['life'] = trim(pts[k0:])
    else:
        res['life'] = None

    def supp(r, rad):
        if not r: return
        for q in r['pts']:
            x0 = int(round(q[0])); y0 = int(round(q[1]))
            for y in range(y0-rad, y0+rad+1):
                for x in range(x0-rad, x0+rad+1):
                    if 0 <= x < S and 0 <= y < S: p[y, x] *= .15
    p0 = p.copy(); supp(res['life'], 2)
    hpr = lambda x: lerp(146, 160, (x-126)/50)
    hy = dp(124, 188, lambda x: hpr(x)-18, lambda x: hpr(x)+16, hpr, lambda x, y: pk(x, y), 16, .12, 2)
    res['heart'] = trim([(124+k, y) for k, y in enumerate(hy)], .4, 4) if hy else None
    hmap = {}
    if res['heart']:
        for q in res['heart']['pts']: hmap[q[0]] = q[1]
    dpr = lambda x: lerp(146, 178, (x-98)/52)
    def head_run():
        dy = dp(90, 186, lambda x: max(dpr(x)-16, hmap[x]+4 if x in hmap else 0), lambda x: dpr(x)+22, dpr, lambda x, y: pk(x, y), 14, .12, 2)
        return trim([(90+k, y) for k, y in enumerate(dy)]) if dy else None
    hd = head_run()
    if not (hd and hd['len'] >= 28 and hd['mean'] >= .5):
        p[:] = p0
        h2 = head_run()
        if h2 and res['life']:
            near = 0
            for q in h2['pts']:
                m_ = min(math.hypot(q[0]-r[0], q[1]-r[1]) for r in res['life']['pts'])
                if m_ < 3.5: near += 1
            if len(h2['pts']) - near >= 10: hd = h2
        elif h2:
            hd = h2
    res['head'] = hd
    for k in ('heart', 'head', 'life'):
        r = res[k]
        if r and (r['len'] < 16 or r['mean'] < .42): res[k] = None
    for k in ('heart', 'head', 'life'):
        r = res[k]
        if not r: continue
        a = r['pts']; n = len(a)
        sm = [(sum(a[j][0] for j in range(max(0, i-4), min(n-1, i+4)+1))/(min(n-1, i+4)-max(0, i-4)+1),
               sum(a[j][1] for j in range(max(0, i-4), min(n-1, i+4)+1))/(min(n-1, i+4)-max(0, i-4)+1)) for i in range(n)]
        r['pts'] = [q for i, q in enumerate(sm) if i % 3 == 0 or i == n-1]
    return res


def analyze(path):
    """Retorna (imagem PIL RGB desenhada, dict linhas, info) ou levanta ValueError com motivo."""
    pil = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    if max(pil.size) > 1600:
        pil.thumbnail((1600, 1600))
    W, H = pil.size
    rgb = np.array(pil)
    det = hl().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not det.hand_landmarks:
        raise ValueError('sem_mao')
    lm = det.hand_landmarks[0]
    P = np.array([[q.x*W, q.y*H] for q in lm])
    T = np.array([[TC[2*i]*S, TC[2*i+1]*S] for i in range(21)])
    best = None
    for flip_ in (True, False):   # o gabarito espera o polegar à esquerda; testa os dois lados
        pts = np.array([[((1-q.x) if flip_ else q.x)*W, q.y*H] for q in lm])
        sc = min(1, 1.2*np.hypot(*(T[5]-T[17])) / np.hypot(*(pts[5]-pts[17])))
        h, _ = cv2.findHomography((pts*sc).astype(np.float32), T.astype(np.float32), cv2.RANSAC, 5.0)
        if h is None: continue
        proj = cv2.perspectiveTransform((pts*sc)[None].astype(np.float32), h)[0]
        err = float(np.mean(np.hypot(*(proj-T).T)))
        if best is None or err < best[0]:
            best = (err, flip_, sc, h)
    if best is None:
        raise ValueError('sem_homografia')
    err, flip_, sc, h = best
    cw = max(1, int(W*sc)); ch = max(1, int(H*sc))
    src = cv2.flip(rgb, 1) if flip_ else rgb
    small = cv2.resize(src, (cw, ch), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
    hi = np.linalg.inv(h)
    warp = cv2.warpPerspective(small, h, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    hsv = cv2.cvtColor(warp, cv2.COLOR_RGB2HSV)
    skin = (hsv[..., 0] <= 50) & (hsv[..., 1] >= 20) & (hsv[..., 2] >= 80) & (warp[..., 1] > 10)
    img = warp.copy(); img[~skin] = 255
    x = (img.astype(np.float32)/255).transpose(2, 0, 1)[None]
    y = sess().run(['output'], {'input': x})[0]
    ln = ai_lines(y)
    out = {}
    for k in ('heart', 'head', 'life'):
        r = ln.get(k)
        if not r: continue
        a = np.array(r['pts'], np.float64)
        q = cv2.perspectiveTransform(a[None], hi)[0]
        out[k] = [((W - qx/sc) if flip_ else qx/sc, qy/sc) for qx, qy in q]
    P = np.array([[q.x*W, q.y*H] for q in lm])
    return pil, P, out


COL = {'heart': (255, 107, 107), 'head': (242, 207, 107), 'life': (63, 208, 140)}
NOME = {'heart': 'Coração', 'head': 'Cabeça', 'life': 'Vida'}


def render(pil, P, lines, out_path, ow=900):
    """Mão em pé, recortada, com as linhas desenhadas (porte de aiGeo)."""
    W0, H0 = pil.size
    vy = P[9]-P[0]; Lp = float(np.hypot(*vy))
    ey = vy/Lp
    rot = -math.pi/2 - math.atan2(ey[1], ey[0]); rot = math.atan2(math.sin(rot), math.cos(rot))
    if abs(rot) < 12*math.pi/180: rot = 0
    piv = np.array([(P[:, 0].min()+P[:, 0].max())/2, (P[:, 1].min()+P[:, 1].max())/2])
    cr, sr = math.cos(rot), math.sin(rot)
    def rp(p):
        x, y = p[0]-piv[0], p[1]-piv[1]; return (x*cr - y*sr, x*sr + y*cr)
    Q = np.array([rp(p) for p in P])
    bx0, bx1 = Q[:, 0].min(), Q[:, 0].max(); by0 = Q[:, 1].min(); by1 = Q[:, 1].max() + Lp*.12
    AR = .72
    chh = max((by1-by0)*1.1, (bx1-bx0)*1.1/AR); cww = chh*AR
    ccx, ccy = (bx0+bx1)/2, (by0+by1)/2
    OW = ow; OH = round(ow/AR); k = OW/cww
    # matriz imagem original -> saída: rotaciona em torno de piv, centra em cc
    # saída = ((R(p-piv) - cc)*k + O/2)
    R = np.array([[cr, -sr], [sr, cr]])
    A = k*R
    b = k*(-R@piv - np.array([ccx, ccy])) + np.array([OW/2, OH/2])
    M = np.hstack([A, b[:, None]]).astype(np.float32)
    base = cv2.warpAffine(np.array(pil), M, (OW, OH), flags=cv2.INTER_LANCZOS4, borderValue=(31, 58, 69))
    im = Image.fromarray(base).convert('RGBA')
    ov = Image.new('RGBA', im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    def tr(p):
        q = A@np.array(p) + b; return (float(q[0]), float(q[1]))
    sw = max(5, int(Lp*k*.022))
    for n in ('life', 'head', 'heart'):
        if n in lines and len(lines[n]) > 3:
            pts = [tr(q) for q in lines[n]]
            d.line(pts, fill=COL[n]+(70,), width=sw*3, joint='curve')   # brilho
            d.line(pts, fill=COL[n]+(255,), width=sw, joint='curve')
            for q in (pts[0], pts[-1]):
                d.ellipse([q[0]-sw*.8, q[1]-sw*.8, q[0]+sw*.8, q[1]+sw*.8], fill=COL[n]+(255,))
    im = Image.alpha_composite(im, ov).convert('RGB')
    im.save(out_path, 'JPEG', quality=88)
    return out_path


if __name__ == '__main__':
    pil, P, ln = analyze(sys.argv[1])
    print({k: len(v) for k, v in ln.items()})
    render(pil, P, ln, sys.argv[2])
