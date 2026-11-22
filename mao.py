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


def ai_lines(L, force=True):
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

    res = {}; raw = {}
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
        raw['life'] = pts[k0:]
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
    if hy: raw['heart'] = [(124+k, y) for k, y in enumerate(hy) if 128 <= 124+k <= 186]
    res['heart'] = trim([(124+k, y) for k, y in enumerate(hy)], .4, 4) if hy else None
    hmap = {}
    if res['heart']:
        for q in res['heart']['pts']: hmap[q[0]] = q[1]
    dpr = lambda x: lerp(146, 178, (x-98)/52)
    def head_run():
        dy = dp(90, 186, lambda x: max(dpr(x)-16, hmap[x]+4 if x in hmap else 0), lambda x: dpr(x)+22, dpr, lambda x, y: pk(x, y), 14, .12, 2)
        if dy: raw['head'] = [(90+k, y) for k, y in enumerate(dy) if 100 <= 90+k <= 160]
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
    if force:
        prior = {'heart': [(x, hpr(x)) for x in range(128, 181, 2)],
                 'head': [(x, dpr(x)) for x in range(102, 151, 2)],
                 'life': [pol(t, 39) for t in range(-75, 52, 4)]}
        def sobrepoe(pts, outros):
            # fracao de pontos a menos de 7 px de outra linha ja achada
            if not outros: return 0.0
            n = sum(1 for q in pts if min(math.hypot(q[0]-o[0], q[1]-o[1]) for o in outros) < 7)
            return n / len(pts)
        for k in ('heart', 'head', 'life'):
            if res[k]: continue
            outros = [q for j in ('heart', 'head', 'life') if res[j] for q in res[j]['pts']]
            pts = raw.get(k) or []
            if len(pts) < 12 or sobrepoe(pts, outros) > .25: pts = prior[k]
            res[k] = {'pts': pts, 'len': 0, 'mean': 0, 'forcada': True}
    # pos-processamento: dentro da palma, comprimento minimo e sem sobreposicao
    PGf = PG.astype(np.float32).reshape(-1, 1, 2)
    MARG, DMIN = 5, 7
    MINL = {'heart': 45, 'head': 45, 'life': 55}
    def plen(a):
        return sum(math.hypot(a[i][0]-a[i-1][0], a[i][1]-a[i-1][1]) for i in range(1, len(a)))
    RL = {'D': DMIN, 'M': MARG}
    def melhor(pts, outros):
        best = []; cur = []
        for q in pts:
            ok = cv2.pointPolygonTest(PGf, (float(q[0]), float(q[1])), True) >= RL['M'] and \
                 (not outros or min(math.hypot(q[0]-o[0], q[1]-o[1]) for o in outros) >= RL['D'])
            if ok: cur.append(q)
            else:
                if plen(cur) > plen(best): best = cur
                cur = []
        return cur if plen(cur) > plen(best) else best
    if force:
        ordem = sorted(('life', 'heart', 'head'), key=lambda k: bool(res[k].get('forcada')))
        feitas = {}
        for k in ordem:
            r = res[k]; outros = [q for v in feitas.values() for q in v]
            RL['D'], RL['M'] = DMIN, MARG
            seg = melhor(r['pts'], outros)
            for d_, m_ in ((DMIN, MARG), (5, 4), (4, 3), (3, 2)):
                RL['D'], RL['M'] = d_, m_
                s2 = melhor(r['pts'], outros)
                if plen(s2) > plen(seg): seg = s2
                if plen(seg) >= MINL[k]*.95: break
                cand = None; bl = plen(seg)
                for sh in (0, 5, -5, 10, -10, 15, -15, 20, -20):
                    pp = [(x, y+sh) if k != 'life' else (x+sh, y) for x, y in prior[k]]
                    sg = melhor(pp, outros)
                    if plen(sg) >= MINL[k]*.95: cand = sg; break
                    if plen(sg) > bl: bl = plen(sg); cand = sg
                if cand and plen(cand) > plen(seg):
                    seg = cand; r['forcada'] = True
                if plen(seg) >= MINL[k]*.95: break
            if len(seg) >= 2: r['pts'] = seg
            feitas[k] = r['pts']
    for k in ('heart', 'head', 'life'):
        r = res[k]
        if not r: continue
        a = r['pts']; n = len(a)
        sm = [(sum(a[j][0] for j in range(max(0, i-4), min(n-1, i+4)+1))/(min(n-1, i+4)-max(0, i-4)+1),
               sum(a[j][1] for j in range(max(0, i-4), min(n-1, i+4)+1))/(min(n-1, i+4)-max(0, i-4)+1)) for i in range(n)]
        r['pts'] = [q for i, q in enumerate(sm) if i % 3 == 0 or i == n-1]
    return res


# ============ METODO POR TRACOS: esqueleto do mapa da rede -> grafo -> linhas inteiras com forma anatomica ============
tr_S = 256
tr_PG = [[80,176],[96,146],[108,126],[130,120],[148,125],[166,136],[186,150],[186,200],[168,236],[134,246],[106,232],[88,204]]

def tr_lerp(a, b, t): return a + (b-a)*t
def tr_hpr(x): return tr_lerp(146, 160, (x-126)/50)
def tr_dpr(x): return tr_lerp(146, 178, (x-98)/52)
tr_CX, tr_CY = 90, 195
def tr_pol(t, r=39):
    a = t*math.pi/180; return (tr_CX + r*math.cos(a), tr_CY + r*math.sin(a))

def tr_zhang_suen(B):
    I = B.astype(np.uint8).copy()
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            P = np.pad(I, 1)
            p2 = P[:-2, 1:-1]; p3 = P[:-2, 2:]; p4 = P[1:-1, 2:]; p5 = P[2:, 2:]
            p6 = P[2:, 1:-1]; p7 = P[2:, :-2]; p8 = P[1:-1, :-2]; p9 = P[:-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            Bn = sum(n.astype(int) for n in nb)
            seq = nb + [p2]
            A = sum(((seq[i] == 0) & (seq[i+1] == 1)).astype(int) for i in range(8))
            if step == 0:
                c = (p2*p4*p6 == 0) & (p4*p6*p8 == 0)
            else:
                c = (p2*p4*p8 == 0) & (p2*p6*p8 == 0)
            m = (I == 1) & (Bn >= 2) & (Bn <= 6) & (A == 1) & c
            if m.any():
                I[m] = 0; changed = True
    return I

tr_N8 = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]

def tr_components(B, minsize):
    H, W = B.shape; lab = np.zeros(B.shape, np.int32); n = 0; out = B.copy()
    for y in range(H):
        for x in range(W):
            if B[y, x] and not lab[y, x]:
                n += 1; st = [(x, y)]; lab[y, x] = n; pix = []
                while st:
                    cx, cy = st.pop(); pix.append((cx, cy))
                    for dx, dy in tr_N8:
                        nx, ny = cx+dx, cy+dy
                        if 0 <= nx < W and 0 <= ny < H and B[ny, nx] and not lab[ny, nx]:
                            lab[ny, nx] = n; st.append((nx, ny))
                if len(pix) < minsize:
                    for cx, cy in pix: out[cy, cx] = 0
    return out

tr_RING = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]

def tr_graph(Sk):
    pts = set((int(x), int(y)) for y, x in zip(*np.nonzero(Sk)))
    def nbrs(p): return [(p[0]+dx, p[1]+dy) for dx, dy in tr_N8 if (p[0]+dx, p[1]+dy) in pts]
    def cn(p):
        r = [((p[0]+dx, p[1]+dy) in pts) for dx, dy in tr_RING]
        return sum(1 for i in range(8) if r[i] and not r[(i+1) % 8])
    # poda de espinhos curtos (ate 5 px)
    for _ in range(3):
        ends = sorted([p for p in pts if cn(p) == 1], key=lambda q: (q[1], q[0])); rem = False
        for e in ends:
            chain = [e]; vis = {e}; cur = e
            while True:
                nn = [q for q in nbrs(cur) if q not in vis]
                if not nn: break
                nn.sort(key=lambda r: abs(r[0]-cur[0]) + abs(r[1]-cur[1]))
                nx = nn[0]
                if cn(nx) >= 3:
                    if len(chain) <= 5:
                        for c in chain: pts.discard(c)
                        rem = True
                    break
                chain.append(nx); vis.add(nx); cur = nx
                if len(chain) > 5: break
        if not rem: break
    C = {p: cn(p) for p in pts}
    nodes = {p for p in pts if C[p] != 2}
    nodes_s = sorted(nodes, key=lambda q: (q[1], q[0]))
    cid = {}; k = 0
    for p in nodes_s:
        if p in cid: continue
        k += 1; st = [p]; cid[p] = k
        while st:
            c = st.pop()
            for q in nbrs(c):
                if q in nodes and q not in cid: cid[q] = k; st.append(q)
    edges = []; vis = set()
    for p in nodes_s:
        for q in nbrs(p):
            if q in nodes or q in vis: continue
            chain = [p, q]; vis.add(q); cur = q
            while True:
                nn = [r for r in nbrs(cur) if r not in vis and r != chain[-2]]
                nd = [r for r in nn if r in nodes and cid[r] != cid[p] or (r in nodes and len(chain) > 3)]
                free = [r for r in nn if r not in nodes]
                if free:
                    free.sort(key=lambda r: abs(r[0]-cur[0]) + abs(r[1]-cur[1]))
                    cur = free[0]; vis.add(cur); chain.append(cur); continue
                if nd:
                    chain.append(nd[0]); edges.append((cid[p], cid[nd[0]], chain))
                break
    return edges

def tr_plen(a):
    return sum(math.hypot(a[i][0]-a[i-1][0], a[i][1]-a[i-1][1]) for i in range(1, len(a)))

def tr_dir_at(a, start, n=6):
    if start:
        j = min(n, len(a)-1); return math.atan2(a[j][1]-a[0][1], a[j][0]-a[0][0])
    j = max(0, len(a)-1-n); return math.atan2(a[-1][1]-a[j][1], a[-1][0]-a[j][0])

def tr_paths(edges, maxturn=50, maxe=6):
    adj = {}
    for i, (a, b, ch) in enumerate(edges):
        adj.setdefault(a, []).append((i, b, ch)); adj.setdefault(b, []).append((i, a, ch[::-1]))
    out = []
    def dfs(node, pts, used, ne):
        if len(out) > 4000: return
        if tr_plen(pts) >= 15: out.append(list(pts))
        if ne >= maxe: return
        for i, nb, ch in adj.get(node, []):
            if i in used: continue
            if pts:
                d1 = tr_dir_at(pts, False); d2 = tr_dir_at(ch, True)
                da = abs(math.atan2(math.sin(d2-d1), math.cos(d2-d1)))
                if da > math.radians(maxturn): continue
            dfs(nb, pts + ch[1:] if pts else list(ch), used | {i}, ne+1)
    for n in adj:
        dfs(n, [], frozenset(), 0)
    # quebra caminhos em dobras bruscas (> 55 graus) e usa os pedacos como candidatos
    extra = []
    for p in out:
        cuts = [0]; w = 5
        for i in range(w, len(p)-w):
            a1 = math.atan2(p[i][1]-p[i-w][1], p[i][0]-p[i-w][0]); a2 = math.atan2(p[i+w][1]-p[i][1], p[i+w][0]-p[i][0])
            if abs(math.atan2(math.sin(a2-a1), math.cos(a2-a1))) > math.radians(55) and i - cuts[-1] > 3: cuts.append(i)
        if len(cuts) > 1:
            cuts.append(len(p)-1)
            for a, b in zip(cuts[:-1], cuts[1:]):
                q = p[a+2:b-1] if a else p[:b-1]
                if tr_plen(q) >= 15: extra.append(q)
    out += extra
    # remove duplicados (mesmo caminho invertido)
    uniq = {}
    for p in out:
        key = (tuple(p[0]), tuple(p[-1]), len(p))
        rk = (tuple(p[-1]), tuple(p[0]), len(p))
        if key in uniq or rk in uniq: continue
        uniq[key] = p
    return list(uniq.values())

def tr_smooth(a, it=3):
    a = [tuple(map(float, q)) for q in a]
    for _ in range(it):
        if len(a) < 3: break
        a = [a[0]] + [((a[i-1][0]+2*a[i][0]+a[i+1][0])/4, (a[i-1][1]+2*a[i][1]+a[i+1][1])/4) for i in range(1, len(a)-1)] + [a[-1]]
    return a

def tr_mind(q, curve):
    return min(math.hypot(q[0]-c[0], q[1]-c[1]) for c in curve)

tr_PRI = {'heart': [(x, tr_hpr(x)) for x in range(108, 192, 2)],
       'head': [(x, tr_dpr(x)) for x in range(94, 176, 2)],
       'life': [tr_pol(t) for t in range(-100, 62, 3)]}

def tr_feats(p, P):
    xs = [q[0] for q in p]; ys = [q[1] for q in p]
    mp = float(np.mean([P[int(q[1]), int(q[0])] for q in p]))
    return dict(L=tr_plen(p), mp=mp, x0=min(xs), x1=max(xs), y0=min(ys), y1=max(ys),
                ch=math.atan2(p[-1][1]-p[0][1], p[-1][0]-p[0][0]))

def tr_orient(p, kind):
    # heart/head: da esquerda (polegar) para a direita; life: de cima para baixo
    if kind == 'life':
        return p if p[0][1] <= p[-1][1] else p[::-1]
    return p if p[0][0] <= p[-1][0] else p[::-1]

def tr_max_turn(p, w=5):
    m = 0.0
    for i in range(w, len(p)-w):
        a1 = math.atan2(p[i][1]-p[i-w][1], p[i][0]-p[i-w][0]); a2 = math.atan2(p[i+w][1]-p[i][1], p[i+w][0]-p[i][0])
        m = max(m, abs(math.atan2(math.sin(a2-a1), math.cos(a2-a1))))
    return math.degrees(m)

def tr_trim_steep(p, lim=50, w=6):
    # tira pontas muito inclinadas (vinco de dedo) do coracao/cabeca
    def ang(i, j): return abs(math.degrees(math.atan2(p[j][1]-p[i][1], p[j][0]-p[i][0])))
    a = 0
    while a+w < len(p) and min(ang(a, a+w), 180-ang(a, a+w)) > lim: a += 1
    b = len(p)-1
    while b-w > a and min(ang(b-w, b), 180-ang(b-w, b)) > lim: b -= 1
    return p[a:b+1]

def tr_score(p, kind, P):
    p = tr_orient(p, kind)
    if kind in ('heart', 'head'): p = tr_trim_steep(p, 50 if kind == 'heart' else 60)
    if len(p) < 10 or tr_max_turn(p) > 55: return None
    f = tr_feats(p, P)
    dpri = float(np.mean([tr_mind(q, tr_PRI[kind]) for q in p[::2]]))
    ang = math.degrees(math.atan2(p[-1][1]-p[0][1], p[-1][0]-p[0][0]))
    if kind == 'heart':
        if f['x1'] < 150 or f['x1']-f['x0'] < 28 or not (-40 <= ang <= 40): return None
    elif kind == 'head':
        if f['x0'] > 128 or f['x1']-f['x0'] < 22 or not (-15 <= ang <= 52): return None
        if (f['x1']-f['x0']) < .9*(f['y1']-f['y0']): return None
        if f['y1'] > 215: return None
    else:
        if f['y1']-f['y0'] < 30 or f['y0'] > 175 or f['x0'] > 135: return None
        if not (35 <= ang <= 140): return None
    if dpri > 22: return None
    return f['L']*(.4+f['mp']) - 1.6*dpri*(1 + f['L']/60), p

def tr_overlap(a, b, d=3.0):
    if not a or not b: return 0.0
    n = sum(1 for q in a[::2] if tr_mind(q, b) < d)
    return n / max(1, len(a[::2]))

def tr_strip_near(a, others, d=3.0):
    best = []; cur = []
    for q in a:
        if any(tr_mind(q, o) < d for o in others if o): 
            if tr_plen(cur) > tr_plen(best): best = cur
            cur = []
        else: cur.append(q)
    return cur if tr_plen(cur) > tr_plen(best) else best

def tr_tail_after(d, l, dd=3.0):
    # cabeca pode nascer junto da vida: ignora o trecho inicial compartilhado
    i = 0
    while i < len(d) and tr_mind(d[i], l) < dd: i += 1
    return d[i:] if i < len(d) else d[-1:]

def tr_below(head, heart):
    hx = {}
    for q in heart: hx.setdefault(int(round(q[0])), []).append(q[1])
    n = c = 0
    for q in head:
        k = int(round(q[0]))
        if k in hx:
            n += 1
            if q[1] > min(hx[k]) - 1: c += 1
    return n == 0 or c / n > .7

def tr_lines(P, thr=.3, dbg=None):
    B = (P > thr)
    B = tr_components(B, 12)
    Sk = tr_zhang_suen(B)
    E = tr_graph(Sk)
    C = tr_paths(E)
    cands = {k: [] for k in ('heart', 'head', 'life')}
    for p in C:
        for k in cands:
            r = tr_score(p, k, P)
            if r: cands[k].append(r)
    for k in cands: cands[k].sort(key=lambda t: -t[0]); cands[k] = cands[k][:10]
    if dbg is not None: dbg.update(dict(B=B, Sk=Sk, E=E, C=C, cands=cands))
    best = None
    L_ = cands['life'] + [(0, None)]; H_ = cands['heart'] + [(0, None)]; D_ = cands['head'] + [(0, None)]
    for sl, l in L_:
        for sh, h in H_:
            if l and h and tr_overlap(h, l) > .2: continue
            for sd, d in D_:
                if d is not None:
                    if l and tr_overlap(tr_tail_after(d, l), l) > .2: continue
                    if h and tr_overlap(d, h) > .25: continue
                    if h and not tr_below(d, h): continue
                tot = sl + sh + sd
                if best is None or tot > best[0]: best = (tot, l, h, d)
    _, l, h, d = best
    out = {}
    if l: out['life'] = tr_smooth(l)
    if h: out['heart'] = tr_smooth(tr_strip_near(h, [l], 2.5) if l else h)
    if d:
        dd = tr_strip_near(d, [l, h], 2.5)
        if tr_plen(dd) >= 20: out['head'] = tr_smooth(dd)
    return out


_TRM = None
def tr_mask():
    """mascara da palma (mesmo teste par-impar do site)."""
    global _TRM
    if _TRM is None:
        m = np.zeros((S, S))
        for v in range(S):
            for u in range(S):
                c = False; j = len(tr_PG)-1
                for i in range(len(tr_PG)):
                    a = tr_PG[i]; b = tr_PG[j]
                    if (a[1] > v) != (b[1] > v) and u < (b[0]-a[0])*(v-a[1])/(b[1]-a[1])+a[0]: c = not c
                    j = i
                m[v, u] = 1.0 if c else 0.0
        _TRM = m
    return _TRM


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
    # a palma (poligono do gabarito) sempre fica; fora dela vale a regra de cor da pele.
    # isso evita apagar as linhas em maos claras/rosadas ou com brilho (saturacao baixa)
    palm = np.zeros((S, S), np.uint8)
    cv2.fillPoly(palm, [np.array([[80,176],[96,146],[108,126],[130,120],[148,125],[166,136],[186,150],[186,200],[168,236],[134,246],[106,232],[88,204]], np.int32)], 1)
    keep = skin | (palm > 0)
    img = warp.copy(); img[~keep] = 255
    x = (img.astype(np.float32)/255).transpose(2, 0, 1)[None]
    y = sess().run(['output'], {'input': x})[0]
    ln = ai_lines(y)
    # metodo por tracos: cada linha encontrada inteira no mapa da rede substitui a do metodo antigo
    try:
        pm = 1 / (1 + np.exp(-y.astype(np.float64).reshape(S, S)))
        tv = tr_lines(pm * tr_mask())
        for k_ in ('heart', 'head', 'life'):
            if k_ in tv: ln[k_] = {'pts': tv[k_], 'len': 0, 'mean': 1}
    except Exception:
        pass
    out = Linhas()
    for k in ('heart', 'head', 'life'):
        r = ln.get(k)
        if not r: continue
        if r.get('forcada'): out.forcadas.add(k)
        a = np.array(r['pts'], np.float64)
        q = cv2.perspectiveTransform(a[None], hi)[0]
        out[k] = [((W - qx/sc) if flip_ else qx/sc, qy/sc) for qx, qy in q]
    P = np.array([[q.x*W, q.y*H] for q in lm])
    # recorta as linhas ao casco da palma (landmarks da base da mao), para nao sairem da mao
    hull = cv2.convexHull(P[[0, 1, 2, 5, 9, 13, 17]].astype(np.float32))[:, 0, :]
    cen = hull.mean(0); hull = (cen + (hull - cen)*1.05).astype(np.float32).reshape(-1, 1, 2)
    for k in list(out.keys()):
        n0 = len(out[k]); pick = None
        for fx in (1, 1.08, 1.16, 1.25):
            if pick is not None and len(pick) >= n0*.8: break
            H2 = (cen + (hull.reshape(-1,2) - cen)*fx).astype(np.float32).reshape(-1, 1, 2)
            seg = []; best = []
            for q in out[k] + [None]:
                if q is not None and cv2.pointPolygonTest(H2, (float(q[0]), float(q[1])), False) >= 0: seg.append(q)
                else:
                    if len(seg) > len(best): best = seg
                    seg = []
            if pick is None or len(best) > len(pick): pick = best
        if len(pick) >= 4: out[k] = pick
    return pil, P, out


class Linhas(dict):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.forcadas = set()


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
    fc = getattr(lines, 'forcadas', set())
    for n in sorted(('life', 'head', 'heart'), key=lambda x: x not in fc):
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
