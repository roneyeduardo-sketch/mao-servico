"""Serviço /mao: foto da palma -> linhas traçadas + leitura. Variáveis: MAO_TOKEN, OPENAI_API_KEY, PUBLIC_URL, MODEL."""
import os, json, time, uuid, hmac, glob, urllib.request
from flask import Flask, request, jsonify, send_file, abort
import mao

app = Flask(__name__)
import logging, sys
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger('mao')

@app.after_request
def _log(r):
    log.info('REQ %s %s -> %s', request.method, request.path, r.status_code)
    return r

OUT = os.environ.get('OUT_DIR', '/tmp/mao_out'); os.makedirs(OUT, exist_ok=True)
MODEL = os.environ.get('MODEL', 'gpt-5.4-mini')
LINHA = {'heart': 'linha do coração (amor, afeto, vínculos)', 'head': 'linha da cabeça (mente, decisões)', 'life': 'linha da vida (vitalidade, energia, mudanças)'}


def faxina():
    for f in glob.glob(OUT + '/*.jpg'):
        if time.time() - os.path.getmtime(f) > 86400: os.remove(f)


def baixar(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=20) as r:
        b = r.read(15_000_000)
    p = f'{OUT}/in_{uuid.uuid4().hex}.bin'
    open(p, 'wb').write(b); return p


def leitura(nome, achadas, lang, interesse=''):
    key = os.environ.get('OPENAI_API_KEY')
    tem = ', '.join(LINHA[k] for k in achadas) or 'nenhuma'
    falta = ', '.join(LINHA[k] for k in LINHA if k not in achadas) or 'nenhuma'
    system = (
        'Você é uma leitora de palma e tarô acolhedora, em português do Brasil, falando por WhatsApp com uma cliente. '
        'Tom caloroso, íntimo e específico, como quem conversa. Não prometa certezas nem dê conselho médico, financeiro ou jurídico. '
        'Você não vê a imagem: nunca descreva detalhes visuais (cruzamentos, ilhas, ramificações). Interprete pelo simbolismo de cada linha. '
        'Responda SOMENTE JSON com as chaves: intro, coracao, cabeca, vida, sintese, convite, resumo. '
        'Cada texto é uma mensagem separada de WhatsApp e traz ideias NOVAS: nunca repita uma ideia, expressão ou conselho já dito em outra parte. '
        'intro: 250 a 350 caracteres, acolhe pelo nome e, se houver interesse, mostra que a leitura vai olhar para ele. '
        'coracao: começa com "🔴 *Linha do coração*" e uma quebra de linha, depois 600 a 800 caracteres sobre o jeito de amar, de se entregar, de lidar com perdas e o que a pessoa precisa receber no amor. '
        'cabeca: começa com "🟡 *Linha da cabeça*" e uma quebra de linha, depois 600 a 800 caracteres sobre como pensa, decide e interpreta o que sente, e como isso aparece nos relacionamentos. '
        'vida: começa com "🟢 *Linha da vida*" e uma quebra de linha, depois 600 a 800 caracteres sobre energia, fases, recomeços e o que sustenta a pessoa nos momentos de mudança. '
        'sintese: começa com "✨ *O que as três linhas dizem juntas*" e uma quebra de linha, depois 500 a 700 caracteres que ligam as três linhas, apontam a tensão ou o ponto forte entre elas e dão um conselho prático e gentil para os próximos dias. '
        'convite: 200 a 300 caracteres, convida para uma leitura de tarô completa, ligada ao interesse da cliente, sem pressão. '
        'resumo: uma frase de até 160 caracteres. '
        'Use *negrito* do WhatsApp só nos títulos. Parágrafos curtos. Não use travessão nem hífen como pontuação.')
    user = (f'Nome da cliente: {nome or "querida"}. '
            f'Interesse ou contexto que ela contou: {interesse or "não informado, foque em amor e relacionamentos"}. '
            f'Linhas traçadas na foto: {tem}. Linhas menos nítidas: {falta}. '
            'Escreva as três partes das linhas mesmo assim, sem mencionar falhas da foto.')
    body = json.dumps({'model': MODEL, 'response_format': {'type': 'json_object'},
                       'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}).encode()
    req = urllib.request.Request('https://api.openai.com/v1/chat/completions', body,
                                 {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    with urllib.request.urlopen(req, timeout=100) as r:
        j = json.load(r)
    return json.loads(j['choices'][0]['message']['content'])


def parse_body(raw):
    """Aceita JSON valido ou o formato da Leona sem aspas: {k:v,k2:v2}."""
    import re
    try:
        d = json.loads(raw)
        if isinstance(d, dict): return d
    except Exception:
        pass
    ms = list(re.finditer(r'[{,]\s*(\w+):', raw))
    d = {}
    for i, m in enumerate(ms):
        fim = ms[i + 1].start() if i + 1 < len(ms) else len(raw)
        d[m.group(1)] = raw[m.end():fim].strip().rstrip('}').strip()
    return d


def auth():
    tok = os.environ.get('MAO_TOKEN', '')
    got = (request.headers.get('Authorization') or '').removeprefix('Bearer ').strip()
    if not tok or not hmac.compare_digest(got, tok): abort(401)


def achar_foto(d, raw):
    import re
    for k in ('foto', 'foto2', 'foto3'):
        v = str(d.get(k) or '').strip()
        if v.startswith('http'): return v
    m = re.search(r'https?://[^\s"\',}]+', raw)
    return m.group(0) if m else ''


def processar(d, foto, base):
    """Retorna (dict, status http)."""
    faxina()
    try:
        p = baixar(foto)
    except Exception as e:
        log.info('ERRO download %s', e)
        return {'erro': 'download', 'detalhe': str(e)[:120]}, 422
    try:
        pil, P, ln = mao.analyze(p)
        log.info('LINHAS %s', sorted(ln.keys()))
        if len(ln) < 2: return {'erro': 'poucas_linhas'}, 422
        nome = uuid.uuid4().hex + '.jpg'
        mao.render(pil, P, ln, f'{OUT}/{nome}')
    except ValueError as e:
        log.info('ERRO valor %s', e)
        return {'erro': str(e)}, 422
    except Exception as e:
        log.exception('ERRO processamento')
        return {'erro': 'processamento', 'detalhe': str(e)[:120]}, 500
    finally:
        try: os.remove(p)
        except OSError: pass
    out = {'imagem': f'{base}/img/{nome}', 'linhas': sorted(ln.keys())}
    try:
        r = leitura(d.get('nome'), sorted(ln.keys()), d.get('lang', 'pt-BR'), str(d.get('interesse') or '')[:200])
        partes = ('intro', 'coracao', 'cabeca', 'vida', 'sintese', 'convite')
        for k in partes: out[k] = str(r.get(k) or '').strip()
        out['leitura'] = '\n\n'.join(out[k] for k in partes if out[k]); out['resumo'] = str(r.get('resumo') or '')
    except Exception as e:
        [out.setdefault(k, '') for k in ('intro', 'coracao', 'cabeca', 'vida', 'sintese', 'convite')]; out['leitura'] = ''; out['resumo'] = ''; out['erro_leitura'] = str(e)[:100]
    return out, 200


def base_url():
    return os.environ.get('PUBLIC_URL', request.host_url).rstrip('/')


@app.post('/mao')
def tracar():
    auth()
    raw = request.get_data(as_text=True)
    log.info('RAW %s', raw[:600])
    d = parse_body(raw)
    foto = achar_foto(d, raw)
    log.info('FOTO %s', foto[:200])
    if not foto.startswith('http'): return jsonify(erro='foto_invalida'), 400
    out, st = processar(d, foto, base_url())
    return jsonify(out), st


JOBS = {}


@app.post('/mao/start')
def start():
    """Assincrono: responde na hora e processa em segundo plano. Resultado em GET /mao/result?tel=..."""
    import threading
    auth()
    raw = request.get_data(as_text=True)
    log.info('RAW %s', raw[:600])
    d = parse_body(raw)
    foto = achar_foto(d, raw)
    tel = ''.join(c for c in str(d.get('telefone') or '') if c.isdigit())
    if not foto.startswith('http') or not tel: return jsonify(erro='foto_invalida'), 400
    base = base_url()
    JOBS[tel] = {'estado': 'processando', 't': time.time()}

    def run():
        try:
            out, st = processar(d, foto, base)
            JOBS[tel] = {'estado': 'ok' if st == 200 else 'erro', 'out': out, 't': time.time()}
        except Exception as e:
            log.exception('job')
            JOBS[tel] = {'estado': 'erro', 'out': {'erro': str(e)[:100]}, 't': time.time()}

    threading.Thread(target=run, daemon=True).start()
    return jsonify(ok=True)


@app.get('/mao/result')
def result():
    auth()
    tel = ''.join(c for c in request.args.get('tel', '') if c.isdigit())
    j = JOBS.get(tel)
    if not j or j['estado'] != 'ok': return jsonify(estado=(j or {}).get('estado', 'nada')), 404
    return jsonify(j['out'])


@app.get('/img/<n>')
def img(n):
    if not n.endswith('.jpg') or '/' in n or '..' in n: abort(404)
    f = f'{OUT}/{n}'
    if not os.path.exists(f): abort(404)
    return send_file(f, mimetype='image/jpeg')


@app.get('/')
def ok(): return 'ok'
