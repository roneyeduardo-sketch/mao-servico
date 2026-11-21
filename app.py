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


def leitura(nome, achadas, lang):
    key = os.environ.get('OPENAI_API_KEY')
    nomes = ', '.join(LINHA[k] for k in achadas) or 'a palma de forma geral'
    system = ('Você é uma leitora de palma e tarô acolhedora, em português do Brasil, que fala por WhatsApp. '
              'Foco em amor e relacionamentos. Tom caloroso, sem prometer certezas nem dar conselho médico, financeiro ou jurídico. '
              'Responda SOMENTE JSON: {"leitura": texto de 900 a 1300 caracteres, parágrafos curtos separados por linha em branco, '
              'usando *negrito* do WhatsApp nos títulos de cada linha, "resumo": uma frase de até 160 caracteres}. '
              'Não use travessão nem hífen como pontuação.')
    user = f'Nome da cliente: {nome or "querida"}. Linhas traçadas na foto dela: {nomes}. Escreva a leitura falando de cada linha traçada e feche com um convite para uma leitura de tarô completa.'
    body = json.dumps({'model': MODEL, 'response_format': {'type': 'json_object'},
                       'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}).encode()
    req = urllib.request.Request('https://api.openai.com/v1/chat/completions', body,
                                 {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    with urllib.request.urlopen(req, timeout=60) as r:
        j = json.load(r)
    return json.loads(j['choices'][0]['message']['content'])


@app.post('/mao')
def tracar():
    tok = os.environ.get('MAO_TOKEN', '')
    got = (request.headers.get('Authorization') or '').removeprefix('Bearer ').strip()
    if not tok or not hmac.compare_digest(got, tok): abort(401)
    d = request.get_json(silent=True) or {}
    foto = (d.get('foto') or '').strip()
    log.info('FOTO %s', foto[:200])
    if not foto.startswith('http'): return jsonify(erro='foto_invalida'), 400
    faxina()
    try:
        p = baixar(foto)
    except Exception as e:
        log.info('ERRO download %s', e)
        return jsonify(erro='download', detalhe=str(e)[:120]), 422
    try:
        pil, P, ln = mao.analyze(p)
        log.info('LINHAS %s', sorted(ln.keys()))
        if len(ln) < 2: return jsonify(erro='poucas_linhas'), 422
        nome = uuid.uuid4().hex + '.jpg'
        mao.render(pil, P, ln, f'{OUT}/{nome}')
    except ValueError as e:
        log.info('ERRO valor %s', e)
        return jsonify(erro=str(e)), 422
    except Exception as e:
        log.exception('ERRO processamento')
        return jsonify(erro='processamento', detalhe=str(e)[:120]), 500
    finally:
        try: os.remove(p)
        except OSError: pass
    base = os.environ.get('PUBLIC_URL', request.host_url).rstrip('/')
    out = {'imagem': f'{base}/img/{nome}', 'linhas': sorted(ln.keys())}
    try:
        r = leitura(d.get('nome'), sorted(ln.keys()), d.get('lang', 'pt-BR'))
        out['leitura'] = r.get('leitura', ''); out['resumo'] = r.get('resumo', '')
    except Exception as e:
        out['leitura'] = ''; out['resumo'] = ''; out['erro_leitura'] = str(e)[:100]
    return jsonify(out)


@app.get('/img/<n>')
def img(n):
    if not n.endswith('.jpg') or '/' in n or '..' in n: abort(404)
    f = f'{OUT}/{n}'
    if not os.path.exists(f): abort(404)
    return send_file(f, mimetype='image/jpeg')


@app.get('/')
def ok(): return 'ok'
