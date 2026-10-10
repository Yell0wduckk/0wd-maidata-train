"""Build a small, auditable set of long-distance attention examples."""
import argparse
import csv
import html
import io
import json
from pathlib import Path
import zipfile
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    root = parser.parse_args().root
    with (root/'attention-over-30-final.csv').open(encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if r['scope'] == 'final_prediction']
    journal = json.loads((root/'journal.json').read_text(encoding='utf8'))
    order = {node: i for i, node in enumerate(journal['final_history_node_ids'])}
    for r in rows:
        r['probability'] = float(r['probability'])
        r['query_seconds'] = float(r['query_seconds'])
        r['key_seconds'] = float(r['key_seconds'])
        r['distance_seconds'] = r['key_seconds'] - r['query_seconds']
    choices = [
        ('回看123秒前的音频', 'decoder_cross_L5', '1', -1),
        ('回看83秒前的音频', 'decoder_cross_L8', '4', -1),
        ('读取53秒后的音频', 'decoder_cross_L8', '1', 1),
        ('回看41秒前的谱面分支', 'decoder_self_L6', '5', -1),
        ('回看66秒前的谱面分支', 'decoder_self_L3', '1', -1),
    ]
    examples = []
    with zipfile.ZipFile(root/'raw-attention.zip') as z:
        for title, layer, head, direction in choices:
            candidates = [r for r in rows if r['layer'] == layer and r['head'] == head
                          and direction*r['distance_seconds'] > 30
                          and (r['kind'] != 'decoder_self' or r['key_event'] in ('NOTE', 'MOMENT', 'SLIDE_BRANCH'))]
            r = dict(max(candidates, key=lambda r: r['probability']))
            r['title'] = title
            if r['kind'] == 'decoder_self':
                r['history_row_distance'] = order[int(r['query_node_id'])] - order[int(r['key_node_id'])]
            with np.load(io.BytesIO(z.read(r['record_path'])), allow_pickle=False) as a:
                key = ('self_' if r['kind'] == 'decoder_self' else 'cross_') + f"{int(layer.split('L')[-1])-1:02d}"
                b, q, k = (int(r[x]) for x in ('batch_index', 'query_index', 'key_index'))
                assert float(a[key][b, int(head)-1, q, k]) == r['probability']
                assert a['query_valid'][b, q]
            examples.append(r)
    (root/'distant-examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding='utf8')
    cards = []
    for i, r in enumerate(examples, 1):
        q, k = r['query_seconds'], r['key_seconds']
        rows_text = f"，相隔 {r['history_row_distance']} 个历史输入行位置" if 'history_row_distance' in r else ''
        cards.append(f'''<section><h2>{i}. {r['title']}</h2>
<p><b>{r['layer']} · 头 {r['head']} · {r['probability']*100:.2f}%</b></p>
<p>Query <b>{q:.2f}s</b> → Key <b>{k:.2f}s</b>；距离 {abs(k-q):.2f}s{rows_text}。</p>
<p>{r['input_event']} → 预测 {r['predicted_event']}；关注的 Key：{r['key_event'] or '音频 P0 token'}。</p>
<svg viewBox="0 0 1000 80" role="img" aria-label="Query和Key在全曲时间轴的位置">
<line x1="30" x2="970" y1="40" y2="40" stroke="#ccd6df" stroke-width="3"/>
<line x1="{30+q/151.5*940:.2f}" x2="{30+k/151.5*940:.2f}" y1="40" y2="40" stroke="#5573ac" stroke-width="4"/>
<circle cx="{30+q/151.5*940:.2f}" cy="40" r="7" fill="#c74d43"/>
<circle cx="{30+k/151.5*940:.2f}" cy="40" r="7" fill="#187994"/>
<text x="{30+q/151.5*940:.2f}" y="23" text-anchor="middle">Q {q:.2f}s</text>
<text x="{30+k/151.5*940:.2f}" y="70" text-anchor="middle">K {k:.2f}s</text></svg>
<button data-time="{q}">定位 Query</button> <button data-time="{k}">定位 Key</button>
<details><summary>原始记录索引</summary><pre>{html.escape(json.dumps(r,ensure_ascii=False,indent=2))}</pre></details></section>''')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>11739 · 远距离注意力实例</title><style>body{margin:0;background:#f3f6fa;color:#223345;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:960px;margin:auto;padding:24px}header{position:sticky;top:0;background:#f3f6fa;z-index:1;padding:10px 0}audio{width:100%}section{background:white;padding:20px;margin:20px 0;border-radius:10px}h1{font-size:26px}h2{font-size:20px}button{padding:8px 16px;border:1px solid #ccd6df;background:#f4f8ff;border-radius:5px;cursor:pointer}a{color:#175aaa}pre{font-size:12px;white-space:pre-wrap}svg{width:100%;max-height:90px}svg text{font-size:15px}</style><main>
<a href="index.html">时间轴热力图</a> · <a href="attention-over-30.html">完整明细（可选）</a> · <a href="tail-diagnostic.html">尾部生成诊断</a>
<h1>Decoder 确实会关注远处：五个实际例子</h1>
<p>均来自本次11739最终接受的生成路径，单层、单头、单个Key的概率；以相距超过30秒筛选并选取高权重实例，不代表平均行为。谱面例子排除了起始／窗口控制行。</p>
<header><audio id="audio" controls preload="metadata" src="audio.mp3"></audio><span id="position">点击定位按钮后，可按播放试听。</span></header>
''' + ''.join(cards) + '''<p>Q 是已知输入行的时间，K 是读取目标的时间；音频可读取后续全曲，自注意力仅读取历史。高权重证明该头读取了远处，但还不能证明这次读取改善了谱面，需消融实验才能判断。</p>
<p>“此时间格没有查询行”指该0.5秒区间没有用于汇总的有效query，因此无注意力分布可显示；不等于该处音频无效或注意力为零。</p>
</main><script>const audio=document.getElementById('audio');document.querySelectorAll('button[data-time]').forEach(b=>b.onclick=()=>{audio.currentTime=Number(b.dataset.time);});audio.ontimeupdate=()=>document.getElementById('position').textContent='当前位置 '+audio.currentTime.toFixed(2)+' 秒';</script></html>'''
    (root/'distant-examples.html').write_text(page, encoding='utf8')
    print(json.dumps(examples, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
