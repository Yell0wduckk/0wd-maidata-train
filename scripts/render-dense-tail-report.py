"""Summarize the fixed-prefix probes and accepted controlled suffix runs."""
import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path


NAMES={'local_audio_only':'仅局部音频','no_direct_audio':'关闭直接音频','no_history_notes':'屏蔽历史NOTE／分支'}


def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def row_text(r):
    if 'error' in r:return '错误：'+r['error']
    ev=r['event']
    if ev=='MOMENT':return f"MOMENT Δ{r['delta_tick']} tick"
    if ev=='SEGMENT_END':return '结束窗口'
    return ' '.join(str(r[k]) for k in ('event','kind','pad','start_pad','shape','end_offset','hold_duration_tick','branch_duration_tick','flags') if k in r and r[k]!='')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('viewer',type=Path);p.add_argument('experiment',type=Path)
    a=p.parse_args();r,e=a.viewer,a.experiment
    attention=read(r/'tail-attention.json');probes=read(e/'probe/fixed-prefix-probes.json')
    base=read(e/'probe/inference/generation.json');reference=read(e/'probe/accepted-rows.json')
    cut=41808;body=attention['original_last_body_end_seconds']
    stats={};invariants={};all_rows={'baseline':reference}
    for mode in ['baseline',*NAMES]:
        rows=reference if mode=='baseline' else read(e/(mode+'-v2')/'accepted-rows.json')
        all_rows[mode]=rows
        tail=[x for x in rows if x['anchor_tick']>cut]
        notes=[x for x in tail if x['row']['event']=='NOTE']
        stats[mode]=dict(events=dict(Counter(x['row']['event'] for x in tail)),notes=len(notes),
            last_note_seconds=max(x['anchor_tick'] for x in notes)/96*60/184,
            notes_after_original_body=sum(x['anchor_tick']/96*60/184>body+1e-6 for x in notes),
            notes_after_149=sum(x['anchor_tick']/96*60/184>=149 for x in notes),
            notes_per_window=dict(Counter(x['index'] for x in notes)))
        if mode!='baseline':
            g=read(e/(mode+'-v2')/'inference/generation.json');completion=read(e/(mode+'-v2')/'completion.json')
            invariants[mode]=dict(sampling_same=g['sampling']==base['sampling'],checkpoint_same=g['checkpoint_sha256']==base['checkpoint_sha256'],
                latent_all_windows_exact=[x['style'] for x in g['windows']]==[x['style'] for x in base['windows']],
                prefix_predictions_exact=rows[:1404]==reference[:1404],complete=completion['exit_code']==0 and not g['truncated'])
            assert all(invariants[mode].values())
    keyed=defaultdict(list)
    for probe in probes:
        row=probe['baseline']['row'];tick=probe['tick']+(row.get('delta_tick',0) if row['event']=='MOMENT' else 0)
        keyed[(tick,json.dumps(row,sort_keys=True))].append(probe)
    aligned=[]
    for q in attention['queries']:
        candidates=keyed[(q['anchor_tick'],json.dumps(q['row'],sort_keys=True))]
        probe=candidates.pop(0) if candidates else None
        aligned.append(dict(attention=q,probe=probe))
    notes=[x for x in aligned if x['attention']['row']['event']=='NOTE']
    assert len(notes)==15 and all(x['probe'] for x in notes)
    changed={mode:sum(x['probe']['conditions'][mode]['row']!=x['attention']['row'] for x in notes) for mode in NAMES}
    summary=dict(stats=stats,invariants=invariants,changed_notes_out_of_15=changed,
        fixed_prefix_probe_count=len(probes),all_baseline_repeat_rows_exact=True,
        unmatched_tail_decisions=[x['attention']['query_node_id'] for x in aligned if x['probe'] is None],
        sampling=base['sampling'],audio_rms=read(e/'audio-tail-rms.json'),aligned=aligned)
    assert all(x['conditions']['baseline_repeat']['row']==x['baseline']['row'] for x in probes)
    save(e/'summary.json',summary);save(e/'invariant-checks.json',invariants)
    lines=['# 11739 dense B50 尾部生成诊断','',
        '原谱面最后 onset：142.01087s；最后滑条结束：147.39130s；推理音频采样边界：150.97829s。',
        '142–147s仍有显著声音，149s段RMS约-50dBFS、150s段约-64dBFS。基线尾部15个NOTE，其中11个晚于原谱面最后滑条结束。','',
        '| 条件 | 固定前缀下改变的NOTE/15 | 自由延续尾部NOTE数 | 最后NOTE秒 |',
        '|---|---:|---:|---:|']
    for mode in ['baseline',*NAMES]:
        lines.append(f"| {NAMES.get(mode,'原基线')} | {changed.get(mode,0)} | {stats[mode]['notes']} | {stats[mode]['last_note_seconds']:.5f} |")
    lines+=['','全部保留干预前1404个已接受预测（含窗口结束行），checkpoint、sampling和29个窗口latent逐项相同。',
        '单步对照每次恢复原缓存和前缀，做34个决策探针；重复基线逐行相同，完整基线仍为原1406行。',
        '自由延续从精确边界行之后持续施加干预；三组均正常完成、无截断。','',
        '远处音频指原窗口[start-4,start+18)拍之外的decoder P0 keys；全部直接音频指八层decoder音频残差、router音频残差和delta候选音频项全部关闭；历史NOTE含SLIDE_BRANCH，屏蔽decoder和router中严格早于query的这些keys。',
        '当前query自身、MOMENT／控制行、固定latent及干预前KV保留。音频encoder局部特征已经融合全曲信息，旧KV和latent也可携带音频／历史信息；这些是直接读取通路干预，不等于抹去所有间接信息。','',
        '## 结束机制',
        '基线在143.66848s结束第27号窗口，生成器仍于146.08696s开启第28号窗口，后者产生13个尾部NOTE。SEGMENT_END仅结束窗口，没有整曲EOS。',
        '第28号窗口首query固定前缀下：基线MOMENT 98.21194%、SEGMENT_END 1.78806%；三种干预的继续MOMENT概率仍约98%。语法允许直接结束，模型却继续。',
        '训练分窗由最后onset确定window_count；生成分窗由音频采样终点确定。本曲原谱面应有28个监督窗口，生成循环有29个。这是尾窗监督与生成范围的结构差异，尚未做重训修复实验。','',
        '## 每个尾部NOTE的关注及单步预测',
        '远距离示例从当前query之前超过16拍的key中选取最高单头概率；谱面key仅NOTE／SLIDE_BRANCH，未混入控制行。不是各头均值，也不代表全层总贡献。','',
        '| 秒 | 基线 | 远音频Key秒（层/头/概率） | 历史NOTE Key秒（层/头/概率） | 仅局部音频 | 无直接音频 | 无历史NOTE |',
        '|---|---|---|---|---|---|---|']
    table=[]
    for x in notes:
        q,probe=x['attention'],x['probe'];cols=[f"{q['output_seconds']:.5f}",row_text(q['row'])]
        for kind in ('cross','self'):
            v=q['distant_semantic_examples'][kind]
            cols.append(f"{v['seconds']:.2f}（L{v['layer']}/H{v['head']}/{v['probability']*100:.2f}%）")
        cols += [row_text(probe['conditions'][mode]['row']) for mode in NAMES]
        lines.append('| '+' | '.join(cols)+' |');table.append((q,cols))
    lines+=['','## 证据与范围',
        '固定模型源码642ace5e2b23fafc6dfcf34eb174cd0882819a0a，dense B50 epoch49／step1147；完整checkpoint SHA见各组generation.json。',
        '远端独立目录：/root/autodl-tmp/dense-11739-tail-diagnostic-20261004；正式结果仅probe及*-v2。首轮自由延续按时间切换，影响边界同拍行，被前缀校验拒绝；保留失败目录，不用于结论。',
        'fixed-prefix-probes.json保留完整选择trace。tail-attention.json保留31条尾部输出对应query的八层self／cross各头均值Top5、单头最大值及局部／远处概率质量。',
        '末端Physical约束路径有尾部控制决策未进入单步探针，summary.json显式列出；15个NOTE均覆盖。',
        '未修改训练／生成核心代码、未重新训练；结果仅针对本曲本seed，不能据此声称屏蔽某通路普遍改善或恶化谱面质量。']
    text='\n'.join(lines)+'\n';(e/'report.md').write_text(text,encoding='utf8')
    (r/'tail-report.md').write_text(text,encoding='utf8')
    save(r/'tail-summary.json',summary)
    rows=''.join('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cols)+'</tr>' for q,cols in table)
    details=[]
    for x in aligned:
        q=x['attention'];bodylines=[]
        for layer in q['layers']:
            top=layer['head_mean_top'][:3]
            bodylines.append(f"{layer['kind']} L{layer['layer']}: "+'；'.join(f"{v['seconds']:.2f}s {v.get('event','音频')} {v['mean_probability']*100:.2f}%" for v in top))
        details.append(f"<details><summary>{q['output_seconds']:.3f}s · {html.escape(row_text(q['row']))} · Query {q['query_seconds']:.3f}s</summary><button data-t='{q['query_seconds']}'>定位Query</button><p>下列为各层8头均值Top3；原始记录 {q['raw_path']}，query node {q['query_node_id']}。</p><pre>{html.escape(chr(10).join(bodylines))}</pre></details>")
    statsrows=''.join(f"<tr><td>{NAMES.get(m,'基线')}</td><td>{changed.get(m,0)}/15</td><td>{stats[m]['notes']}</td><td>{stats[m]['last_note_seconds']:.3f}s</td></tr>" for m in ['baseline',*NAMES])
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>11739 · 尾部生成诊断</title>
<style>body{margin:0;background:#f4f6fa;color:#213547;font:15px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1450px;margin:auto;padding:24px}h1{font-size:27px}h2{font-size:20px;margin-top:30px}a{color:#175da7}audio{width:100%}table{border-collapse:collapse;background:white;width:100%}td,th{border:1px solid #dde3eb;padding:9px;text-align:left}th{background:#e7edf5}.wide{overflow:auto}details{background:white;border:1px solid #dde3eb;margin:9px 0;padding:12px}summary{cursor:pointer}pre{white-space:pre-wrap}button{padding:6px 12px;cursor:pointer}.callout{padding:16px;background:#e8f0fc;border-left:4px solid #477cb9}</style><main>
<a href="index.html">注意力热力图</a> · <a href="distant-examples.html">远距离实例</a> · <a href="tail-report.md">完整报告</a> · <a href="tail-summary.json">实验明细 JSON</a>
<h1>11739：为什么最后一个音符后还在生成？</h1>
<p class="callout">基线在原谱面最后按键后生成15个NOTE；屏蔽远处音频、全部直接音频、历史NOTE后，尾部没有消失。主要发现是：结束窗口不等于结束整曲，程序会继续开启音频尾部的新窗口。</p>
<audio id="audio" controls preload="metadata" src="audio.mp3"></audio><p id="position">可拖动试听142–151秒的尾部。</p>
<p>原谱面最后按键 <b>142.011s</b>；最后滑条结束 <b>147.391s</b>；音频文件结束 <b>150.978s</b>。142–147s仍有显著声音，149s约−50dBFS、150s约−64dBFS。基线有11个NOTE晚于原谱面全部长音符结束。</p>
<h2>严格控制变量结果</h2><table><tr><th>条件</th><th>固定前缀下改变的NOTE</th><th>自由延续尾部NOTE数</th><th>最后NOTE</th></tr>'''+statsrows+'''</table>
<p>所有组保持1404个已接受预测的原前缀、相同checkpoint／seed／temperature／全部29窗latent。基线重复逐行一致；三组自由延续完整结束，无截断。</p>
<p>“仅局部音频”保留原窗口前4拍至后18拍的P0；“关闭直接音频”关闭decoder、router的音频残差及delta音频候选项；“历史NOTE”同时屏蔽decoder和router的过去NOTE／SLIDE_BRANCH keys，保留当前query、MOMENT和控制行。</p>
<p><b>边界：</b>固定latent、旧KV以及融合过全曲信息的局部特征仍在，因此不等于彻底移除一切音频／历史信息。</p>
<h2>停止机制</h2><p>143.668s模型已输出SEGMENT_END，但它只结束第27号窗口。生成器在146.087s开启第28号窗口，产生13个尾部NOTE。新窗口首query给继续MOMENT约98.21%、结束约1.79%；三种单步干预后继续概率仍约98%。</p>
<p>训练窗口只建到最后onset，生成窗口则延伸到音频采样终点；本曲对应28个监督窗口和29个生成窗口。没有整曲EOS，尾窗停止监督与推理范围存在差异。此处尚未重训验证修复。</p>
<h2>15个尾部NOTE：远处关注和固定前缀预测</h2><p>下表远处示例均为相隔超过16拍的最高单头权重；self只选真实NOTE／分支。具体关注不等于因果贡献。</p><div class="wide"><table><tr><th>秒</th><th>基线</th><th>远音频Key秒（层/头/概率）</th><th>历史NOTE Key秒（层/头/概率）</th><th>仅局部音频</th><th>无直接音频</th><th>无历史NOTE</th></tr>'''+rows+'''</table></div>
<h2>全部尾部行对应的Query关注位置</h2>'''+''.join(details)+'''<p>完整原始索引和实验定义见本次诊断report.md、summary.json及fixed-prefix-probes.json。末端部分强制终止控制路径未采集单步探针；所有15个NOTE均覆盖。</p></main>
<script>const a=document.getElementById('audio');document.querySelectorAll('button[data-t]').forEach(b=>b.onclick=()=>{a.currentTime=Number(b.dataset.t)});a.ontimeupdate=()=>document.getElementById('position').textContent='当前位置 '+a.currentTime.toFixed(3)+'s';</script></html>'''
    (r/'tail-diagnostic.html').write_text(page,encoding='utf8')
    print(json.dumps(dict(stats=stats,changed=changed,invariants=invariants,unmatched=summary['unmatched_tail_decisions']),ensure_ascii=False))


if __name__=='__main__':main()
