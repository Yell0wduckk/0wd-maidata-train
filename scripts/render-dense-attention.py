"""Make a paginated audio-aligned HTML explorer and a multi-page PDF."""
import argparse
import base64
import gzip
import json
import math
from pathlib import Path
import shutil

import numpy as np


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>11739 · Dense B50 注意力</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f6f8fa;color:#17212d;font:15px/1.55 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1280px;margin:auto;padding:30px 32px 60px}h1{font-size:27px;margin:0 0 4px}h2{font-size:18px;margin:14px 0 8px}.sub{color:#556372;margin:4px 0 16px}.bar{display:flex;gap:15px;align-items:center;flex-wrap:wrap;padding:15px 0}label{display:flex;gap:7px;align-items:center}select,button{font:inherit;border:1px solid #c4cdd6;border-radius:6px;background:white;padding:6px 10px;color:inherit}button:disabled{opacity:.4}button{cursor:pointer}audio{width:100%;height:40px}.panel{background:#fff;padding:12px 16px;margin:14px 0;border:1px solid #e0e5ea;border-radius:8px}.plot{width:100%;position:relative}canvas{display:block;width:100%;height:auto}.legend{display:flex;align-items:center;gap:12px;font-size:13px;color:#465668}.gradient{width:180px;height:12px;background:linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)}.note{color:#506174;font-size:13px}.tip{position:fixed;display:none;pointer-events:none;background:#152436;color:white;border-radius:5px;padding:8px 12px;z-index:3;white-space:pre;font-size:13px}a{color:#145cb6}details{margin:18px 0;background:white;padding:12px 16px;border-radius:8px}#status{font-weight:500;color:#247155}.axis-note{font-size:13px;margin:0;color:#596d7b}@media(max-width:700px){main{padding:18px 10px}.panel{padding:8px 5px}h1{font-size:22px}.bar{gap:8px}select{max-width:230px}}
</style></head><body><main>
<h1>11739 · Dense A90 → B50</h1>
<p><a href="distant-examples.html">五个远距离关注实例</a> · <a href="attention-over-30.html">查看全部注意力概率 &gt; 30% 的记录 →</a></p>
<p class="sub">真实自由生成的注意力概率 · 与原音频时间轴对齐 · 全部层与全部 8 个头</p>
<audio id="audio" controls preload="metadata" src="audio.mp3"></audio>
<div class="bar"><button id="prev">上一页</button><label>时间页 <select id="page"></select></label><button id="next">下一页</button><button id="playPage">从本页播放</button><span id="status">正在读取概率数据…</span></div>
<div class="bar"><label>音频自注意力 <select id="audioLayer"></select></label><label>Decoder <select id="decoderLayer"><option value="mean">八层均值</option></select></label><label>注意力头 <select id="head"><option value="mean">8 头均值</option></select></label></div>
<p class="axis-note">横轴＝query 当前已知位置（音频秒）；纵轴＝被关注的 key 位置（音频秒）。每页仅缩放横轴，纵轴始终保留全曲，远距离关注不会被裁掉。</p>
<div class="panel"><h2>音频波形与生成事件</h2><div class="plot"><canvas id="wave" width="1200" height="170" aria-label="音频波形和生成音符时刻"></canvas></div></div>
<div class="panel"><h2 id="audioTitle">音频自注意力</h2><canvas id="audioMap" width="1200" height="335" aria-label="音频自注意力热力图"></canvas></div>
<div class="panel"><h2 id="crossTitle">Decoder → 全曲 P0 音频</h2><canvas id="crossMap" width="1200" height="335" aria-label="Decoder 音频交叉注意力热力图"></canvas></div>
<div class="panel"><h2 id="selfTitle">Decoder → 已接受谱面历史</h2><canvas id="selfMap" width="1200" height="335" aria-label="Decoder 因果自注意力热力图"></canvas></div>
<div class="legend"><span>概率质量（对数色阶）</span><span class="gradient"></span><span>0.01% → 100%；零为浅灰，无 query 为灰色</span></div>
<p class="note" id="metadata"></p>
<details open><summary>读图与记录口径</summary><p>每个格子表示落入该 0.5 秒 key 区间的概率之和，再对该 query 时间区间内的查询取平均；层均值和头均值均为等权。没有把页面之外的 key 重新归一化。鼠标悬停可查看具体概率，单击图中横向位置可定位音频。红色竖线跟随音频播放；橙色虚线仅标出 key＝query 的时间参考，不是概率曲线。</p><p>Decoder 横轴是已经接受的输入行位置，而不是尚未生成的下一 MOMENT 时间；同拍多行在展示中合并，原始记录仍保留行号与严格先后顺序。默认仅统计最终接受的预测 query，回滚和被替换尝试留在原始包与 journal 中。</p><p>音频包括 2 层独立全曲编码器，以及 E0/E1/E2 局部编码器。局部图只选各窗口中央 16 拍的有效 query 以免重复计算重叠上下文，但保留它们全部有效 key。Decoder 交叉注意力此处专指音频 P0，不混入四槽 latent 或独立 router。</p><p>分数是实际 post-RoPE Q/K 与实际 mask 重建的 softmax 概率，旁路不替换原 SDPA 输出。原始包逐头保存 FP16 概率；时间汇总保存 FP32。页面用 uint16 编码展示概率，实测量化误差见下方。注意力概率不等于因果贡献或模型重要性。</p><p id="validation"></p><p><a href="attention-report.pdf">多页 PDF</a> · <a href="raw-attention.zip">逐层逐头原始概率</a> · <a href="journal.json">输入行／回滚／输出对应记录</a> · <a href="aggregation.json">汇总与来源</a> · <a href="inference/maidata.txt">生成谱面</a></p></details>
</main><div id="tip" class="tip"></div><script src="viewer-data.js"></script><script>
(async()=>{
'use strict';const M=window.ATTENTION_META,D=window.ATTENTION_DATA,$=id=>document.getElementById(id);
const n=M.bins,dt=M.bin_seconds,T=M.axis_end_seconds,labels=M.streams.map(s=>s.label),mat={},count=M.counts;
const decoderKeys=kind=>Array.from({length:8},(_,i)=>`decoder_${kind}_L${i+1}`),globalKeys=labels.filter(x=>x.startsWith('song_audio_encoder.blocks'));
for(let h=1;h<=8;h++){ $('head').add(new Option(`头 ${h}`,h-1));$('decoderLayer').add(new Option(`第 ${h} 层`,h));}
$('audioLayer').add(new Option('全曲编码器 · 两层均值','global_mean'));
for(const label of labels.filter(x=>!x.startsWith('decoder_'))){let title=label.startsWith('song_audio_encoder')?'全曲编码器 '+(Number(label.match(/blocks\.(\d+)/)[1])+1)+' 层':label.replace('audio_encoder.stages.','E').replace(/\.blocks\.(\d+)\.attention/,(_,x)=>' / Block '+(Number(x)+1));$('audioLayer').add(new Option(title,label));}
$('page').add(new Option('全曲总览','-1'));for(let i=0;i<Math.ceil(T/20);i++)$('page').add(new Option(`${i+1} · ${i*20}–${Math.min(T,(i+1)*20).toFixed(1)} 秒`,i));$('page').value='0';
const bytes=Uint8Array.from(atob(D),c=>c.charCodeAt(0));const raw=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();const values=new Uint16Array(raw);
for(let i=0;i<labels.length;i++)mat[labels[i]]=values.subarray(i*8*n*n,(i+1)*8*n*n);
let selected={},domain=[0,Math.min(20,T)],drawPending=false;
const palettes=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
function color(v){if(v<=0)return '#f3f5f6';let t=Math.max(0,Math.min(1,(Math.log10(v)+4)/4))*4,j=Math.min(3,Math.floor(t)),f=t-j;return `rgb(${palettes[j].map((x,k)=>Math.round(x*(1-f)+palettes[j+1][k]*f)).join(',')})`;}
function combine(keys){const out=new Float32Array(n*n),valid=new Uint8Array(n),head=$('head').value;for(let q=0;q<n;q++){const available=keys.filter(k=>count[k][q]>0);if(!available.length)continue;valid[q]=1;for(const key of available){const hs=head==='mean'?[0,1,2,3,4,5,6,7]:[Number(head)];for(const h of hs)for(let k=0;k<n;k++)out[q*n+k]+=mat[key][(h*n+q)*n+k]/65535/available.length/hs.length;}}return {out,valid};}
function axes(ctx,w,h,ylabel){let box={x:78,y:12,w:w-100,h:h-62};ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);ctx.strokeStyle='#bac5ce';ctx.lineWidth=1;ctx.strokeRect(box.x,box.y,box.w,box.h);ctx.font='13px system-ui';ctx.fillStyle='#344b5d';ctx.textAlign='center';for(let i=0;i<=5;i++){let t=domain[0]+i*(domain[1]-domain[0])/5,x=box.x+i*box.w/5;ctx.fillText(t.toFixed(1),x,h-31);}ctx.fillText('query · 音频时间 (s)',box.x+box.w/2,h-7);if(ylabel){ctx.textAlign='right';for(let i=0;i<=4;i++)ctx.fillText((i*T/4).toFixed(1),box.x-8,box.y+i*box.h/4+5);ctx.save();ctx.translate(17,box.y+box.h/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillText('key · 音频时间 (s)',0,0);ctx.restore();}return box;}
function cursor(ctx,box){const t=$('audio').currentTime;if(t<domain[0]||t>domain[1])return;const x=box.x+(t-domain[0])/(domain[1]-domain[0])*box.w;ctx.strokeStyle='#e44932';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x,box.y);ctx.lineTo(x,box.y+box.h);ctx.stroke();}
function heat(id,data){const c=$(id),ctx=c.getContext('2d'),b=axes(ctx,c.width,c.height,true);ctx.save();ctx.beginPath();ctx.rect(b.x,b.y,b.w,b.h);ctx.clip();for(let q=Math.max(0,Math.floor(domain[0]/dt));q<Math.min(n,Math.ceil(domain[1]/dt));q++){const x=b.x+(q*dt-domain[0])/(domain[1]-domain[0])*b.w,ww=dt/(domain[1]-domain[0])*b.w+.3;for(let k=0;k<n;k++){ctx.fillStyle=data.valid[q]?color(data.out[q*n+k]):'#e2e7eb';ctx.fillRect(x,b.y+k*dt/T*b.h,ww,dt/T*b.h+.25);}}ctx.strokeStyle='#f8a56a';ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(b.x,b.y+domain[0]/T*b.h);ctx.lineTo(b.x+b.w,b.y+domain[1]/T*b.h);ctx.stroke();ctx.setLineDash([]);cursor(ctx,b);ctx.restore();c._box=b;c._data=data;}
function wave(){const c=$('wave'),ctx=c.getContext('2d'),b=axes(ctx,c.width,c.height,false),wave=M.waveform;ctx.fillStyle='#7da6ba';for(let i=Math.max(0,Math.floor(domain[0]/wave.envelope_step_seconds));i<Math.min(wave.maximum.length,Math.ceil(domain[1]/wave.envelope_step_seconds));i++){let x=b.x+(i*wave.envelope_step_seconds-domain[0])/(domain[1]-domain[0])*b.w;ctx.fillRect(x,b.y+30-wave.maximum[i]*25,1.5,Math.max(.5,(wave.maximum[i]-wave.minimum[i])*25));}ctx.fillStyle='#8460a8';for(const row of M.predictions){if(!['NOTE','SLIDE_BRANCH'].includes(row.row.event))continue;const t=row.output_seconds;if(t>=domain[0]&&t<domain[1]){const x=b.x+(t-domain[0])/(domain[1]-domain[0])*b.w;ctx.fillRect(x,b.y+68,1,13);}}ctx.fillStyle='#344b5d';ctx.textAlign='left';ctx.fillText('波形',b.x+4,b.y+13);ctx.fillText('生成 NOTE / BRANCH',b.x+4,b.y+64);cursor(ctx,b);c._box=b;}
function draw(recompute=true){const page=Number($('page').value);domain=page<0?[0,T]:[page*20,Math.min(T,(page+1)*20)];if(recompute){const layer=$('decoderLayer').value,akey=$('audioLayer').value;selected={audio:combine(akey==='global_mean'?globalKeys:[akey]),cross:combine(layer==='mean'?decoderKeys('cross'):[`decoder_cross_L${layer}`]),self:combine(layer==='mean'?decoderKeys('self'):[`decoder_self_L${layer}`])};}$('audioTitle').textContent='音频自注意力 · '+$('audioLayer').selectedOptions[0].text;$('crossTitle').textContent='Decoder → 全曲 P0 音频 · '+$('decoderLayer').selectedOptions[0].text;$('selfTitle').textContent='Decoder → 已接受谱面历史 · '+$('decoderLayer').selectedOptions[0].text;wave();heat('audioMap',selected.audio);heat('crossMap',selected.cross);heat('selfMap',selected.self);$('prev').disabled=page<=-1;$('next').disabled=page>=Math.ceil(T/20)-1;}
for(const id of ['page','audioLayer','decoderLayer','head'])$(id).onchange=()=>draw();
$('prev').onclick=()=>{$('page').value=Number($('page').value)-1;draw();};$('next').onclick=()=>{$('page').value=Number($('page').value)+1;draw();};$('playPage').onclick=()=>{$('audio').currentTime=domain[0];$('audio').play().catch(()=>{});};
$('audio').ontimeupdate=()=>{if(!drawPending){drawPending=true;requestAnimationFrame(()=>{draw(false);drawPending=false;});}};
for(const id of ['wave','audioMap','crossMap','selfMap']){const c=$(id);function point(e){const rect=c.getBoundingClientRect(),x=(e.clientX-rect.left)*c.width/rect.width,y=(e.clientY-rect.top)*c.height/rect.height,b=c._box;return {x,y,b,t:domain[0]+(x-b.x)/b.w*(domain[1]-domain[0])};}c.onclick=e=>{const p=point(e);if(p.x>=p.b.x&&p.x<=p.b.x+p.b.w)$('audio').currentTime=Math.max(0,Math.min(T,p.t));};c.onmousemove=e=>{if(!c._data)return;const p=point(e),q=Math.floor(p.t/dt),k=Math.floor((p.y-p.b.y)/p.b.h*T/dt);if(q<0||q>=n||k<0||k>=n||p.x<p.b.x||p.x>p.b.x+p.b.w){$('tip').style.display='none';return;}const value=c._data.out[q*n+k];$('tip').textContent=`query ${ (q*dt).toFixed(1)}–${((q+1)*dt).toFixed(1)} s\nkey ${ (k*dt).toFixed(1)}–${((k+1)*dt).toFixed(1)} s\n${c._data.valid[q]?'概率 '+(value*100).toFixed(4)+'%':'此0.5秒时间格没有用于统计的查询行（不是概率为零）'}`;$('tip').style.display='block';$('tip').style.left=Math.min(e.clientX+14,window.innerWidth-245)+'px';$('tip').style.top=Math.min(e.clientY+14,window.innerHeight-100)+'px';};c.onmouseleave=()=>$('tip').style.display='none';}
$('metadata').textContent=`音频 ${M.duration_seconds.toFixed(2)} 秒（尾部有效 beat token 中心可超出音频末采样，坐标轴保留到 ${T.toFixed(1)} 秒） · ${M.generation.rows} 行 / ${M.generation.moments} MOMENT · B50 step ${M.generation.checkpoint_step} · checkpoint ${M.generation.checkpoint_sha256.slice(0,12)} · ${M.statistics.accepted_queries} 个最终预测 query`;
$('validation').textContent=`验证：与未插桩 B50 的生成 rows ${M.completion.same_rows_as_uninstrumented_b50?'逐项相同':'存在差异，见 completion.json'}；masked 概率最大值 ${M.statistics.masked_probability_max}；严格行顺序下未来谱面概率最大值 ${M.statistics.future_causal_probability_max}。FP16 存储后单头概率和最大误差 ${M.statistics.stored_row_sum_max_error.toExponential(2)}；页面编码最大绝对误差 ${M.quantization_max_abs.toExponential(2)}。无截断；Physical 绕过与重试记录保留在 generation.json。`;
$('status').textContent='已载入 · 共 '+Math.ceil(T/20)+' 个时间页';draw();window.attentionViewerReady=true;
})().catch(e=>{document.getElementById('status').textContent='读取失败：'+e;console.error(e);});
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--audio', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    report = json.loads((root/'aggregation.json').read_text(encoding='utf8'))
    journal = json.loads((root/'journal.json').read_text(encoding='utf8'))
    complete = json.loads((root/'completion.json').read_text(encoding='utf8'))
    from importlib.machinery import SourceFileLoader
    seconds = SourceFileLoader('aggregate',str(Path(__file__).with_name('aggregate-dense-attention.py'))).load_module().seconds
    arrays = np.load(root/'time-binned-attention.npz', allow_pickle=False)
    labels = [row['label'] for row in report['streams']]
    matrices = np.stack([arrays[label] for label in labels])
    quantized = np.rint(np.clip(matrices,0,1)*65535).astype('<u2')
    quantization_error = float(np.max(np.abs(quantized.astype(np.float32)/65535 - matrices)))
    packed = quantized.tobytes()
    metadata = dict(report, completion=complete, quantization_max_abs=quantization_error, counts={label:arrays[label+'__counts'].tolist() for label in labels},
        predictions=[dict(row, output_seconds=float(seconds(np.array(row['anchor_tick']/96), report['timing']))) for row in journal['accepted_predictions']])
    (root/'viewer-data.js').write_text('window.ATTENTION_META='+json.dumps(metadata,ensure_ascii=False,separators=(',',':'))+';\nwindow.ATTENTION_DATA="'+base64.b64encode(gzip.compress(packed,compresslevel=6)).decode()+'";\n',encoding='utf8')
    (root/'index.html').write_text(HTML,encoding='utf8')
    if args.audio.resolve() != (root/'audio.mp3').resolve():
        shutil.copy2(args.audio, root/'audio.mp3')
    for helper in (Path(__file__).parent/'attention-viewer').iterdir():
        if helper.is_file():
            shutil.copy2(helper, root/helper.name)
    make_pdf(root, report, arrays, metadata['predictions'])
    print(json.dumps(dict(pages=math.ceil(report['duration_seconds']/20), streams=len(labels),
        html=str(root/'index.html'), pdf=str(root/'attention-report.pdf'))))


def make_pdf(root, report, arrays, predictions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LogNorm
    dt, duration = report['bin_seconds'], report['axis_end_seconds']
    n = report['bins']
    labels = [s['label'] for s in report['streams']]
    gkeys = [x for x in labels if x.startswith('song_audio_encoder.blocks')]
    def mean(keys):
        masks = np.stack([arrays[k+'__counts']>0 for k in keys])
        value = np.stack([arrays[k].mean(0) for k in keys])
        out=(value*masks[:,:,None]).sum(0)/np.maximum(1,masks.sum(0))[:,None]
        return np.ma.array(out,mask=np.broadcast_to(~masks.any(0)[:,None],out.shape))
    averages = [mean(gkeys),mean([f'decoder_cross_L{i}' for i in range(1,9)]),mean([f'decoder_self_L{i}' for i in range(1,9)])]
    titles=['Audio encoder self attention (2 global layers, 8 heads averaged)',
            'Decoder cross attention to full-song P0 (8 layers, 8 heads averaged)',
            'Decoder causal self attention (8 layers, 8 heads averaged)']
    wave=report['waveform'];wt=np.arange(len(wave['maximum']))*wave['envelope_step_seconds']
    note_times=[x['output_seconds'] for x in predictions if x['row']['event'] in ('NOTE','SLIDE_BRANCH')]
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#dddddd');cmap.set_under('#f2f5f6')
    with PdfPages(root/'attention-report.pdf') as pdf:
        def page(start,end,values,heading,names,number,png=False):
            fig,axs=plt.subplots(len(values)+1,1,figsize=(13,10),sharex=True,
                gridspec_kw={'height_ratios':[.6]+[1]*len(values)})
            fig.subplots_adjust(left=.085,right=.9,top=.89,bottom=.08,hspace=.34)
            fig.suptitle(heading,fontsize=17,y=.97)
            fig.text(.085,.932,f'11739 | Dense A90 / B50 | {start:.1f} - {end:.1f} s | 0.5-s bins | final accepted queries',fontsize=10)
            axs[0].fill_between(wt,wave['minimum'],wave['maximum'],color='#7196ab',linewidth=0)
            axs[0].vlines(note_times,-1.15,-.9,color='#986bba',linewidth=.5)
            axs[0].set_ylim(-1.3,1);axs[0].set_ylabel('Audio / notes')
            for ax,z,title in zip(axs[1:],values,names):
                im=ax.imshow(np.ma.maximum(z.T,1e-8),origin='upper',aspect='auto',extent=[0,n*dt,n*dt,0],
                    interpolation='nearest',cmap=cmap,norm=LogNorm(vmin=1e-4,vmax=1))
                ax.plot([start,end],[start,end],color='#e79766',lw=.7,ls='--')
                ax.set_ylim(duration,0);ax.set_title(title,fontsize=11,loc='left');ax.set_ylabel('Key time (s)')
            for ax in axs:ax.set_xlim(start,end);ax.tick_params(labelsize=9)
            axs[-1].set_xlabel('Query known-anchor time on the audio timeline (s)',fontsize=11)
            cb=fig.colorbar(im,cax=fig.add_axes([.92,.2,.015,.55]));cb.set_label('Attention probability mass / key time bin')
            fig.text(.085,.025,'Keys summed; queries averaged. No renormalization after display cropping. Same-beat row order is retained in raw records.',fontsize=9)
            pdf.savefig(fig)
            if png:fig.savefig(root/f'page-{number:02d}.png',dpi=140)
            plt.close(fig)
        page(0,duration,averages,'Full-song attention overview',titles,0,True)
        for i in range(math.ceil(duration/20)):
            page(i*20,min(duration,(i+1)*20),averages,f'Time page {i+1}',titles,i+1,True)
        for layer in range(1,9):
            page(0,duration,[mean([f'decoder_cross_L{layer}']),mean([f'decoder_self_L{layer}'])],
                 f'Decoder layer {layer}', ['Audio cross attention / 8-head mean','Causal self attention / 8-head mean'],20+layer)
        for stage in range(3):
            keys=[x for x in labels if f'.stages.{stage}.' in x]
            if keys:
                page(0,duration,[mean([key]) for key in keys],f'Local audio encoder E{stage}',
                     [key+' / 8-head mean' for key in keys],40+stage)


if __name__ == '__main__':
    main()
