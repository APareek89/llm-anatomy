import './ui/style.css';
import {loadModelData,loadTile,loadPrefetchedTile,downloadStatus,computeStats,matrixShape,type ModelManifest,type TensorMeta,type WeightTile,type TileFile} from './data/hf';
import {AnatomyScene,type Selection} from './scene/anatomy';
import {esc,count,numeric,components,describeTensor,glossary,tourStops} from './ui/content';
import {Playback} from './playback/controller';
import {ModelWorker} from './playback/client';
import type {RecordedStep,TensorView} from './model/micro/tensor';
import type {InferenceResult,TrainResult,MicroConfig,Prediction} from './model/micro/model';

const icons:Record<string,string>={
 search:'<circle cx="10" cy="10" r="6"/><path d="m15 15 5 5"/>',home:'<path d="m3 10 9-7 9 7v10H3Z"/><path d="M9 20v-7h6v7"/>',back:'<path d="m14 5-7 7 7 7"/>',forward:'<path d="m10 5 7 7-7 7"/>',play:'<path d="m8 4 12 8-12 8Z"/>',pause:'<path d="M8 4v16M16 4v16"/>',book:'<path d="M12 5v15M12 5C8 2 3 3 3 3v15s5-1 9 2c4-3 9-2 9-2V3s-5-1-9 2Z"/>',close:'<path d="m6 6 12 12M18 6 6 18"/>',expand:'<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5"/>',reset:'<path d="M4 10a8 8 0 1 1 1 8M4 4v6h6"/>',arrow:'<path d="M4 12h16m-6-6 6 6-6 6"/>',layers:'<path d="m3 8 9-5 9 5-9 5-9-5Zm0 5 9 5 9-5M3 18l9 5 9-5"/>'
};
const icon=(name:string)=>`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name]??icons.arrow}</svg>`;
const $=<T extends HTMLElement=HTMLElement>(id:string)=>document.getElementById(id) as T;
const on=(id:string,fn:(event:Event)=>void)=>$(id)?.addEventListener('click',event=>{try{fn(event);}catch(e){fail(e);}});
let manifest:ModelManifest,scene:AnatomyScene,selection:Selection={kind:'model'},currentTile:WeightTile|undefined;
let mode:'explore'|'inference'|'training'='explore',micro:{config:MicroConfig;vocab:string[];parameterCount:number;parameters:{name:string;shape:number[]}[];step:number;corpus:string[]}|undefined;
let busy=false,traceResult:InferenceResult|undefined,lastTrain:TrainResult|undefined,lossHistory:{step:number;loss:number;accuracy:number}[]=[],evaluation:{loss:number;accuracy:number}|undefined;
let requestGeneration=0,tokenRequest=0,toastTimer:ReturnType<typeof setTimeout>,tourIndex=-1,tourRunning=false;
const worker=new ModelWorker();
const tokenizerWorker=new Worker(new URL('./data/tokenizer.worker.ts',import.meta.url),{type:'module'});
let microReady:Promise<void>;
const playback=new Playback((step,index,total,playing)=>updatePlayback(step,index,total,playing),()=>{
  if(mode==='inference'&&traceResult){$('annotation').classList.remove('hidden');$('annotation').innerHTML=`Chosen next token: <strong>${esc(traceResult.prediction)}</strong> <span class="tag micro">[micro]</span>`;}
});

function toast(message:string,error=false){clearTimeout(toastTimer);let box=$('toast');if(!box){box=document.createElement('div');box.id='toast';document.body.appendChild(box);}box.className=`toast${error?' error':''}`;box.textContent=message;box.classList.remove('hidden');toastTimer=setTimeout(()=>box.classList.add('hidden'),error?12000:5000);}
function fail(error:unknown){console.error(error);toast(error instanceof Error?error.message:String(error),true);}
function componentInfo(key:string){return components[key]??components[key==='deltaNet'?'delta':key==='lmHead'?'lm_head':key]??{label:key,color:'#8994a3'};}
function getMeta(name:string):TensorMeta|undefined{
  if(mode==='explore')return manifest.tensors[name];
  const p=micro?.parameters.find(p=>p.name===name);if(!p)return;const params=p.shape.reduce((a,b)=>a*b,1),match=/layers\.(\d+)/.exec(name);
  return {name,shape:p.shape,dtype:'Float64',params,bytes:params*8,shard:'Micro-Qwen worker',component:name.includes('mlp')?'ffn':name.includes('linear_attn')?'deltaNet':name.includes('self_attn')?'attention':'norms',layer:match?+match[1]:undefined,dataOffsets:[0,0],absoluteOffsets:[0,0]};
}
function shell(){
  $('app').innerHTML=`
  <header class="topbar"><div class="brand"><div class="brandmark">Q</div><div><div class="brand-name">Qwen <span style="font-weight:350;color:#9caeba">anatomy</span></div><div class="brand-sub">INSIDE A LANGUAGE MODEL</div></div></div>
    <nav class="mode-switch" aria-label="Explorer mode"><button data-mode="explore" class="active">Explore</button><button data-mode="inference">Inference</button><button data-mode="training">Training</button></nav>
    <div class="top-actions"><button id="source-button" class="ghost source-link">Data & sources</button><button id="glossary-button" class="ghost" title="Open glossary">${icon('book')} <span>Glossary</span></button><button id="tour-button">Take the tour ${icon('arrow')}</button></div>
  </header>
  <main id="stage"></main><div id="breadcrumb" class="breadcrumb"></div>
  <div class="stage-caption"><div class="eyebrow" id="stage-eyebrow">THE REAL CHECKPOINT</div><h1 id="stage-title">One model.<br>Billions of decisions.</h1><p id="stage-description">Orbit the architecture. Open a layer.<br>Get all the way down to a single weight.</p></div>
  <div class="stage-pills"><span id="source-pill" class="tag config">[config] structure</span><span id="lod-pill" class="tag">LOD 0 · model</span></div>
  <aside id="sidebar" class="sidebar" aria-label="Model structure"></aside>
  <aside id="inspector" class="inspector" aria-label="Inspector and calculation lens"><div id="inspector-head" class="inspector-head"></div><div id="inspector-body" class="inspector-body"></div></aside>
  <div class="stage-tools"><button id="back-button" title="Back one level">${icon('back')} Back</button><button id="home-button" title="Fit the whole model">${icon('home')} Model</button><button id="fullscreen-button" title="Toggle full screen">${icon('expand')}</button></div>
  <div class="stage-hint">DRAG TO ORBIT &nbsp;·&nbsp; SCROLL TO ZOOM &nbsp;·&nbsp; DOUBLE-CLICK TO FOCUS &nbsp;·&nbsp; ESC TO GO BACK</div>
  <div id="annotation" class="annotation hidden"></div>
  <section id="bottom-panel" class="bottom-panel"></section>
  <footer class="statusbar"><span id="status-left">PINNED CHECKPOINT · ${manifest.revision.slice(0,8)} · ${Object.keys(manifest.tensors).length.toLocaleString()} TENSORS</span><span id="status-right">INITIALIZING 3D</span></footer>`;
  document.querySelectorAll<HTMLButtonElement>('[data-mode]').forEach(b=>b.onclick=()=>void setMode(b.dataset.mode as typeof mode).catch(fail));
  on('home-button',()=>{scene.showModel();});on('back-button',()=>scene.back());
  on('fullscreen-button',()=>{if(document.fullscreenElement)void document.exitFullscreen();else void document.documentElement.requestFullscreen().catch(fail);});
  on('glossary-button',showGlossary);on('source-button',showSources);on('tour-button',()=>void startTour());
  window.addEventListener('keydown',event=>{if(event.key==='Escape'){closeModal();if(tourIndex>=0)endTour();}if(event.code==='Space'&&!['INPUT','SELECT','TEXTAREA','BUTTON'].includes((event.target as HTMLElement).tagName)&&mode!=='explore'){event.preventDefault();playback.toggle();}});
}
function renderOverviewSidebar(){
  const sorted=Object.entries(manifest.components).sort((a,b)=>b[1].params-a[1].params),c=manifest.config.text_config;
  $('sidebar').innerHTML=`<div class="spread"><span class="eyebrow">MODEL ANATOMY</span><span class="tag config">[config]</span></div><h2>Qwen3.8–27B</h2><p class="tiny muted">Hybrid language + vision model</p>
    <div class="model-total">${(manifest.totalParams/1e9).toFixed(2)}<span>billion parameters</span></div>
    <div class="mini-facts"><span>${c.num_hidden_layers} layers</span><span>${c.hidden_size.toLocaleString()} hidden</span><span>BF16</span></div>
    <div class="param-bar" title="Parameter share computed from all tensor headers">${sorted.map(([k,v])=>`<span style="width:${v.params/manifest.totalParams*100}%;background:${componentInfo(k).color}"></span>`).join('')}</div>
    <div class="component-list">${sorted.map(([k,v])=>`<div class="component-line"><i style="background:${componentInfo(k).color}"></i><span class="label">${componentInfo(k).label}</span><span class="value mono">${count(v.params)}</span></div>`).join('')}</div>
    <div class="structure-section"><div class="spread"><span class="eyebrow">THE LAYER STACK</span><span class="tiny muted">${c.num_hidden_layers} layers</span></div><div class="search">${icon('search')}<input id="tensor-search" placeholder="Find layer or tensor…" aria-label="Find layer or tensor" autocomplete="off"></div><div id="layer-list" class="layer-list"></div></div>
    <div class="sidebar-footer">Three DeltaNet layers, then one full-attention layer. Repeated ${c.num_hidden_layers/4} times.</div>`;
  renderSearch('');$('tensor-search').addEventListener('input',e=>renderSearch((e.target as HTMLInputElement).value));
  $('tensor-search').addEventListener('keydown',e=>{if((e as KeyboardEvent).key==='Enter')$('layer-list').querySelector<HTMLButtonElement>('button')?.click();});
}
function renderSearch(query:string){
  const q=query.trim().toLowerCase(),numericMatch=/^(?:layer\s*)?(\d+)$/.exec(q);
  const list=$('layer-list');
  if(!q||numericMatch){const layers=manifest.layers.filter(l=>!numericMatch||l.index===+numericMatch[1]);list.innerHTML=layers.map(l=>`<button class="layer-row ${l.kind==='full_attention'?'attention':''} ${selection.layer===l.index?'active':''}" data-layer="${l.index}"><span class="layer-number mono">${String(l.index).padStart(2,'0')}</span><i></i><span>Layer ${l.index}</span><span class="kind">${l.kind==='full_attention'?'attention':'DeltaNet'}</span></button>`).join('');list.querySelectorAll<HTMLButtonElement>('[data-layer]').forEach(b=>b.onclick=()=>scene.showLayer(Number(b.dataset.layer)));}
  else {const names=Object.keys(manifest.tensors).filter(n=>n.toLowerCase().includes(q)).slice(0,60);list.innerHTML=names.length?names.map(n=>`<button class="tensor-button" data-name="${esc(n)}" style="width:100%;margin-bottom:4px"><span>${esc(n.replace('model.language_model.',''))}</span><small>${manifest.tensors[n].shape.join(' × ')}</small></button>`).join(''):'<p class="tiny muted">No matching tensor. Try “q_proj” or “layer 40”.</p>';list.querySelectorAll<HTMLButtonElement>('[data-name]').forEach(b=>b.onclick=()=>scene.showTensor(manifest.tensors[b.dataset.name!]));}
}
function renderExploreBottom(){
  $('bottom-panel').innerHTML=`<div class="explore-bottom"><div><div class="eyebrow">FOLLOW YOUR CURIOSITY</div><h3>Choose a place to begin</h3><p>Real structure. Real weight samples.</p></div><div class="quick-stops"><button id="quick-delta"><span>01 / RECURRENT MEMORY</span><strong>Inside DeltaNet ${icon('arrow')}</strong></button><button id="quick-attention"><span>02 / CONTEXT & HEADS</span><strong>Inside attention ${icon('arrow')}</strong></button><button id="quick-weight"><span>03 / THE REAL NUMBERS</span><strong>Inspect a weight ${icon('arrow')}</strong></button></div><div class="download-meta" id="budget"></div></div>`;
  on('quick-delta',()=>scene.showLayer(0));on('quick-attention',()=>scene.showLayer(3));on('quick-weight',()=>scene.showTensor(manifest.tensors['model.language_model.layers.3.self_attn.q_proj.weight']));void refreshBudget();
}
async function refreshBudget(){const b=await downloadStatus();if($('budget'))$('budget').innerHTML=`<strong class="mono">${(b.used/1e6).toFixed(1)} / ${(b.limit/1e6).toFixed(0)} MB</strong><br>Data budget used<br><span class="dot"></span>Small tiles, fetched on demand`;}
function selectObject(s:Selection){
  document.querySelector('.stage-caption')?.classList.toggle('hidden',s.kind!=='model');
  if(s.kind==='cell'&&currentTile&&currentTile.name===s.tensor){const tile=currentTile;selection=s;renderBreadcrumb();renderInspector(s);renderTile(tile);showCellValue(tile,s.row??tile.row,s.col??tile.col);return;}
  selection=s;currentTile=undefined;requestGeneration++;
  $('source-pill').className=`tag ${mode==='explore'?'config':'micro'}`;$('source-pill').textContent=mode==='explore'?'[config] structure':'[micro] live computation';
  renderBreadcrumb();if(mode==='explore')renderInspector(s);else if(s.kind==='tensor'||s.kind==='cell')renderInspector(s);
  document.querySelectorAll<HTMLButtonElement>('[data-layer]').forEach(b=>b.classList.toggle('active',Number(b.dataset.layer)===s.layer));
  if(s.kind==='tensor'&&s.tensor){const meta=getMeta(s.tensor);if(!meta)return;
    if(mode!=='explore'){void fetchMicroTile(meta).catch(fail);return;}
    const row=s.row??0,col=s.col??0;const local=manifest.tiles.find(f=>f.name===meta.name&&f.row<=row&&f.col<=col&&f.row+f.rows>row&&f.col+f.cols>col);
    if(local)void fetchCurrentTile(row,col).catch(fail);
  }
}
function renderBreadcrumb(){
  const pieces=[`<button id="crumb-model">${mode==='explore'?'Qwen3.8–27B':'Micro-Qwen'}</button>`];
  if(selection.layer!==undefined)pieces.push(`<span>›</span><button id="crumb-layer">Layer ${selection.layer}</button>`);
  if(selection.tensor)pieces.push(`<span>›</span><span class="current mono">${esc(selection.tensor.split('.').slice(-2).join('.'))}</span>`);
  if(selection.kind==='cell')pieces.push(`<span>›</span><span class="current mono">(${selection.row}, ${selection.col})</span>`);
  $('breadcrumb').innerHTML=pieces.join('');on('crumb-model',()=>scene.showModel());if($('crumb-layer'))on('crumb-layer',()=>scene.showLayer(selection.layer!));
}
function inspectorTitle(eyebrow:string,tag='config'){ $('inspector-head').innerHTML=`<div class="spread"><span class="eyebrow">${eyebrow}</span><span class="tag ${tag}">[${tag}]</span></div>`; }
function renderInspector(s:Selection){
  const source=mode==='explore'?'config':'micro',body=$('inspector-body');
  if(s.kind==='model'){
    inspectorTitle('A CLOSER LOOK');const c=manifest.config.text_config,ffn=manifest.components.ffn.params/manifest.totalParams*100;
    body.innerHTML=`<h2>A language model,<br>opened up.</h2><p>Each layer reads a running representation, adds context, and refines its features. The next layer continues the work.</p><div class="info-grid"><div><small>Learned parameters</small><strong>${manifest.totalParams.toLocaleString()}</strong></div><div><small>Checkpoint storage</small><strong>${(manifest.totalBytes/1e9).toFixed(2)} GB</strong></div><div><small>Vocabulary</small><strong>${c.vocab_size.toLocaleString()} tokens</strong></div><div><small>Native context limit</small><strong>${c.max_position_embeddings.toLocaleString()}</strong></div></div><div class="callout">${ffn.toFixed(1)}% of the parameters live in the feed-forward blocks. Their job is to transform features at each token position.</div><h3>What the colours mean</h3><p><span style="color:var(--green)">DeltaNet</span> carries a compact recurrent memory. <span style="color:var(--violet)">Full attention</span> compares token pairs. <span style="color:var(--sand)">Feed-forward</span> blocks transform those features.</p><h3>Start close. Go deeper.</h3><p>Click any layer to open its tensors. Double-click a tensor to inspect a real weight tile, then zoom to individual cells.</p><div class="subtle-divider"></div><p class="tiny">Live inference and training use a separate, clearly labelled 8-layer Micro-Qwen. The 27B checkpoint is explored through its actual metadata and weight tiles.</p><button id="try-simulation" class="primary" style="width:100%;margin-top:16px">Watch a forward pass ${icon('arrow')}</button>`;
    on('try-simulation',()=>void setMode('inference').catch(fail));return;
  }
  if(s.kind==='layer'){
    const l=s.layer!,names=mode==='explore'?manifest.layers[l].names:micro!.parameters.filter(p=>p.name.startsWith(`layers.${l}.`)).map(p=>p.name),params=names.reduce((a,n)=>a+(getMeta(n)?.params??0),0),full=l%4===3;
    inspectorTitle('LAYER ANATOMY',source);body.innerHTML=`<h2>Layer ${String(l).padStart(2,'0')}<br><span style="color:${full?'var(--violet)':'var(--green)'}">${full?'Gated attention':'Gated DeltaNet'}</span></h2><p>${full?'Queries compare with earlier keys. A probability-weighted sum reads their values, then a gate controls the output.':'A compact state remembers earlier tokens. Each new key and value corrects what that memory predicts.'}</p><div class="info-grid"><div><small>Parameters in layer</small><strong>${count(params)}</strong></div><div><small>Named tensors</small><strong>${names.length}</strong></div></div><div class="math">residual → RMSNorm → mixer → add<br>→ RMSNorm → feed-forward → add</div><h3>Open a tensor</h3><div class="tensor-list">${names.map(n=>`<button class="tensor-button" data-inspect="${esc(n)}"><span>${esc(n.split('.').slice(-3).join('.'))}</span><small>${getMeta(n)!.shape.join(' × ')} · ${count(getMeta(n)!.params)} parameters</small></button>`).join('')}</div>`;
    body.querySelectorAll<HTMLButtonElement>('[data-inspect]').forEach(b=>b.onclick=()=>scene.showTensor(getMeta(b.dataset.inspect!)!));return;
  }
  const meta=s.tensor?getMeta(s.tensor):undefined;if(!meta){inspectorTitle('TENSOR UNAVAILABLE');body.innerHTML='<p>This selection has no verified tensor metadata.</p>';return;}
  const d=describeTensor(meta),[rows,cols]=matrixShape(meta),isQ=meta.name.includes('self_attn.q_proj'),isK=/self_attn\.[kv]_proj/.test(meta.name),headDim=mode==='explore'?manifest.config.text_config.head_dim:micro!.config.headDim,heads=isQ?(mode==='explore'?manifest.config.text_config.num_attention_heads:micro!.config.queryHeads):isK?(mode==='explore'?manifest.config.text_config.num_key_value_heads:micro!.config.kvHeads):0;
  inspectorTitle(s.kind==='cell'?'ONE LEARNED NUMBER':'TENSOR INSPECTOR',source);
  body.innerHTML=`<h2>${esc(d.title)}</h2><div class="tensor-name mono">${esc(meta.name)}</div><p style="margin-top:12px">${esc(d.what)}</p><div class="info-grid"><div><small>Shape · output × input</small><strong>${meta.shape.join(' × ')}</strong></div><div><small>Storage dtype</small><strong>${meta.dtype}</strong></div><div><small>Parameters</small><strong>${count(meta.params)}</strong></div><div><small>Tensor storage</small><strong>${(meta.bytes/1e6).toFixed(3)} MB</strong></div></div><div id="cell-value"></div><div id="tile-state"><div class="banner">Not loaded — click to fetch.<br><span class="tiny">No values are invented for this tensor.</span></div></div>
    ${heads?`<h3>${isQ?'Query + gate':'Key / value'} head bands</h3><p class="tiny">${isQ?`${heads} heads. Each band has ${headDim} query rows, then ${headDim} output-gate rows.`:`${heads} heads. ${headDim} rows per head; each is shared by several queries.`}</p><div class="head-bands">${Array.from({length:heads},(_,h)=>`<button data-head="${h}" title="Inspect head ${h}">${h}</button>`).join('')}</div>`:''}
    <div class="tile-controls"><label>Output row<input id="tile-row" type="number" min="0" max="${rows-1}" value="${s.row??0}" step="256"></label><label>Input column<input id="tile-col" type="number" min="0" max="${cols-1}" value="${s.col??0}" step="256"></label><button id="fetch-tile" class="primary">${mode==='explore'?'Fetch real tile':'Read micro weights'} ${icon('arrow')}</button></div><div id="sparse-rows"></div>
    <h3>Why it exists</h3><p>${esc(d.why)}</p><div class="math">${esc(d.math)}</div><h3>Think of it as…</h3><p>${esc(d.analogy)}</p><p class="tiny" style="margin-top:14px">Columns select input features. Rows contribute to output features. ${meta.shape.length>2?'Extra tensor dimensions are flattened into columns for viewing.':''}</p>`;
  on('fetch-tile',()=>void (mode==='explore'?fetchCurrentTile():fetchMicroTile(meta)).catch(fail));
  body.querySelectorAll<HTMLButtonElement>('[data-head]').forEach(b=>b.onclick=()=>{const r=Number(b.dataset.head)*headDim*(isQ?2:1);$<HTMLInputElement>('tile-row').value=String(r);void(mode==='explore'?fetchCurrentTile(r,0):fetchMicroTile(meta)).catch(fail);body.querySelectorAll('[data-head]').forEach(x=>x.classList.remove('active'));b.classList.add('active');});
  const sparse=mode==='explore'?manifest.tiles.filter(t=>t.name===meta.name&&t.rows===1):[];
  if(sparse.length){$('sparse-rows').innerHTML=`<h3>Prefetched token rows</h3><p class="tiny">Full rows for the sample sentence${sparse.length>10?' and 64 seeded random token IDs':''}.</p><select id="sparse-select" style="width:100%;margin-top:7px"><option value="">Choose a loaded token ID…</option>${sparse.map(f=>`<option value="${f.id}">Token ${f.row} · ${f.cols} input features</option>`).join('')}</select>`;$('sparse-select').addEventListener('change',()=>{const file=sparse.find(f=>f.id===$<HTMLSelectElement>('sparse-select').value);if(file)void fetchSparse(file).catch(fail);});}
}
async function fetchSparse(file:TileFile){const generation=++requestGeneration;loading(0);const tile=await loadPrefetchedTile(file);if(generation!==requestGeneration)return;currentTile=tile;scene.setTile(tile);renderTile(tile);}
function loading(f:number){const node=$('tile-state');if(node)node.innerHTML=`<div class="spread"><span class="tiny muted">Fetching verified weight values</span><span class="tiny mono">${Math.round(f*100)}%</span></div><div class="progress"><span style="width:${f*100}%"></span></div>`;(scene as AnatomyScene&{setLoading?:(progress:number)=>void}).setLoading?.(f);}
async function fetchCurrentTile(row?:number,col?:number){
  const name=selection.tensor;if(!name)return;const generation=++requestGeneration;
  row??=Number($<HTMLInputElement>('tile-row')?.value??0);col??=Number($<HTMLInputElement>('tile-col')?.value??0);loading(0);
  try{const tile=await loadTile(name,row,col,f=>{if(generation===requestGeneration)loading(f);});if(generation!==requestGeneration)return;currentTile=tile;scene.setTile(tile);renderTile(tile);void refreshBudget();}
  catch(error){if(generation===requestGeneration&&$('tile-state'))$('tile-state').innerHTML=`<div class="banner" style="border-color:#7d5146;color:#d2a79f;background:#251b19">${esc(error instanceof Error?error.message:String(error))}</div>`;throw error;}
}
async function fetchMicroTile(meta:TensorMeta){
  const generation=++requestGeneration,row=Number($<HTMLInputElement>('tile-row')?.value??0),col=Number($<HTMLInputElement>('tile-col')?.value??0),response=await worker.call('parameter',{name:meta.name});if(generation!==requestGeneration)return;
  const [height,width]=matrixShape(meta),r=Math.max(0,Math.min(height-1,Math.floor(row))),c=Math.max(0,Math.min(width-1,Math.floor(col))),rows=Math.min(256,height-r),cols=Math.min(256,width-c),values=new Float32Array(rows*cols);for(let j=0;j<rows;j++)values.set(response.result.values.slice((r+j)*width+c,(r+j)*width+c+cols),j*cols);
  const tile:WeightTile={name:meta.name,row:r,col:c,rows,cols,values,source:'micro',origin:'micro-worker',stats:computeStats(values)};currentTile=tile;scene.setTile(tile);renderTile(tile);
}
function renderTile(tile:WeightTile){
  if(!$('tile-state'))return;const s=tile.stats,tag=tile.source;
  $('source-pill').className=`tag ${tag}`;$('source-pill').textContent=`[${tag}] weight values`;
  $('tile-state').innerHTML=`<div class="spread"><span class="tag ${tag}">[${tag}] loaded values</span><span class="tiny muted">${tile.rows} × ${tile.cols}</span></div><canvas id="tile-preview" class="heatmap-preview" width="256" height="160" aria-label="Loaded weight heatmap"></canvas><p class="tiny">Rows ${tile.row}–${tile.row+tile.rows-1} · columns ${tile.col}–${tile.col+tile.cols-1}<br>${tile.source==='real'?esc(tile.origin):'Micro worker values · Float32 display copy'}</p><div class="stats-grid"><div><small>Mean</small><strong>${numeric(s.mean)}</strong></div><div><small>Std. deviation</small><strong>${numeric(s.std)}</strong></div><div><small>Minimum</small><strong>${numeric(s.min)}</strong></div><div><small>Maximum</small><strong>${numeric(s.max)}</strong></div><div><small>Near zero · |w| &lt; .001</small><strong>${(s.nearZeroFraction*100).toFixed(2)}%</strong></div><div><small>Values in this tile</small><strong>${s.count.toLocaleString()}</strong></div></div><p class="tiny muted">Statistics describe this loaded tile only. Symmetric colour scale uses its 1st / 99th percentiles.</p>`;
  drawMatrix($<HTMLCanvasElement>('tile-preview'),tile.values,[tile.rows,tile.cols]);
  if($('tile-row'))$<HTMLInputElement>('tile-row').value=String(tile.row);if($('tile-col'))$<HTMLInputElement>('tile-col').value=String(tile.col);
  const preview=$<HTMLCanvasElement>('tile-preview');preview.onmousemove=e=>{const rect=preview.getBoundingClientRect(),row=tile.row+Math.min(tile.rows-1,Math.floor((e.clientY-rect.top)/rect.height*tile.rows)),col=tile.col+Math.min(tile.cols-1,Math.floor((e.clientX-rect.left)/rect.width*tile.cols));showCellValue(tile,row,col);};
}
function showCellValue(tile:WeightTile,row:number,col:number){const index=(row-tile.row)*tile.cols+col-tile.col,value=tile.values[index];if(value===undefined)return;const node=$('cell-value');if(node)node.innerHTML=`<div class="callout"><span class="tag ${tile.source}">[${tile.source}]</span><div class="mono" style="font-size:16px;margin:9px 0">w[${row}, ${col}] = ${numeric(value)}</div>Multiplies input feature <strong>${col}</strong> and contributes to output feature <strong>${row}</strong>.</div>`;}
function drawMatrix(canvas:HTMLCanvasElement|undefined,values:ArrayLike<number>,shape:number[],mask?:boolean[],scale?:number){
  if(!canvas)return;const ctx=canvas.getContext('2d')!,rows=shape.length>1?shape[0]:1,cols=shape.length>1?shape.slice(1).reduce((a,b)=>a*b,1):shape[0];
  const finite=Array.from(values).filter(Number.isFinite).sort((a,b)=>a-b),max=scale??Math.max(1e-9,Math.abs(finite[Math.floor(finite.length*.01)]??0),Math.abs(finite[Math.floor(finite.length*.99)]??0));
  const w=canvas.width,h=canvas.height;ctx.fillStyle='#0c141c';ctx.fillRect(0,0,w,h);const rr=Math.min(rows,256),cc=Math.min(cols,256),cw=w/cc,ch=h/rr;
  for(let r=0;r<rr;r++)for(let c=0;c<cc;c++){const i=Math.floor(r*rows/rr)*cols+Math.floor(c*cols/cc),x=values[i];if(mask?.[i]||!Number.isFinite(x)){ctx.fillStyle='#27303a';ctx.fillRect(c*cw,r*ch,cw+.3,ch+.3);if(cw>20&&ch>20){ctx.fillStyle='#647180';ctx.font='9px monospace';ctx.textAlign='center';ctx.fillText('×',(c+.5)*cw,(r+.5)*ch+3);}continue;}const a=Math.min(1,Math.abs(x)/max);ctx.fillStyle=x>=0?`rgb(${Math.round(20+a*78)},${Math.round(32+a*174)},${Math.round(43+a*137)})`:`rgb(${Math.round(25+a*139)},${Math.round(30+a*109)},${Math.round(45+a*174)})`;ctx.fillRect(c*cw,r*ch,Math.max(1,cw-.6),Math.max(1,ch-.6));if(cw>45&&ch>23){ctx.fillStyle=a>.5?'#0b1720':'#cad7df';ctx.textAlign='center';ctx.font='9px monospace';ctx.fillText(x.toFixed(3),(c+.5)*cw,(r+.5)*ch+3);}}
}

async function setMode(next:typeof mode){
  if(busy){toast('Let the current calculation finish, or stop the fast training run.');return;}
  playback.pause();mode=next;document.body.classList.toggle('sim-mode',mode!=='explore');document.querySelectorAll<HTMLButtonElement>('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode));$('annotation').classList.add('hidden');
  if(mode==='explore'){
    scene.setMicro(false);scene.showModel();renderOverviewSidebar();renderExploreBottom();renderInspector({kind:'model'});
    $('stage-eyebrow').textContent='THE REAL CHECKPOINT';$('stage-title').innerHTML='One model.<br>Billions of decisions.';$('stage-description').innerHTML='Orbit the architecture. Open a layer.<br>Get all the way down to a single weight.';$('source-pill').className='tag config';$('source-pill').textContent='[config] structure';return;
  }
  $('stage-eyebrow').textContent='THE WORKING LEARNING MODEL';$('stage-title').innerHTML=mode==='inference'?'Follow a thought.<br>One operation at a time.':'Watch the model<br>learn from its mistakes.';$('stage-description').innerHTML='Micro-Qwen · 8 layers · actual calculations<br>A smaller model from the same family.';$('source-pill').className='tag micro';$('source-pill').textContent='[micro] live computation';
  await microReady;scene.setMicro(true,micro!.parameters);scene.showModel();renderSimulationSidebar();renderSimulationBottom();
  if(mode==='inference')await runInference(false);else {playback.load([]);inspectorTitle('CALCULATION LENS','micro');$('inspector-body').innerHTML='<h2>Learning, made visible.</h2><p>Run one animated step to see the batch, forward pass, loss, gradients, and exact weight updates.</p><div class="callout">Gradients trace the loss backward through its dependencies. The optimizer uses them to make a small change to each learned weight.</div><p>The starter has already trained for 300 steps. Reset weights to watch learning from the same seed.</p><button id="start-one" class="primary" style="width:100%;margin-top:18px">Animate one training step</button>';on('start-one',()=>void train(1,true).catch(fail));}
}
function renderSimulationBottom(){
  const training=mode==='training';
  $('bottom-panel').innerHTML=`<div class="sim-settings"><label class="prompt-field">${training?'Probe prompt':'Prompt'}<input id="prompt" value="${esc(traceResult?.prompt??'the cat sat on')}" aria-label="Simulation prompt"></label>
    ${training?'<label>Learning rate<input id="learning-rate" class="small-input" type="number" value="0.05" min="0.0001" max="1" step="0.01"></label><label>Batch<input id="batch-size" class="small-input" type="number" value="4" min="1" max="16"></label><button id="train-one" class="primary">Animate 1 step</button><label>Fast steps<input id="train-count" class="small-input" type="number" value="300" min="1" max="10000"></label><button id="train-fast">Train 300 fast</button><button id="stop-training" class="hidden">Stop</button><button id="reset-weights" title="Reset to seed 42">Reset weights</button>':'<label>Temperature<input id="temperature" class="small-input" type="number" value="0" min="0" max="3" step="0.1"></label><button id="run-inference" class="primary">Run forward pass '+icon('arrow')+'</button><button id="next-token" title="Append the chosen token and run another pass">Append next token</button><button id="open-tokenizer">Tokenizer</button>'}
    <span class="tag micro">[micro]</span></div>
    <div class="sim-controls"><div class="transport"><button id="step-back" title="Previous recorded operation">${icon('back')}</button><button id="play" class="play" title="Play or pause">${icon('play')}</button><button id="step-forward" title="Next recorded operation">${icon('forward')}</button></div><div class="scrubber-wrap"><input id="scrubber" type="range" min="0" max="1" value="0" step="1" aria-label="Recorded operation scrubber"><div class="step-caption"><span id="step-label">Prepare a calculation to begin</span><span id="step-number">0 / 0</span></div></div><label class="speed-wrap">Speed<input id="speed" type="range" min="0.05" max="5" value="${playback.speed}" step="0.05" aria-label="Playback speed"><span id="speed-value">${playback.speed.toFixed(2)}×</span></label><select id="jump-layer" class="jump-select" aria-label="Jump to recorded layer"><option value="">Jump to layer</option>${Array.from({length:micro!.config.layers},(_,i)=>`<option value="${i}">Layer ${i} · ${i%4===3?'attention':'DeltaNet'}</option>`).join('')}<option value="final">Final norm + logits</option>${training?'<option value="backward">Backward gradients</option><option value="update">Weight updates</option>':''}</select><label class="follow-label"><input id="follow-camera" type="checkbox" checked> Follow operation</label></div>`;
  on('play',()=>playback.toggle());on('step-back',()=>playback.step(-1));on('step-forward',()=>playback.step(1));
  $('scrubber').addEventListener('input',()=>{playback.pause();playback.seek(Number($<HTMLInputElement>('scrubber').value));});
  $('speed').addEventListener('input',()=>{const speed=Number($<HTMLInputElement>('speed').value);playback.setSpeed(speed);$('speed-value').textContent=`${speed.toFixed(2)}×`;});
  $('follow-camera').addEventListener('change',()=>{scene.setFollow($<HTMLInputElement>('follow-camera').checked);});
  $('jump-layer').addEventListener('change',()=>{const value=$<HTMLSelectElement>('jump-layer').value;const index=playback.steps.findIndex(s=>value==='final'?s.layer===micro!.config.layers:value==='backward'||value==='update'?s.phase===value:s.layer===Number(value));if(index>=0){playback.pause();playback.seek(index);}else toast('Run a recorded calculation first.');});
  if(training){on('train-one',()=>void train(1,true).catch(fail));on('train-fast',()=>void train(Number($<HTMLInputElement>('train-count').value),false).catch(fail));on('stop-training',()=>void worker.call('stop').catch(fail));on('reset-weights',()=>void resetWeights().catch(fail));$('train-count').addEventListener('input',()=>{$('train-fast').textContent=`Train ${$<HTMLInputElement>('train-count').value} fast`;});}
  else {on('run-inference',()=>void runInference(true).catch(fail));on('open-tokenizer',showTokenizer);on('next-token',()=>{if(traceResult){const prompt=$<HTMLInputElement>('prompt');prompt.value=`${prompt.value.trim()} ${traceResult.prediction}`;void runInference(true).catch(fail);}});}
  $('prompt').addEventListener('keydown',e=>{if((e as KeyboardEvent).key==='Enter')void runInference(false).catch(fail);});
}
function predictionBars(top:Prediction[],limit=7){return top.slice(0,limit).map(p=>`<div class="prediction-row"><span>${esc(p.token)}</span><div class="track"><span style="width:${Math.max(.1,p.probability*100)}%"></span></div><span class="percent mono">${(p.probability*100).toFixed(1)}%</span></div>`).join('');}
function renderSimulationSidebar(){
  if(!micro)return;
  const training=mode==='training';$('sidebar').innerHTML=`<div class="spread"><span class="eyebrow">${training?'LEARNING LAB':'TOKEN JOURNEY'}</span><span class="tag micro">[micro]</span></div><h2>Micro-Qwen</h2><p class="tiny muted">${count(micro.parameterCount)} parameters · ${micro.config.layers} layers</p><div class="mini-facts"><span>Seed ${micro.config.seed}</span><span id="iteration-label">${micro.step} trained steps</span></div><div class="sim-sidebar-inner" id="sim-sidebar-inner">
  ${training?`<div class="training-metrics"><div><small class="tiny">Corpus loss</small><strong id="corpus-loss">${evaluation?evaluation.loss.toFixed(3):'—'}</strong></div><div><small class="tiny">Corpus accuracy</small><strong id="corpus-accuracy">${evaluation?(evaluation.accuracy*100).toFixed(1)+'%':'—'}</strong></div></div><canvas class="loss-chart" id="loss-chart" width="420" height="190" aria-label="Training batch loss by optimizer step"></canvas><div class="loss-legend"><span>Batch loss</span><span>Optimizer step →</span></div><p class="tiny muted" id="train-status" style="margin-top:12px">${micro.step===300?'Loaded actual 300-step starter. Reset to train from seed.':micro.step===0?'Random weights · ready to learn.':'Metrics from the current model.'}</p><div id="probe-bars"></div><h3>Biggest movers</h3><div id="movers"><p class="tiny muted">Run a step to measure which weights change most.</p></div>`:`<h3>Input words</h3><div class="token-row" id="micro-tokens"></div><div class="spread"><h3>Real Qwen tokens</h3><span class="tag real">[real]</span></div><div class="token-row real" id="real-tokens"><p class="tiny muted">Preparing the real tokenizer…</p></div><p class="tiny muted">Different tokenizers; different IDs. The simulation uses the micro vocabulary.</p><div class="subtle-divider"></div><div class="eyebrow">NEXT TOKEN DISTRIBUTION</div><div class="chosen-word" id="chosen-word">—</div><div id="prediction-bars"></div><div class="subtle-divider"></div><p class="tiny muted">Weights are fixed during inference. Activations and recurrent memory change as tokens move through the model.</p>`}
  </div><div class="sidebar-footer">Same architecture family. Smaller dimensions.<br>Every displayed calculation uses actual micro values.</div>`;
  if(training){drawLossChart();renderTrainingDetails();}else if(traceResult)renderPrediction(traceResult);
}
function renderPrediction(result:InferenceResult){
  if(!$('micro-tokens'))return;$('micro-tokens').innerHTML=result.tokens.map((word,i)=>`<span class="token-chip">${esc(word)}<small>${result.ids[i]}</small></span>`).join('');
  $('chosen-word').textContent=result.prediction;$('prediction-bars').innerHTML=predictionBars(result.top,10);
  const id=++tokenRequest;tokenizerWorker.postMessage({id,text:result.prompt});
}
tokenizerWorker.onmessage=event=>{const {id,pieces,error}=event.data;if(id!==tokenRequest||!$('real-tokens'))return;$('real-tokens').innerHTML=error?`<p class="tiny">${esc(error)}</p>`:pieces.map((p:{id:number;text:string;token:string})=>`<span class="token-chip" title="${esc(p.token)}">${esc(p.text.replace(/ /g,'␣'))}<small>${p.id}</small></span>`).join('');};
async function runInference(autoPlay:boolean){
  if(busy)return;await microReady;busy=true;playback.pause();const button=$<HTMLButtonElement>('run-inference');if(button){button.disabled=true;button.textContent='Calculating…';}
  try{const prompt=$<HTMLInputElement>('prompt')?.value??'the cat sat on',temperature=Number($<HTMLInputElement>('temperature')?.value??0);if(!Number.isFinite(temperature)||temperature<0||temperature>3)throw new Error('Temperature must be between 0 and 3.');const message=await worker.call('infer',{prompt,temperature,record:true});traceResult=message.result;scene.showModel();renderPrediction(traceResult!);playback.load(traceResult!.steps);if(autoPlay)playback.play();}
  finally{busy=false;if(button){button.disabled=false;button.innerHTML=`Run forward pass ${icon('arrow')}`;}}
}
function updatePlayback(step:RecordedStep|undefined,index:number,total:number,playing:boolean){
  if(!$('scrubber'))return;const scrub=$<HTMLInputElement>('scrubber');scrub.max=String(Math.max(0,total-1));scrub.value=String(index);$('step-label').textContent=step?`${step.phase==='forward'?'→':step.phase==='backward'?'←':'↻'} ${step.name}`:'Prepare a recorded calculation';$('step-number').textContent=total?`${index+1} / ${total}`:'0 / 0';$('play').innerHTML=icon(playing?'pause':'play');$('play').setAttribute('aria-label',playing?'Pause playback':'Play playback');
  if(step&&mode!=='explore'){renderLens(step);scene.setActiveStep(step);$('annotation').classList.remove('hidden');$('annotation').innerHTML=`${step.layer>=0&&step.layer<micro!.config.layers?`Layer ${step.layer} · `:''}${esc(step.name)} <span class="tag micro">[micro]</span>`;}
}
function tensorView(view:TensorView,index:number,kind:'in'|'out'){
  const vals=view.values.slice(0,8);return `<div class="tensor-view"><div class="spread"><span class="name">${esc(view.name)}</span><span class="shape mono">[${view.shape.join(' × ')}]</span></div><div class="value-strip">${vals.map(v=>`<span title="${v}">${numeric(v)}</span>`).join('')}</div>${view.values.length>8?`<button class="ghost tiny" data-expand="${kind}:${index}" style="padding:5px 0">Inspect all ${view.values.length.toLocaleString()} values ${icon('arrow')}</button>`:''}</div>`;
}
function renderLens(step:RecordedStep){
  inspectorTitle('CALCULATION LENS','micro');const body=$('inspector-body');
  const matrix=step.matrix??(step.phase!=='forward'&&step.outputs[0]?{shape:step.outputs[0].shape,values:step.outputs[0].values}:undefined);
  body.innerHTML=`<div class="spread"><span class="phase-badge ${step.phase}">${step.phase==='update'?'weight update':step.phase+' pass'}</span><span class="tiny muted">${step.layer>=0?`Layer ${step.layer}`:'Input stage'}</span></div><h2 class="lens-title">${esc(step.name)}</h2><p>${esc(step.explanation)}</p>${matrix?'<canvas id="lens-matrix" class="lens-matrix" width="560" height="320" aria-label="Actual recorded operation matrix"></canvas><p class="tiny muted" id="matrix-caption"></p>':''}<h3>Inputs</h3>${step.inputs.length?step.inputs.map((v,i)=>tensorView(v,i,'in')).join(''):'<p class="tiny muted">The input is the prompt or training batch.</p>'}<h3>The operation</h3><div class="math">${esc(step.formula)}</div><h3>Check one calculation</h3><div class="math arithmetic">${esc(step.arithmetic)}</div><p class="tiny muted" style="margin-top:6px">Arithmetic rounded to 7 significant digits. Stored reference values use Float64.</p><h3>Output</h3>${step.outputs.map((v,i)=>tensorView(v,i,'out')).join('')}${step.formula==='Y = X Wᵀ'?'<button id="gpu-verify" style="width:100%;margin-top:12px">Recompute this projection on WebGPU</button><p id="gpu-result" class="tiny muted" style="margin-top:8px"></p>':''}`;
  if(matrix){drawMatrix($<HTMLCanvasElement>('lens-matrix'),matrix.values,matrix.shape,matrix.mask);$('matrix-caption').textContent=`${matrix.shape.join(' × ')} · ${step.name.includes('attention')||step.name.includes('Q · K')?'head 0 · future positions are masked':step.name.includes('state')?'DeltaNet head 0 · actual recurrent state':'actual values'} · [micro]`;}
  if(step.phase==='update'&&step.inputs.length>=2){
    const before=step.inputs[0],gradient=step.inputs[1],after=step.outputs[0],width=before.shape.slice(1).reduce((a,b)=>a*b,1),movers=before.values.map((v,i)=>({i,change:Math.abs(after.values[i]-v)})).sort((a,b)=>b.change-a.change).slice(0,5);
    const max=Math.max(1e-9,...before.values.map(Math.abs),...after.values.map(Math.abs));
    const comparison=document.createElement('div');comparison.innerHTML=`<h3>The same weights, before and after</h3><div class="weight-compare"><div><span>Before</span><canvas id="weight-before" width="240" height="160"></canvas></div><div><span>Gradient</span><canvas id="weight-gradient" width="240" height="160"></canvas></div><div><span>After</span><canvas id="weight-after" width="240" height="160"></canvas></div></div><p class="tiny muted">All three use the same symmetric scale: ±${max.toPrecision(3)}.</p><h3>Cells that moved most</h3>${movers.map(m=>`<div class="mover changed-cell mono">[${Math.floor(m.i/width)}, ${m.i%width}] ${numeric(before.values[m.i])} → ${numeric(after.values[m.i])}<span>|Δw| = ${m.change.toExponential(4)}</span></div>`).join('')}`;body.insertBefore(comparison,body.querySelector('h3'));
    drawMatrix($<HTMLCanvasElement>('weight-before'),before.values,before.shape,undefined,max);drawMatrix($<HTMLCanvasElement>('weight-gradient'),gradient.values,gradient.shape,undefined,max);drawMatrix($<HTMLCanvasElement>('weight-after'),after.values,after.shape,undefined,max);
  }
  body.querySelectorAll<HTMLButtonElement>('[data-expand]').forEach(b=>b.onclick=()=>{const [kind,index]=b.dataset.expand!.split(':'),view=(kind==='in'?step.inputs:step.outputs)[Number(index)];showTensorValues(view);});
  if($('gpu-verify'))on('gpu-verify',()=>void verifyGPU(step).catch(fail));
  body.scrollTop=0;
}
function showTensorValues(view:TensorView){
  showModal(view.name,`<p class="tiny muted">[micro] · shape ${view.shape.join(' × ')} · ${view.values.length.toLocaleString()} actual values. Choose a flat index to inspect an exact cell.</p><canvas id="expanded-matrix" class="lens-matrix" width="1000" height="500"></canvas><div class="row"><label class="tiny">Flat index <input id="value-index" type="number" value="0" min="0" max="${view.values.length-1}" style="width:130px;margin-left:8px"></label><strong id="exact-value" class="mono" style="font-size:17px"></strong></div><p class="tiny muted" id="exact-gradient" style="margin-top:12px"></p>`);
  drawMatrix($<HTMLCanvasElement>('expanded-matrix'),view.values,view.shape);const update=()=>{const i=Math.max(0,Math.min(view.values.length-1,Math.floor(Number($<HTMLInputElement>('value-index').value))));$('exact-value').textContent=String(view.values[i]);$('exact-gradient').textContent=view.gradient?`Recorded gradient at this coordinate: ${view.gradient[i]}`:'Gradients are recorded during training.';};$('value-index').addEventListener('input',update);update();
}
async function verifyGPU(step:RecordedStep){
  const [x,w]=step.inputs;if(!x||!w)return;const b=$<HTMLButtonElement>('gpu-verify');b.disabled=true;
  try{const response=await worker.call('gpu',{input:x.values,weight:w.values,rows:x.shape[0],inputDim:x.shape[1],outputDim:w.shape[0],expected:step.outputs[0].values}),r=response.result;if($('gpu-result'))$('gpu-result').textContent=r.available?`Actual WebGPU Float32 result. Max absolute difference from the Float64 reference: ${r.maxAbsoluteError.toExponential(3)}. Dispatch + readback: ${r.milliseconds.toFixed(2)} ms.`:r.reason;}
  finally{if(b.isConnected)b.disabled=false;}
}
function setTrainingBusy(value:boolean){busy=value;for(const id of ['train-one','train-fast','reset-weights','start-one'])if($(id))$<HTMLButtonElement>(id).disabled=value;$('stop-training')?.classList.toggle('hidden',!value);}
function mergeHistory(history:{step:number;loss:number;accuracy:number}[]){const byStep=new Map(lossHistory.map(p=>[p.step,p]));history.forEach(p=>byStep.set(p.step,p));lossHistory=[...byStep.values()].sort((a,b)=>a.step-b.step);}
async function train(steps:number,record:boolean){
  if(busy)return;await microReady;if(!Number.isInteger(steps)||steps<1||steps>10000)throw new Error('Choose between 1 and 10,000 training steps.');
  const learningRate=Number($<HTMLInputElement>('learning-rate').value),batchSize=Number($<HTMLInputElement>('batch-size').value);if(!Number.isFinite(learningRate)||learningRate<=0||learningRate>1||!Number.isInteger(batchSize)||batchSize<1||batchSize>16)throw new Error('Use a learning rate in (0, 1] and a batch size from 1 to 16.');
  playback.pause();setTrainingBusy(true);scene.showModel();
  try{const message=await worker.call('train',{steps,record,learningRate,batchSize},progress=>{if(record)return;micro!.step=progress.result.step;lastTrain=progress.result;mergeHistory(progress.history);renderTrainingDetails();if($('train-status'))$('train-status').textContent=`Training ${progress.completed} / ${progress.total} · ${(progress.milliseconds/1000).toFixed(1)} s`;});
    if(message.result){lastTrain=message.result;micro!.step=lastTrain!.step;mergeHistory(message.history);}evaluation=message.evaluation;renderTrainingDetails();if($('train-status'))$('train-status').textContent=`${message.cancelled?'Stopped':'Finished'} · ${message.history.length} steps in ${(message.milliseconds/1000).toFixed(2)} s · actual worker computation`;
    if(record&&lastTrain){playback.load(lastTrain.steps);playback.play();}else toast(`Training ${message.cancelled?'stopped':'complete'}. Corpus loss ${evaluation!.loss.toFixed(3)}; ${(evaluation!.accuracy*100).toFixed(1)}% accuracy.`);
  }finally{setTrainingBusy(false);}
}
async function resetWeights(){if(busy)return;playback.pause();const message=await worker.call('reset',{seed:42});micro=message;lossHistory=[];lastTrain=undefined;traceResult=undefined;const result=await worker.call('evaluate');evaluation=result.result;renderSimulationSidebar();playback.load([]);inspectorTitle('RESET TO SEED 42','micro');$('inspector-body').innerHTML='<h2>Before learning.</h2><p>These are deterministic random weights. Run 300 fast steps to learn the cat, dog and bird corpus, or animate a single step to inspect how learning works.</p><div class="callout">The same seed, corpus and sequence of training calls produce the same weights and predictions.</div>';scene.setMicro(true,micro!.parameters);scene.showModel();toast('Reset every micro parameter to seed 42. Training step 0.');}
function renderTrainingDetails(){
  if(!$('probe-bars')||!micro)return;$('iteration-label').textContent=`${micro.step} trained steps`;
  if(evaluation){$('corpus-loss').textContent=evaluation.loss.toFixed(3);$('corpus-accuracy').textContent=`${(evaluation.accuracy*100).toFixed(1)}%`;}
  if(lastTrain){$('probe-bars').innerHTML=Object.entries(lastTrain.predictions).map(([prompt,top])=>`<h3>${esc(prompt)} ___</h3>${predictionBars(top,3)}`).join('');$('movers').innerHTML=lastTrain.movers.map(m=>`<div class="mover mono">${esc(m.name)}<span>‖ΔW‖ ${m.changeNorm.toExponential(3)}</span></div>`).join('');}
  drawLossChart();
}
function drawLossChart(){
  const canvas=$<HTMLCanvasElement>('loss-chart');if(!canvas)return;const ctx=canvas.getContext('2d')!,w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);ctx.fillStyle='#8293a1';ctx.font='15px monospace';if(!lossHistory.length){ctx.fillText('No updates yet',20,100);return;}
  const max=Math.max(1,...lossHistory.map(x=>x.loss))*1.05,minStep=lossHistory[0].step,maxStep=Math.max(minStep+1,lossHistory.at(-1)!.step),left=48,right=w-12,top=10,bottom=h-25;
  for(let j=0;j<3;j++){const y=top+(bottom-top)*j/2;ctx.strokeStyle='#283540';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.fillStyle='#667989';ctx.textAlign='left';ctx.fillText((max*(1-j/2)).toFixed(1),2,y+5);}
  ctx.strokeStyle='#67d7bd';ctx.lineWidth=2;ctx.beginPath();lossHistory.forEach((p,i)=>{const x=left+(p.step-minStep)/(maxStep-minStep)*(right-left),y=bottom-p.loss/max*(bottom-top);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();ctx.fillStyle='#8293a1';ctx.textAlign='left';ctx.fillText(String(minStep),left,h-3);ctx.textAlign='right';ctx.fillText(String(maxStep),right,h-3);
}
function closeModal(){$('modal-backdrop')?.remove();}
function showModal(title:string,body:string){closeModal();const backdrop=document.createElement('div');backdrop.id='modal-backdrop';backdrop.className='modal-backdrop';backdrop.innerHTML=`<section class="modal" role="dialog" aria-modal="true" aria-label="${esc(title)}"><header><h2>${esc(title)}</h2><button id="close-modal" class="icon" aria-label="Close dialog">${icon('close')}</button></header><div class="modal-body">${body}</div></section>`;document.body.appendChild(backdrop);on('close-modal',closeModal);backdrop.addEventListener('click',event=>{if(event.target===backdrop)closeModal();});$('close-modal').focus();}
function showGlossary(){showModal('The language behind the model',`<p class="muted" style="font-size:12px;line-height:1.7;margin-bottom:24px">Short definitions for the things you can see and change in this explorer.</p><dl class="glossary-grid">${glossary.map(([term,text])=>`<div><dt>${esc(term)}</dt><dd>${esc(text)}</dd></div>`).join('')}</dl>`);}
function showTokenizer(){
  showModal('Real Qwen tokenizer',`<p class="muted" style="line-height:1.7">Type text in any language. These are IDs from the actual pinned Qwen tokenizer, separate from Micro-Qwen’s 64-token simulation context.</p><textarea id="tokenizer-text" aria-label="Text for real Qwen tokenizer" rows="4" style="width:100%;margin:17px 0;background:#0b131b;color:#e0e9ef;border:1px solid #394d5b;border-radius:6px;padding:12px;font:13px inherit">${esc($<HTMLInputElement>('prompt')?.value??'the cat sat on the mat')}</textarea><button id="tokenize-real" class="primary">Tokenize text [real]</button><div class="token-row real" id="tokenizer-result" style="margin-top:20px"></div><p class="tiny muted" id="tokenizer-summary"></p>`);
  const tokenWorker=new Worker(new URL('./data/tokenizer.worker.ts',import.meta.url),{type:'module'});let id=0;
  tokenWorker.onmessage=e=>{if(!$('tokenizer-result')){tokenWorker.terminate();return;}if(e.data.id!==id)return;const {pieces,error}=e.data;if(error){$('tokenizer-result').textContent=error;return;}$('tokenizer-result').innerHTML=pieces.map((p:{id:number;text:string;token:string})=>`<span class="token-chip" title="${esc(p.token)}">${esc(p.text.replace(/ /g,'␣'))}<small>${p.id}</small></span>`).join('');$('tokenizer-summary').textContent=`${pieces.length} real tokens · NFC Unicode normalization · exact checkpoint vocabulary`;};
  const submit=()=>tokenWorker.postMessage({id:++id,text:$<HTMLTextAreaElement>('tokenizer-text').value});on('tokenize-real',submit);submit();
  const observer=new MutationObserver(()=>{if(!$('tokenizer-result')){tokenWorker.terminate();observer.disconnect();}});observer.observe(document.body,{childList:true});
}
function showSources(){showModal('Real numbers. Clear boundaries.',`<div class="stack"><p><span class="tag real">[real]</span> Weight values come from selected ranges of the pinned Hugging Face safetensors checkpoint. Token IDs come from its actual tokenizer.</p><p><span class="tag config">[config]</span> Dimensions, parameter counts, storage totals and scene geometry come from the config, index and all 18 shard headers.</p><p><span class="tag micro">[micro]</span> Live inference and training use a 278,932-parameter, 8-layer model with explicit gradients. The starter was actually trained for 300 steps. Reset restarts from seed 42.</p><div class="math">Checkpoint: ${manifest.repo}<br>Revision: ${manifest.revision}<br>Verified tensors: ${Object.keys(manifest.tensors).length}<br>Actual source-data download: ${(manifest.budget.downloadedBytes/1e6).toFixed(3)} MB / 150 MB</div><h3>Source files</h3><p><a target="_blank" rel="noreferrer" href="https://huggingface.co/${manifest.repo}/blob/${manifest.revision}/config.json">Model configuration</a> · <a target="_blank" rel="noreferrer" href="https://huggingface.co/${manifest.repo}/blob/${manifest.revision}/model.safetensors.index.json">Weight index</a> · <a target="_blank" rel="noreferrer" href="https://github.com/huggingface/transformers/blob/bd15bc95a89e728bbc1224084eb3b5829428c353/src/transformers/models/qwen3_5/modeling_qwen3_5.py">Reference implementation</a></p><h3>Scope</h3><p>Micro-Qwen keeps the hybrid architecture, grouped-query attention, partial rotary positions, gated DeltaNet recurrence and gated feed-forward blocks. Its dimensions and vocabulary are intentionally small. Vision and multi-token prediction are shown in the real structure but omitted from Micro-Qwen.</p><p>The 3D renderer uses WebGL2. Deterministic model math runs on the CPU in a worker. The calculation lens can independently execute selected projections with WebGPU and report the numerical difference.</p><p>A training trace expands every operation of the first batch sentence, shows each sentence’s loss, and records the averaged gradients and updates for every parameter tensor. Head-0 matrices are shown for clarity; stored operations retain the other head values.</p></div>`);}
async function startTour(){if(busy){toast('Finish the current calculation before starting the tour.');return;}tourIndex=0;await runTourStop();}
function endTour(){tourIndex=-1;$('tour-card')?.remove();tourRunning=false;}
async function runTourStop(){
  if(tourIndex<0||tourRunning)return;tourRunning=true;const stop=tourStops[tourIndex];$('tour-card')?.remove();
  const card=document.createElement('section');card.id='tour-card';card.className='tour-card';card.innerHTML=`<div class="spread"><span class="eyebrow">GUIDED TOUR · ${String(tourIndex+1).padStart(2,'0')} / 12</span><button id="end-tour" class="icon ghost" aria-label="Close tour">${icon('close')}</button></div><h2>${esc(stop.title)}</h2><p>${esc(stop.body)}</p><div class="tour-progress">${tourStops.map((_,i)=>`<i class="${i<=tourIndex?'done':''}"></i>`).join('')}</div><footer><button id="tour-prev" ${tourIndex===0?'disabled':''}>${icon('back')} Previous</button><button id="tour-next" class="primary" disabled>${tourIndex===11?'Finish tour':'Next stop'} ${icon('arrow')}</button></footer>`;document.body.appendChild(card);on('end-tour',endTour);
  try{
    switch(stop.action){
      case 'model':case 'memory':case 'finish':await setMode('explore');scene.showModel();break;
      case 'embedding':await setMode('explore');scene.showTensor(manifest.tensors['model.language_model.embed_tokens.weight']);{const f=manifest.tiles.find(f=>f.name==='model.language_model.embed_tokens.weight'&&f.row===manifest.defaultTokenIds[0]);if(f)await fetchSparse(f);}break;
      case 'delta':await setMode('explore');scene.showLayer(0);break;
      case 'attention':await setMode('explore');scene.showLayer(3);break;
      case 'ffn':await setMode('explore');scene.showTensor(manifest.tensors['model.language_model.layers.0.mlp.gate_proj.weight']);break;
      case 'residual':await setMode('explore');scene.showLayer(0);break;
      case 'inference':await setMode('inference');playback.seek(Math.max(0,playback.steps.findIndex(s=>s.name==='LM head logits')));break;
      case 'sampling':if(mode!=='inference')await setMode('inference');playback.seek(playback.steps.length-1);break;
      case 'training':await setMode('training');await train(1,true);playback.pause();playback.seek(0);break;
      case 'gradient':if(mode!=='training')await setMode('training');{let i=playback.steps.findIndex(s=>s.phase==='backward');if(i<0){await train(1,true);playback.pause();i=playback.steps.findIndex(s=>s.phase==='backward');}playback.seek(Math.max(0,i));}break;
    }
  }catch(error){fail(error);}finally{tourRunning=false;if($('tour-next'))$<HTMLButtonElement>('tour-next').disabled=false;}
  on('tour-prev',()=>{if(tourIndex>0){tourIndex--;void runTourStop();}});on('tour-next',()=>{if(tourIndex>=11)endTour();else{tourIndex++;void runTourStop();}});
}
async function boot(){
  manifest=await loadModelData();shell();renderOverviewSidebar();renderExploreBottom();
  scene=new AnatomyScene($('stage'),manifest,{onSelect:selectObject,onStats:s=>{$('status-right').textContent=`${s.renderer.toUpperCase()} · ${Math.round(s.fps)} FPS · ${s.instances.toLocaleString()} OBJECTS / CELLS`;$('lod-pill').textContent=`LOD ${s.lod} · ${['model','layer','tensor','cells'][s.lod]??'detail'}`;}});renderInspector({kind:'model'});
  microReady=(async()=>{try{micro=await worker.call('init',{checkpointUrl:`${import.meta.env.BASE_URL}data/micro-checkpoint.json`});const benchmark=await fetch(`${import.meta.env.BASE_URL}data/micro-benchmark.json`).then(r=>r.json());lossHistory=benchmark.history??[];evaluation=benchmark.final??benchmark.after;}
    catch(error){console.warn('Starter unavailable; initialized seed42',error);micro=await worker.call('init',{seed:42});evaluation=(await worker.call('evaluate')).result;toast('The trained starter was unavailable. Micro-Qwen is ready with seed 42 random weights.');}})();
  // Deliberately expose only read-only runtime telemetry for reproducible browser QA.
  Object.defineProperty(window,'anatomyStatus',{get:()=>({mode,selection:{...selection},source:currentTile?.source,loadedTile:currentTile?{name:currentTile.name,row:currentTile.row,col:currentTile.col,rows:currentTile.rows,cols:currentTile.cols}:null,step:playback.index,steps:playback.steps.length,microStep:micro?.step,playing:playback.playing,busy})});
}
void boot().catch(error=>{$('app').innerHTML=`<div class="error-screen"><h1>The explorer could not start</h1><p>${esc(error instanceof Error?error.message:String(error))}</p><pre>npm install\nnpm run prefetch\nnpm run dev</pre><p>If your browser disables hardware rendering, enable WebGL2 and reload.</p></div>`;console.error(error);});
