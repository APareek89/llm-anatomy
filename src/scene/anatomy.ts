import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { ModelManifest, TensorMeta, WeightTile } from '../data/hf';
import { axesMeaning, cellMeaning } from '../ui/content';

export interface Selection { kind:'model'|'layer'|'tensor'|'cell'; layer?:number; tensor?:string; row?:number; col?:number }
interface Callbacks { onSelect:(selection:Selection)=>void; onStats?:(stats:{fps:number;renderer:string;lod:number;instances:number})=>void; onTileRequest?:(name:string,row:number,col:number)=>Promise<void> }
interface StepLike { name?:string; layer?:number; phase?:string; parameter?:string; matrix?:{shape:number[];values:number[];mask?:boolean[]}; inputs?:{name:string;shape:number[];values:number[]}[]; outputs?:{name:string;shape:number[];values:number[]}[] }
export type MicroFamily='qwen'|'gpt-oss'|'llama'|'gemma';
export interface MicroSceneConfig { layers:number;hidden:number;queryHeads:number;kvHeads:number;headDim:number;localWindow?:number;experts?:number;expertsPerToken?:number;deltaKeyHeads?:number;deltaValueHeads?:number;deltaKeyDim?:number;deltaValueDim?:number; }
type HitMesh = THREE.Mesh | THREE.InstancedMesh;
const C={delta:0x54dccb,attention:0xa493ff,ffn:0x898174,norm:0x7e929d,embedding:0x739dbc,vision:0x577e72,mtp:0xa58ac5,head:0xb4a487};
const v=(x:number,y:number,z:number)=>new THREE.Vector3(x,y,z);
const short=(name:string)=>name.replace(/^model\.language_model\./,'').replace(/^model\.visual\./,'vision.');
const leaf=(name:string)=>name.split('.').slice(-2).join('.');
const count=(n:number)=>n>=1e9?`${(n/1e9).toFixed(2)}B`:n>=1e6?`${(n/1e6).toFixed(1)}M`:n>=1e3?`${(n/1e3).toFixed(1)}K`:`${n}`;
function colorFor(name:string){return name.includes('linear_attn')?C.delta:name.includes('self_attn')?C.attention:name.includes('.mlp.')?C.ffn:name.includes('embed')?C.embedding:name.startsWith('mtp')?C.mtp:name.includes('visual')?C.vision:name.includes('lm_head')?C.head:C.norm;}

/** Real checkpoint geometry and explicitly separate micro-model simulation geometry. */
export class AnatomyScene {
  readonly renderer:THREE.WebGLRenderer;
  readonly camera=new THREE.PerspectiveCamera(38,1,.0001,250);
  readonly scene=new THREE.Scene();
  readonly controls:OrbitControls;
  private content=new THREE.Group();
  private decor=new THREE.Group();
  private overlay=new THREE.Group();
  private interactives:HitMesh[]=[];
  private raycaster=new THREE.Raycaster();
  private pointer=new THREE.Vector2();
  private selection:Selection={kind:'model'};
  private micro=false;
  private microFamily:MicroFamily='qwen';
  private microConfig:MicroSceneConfig={layers:8,hidden:64,queryHeads:6,kvHeads:1,headDim:16,localWindow:4,experts:4,expertsPerToken:2};
  private familyDirty=false;
  private microParameters:{name:string;shape:number[]}[]=[];
  private microLayers=8;
  private frame=0;
  private lastFrame=performance.now();
  private measuredAt=performance.now();
  private measuredFrames=0;
  private slowFor=0;
  private quality=1;
  private lod=0;
  private tween?:{start:number;duration:number;from:THREE.Vector3;to:THREE.Vector3;targetFrom:THREE.Vector3;targetTo:THREE.Vector3};
  private dragStart={x:0,y:0};
  private hovered?:HitMesh;
  private layerCenters=new Map<number,THREE.Vector3>();
  private expertMeshes=new Map<string,THREE.Mesh[]>();
  private lastRouting?:{layer:number;weights:number[]};
  private activeMarker=new THREE.Group();
  private followStep?:StepLike;
  private autoFollow=true;
  private playing=false;
  private playbackSpeed=.5;
  private flowGroup=new THREE.Group();
  private flowStart=0;
  private flowFrom=new THREE.Vector3();
  private flowTo=new THREE.Vector3();
  private observer:ResizeObserver;
  private history:Selection[]=[];
  private disposed=false;
  private tensorMeta?:TensorMeta;
  private tensorTile?:WeightTile;
  private tilePlane?:THREE.Mesh;
  private cellMesh?:THREE.InstancedMesh;
  private cellLabels=new THREE.Group();
  private tileDimensions={width:6.6,height:6.6};
  private tileOrigin=new THREE.Vector3(0,3.8,0);
  private tileGroup=new THREE.Group();
  private previousTileGroup=new THREE.Group();
  private tileSizeOverride?:{width:number;height:number};
  private streamArmed=false;
  private streamPending?:{key:string;name:string;row:number;col:number};
  private streamFailed=new Set<string>();
  private streamRequestedAt=0;
  private loadingGroup=new THREE.Group();
  private highlightGroup=new THREE.Group();
  private highlightKey='';
  private hoveredCell?:string;
  private hoverElement:HTMLDivElement;
  private cellRangeKey='';
  private numericPlane?:THREE.Mesh;
  private updateCellsAt=0;
  private lastLodChange=0;
  private pointerMove=(event:PointerEvent)=>this.onPointerMove(event);
  private pointerLeave=()=>{this.hoverElement.style.display='none';this.highlightGroup.visible=false;};
  private pointerDown=(event:PointerEvent)=>{this.dragStart={x:event.clientX,y:event.clientY};};
  private pointerUp=(event:PointerEvent)=>{if(Math.hypot(event.clientX-this.dragStart.x,event.clientY-this.dragStart.y)<5)this.pick(event,false);};
  private doubleClick=(event:MouseEvent)=>this.pick(event,true);
  private keyDown=(event:KeyboardEvent)=>{if(event.key==='Escape'&&!['INPUT','TEXTAREA','SELECT'].includes((event.target as HTMLElement)?.tagName))this.back();};

  constructor(private container:HTMLElement,private manifest:ModelManifest,private callbacks:Callbacks){
    this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,powerPreference:'high-performance'});
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio,1.75));
    this.renderer.setClearColor(0x070d13,0);
    this.renderer.outputColorSpace=THREE.SRGBColorSpace;
    this.renderer.toneMapping=THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure=.95;
    this.renderer.domElement.setAttribute('aria-label','Interactive three-dimensional language-model architecture. Drag to orbit, scroll to zoom, click a layer, double-click a tensor, Escape to return.');
    this.renderer.domElement.style.cssText='display:block;width:100%;height:100%;touch-action:none;outline:none';
    container.appendChild(this.renderer.domElement);
    this.scene.fog=new THREE.FogExp2(0x070d13,.014);
    this.scene.add(new THREE.HemisphereLight(0xc8e8ef,0x12202f,.65));
    const key=new THREE.DirectionalLight(0xd5f0ff,1.4);key.position.set(6,15,8);this.scene.add(key);
    const rim=new THREE.DirectionalLight(0x758bc8,1.05);rim.position.set(-8,6,-9);this.scene.add(rim);
    const teal=new THREE.PointLight(C.delta,16,22,2);teal.position.set(-5,4,5);this.scene.add(teal);
    this.scene.add(this.decor,this.content,this.overlay,this.activeMarker,this.previousTileGroup,this.tileGroup,this.cellLabels,this.loadingGroup,this.highlightGroup,this.flowGroup);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);
    this.controls.enableDamping=true;this.controls.dampingFactor=.075;this.controls.minDistance=.015;this.controls.maxDistance=65;
    this.controls.maxPolarAngle=Math.PI*.92;this.controls.zoomSpeed=.8;this.controls.panSpeed=.8;this.controls.zoomToCursor=true;
    this.controls.addEventListener('start',()=>{this.tween=undefined;this.streamArmed=true;});
    this.createGround();
    this.camera.position.set(16,12.3,19);this.controls.target.set(.4,5.2,0);
    const canvas=this.renderer.domElement;
    canvas.addEventListener('pointermove',this.pointerMove);canvas.addEventListener('pointerleave',this.pointerLeave);canvas.addEventListener('pointerdown',this.pointerDown);canvas.addEventListener('pointerup',this.pointerUp);canvas.addEventListener('dblclick',this.doubleClick);
    window.addEventListener('keydown',this.keyDown);
    this.hoverElement=document.createElement('div');this.hoverElement.style.cssText='position:absolute;pointer-events:none;display:none;z-index:8;max-width:330px;padding:10px 13px;border:1px solid #40636e;border-radius:9px;background:rgba(9,21,28,.96);color:#c7e0e4;font:12px/1.55 ui-monospace,monospace;box-shadow:0 8px 30px #0008;white-space:pre-line';container.appendChild(this.hoverElement);
    this.observer=new ResizeObserver(()=>this.resize());this.observer.observe(container);this.resize();
    this.showModel();this.animate();
  }

  resize(){const w=Math.max(1,this.container.clientWidth),h=Math.max(1,this.container.clientHeight);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();this.renderer.setSize(w,h,false);}
  private box(width:number,height:number,depth:number,color:number,position:THREE.Vector3,selection?:Selection,opacity=1){
    const geometry=new THREE.BoxGeometry(width,height,depth);
    const material=new THREE.MeshStandardMaterial({color,metalness:.35,roughness:.43,transparent:opacity<1,opacity,emissive:color,emissiveIntensity:.045});
    const mesh=new THREE.Mesh(geometry,material);mesh.position.copy(position);
    const edges=new THREE.LineSegments(new THREE.EdgesGeometry(geometry),new THREE.LineBasicMaterial({color,transparent:true,opacity:.55}));mesh.add(edges);
    if(selection){mesh.userData.selection=selection;this.interactives.push(mesh);}
    this.content.add(mesh);return mesh;
  }
  private label(text:string,position:THREE.Vector3,options:{color?:string;scale?:number;width?:number;opacity?:number;parent?:THREE.Group}={}){
    const canvas=document.createElement('canvas');canvas.height=96;let ctx=canvas.getContext('2d')!;ctx.font='500 44px "SF Pro Display", Inter, -apple-system, sans-serif';canvas.width=Math.max(96,Math.min(2048,Math.ceil(ctx.measureText(text).width)+28));ctx=canvas.getContext('2d')!;ctx.font='500 44px "SF Pro Display", Inter, -apple-system, sans-serif';ctx.textBaseline='middle';ctx.textAlign='center';ctx.fillStyle=options.color??'#b2c5cf';ctx.fillText(text,canvas.width/2,48,canvas.width-24);
    const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;
    const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,transparent:true,opacity:options.opacity??.95,depthTest:false}));
    const s=options.scale??1,w=(options.width??3.2)*s;sprite.scale.set(w,Math.min(.62*s,w*96/canvas.width),1);sprite.position.copy(position);sprite.renderOrder=5;(options.parent??this.content).add(sprite);return sprite;
  }
  private line(points:THREE.Vector3[],color:number,opacity=.5,parent:THREE.Group=this.content){
    const geometry=new THREE.BufferGeometry().setFromPoints(points),line=new THREE.Line(geometry,new THREE.LineBasicMaterial({color,transparent:true,opacity}));parent.add(line);return line;
  }
  private arrow(a:THREE.Vector3,b:THREE.Vector3,color:number){const direction=b.clone().sub(a),length=direction.length();const arrow=new THREE.ArrowHelper(direction.normalize(),a,length,color,.12,.055);this.content.add(arrow);return arrow;}
  private createGround(){
    const plate=new THREE.Mesh(new THREE.CylinderGeometry(6.35,6.6,.14,96),new THREE.MeshStandardMaterial({color:0x0f1b23,metalness:.5,roughness:.58}));plate.position.set(.4,.05,0);this.decor.add(plate);
    for(const radius of [2.2,4.2,6.3,8.7,12]){const pts=Array.from({length:129},(_,i)=>v(.4+Math.cos(i/128*Math.PI*2)*radius,.14,Math.sin(i/128*Math.PI*2)*radius));this.line(pts,0x38515b,radius>6.5?.15:.4,this.decor);}
    for(let a=0;a<Math.PI*2;a+=Math.PI/12)this.line([v(.4+Math.cos(a)*1,.14,Math.sin(a)*1),v(.4+Math.cos(a)*12,.14,Math.sin(a)*12)],0x314853,.2,this.decor);
    this.label('PARAMETER SPACE',v(.6,.23,6.0),{color:'#496371',width:3,parent:this.decor,scale:.9});
  }
  private clear(group:THREE.Group){
    while(group.children.length){const object=group.children[0];group.remove(object);object.traverse(child=>{const item=child as THREE.Mesh;item.geometry?.dispose();const materials=item.material?(Array.isArray(item.material)?item.material:[item.material]):[];materials.forEach(material=>{(material as THREE.MeshBasicMaterial).map?.dispose();material.dispose();});});}
  }
  private resetContent(){this.clear(this.content);this.clear(this.overlay);this.clear(this.tileGroup);this.clear(this.previousTileGroup);this.clear(this.cellLabels);this.clear(this.loadingGroup);this.clear(this.highlightGroup);this.clear(this.flowGroup);this.highlightKey='';this.interactives=[];this.hovered=undefined;this.hoveredCell=undefined;this.hoverElement.style.display='none';this.cellMesh=undefined;this.tilePlane=undefined;this.numericPlane=undefined;this.cellRangeKey='';this.layerCenters.clear();this.expertMeshes.clear();this.lastRouting=undefined;this.tensorMeta=undefined;this.tensorTile=undefined;this.tileSizeOverride=undefined;this.streamPending=undefined;this.streamFailed.clear();this.streamArmed=false;this.activeMarker.visible=false;this.decor.visible=true;}
  private enter(s:Selection,push=true){if(push&&JSON.stringify(s)!==JSON.stringify(this.selection))this.history.push({...this.selection});this.selection={...s};this.callbacks.onSelect({...s});}
  private fly(position:THREE.Vector3,target:THREE.Vector3,duration=950){this.streamArmed=false;this.tween={start:performance.now(),duration,from:this.camera.position.clone(),to:position,targetFrom:this.controls.target.clone(),targetTo:target};}
  private layerData(){
    if(!this.micro)return this.manifest.layers;
    return Array.from({length:this.microLayers},(_,index)=>{const params=this.microParameters.filter(p=>p.name.startsWith(`layers.${index}.`));return {index,kind:this.microLayerKind(index),names:params.map(p=>p.name),params:params.reduce((n,p)=>n+p.shape.reduce((a,b)=>a*b,1),0)||1,bytes:0};});
  }
  private microLayerKind(index:number){return this.microFamily==='qwen'?(index%4===3?'full_attention':'linear_attention'):this.microFamily==='gpt-oss'?(index%2===0?'sliding_attention':'full_attention'):this.microFamily==='gemma'?(index%6===5?'full_attention':'sliding_attention'):'full_attention';}
  private familyName(){return {qwen:'Micro-Qwen','gpt-oss':'Micro-GPT-OSS',llama:'Micro-Llama',gemma:'Micro-Gemma'}[this.microFamily];}
  private headConfig(meta?:TensorMeta){const heads=this.micro?this.microConfig.queryHeads:this.manifest.config.text_config.num_attention_heads,kvHeads=this.micro?this.microConfig.kvHeads:this.manifest.config.text_config.num_key_value_heads,dim=this.micro?this.microConfig.headDim:this.manifest.config.text_config.head_dim;return {heads,kvHeads,dim,gated:meta?meta.shape[0]===heads*dim*2:!this.micro||this.microFamily==='qwen'};}
  private meta(name:string):TensorMeta|undefined{
    if(!this.micro)return this.manifest.tensors[name];
    const p=this.microParameters.find(x=>x.name===name);if(!p)return;const params=p.shape.reduce((a,b)=>a*b,1),layer=/layers\.(\d+)/.exec(name);
    return {name,shape:p.shape,dtype:'Float64 [micro]',shard:'micro-worker',dataOffsets:[0,0],absoluteOffsets:[0,0],params,bytes:params*8,component:name.includes('linear_attn')?'deltaNet':name.includes('self_attn')?'attention':name.includes('mlp')?'ffn':'norms',layer:layer?Number(layer[1]):undefined,...(name.includes('embed_tokens')?{tiedEmbedding:this.microFamily==='llama'||this.microFamily==='gemma',embeddingScale:this.microFamily==='gemma'?Math.sqrt(this.microConfig.hidden):1}:{})};
  }

  showModel(){
    if(this.micro&&this.microFamily!=='qwen'){this.showFamilyModel();return;}
    this.resetContent();this.lod=0;this.enter({kind:'model'});const layers=this.layerData(),total=layers.reduce((n,l)=>n+l.params,0),gap=this.micro?.18:.062,totalBody=this.micro?6.8:6.1,base=.8;
    let y=base;const x=-1.1,width=5.4,depth=3.05;
    this.box(width+.35,.27,depth+.35,C.embedding,v(x,.48,0),{kind:'tensor',tensor:this.micro?'embed_tokens.weight':'model.language_model.embed_tokens.weight'});
    this.label(this.micro?'WORD EMBEDDINGS [micro]':'TOKEN EMBEDDINGS',v(x,.37,2.5),{color:'#8fbfdf',width:3.4,scale:.95});
    for(const layer of layers){
      const h=layer.params/total*totalBody,color=layer.kind==='linear_attention'?C.delta:C.attention;
      const ffnParams=layer.names.filter(n=>n.includes('.mlp.')).reduce((n,name)=>n+(this.meta(name)?.params??0),0),ffnWidth=Math.max(.3,width*(ffnParams/layer.params)),mixerWidth=Math.max(.3,width-ffnWidth-.06);
      const center=y+h/2;this.layerCenters.set(layer.index,v(x,center,0));
      this.box(mixerWidth,h,depth,color,v(x-width/2+mixerWidth/2,center,0),{kind:'layer',layer:layer.index});
      this.box(ffnWidth,h,depth,C.ffn,v(x+width/2-ffnWidth/2,center,0),{kind:'layer',layer:layer.index},.88);
      const front=new THREE.Mesh(new THREE.BoxGeometry(mixerWidth,.008,.013),new THREE.MeshBasicMaterial({color}));front.position.set(x-width/2+mixerWidth/2,center+h/2,depth/2+.012);this.content.add(front);
      if(layer.index%4===3||this.micro)this.label(`${String(layer.index).padStart(2,'0')}`,v(x-width/2-.3,center,depth/2+.03),{color:layer.kind==='full_attention'?'#b6a8fa':'#75a3a0',width:.55,scale:.8});
      y+=h+gap;
      if(layer.index%4===3&&layer.index<layers.length-1){this.line([v(x-width/2-.22,y-.02,-depth/2),v(x-width/2-.22,y-.02,depth/2+.14),v(x-width/2+.2,y-.02,depth/2+.14)],0x79939b,.35);}
    }
    this.box(width,.09,depth,C.norm,v(x,y+.05,0),{kind:'tensor',tensor:this.micro?'norm.weight':'model.language_model.norm.weight'});
    this.box(width+.35,.32,depth+.35,C.head,v(x,y+.36,0),{kind:'tensor',tensor:'lm_head.weight'});
    this.label(this.micro?'LM HEAD [micro]':'LANGUAGE MODEL HEAD',v(x,y+.94,.25),{color:'#e0cba8',width:4.1});
    if(!this.micro){
      this.box(2.75,.2,1.5,C.mtp,v(x,y+1.24,-.25),{kind:'tensor',tensor:'mtp.fc.weight'},.65);this.label('MULTI-TOKEN PREDICTION',v(x,y+1.6,-.25),{color:'#ac99c8',width:3.3,scale:.85});
      const vx=4.22,vz=-1.23,vision=this.manifest.config.vision_config.depth;
      for(let i=0;i<vision;i++)this.box(1.7,.075,1.55,C.vision,v(vx,.72+i*.145,vz),{kind:'tensor',tensor:Object.keys(this.manifest.tensors).find(name=>name.includes(`visual.blocks.${i}.`)&&name.endsWith('qkv.weight'))??Object.keys(this.manifest.tensors).find(name=>name.includes(`visual.blocks.${i}.`))},.72);
      this.box(2.05,.2,1.9,C.vision,v(vx,.43,vz));this.label('VISION ENCODER',v(vx,5.05,vz),{color:'#83aa9d',width:2.7});this.label(`${vision} layers · ${count(this.manifest.components.vision?.params??0)} parameters`,v(vx,4.76,vz),{color:'#62877e',width:3.1,scale:.8});
      this.line([v(vx,.32,vz),v(vx,.32,1.95),v(x,.32,1.95),v(x,.45,1.6)],C.vision,.8);
    }
    this.label(this.micro?'MICRO-QWEN · LIVE SIMULATION':'QWEN 3.8 · 27B',v(x,-.14,3.5),{color:'#d6e5e7',width:4.0,scale:1.05});
    this.label(`${layers.length} layers · ${this.micro?'[micro]':'[config]'} · click to explore`,v(x,-.48,3.5),{color:'#607c89',width:4.2,scale:.85});
    const target=v(.1,this.micro?4.35:5.1,0);this.fly(v(this.micro?18.5:21.8,this.micro?13:15.3,this.micro?22.5:26),target);
  }

  private showFamilyModel(){
    this.resetContent();this.lod=0;this.enter({kind:'model'});
    const layers=this.layerData(),family=this.microFamily,c=this.microConfig,total=layers.reduce((n,l)=>n+l.params,0),accent=family==='gpt-oss'?0x79cdb3:family==='llama'?0x94b9ed:0xe4b784,base=.85,step=7.0/layers.length;
    const embedding=this.meta('embed_tokens.weight'),tied=!this.meta('lm_head.weight');
    this.box(5.8,.24,3.05,C.embedding,v(-.5,.48,0),{kind:'tensor',tensor:'embed_tokens.weight'});
    this.label('TOKEN EMBEDDINGS [micro]',v(-.7,.2,2.2),{color:'#9bbcd7',width:3.2,scale:.85});
    for(const layer of layers){
      const y=base+step*(layer.index+.5),h=Math.max(.28,layer.params/total*4.2),local=layer.kind==='sliding_attention',color=local?(family==='gemma'?0xc29263:0x589f94):(family==='llama'?0x829dc8:0xae9ee9),selection:Selection={kind:'layer',layer:layer.index};this.layerCenters.set(layer.index,v(-.8,y,0));
      this.label(`${String(layer.index).padStart(2,'0')}`,v(-4.05,y,1.55),{color:local?'#c3b393':'#b7b7d3',width:.5,scale:.85});
      if(family==='gpt-oss'){
        const attention=this.box(1.8,h*.72,2.65,color,v(-2.55,y,0),selection,.86);
        this.headBands(attention,1.8,h*.72,c.queryHeads,false);
        this.label(local?`LOCAL · ${c.localWindow} TOKENS`:'GLOBAL ATTENTION',v(-2.55,y+h*.52,1.5),{color:local?'#7ab7aa':'#b7a2dd',width:2.3,scale:.63});
        const router=this.box(.5,h*.63,.5,0xe4bd7b,v(-.72,y,0),{kind:'tensor',layer:layer.index,tensor:`layers.${layer.index}.mlp.router.weight`});router.rotation.z=Math.PI/4;
        this.line([v(-1.64,y,0),v(-.72,y,0)],accent,.7);this.label('ROUTER',v(-.72,y+h*.62,.1),{color:'#d2b582',width:1.3,scale:.6});
        const experts=c.experts??4;
        for(let e=0;e<experts;e++){
          const x=.65+(e%2)*1.42,z=(Math.floor(e/2)-(Math.ceil(experts/2)-1)/2)*1.55,mesh=this.box(1.13,h*.78,1.15,0x72b8a5,v(x,y,z),selection,.7);mesh.userData.expert=e;this.expertMeshes.set(`${layer.index}:${e}`,[mesh]);
          this.line([v(-.47,y,0),v(x-.56,y,z)],0x84baaa,.35);this.label(`E${e}`,v(x,y+h*.48,z+.58),{color:'#a1d2c3',width:.65,scale:.68});
        }
      }else if(family==='llama'){
        // Dense attention and feed-forward branches remain separate, joined at every residual level.
        this.box(2.2,h*.8,2.55,color,v(-2.15,y,0),selection,.88);
        const qLeft=-3.03,qSpan=1.75;
        for(let head=0;head<c.queryHeads;head++){const hx=qLeft+(head+.5)*qSpan/c.queryHeads;this.box(qSpan/c.queryHeads*.54,h*.56,.12,0xb1c7ef,v(hx,y,1.36),selection,.82);this.line([v(hx,y-h*.4,1.38),v(-2.15+(Math.floor(head/(c.queryHeads/c.kvHeads))-(c.kvHeads-1)/2)*.3,y-h*.67,1.38)],0x9cafcc,.5);}
        const matrices=['gate_proj','up_proj','down_proj'];
        for(let m=0;m<matrices.length;m++){const meta=this.meta(`layers.${layer.index}.mlp.${matrices[m]}.weight`),scale=.03125;this.box((meta?.shape[1]??c.hidden)*scale,h*.6,.55,C.ffn,v(.35+m*.2,y,-.78+m*.76),selection,.85);}
        this.line([v(-.98,y,0),v(1.75,y,0)],accent,.55);this.label('DENSE SwiGLU',v(.75,y+h*.5,1.55),{color:'#c4bea7',width:2.3,scale:.7});
        this.label(`${c.queryHeads} Q → ${c.kvHeads} KV`,v(-2.15,y+h*.5,1.5),{color:'#bac9e2',width:1.95,scale:.7});
      }else{
        const width=local?4.8:6.4,depth=local?2.65:3.35,mixerParams=layer.names.filter(n=>n.includes('self_attn')).reduce((n,name)=>n+(this.meta(name)?.params??0),0),attentionWidth=width*mixerParams/layer.params;
        this.box(attentionWidth,h*.73,depth,color,v(-.9-width/2+attentionWidth/2,y,0),selection,.9);this.box(width-attentionWidth-.07,h*.73,depth,C.ffn,v(-.9+attentionWidth/2+.035,y,0),selection,.86);
        const norms=layer.names.filter(n=>n.includes('layernorm.weight'));
        norms.forEach((name,i)=>this.box(.09,h*1.08,.1,0xd9c199,v(-.9-width/2+(i+.5)*width/norms.length,y,depth/2+.11),{kind:'tensor',layer:layer.index,tensor:name},.9));
        this.label(local?`LOCAL WINDOW · ${c.localWindow} TOKENS`:'GLOBAL ATTENTION · WHOLE CONTEXT',v(-.9,y+h*.62,depth/2+.02),{color:local?'#c7ae85':'#c8b8ee',width:local?3.5:5.4,scale:.72});
        if(!local){const points=Array.from({length:65},(_,i)=>v(-.9+Math.cos(i/64*Math.PI*2)*3.55,y,Math.sin(i/64*Math.PI*2)*1.95));this.line(points,0xcbb8f1,.65);}
      }
      if(layer.index<layers.length-1)this.line([v(-3.75,y,0),v(-3.75,y+step,0)],accent,.43);
    }
    const top=base+step*layers.length+.2;this.box(5.65,.1,2.8,C.norm,v(-.6,top,0),{kind:'tensor',tensor:'norm.weight'});this.box(5.8,.27,3.05,C.head,v(-.6,top+.32,0),{kind:'tensor',tensor:tied?'embed_tokens.weight':'lm_head.weight'});
    this.label(tied?'SHARED EMBEDDING → LANGUAGE HEAD':'LANGUAGE MODEL HEAD [micro]',v(-.6,top+.8,.2),{color:'#d5c8ae',width:5.1,scale:.95});
    const subtitle=family==='gpt-oss'?`${c.experts} experts · top ${c.expertsPerToken} routed per token`:family==='llama'?`Dense GQA · ${c.queryHeads} query heads / ${c.kvHeads} KV`:'Five local layers → one global layer';
    this.label(`${this.familyName().toUpperCase()} · ${layers.length} LAYERS`,v(-.7,-.13,3),{color:'#d2dfdf',width:5.2});this.label(`${subtitle} · [micro]`,v(-.7,-.45,3),{color:'#6f8f99',width:5.3,scale:.86});
    if(embedding)this.label(`${count(this.microParameters.reduce((n,p)=>n+p.shape.reduce((a,b)=>a*b,1),0))} learned parameters · actual worker structure`,v(-.7,-.77,3),{color:'#4b6875',width:5.6,scale:.76});
    this.fly(v(18.5,13,22.5),v(.1,4.35,0));
  }

  showLayer(index:number){
    const layer=this.layerData().find(x=>x.index===index);if(!layer)return;
    if(this.micro&&this.microFamily==='gpt-oss'){this.showExpertLayer(index);return;}
    this.resetContent();this.lod=1;this.enter({kind:'layer',layer:index});this.decor.visible=false;
    const full=layer.kind!=='linear_attention',local=layer.kind==='sliding_attention',head=this.headConfig(),attentionTitle=local?'LOCAL ATTENTION':head.gated?'GATED ATTENTION':'GLOBAL ATTENTION',mixerColor=local?0xc49a6a:full?C.attention:C.delta,groups=[{key:'norm',names:layer.names.filter(n=>n.includes('layernorm')),x:-5.05,title:this.micro&&this.microFamily==='gemma'?'PRE / POST NORMS':'NORMALIZE'},{key:'mixer',names:layer.names.filter(n=>n.includes(full?'self_attn':'linear_attn')),x:-.8,title:full?attentionTitle:'GATED DELTANET'},{key:'ffn',names:layer.names.filter(n=>n.includes('.mlp.')),x:4.9,title:'FEED-FORWARD'}];
    const maximumDimension=Math.max(1,...layer.names.flatMap(name=>{const meta=this.meta(name);return meta&&meta.shape.length>=2?[meta.shape[0],meta.shape.slice(1).reduce((a,b)=>a*b,1)]:[];})),matrixScale=2.7/maximumDimension;
    this.label(`LAYER ${String(index).padStart(2,'0')} · ${full?attentionTitle:'GATED DELTANET'}`,v(0,8.25,0),{color:local?'#dfbf8f':full?'#bfafff':'#8ae8d9',width:8,scale:1.2});
    this.label(`${count(layer.params)} parameters · ${this.micro?'[micro]':'[config]'} · every block is a named tensor`,v(0,7.7,0),{color:'#73929f',width:8,scale:.9});
    for(const group of groups){
      this.label(group.title,v(group.x,6.9,0),{color:group.key==='ffn'?'#c9b78f':group.key==='mixer'?(full?'#b5a2f0':'#68cdbc'):'#8ca5b1',width:3.2,scale:.9});
      const cols=group.key==='norm'?(group.names.length>2?2:1):3;
      for(let i=0;i<group.names.length;i++){
        const meta=this.meta(group.names[i]);if(!meta)continue;
        const col=i%cols,row=Math.floor(i/cols),px=group.x+(col-(cols-1)/2)*(group.key==='ffn'?2.1:group.key==='norm'?1.4:1.62),py=(group.key==='ffn'?4.55:5.8)-row*1.38,inDim=meta.shape.slice(1).reduce((a,b)=>a*b,1),outDim=meta.shape[0];
        const vector=meta.shape.length===1,w=vector?.12:inDim*matrixScale,h=vector?Math.max(.08,outDim*matrixScale):outDim*matrixScale,d=vector?.1:.24;
        const mesh=this.box(w,h,d,colorFor(meta.name),v(px,py,.12),{kind:'tensor',layer:index,tensor:meta.name});mesh.userData.tensor=meta.name;
        this.label(group.key==='norm'&&cols>1?meta.name.split('.').at(-2)!.replace('_layernorm',''):leaf(meta.name),v(px,py-h/2-.23,.38),{color:'#a5bbc4',width:group.key==='ffn'?2.8:group.key==='norm'&&cols>1?1.9:2.2,scale:.7});
        this.label(meta.shape.join(' × '),v(px,py-h/2-.48,.4),{color:'#586f7a',width:2.2,scale:.65});
        if(full&&meta.name.includes('q_proj')&&meta.shape.length>=2)this.headBands(mesh,w,h,head.heads,this.headConfig(meta).gated);
        else if(full&&/[kv]_proj/.test(meta.name)&&meta.shape.length>=2)this.headBands(mesh,w,h,head.kvHeads,false);
      }
      this.line([v(group.x-1.6,.76,-.1),v(group.x+1.6,.76,-.1)],group.key==='mixer'?mixerColor:C.ffn,.25);
    }
    this.arrow(v(-4.25,3.0,.55),v(-3.15,3.0,.55),C.norm);this.arrow(v(1.5,2.35,.55),v(3.15,2.35,.55),mixerColor);
    this.line([v(-5.45,1.2,.3),v(-5.45,.38,.3),v(6.5,.38,.3),v(6.5,1.2,.3)],0x7897a4,.6);this.label('RESIDUAL STREAM  →  preserve input + add learned changes',v(.45,.03,.4),{color:'#76929f',width:8,scale:.86});
    this.label(full?`${head.heads/head.kvHeads} query heads share each key / value head${local?` · window ${this.microConfig.localWindow}`:''}`:'State memory is an activation, not a learned weight',v(-.9,1.15,.3),{color:full?'#8f80c0':'#5b9f94',width:5.5,scale:.85});
    if(full){
      const qHeads=head.heads,kvHeads=head.kvHeads,span=3.05,left=-6.15;
      for(let h=0;h<qHeads;h++){const x=left+(h+.5)*span/qHeads,kv=Math.floor(h/(qHeads/kvHeads)),kx=left+(kv+.5)*span/kvHeads;this.line([v(x,2.2,.4),v(kx,1.48,.4)],C.attention,.34);const node=new THREE.Mesh(new THREE.SphereGeometry(.034,8,6),new THREE.MeshBasicMaterial({color:0xcdbdff}));node.position.set(x,2.2,.4);this.content.add(node);}
      for(let h=0;h<kvHeads;h++){const x=left+(h+.5)*span/kvHeads,node=new THREE.Mesh(new THREE.SphereGeometry(.073,10,8),new THREE.MeshBasicMaterial({color:C.attention}));node.position.set(x,1.48,.4);this.content.add(node);}
      this.label(`${qHeads} Q HEADS → ${kvHeads} SHARED KV HEADS`,v(-4.63,1.0,.4),{color:'#8679af',width:3.8,scale:.75});
    }else{
      const p=`layers.${index}.linear_attn`,stateHeads=this.micro?(this.meta(`${p}.A_log`)?.shape[0]??3):this.manifest.config.text_config.linear_num_value_heads,stateDim=this.micro?(this.meta(`${p}.norm.weight`)?.shape[0]??8):this.manifest.config.text_config.linear_value_head_dim;
      this.box(2.45,.84,.15,C.delta,v(-4.85,1.9,.25),undefined,.17);this.label('RECURRENT STATE · ACTIVATION',v(-4.85,2.55,.3),{color:'#75b1a6',width:3.55,scale:.75});this.label(`${stateHeads} heads × ${stateDim} × ${stateDim}`,v(-4.85,1.18,.3),{color:'#548b80',width:3.1,scale:.72});
    }
    this.fly(v(2.0,10.5,27.5),v(.75,4.05,0));
  }
  private showExpertLayer(index:number){
    const layer=this.layerData().find(x=>x.index===index);if(!layer)return;this.resetContent();this.lod=1;this.enter({kind:'layer',layer:index});this.decor.visible=false;
    const c=this.microConfig,local=layer.kind==='sliding_attention',maximum=Math.max(1,...layer.names.flatMap(name=>{const m=this.meta(name);return m?.shape.length===2?m.shape:[];})),scale=1.0/maximum;
    this.label(`LAYER ${index} · ${local?'LOCAL':'GLOBAL'} ATTENTION + MIXTURE OF EXPERTS`,v(0,8.4,0),{color:'#c7e3d4',width:10.4,scale:1.14});
    this.label(`${count(layer.params)} parameters · ${c.expertsPerToken} of ${c.experts} experts per token · [micro]`,v(0,7.87,0),{color:'#829b99',width:8.4,scale:.9});
    this.label(`${local?`LOCAL WINDOW ${c.localWindow}`:'GLOBAL CONTEXT'} · ${c.queryHeads}:${c.kvHeads} GQA`,v(-4.6,7.0,0),{color:'#a6b3d3',width:4.2,scale:.85});
    const projections=layer.names.filter(n=>n.includes('self_attn')&&n.endsWith('.weight'));
    for(let i=0;i<projections.length;i++){
      const meta=this.meta(projections[i])!,x=-5.75+(i%2)*1.82,y=5.8-Math.floor(i/2)*2.12,w=meta.shape[1]*scale,h=meta.shape[0]*scale,mesh=this.box(w,h,.16,C.attention,v(x,y,.1),{kind:'tensor',layer:index,tensor:meta.name});
      this.label(leaf(meta.name),v(x,y-h/2-.22,.25),{color:'#b9b8d1',width:1.9,scale:.72});this.label(meta.shape.join(' × '),v(x,y-h/2-.46,.25),{color:'#768696',width:1.7,scale:.64});
      if(meta.name.includes('q_proj'))this.headBands(mesh,w,h,c.queryHeads,false);else if(/[kv]_proj/.test(meta.name))this.headBands(mesh,w,h,c.kvHeads,false);
      const bias=this.meta(meta.name.replace('.weight','.bias'));if(bias){this.box(.06,h,.1,0xa08eaf,v(x+w/2+.16,y,.13),{kind:'tensor',layer:index,tensor:bias.name});this.label('bias',v(x+w/2+.16,y-h/2-.13,.2),{color:'#746a84',width:.65,scale:.55});}
    }
    const sinks=this.meta(`layers.${index}.self_attn.sinks`);if(sinks){this.box(1.65,.14,.13,0x9786ba,v(-4.8,1.66,.12),{kind:'tensor',layer:index,tensor:sinks.name});this.label(`${sinks.shape[0]} LEARNED ATTENTION SINKS`,v(-4.8,1.22,.2),{color:'#a999c6',width:3.5,scale:.75});}
    const router=this.meta(`layers.${index}.mlp.router.weight`);
    this.box(1.5,.9,.14,0xbc9a62,v(-1.36,4.7,0),undefined,.13);
    if(router){this.box(router.shape[1]*scale,router.shape[0]*scale,.18,0xd5b780,v(-1.36,4.7,.13),{kind:'tensor',layer:index,tensor:router.name});this.label(`${router.shape.join(' × ')} · router.weight`,v(-1.36,4.0,.2),{color:'#bca980',width:2.4,scale:.66});}
    this.label(`TOP ${c.expertsPerToken} ROUTER`,v(-1.36,5.65,.1),{color:'#e3c590',width:2.45,scale:.88});this.label('Token-dependent routing',v(-1.36,5.24,.1),{color:'#8e8269',width:2.7,scale:.68});
    const rb=this.meta(`layers.${index}.mlp.router.bias`);if(rb)this.box(.06,rb.shape[0]*scale,.1,0xb49768,v(-.74,4.7,.15),{kind:'tensor',layer:index,tensor:rb.name});
    this.arrow(v(-3.1,4.7,.25),v(-2.23,4.7,.25),0x9ea1c6);
    for(let e=0;e<(c.experts??4);e++){
      const x=1.55+(e%2)*3.13,y=5.62-Math.floor(e/2)*2.84,meshes:THREE.Mesh[]=[],panel=this.box(2.66,2.21,.15,0x608a7c,v(x,y,-.15),undefined,.11);meshes.push(panel);
      this.label(`EXPERT ${e}`,v(x,y+1.27,0),{color:'#abd0bc',width:2.5,scale:.88});
      for(const [m,part] of ['gate_proj','up_proj','down_proj'].entries()){
        const meta=this.meta(`layers.${index}.mlp.experts.${e}.${part}.weight`);if(!meta)continue;const px=x+(m-1)*.79,w=meta.shape[1]*scale,h=meta.shape[0]*scale,mesh=this.box(w,h,.16,0x6ea38c,v(px,y+.14,.08),{kind:'tensor',layer:index,tensor:meta.name},.75);meshes.push(mesh);
        this.label(part.replace('_proj',''),v(px,y-.76,.2),{color:'#a1b4a7',width:1.0,scale:.62});
        const bias=this.meta(meta.name.replace('.weight','.bias'));if(bias){const bm=this.box(.035,h,.09,0x7e9477,v(px+w/2+.06,y+.14,.12),{kind:'tensor',layer:index,tensor:bias.name},.8);meshes.push(bm);}
      }
      this.expertMeshes.set(`${index}:${e}`,meshes);this.line([v(-.61,4.7,.15),v(x-1.4,y,.15)],0x91ae98,.35);
    }
    const norms=layer.names.filter(n=>n.includes('layernorm.weight'));norms.forEach((name,i)=>{const m=this.meta(name)!;this.box(.06,m.shape[0]*scale,.1,C.norm,v(-1.95+i*1.5,2.35,.15),{kind:'tensor',layer:index,tensor:name});this.label(i?'POST-ATTENTION NORM':'INPUT NORM',v(-1.95+i*1.5,1.72,.2),{color:'#7f959e',width:1.8,scale:.59});});
    this.label('All experts are learned tensors. Playback highlights only the experts selected by actual router values.',v(0,.54,.3),{color:'#749589',width:11.2,scale:.81});
    this.fly(v(2,10.5,30),v(.1,4.1,0));
  }

  private headBands(mesh:THREE.Mesh,width:number,height:number,heads:number,queryGate:boolean){
    for(let h=1;h<heads;h++){const y=-height/2+h*height/heads;const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints([v(-width/2,y,.126),v(width/2,y,.126)]),new THREE.LineBasicMaterial({color:0xddd2ff,transparent:true,opacity:.65}));mesh.add(line);}
    if(queryGate)for(let h=0;h<heads;h++){const band=new THREE.Mesh(new THREE.PlaneGeometry(width,height/heads/2),new THREE.MeshBasicMaterial({color:0x291d48,transparent:true,opacity:.33,side:THREE.DoubleSide}));band.position.set(0,-height/2+(h+.25)*height/heads,.125);mesh.add(band);}
  }

  showTensor(meta:TensorMeta,tile?:WeightTile){
    this.resetContent();this.lod=2;this.enter({kind:'tensor',layer:meta.layer,tensor:meta.name});this.tensorMeta=meta;this.tensorTile=tile;this.decor.visible=false;
    this.renderTensor();
    this.fly(v(.2,4.5,20),v(0,3.8,0));
  }
  setTile(tile:WeightTile){
    if(!this.tensorMeta||this.tensorMeta.name!==tile.name)return;
    const prior=this.tensorTile,size={...this.tileDimensions},pending=this.streamPending,streamed=prior&&pending?.name===tile.name&&pending.row===tile.row&&pending.col===tile.col;
    this.clear(this.previousTileGroup);
    if(streamed){
      const cw=size.width/prior.cols,ch=size.height/prior.rows,width=tile.cols*cw,height=tile.rows*ch,shift=v((size.width-width)/2+(prior.col-tile.col)*cw,(height-size.height)/2+(tile.row-prior.row)*ch,0);
      // Rebase the local tile origin without moving the absolute weight under the camera.
      this.tileSizeOverride={width,height};this.camera.position.add(shift);this.controls.target.add(shift);
      if(this.tween){this.tween.from.add(shift);this.tween.to.add(shift);this.tween.targetFrom.add(shift);this.tween.targetTo.add(shift);}
      const texture=this.heatTexture(prior.values,prior.rows,prior.cols,undefined,prior.stats),mesh=new THREE.Mesh(new THREE.PlaneGeometry(size.width,size.height),new THREE.MeshBasicMaterial({map:texture,side:THREE.DoubleSide,toneMapped:false,fog:false,transparent:true,opacity:.55,depthWrite:false}));
      mesh.position.copy(this.tileOrigin).add(shift).add(v(0,0,-.012));this.previousTileGroup.add(mesh);
    }else{this.tileSizeOverride=undefined;this.streamArmed=false;this.streamPending=undefined;this.streamFailed.clear();}
    this.tensorTile=tile;this.renderTensor();
  }
  setLoading(progress:number){
    this.clear(this.loadingGroup);if(progress>=1||progress<0||!this.tensorMeta)return;
    const radius=.34,geometry=new THREE.TorusGeometry(radius,.022,8,64,Math.max(.08,progress)*Math.PI*2),ring=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color:C.delta}));ring.position.copy(this.tileOrigin).add(v(0,-.7,.18));this.loadingGroup.add(ring);
  }
  private renderTensor(){
    const meta=this.tensorMeta;if(!meta)return;this.clear(this.content);this.clear(this.tileGroup);this.clear(this.cellLabels);this.clear(this.loadingGroup);this.clear(this.highlightGroup);this.highlightKey='';this.interactives=[];this.cellMesh=undefined;this.tilePlane=undefined;this.numericPlane=undefined;this.cellRangeKey='';
    const tile=this.tensorTile,fullRows=meta.shape[0],fullCols=meta.shape.slice(1).reduce((a,b)=>a*b,1),rows=tile?.rows??Math.min(256,fullRows),cols=tile?.cols??Math.min(256,fullCols),width=this.tileSizeOverride?.width??Math.max(.65,6.6*Math.min(1,cols/rows)),height=this.tileSizeOverride?.height??Math.max(.65,6.6*Math.min(1,rows/cols));this.tileDimensions={width,height};
    this.box(width+.13,height+.13,.115,0x1d303b,this.tileOrigin.clone().add(v(0,0,-.075)),undefined,.9);
    const texture=tile?this.heatTexture(tile.values,rows,cols,undefined,tile.stats):this.emptyTexture();
    const plane=new THREE.Mesh(new THREE.PlaneGeometry(width,height),new THREE.MeshBasicMaterial({map:texture,side:THREE.DoubleSide,toneMapped:false,fog:false}));plane.position.copy(this.tileOrigin);plane.userData.selection={kind:'tensor',layer:meta.layer,tensor:meta.name};this.tilePlane=plane;this.tileGroup.add(plane);this.interactives.push(plane);
    this.label(short(meta.name),v(0,8.45,0),{color:'#d1e2e7',width:9.2});
    this.label(`${meta.shape.join(' × ')} · ${this.micro?'[micro]':'[config]'} shape · ${count(meta.params)} parameters`,v(0,8.05,0),{color:'#8aa4af',width:7.5,scale:.85});
    const startRow=tile?.row??0,startCol=tile?.col??0;
    this.label(tile?`[${tile.source}] TILE · rows ${startRow}–${startRow+rows-1} / ${fullRows} · columns ${startCol}–${startCol+cols-1} / ${fullCols}`:'NOT LOADED — CLICK TO FETCH',v(0,7.67,0),{color:tile?'#75d8c7':'#a5b4b9',width:8.5,scale:.78});
    const vector=meta.shape.length===1,columnMeaning=meta.name.includes('embed_tokens')?'EMBEDDING COORDINATES':meta.name.includes('conv1d')?'KERNEL POSITIONS':'INPUT FEATURES';
    this.label(vector?'PARAMETER VECTOR · NO INPUT-COLUMN AXIS':`${columnMeaning}  ${startCol} → ${startCol+cols-1}`,v(0,this.tileOrigin.y-height/2-.34,0),{color:'#87a8b4',width:5.4,scale:.85});
    let rowMeaning='OUTPUT FEATURES';
    if(meta.name.includes('embed'))rowMeaning='TOKEN IDs';else if(meta.name.includes('lm_head'))rowMeaning='VOCABULARY LOGITS';else if(meta.name.includes('conv1d'))rowMeaning='FEATURE CHANNELS';else if(meta.name.includes('.router.'))rowMeaning='EXPERT ROUTING SCORES';else if(meta.name.endsWith('.sinks'))rowMeaning='ATTENTION HEAD SINKS';else if(vector)rowMeaning=meta.name.includes('A_log')||meta.name.includes('dt_bias')?'DELTANET VALUE HEADS':'FEATURE COORDINATES';else if(meta.name.includes('q_proj')){const {dim,gated}=this.headConfig(meta),band=dim*(gated?2:1),head=Math.floor(startRow/band),offset=startRow%band;rowMeaning=`HEAD ${head} · ${gated&&offset>=dim?'GATE':'QUERY'} DIMENSIONS`;}else if(/[kv]_proj/.test(meta.name)){const {dim}=this.headConfig();rowMeaning=`KV HEAD ${Math.floor(startRow/dim)} DIMENSIONS`;}
    this.label(`${rowMeaning}  ${startRow} → ${startRow+rows-1}`,v(0,this.tileOrigin.y-height/2-.67,0),{color:'#70939e',width:7,scale:.78});
    for(const f of [0,.25,.5,.75,1]){
      const row=Math.min(rows-1,Math.floor(f*rows)),col=Math.min(cols-1,Math.floor(f*cols));
      this.label(String(startRow+row),v(-width/2-.36,this.tileOrigin.y+height/2-f*height,0),{color:'#65848e',width:.85,scale:.8});
      if(!vector)this.label(String(startCol+col),v(-width/2+f*width,this.tileOrigin.y+height/2+.18,0),{color:'#65848e',width:.8,scale:.75});
    }
    for(const axis of (vector?['row']:['row','column']) as ('row'|'column')[]){const mesh=new THREE.Mesh(new THREE.PlaneGeometry(axis==='row'?.6:width,axis==='row'?height:.43),new THREE.MeshBasicMaterial({transparent:true,opacity:0,depthWrite:false,side:THREE.DoubleSide}));mesh.position.set(axis==='row'?-width/2-.33:0,axis==='row'?this.tileOrigin.y:this.tileOrigin.y+height/2+.19,.015);mesh.userData.axis=axis;mesh.userData.selection={kind:'tensor',layer:meta.layer,tensor:meta.name};this.content.add(mesh);this.interactives.push(mesh);}
    if(meta.name.includes('.self_attn.')&&/[qkv]_proj/.test(meta.name))this.tensorHeadRuler(meta,tile);
    if(tile){this.colorLegend(tile);this.label('Scroll to resolve weights · drag to pan · Shift+drag to orbit',v(0,-.66,0),{color:'#567680',width:8,scale:.77});}
    else this.label(this.micro?'Micro weights live in the simulation worker':'Not loaded — click to fetch',this.tileOrigin.clone().add(v(0,0,.02)),{color:'#9bb0b7',width:5.9});
  }
  private emptyTexture(){const canvas=document.createElement('canvas');canvas.width=512;canvas.height=512;const ctx=canvas.getContext('2d')!;ctx.fillStyle='#101f28';ctx.fillRect(0,0,512,512);ctx.strokeStyle='#1e323c';ctx.lineWidth=1;for(let i=0;i<=512;i+=32){ctx.beginPath();ctx.moveTo(i,0);ctx.lineTo(i,512);ctx.stroke();ctx.beginPath();ctx.moveTo(0,i);ctx.lineTo(512,i);ctx.stroke();}const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;return texture;}
  private heatColor(value:number,limit:number){
    const x=Math.max(-1,Math.min(1,value/limit)),base=new THREE.Color('#17232e'),end=new THREE.Color(x<0?'#6479e8':'#58e4c4');return base.lerp(end,Math.pow(Math.abs(x),.68));
  }
  private heatTexture(values:ArrayLike<number>,rows:number,cols:number,mask?:boolean[],stats?:{p01:number;p99:number},numbers=false){
    const finite=Array.from(values).filter(Number.isFinite).sort((a,b)=>a-b),limit=Math.max(1e-12,Math.abs(stats?.p01??finite[Math.floor((finite.length-1)*.01)]??1),Math.abs(stats?.p99??finite[Math.floor((finite.length-1)*.99)]??1));
    const pixelScale=numbers&&Math.max(rows,cols)<=20?96:1,canvas=document.createElement('canvas');canvas.width=cols*pixelScale;canvas.height=rows*pixelScale;const ctx=canvas.getContext('2d')!;
    if(pixelScale===1){const pixels=ctx.createImageData(cols,rows);for(let i=0;i<rows*cols;i++){const c=(mask?.[i]||!Number.isFinite(values[i])?new THREE.Color('#35424a'):this.heatColor(values[i],limit)).convertLinearToSRGB();pixels.data.set([Math.round(c.r*255),Math.round(c.g*255),Math.round(c.b*255),255],i*4);}ctx.putImageData(pixels,0,0);}
    else for(let row=0;row<rows;row++)for(let col=0;col<cols;col++){const i=row*cols+col,c=mask?.[i]||!Number.isFinite(values[i])?new THREE.Color('#35424a'):this.heatColor(values[i],limit);ctx.fillStyle=`#${c.getHexString()}`;ctx.fillRect(col*pixelScale,row*pixelScale,pixelScale-2,pixelScale-2);ctx.fillStyle='#e1f1ee';ctx.font='22px ui-monospace,monospace';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(mask?.[i]?'mask':Number.isFinite(values[i])?values[i].toFixed(2):'−∞',(col+.5)*pixelScale,(row+.5)*pixelScale,pixelScale-4);}
    const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;texture.magFilter=THREE.NearestFilter;texture.minFilter=THREE.NearestFilter;return texture;
  }
  private colorLegend(tile:WeightTile){
    const limit=Math.max(Math.abs(tile.stats.p01),Math.abs(tile.stats.p99),1e-12),vals=Array.from({length:100},(_,i)=>(i/99*2-1)*limit),texture=this.heatTexture(vals,1,100,undefined,{p01:-limit,p99:limit}),plane=new THREE.Mesh(new THREE.PlaneGeometry(2.2,.095),new THREE.MeshBasicMaterial({map:texture,side:THREE.DoubleSide}));plane.position.set(0,-.31,0);this.content.add(plane);
    this.label(`${(-limit).toPrecision(3)}     0     ${limit.toPrecision(3)}`,v(0,-.47,0),{color:'#88a7b0',width:3.2,scale:.74});
  }
  private tensorHeadRuler(meta:TensorMeta,tile?:WeightTile){
    const isQ=meta.name.includes('q_proj'),config=this.headConfig(meta),dim=config.dim,heads=isQ?config.heads:config.kvHeads,gated=isQ&&config.gated,band=dim*(gated?2:1),height=6.6,w=.23,x=this.tileDimensions.width/2+.76;
    for(let h=0;h<heads;h++){const y=3.8+height/2-(h+.5)*height/heads,active=tile&&tile.row<((h+1)*band)&&tile.row+tile.rows>h*band;
      const mesh=this.box(w,height/heads*.94,.025,active?0xd4b8ff:0x5d537f,v(x,y,.02),{kind:'tensor',layer:meta.layer,tensor:meta.name,row:h*band,col:tile?.col??0},active?1:.5);mesh.userData.head=h;
      if(gated){const gate=new THREE.Mesh(new THREE.PlaneGeometry(w*.45,height/heads*.94),new THREE.MeshBasicMaterial({color:0x30223e}));gate.position.set(w*.25,0,.014);mesh.add(gate);}
      if(heads<=8||h%4===0||active)this.label(`${h}`,v(x+.38,y,.02),{color:active?'#d9c3ff':'#685e83',width:.5,scale:.72});
    }
    this.label(isQ?(gated?'Q | gate':'QUERY'):'KV',v(x,7.4,.02),{color:'#ac92d6',width:1.35,scale:.75});this.label('HEADS',v(x,7.13,.02),{color:'#695981',width:1.15,scale:.7});
  }
  private updateCellDetail(now:number){
    const tile=this.tensorTile;if(!tile||!this.tensorMeta||!this.tilePlane)return;
    const distance=this.camera.position.distanceTo(this.controls.target),pixelsPerUnit=this.container.clientHeight/(2*Math.tan(THREE.MathUtils.degToRad(this.camera.fov)/2)*Math.max(.005,distance)),cellWidth=this.tileDimensions.width/tile.cols,cellHeight=this.tileDimensions.height/tile.rows,cellPixels=Math.min(cellWidth,cellHeight)*pixelsPerUnit;
    const want=cellPixels>(this.quality<1?10:5);
    if(!want){if(this.cellMesh)this.cellMesh.visible=false;this.cellLabels.visible=false;this.lod=2;return;}
    this.lod=3;this.cellLabels.visible=true;if(this.cellMesh)this.cellMesh.visible=true;
    if(now-this.updateCellsAt<150)return;this.updateCellsAt=now;
    this.raycaster.setFromCamera(new THREE.Vector2(0,0),this.camera);const center=new THREE.Vector3();this.raycaster.ray.intersectPlane(new THREE.Plane(v(0,0,1),-this.tileOrigin.z),center);
    const centerCol=Math.floor((center.x+this.tileDimensions.width/2)/cellWidth),centerRow=Math.floor((this.tileOrigin.y+this.tileDimensions.height/2-center.y)/cellHeight),maxSide=this.quality<1?80:160;
    const halfCols=Math.min(maxSide/2,Math.ceil(this.container.clientWidth/Math.max(cellPixels,1)/2)+3),halfRows=Math.min(maxSide/2,Math.ceil(this.container.clientHeight/Math.max(cellPixels,1)/2)+3);
    const c0=Math.max(0,Math.min(tile.cols-1,centerCol-halfCols)),c1=Math.min(tile.cols,Math.max(1,centerCol+halfCols)),r0=Math.max(0,Math.min(tile.rows-1,centerRow-halfRows)),r1=Math.min(tile.rows,Math.max(1,centerRow+halfRows)),numbers=cellPixels>=40;
    const key=[r0,r1,c0,c1,numbers].join(':');if(this.cellRangeKey===key)return;this.cellRangeKey=key;
    const capacity=25600;
    if(!this.cellMesh){this.cellMesh=new THREE.InstancedMesh(new THREE.PlaneGeometry(cellWidth*.975,cellHeight*.975),new THREE.MeshBasicMaterial({side:THREE.DoubleSide,toneMapped:false,fog:false}),capacity);this.cellMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);this.cellMesh.userData.selection={kind:'cell',tensor:tile.name,layer:this.tensorMeta.layer};this.cellMesh.frustumCulled=false;this.tileGroup.add(this.cellMesh);this.interactives.unshift(this.cellMesh);}
    const matrix=new THREE.Matrix4(),limit=Math.max(Math.abs(tile.stats.p01),Math.abs(tile.stats.p99),1e-12),indices:{row:number;col:number}[]=[];let n=0;
    for(let row=r0;row<r1;row++)for(let col=c0;col<c1;col++){if(n>=capacity)break;matrix.makeTranslation(-this.tileDimensions.width/2+(col+.5)*cellWidth,this.tileOrigin.y+this.tileDimensions.height/2-(row+.5)*cellHeight,.002);this.cellMesh.setMatrixAt(n,matrix);this.cellMesh.setColorAt(n,this.heatColor(tile.values[row*tile.cols+col],limit));indices.push({row:row+tile.row,col:col+tile.col});n++;}
    this.cellMesh.count=n;this.cellMesh.userData.cells=indices;this.cellMesh.instanceMatrix.needsUpdate=true;if(this.cellMesh.instanceColor)this.cellMesh.instanceColor.needsUpdate=true;
    this.clear(this.cellLabels);this.numericPlane=undefined;
    if(numbers&&r1>r0&&c1>c0){const rows=r1-r0,cols=c1-c0,size=96,canvas=document.createElement('canvas');canvas.width=Math.min(4096,cols*size);canvas.height=Math.min(4096,rows*size);const ctx=canvas.getContext('2d')!,sx=canvas.width/cols,sy=canvas.height/rows;ctx.font=`${Math.min(sx,sy)*.21}px ui-monospace,monospace`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillStyle='#e9f8f3';ctx.shadowColor='#081825';ctx.shadowBlur=3;
      for(let row=r0;row<r1;row++)for(let col=c0;col<c1;col++)ctx.fillText(tile.values[row*tile.cols+col].toFixed(4),(col-c0+.5)*sx,(row-r0+.5)*sy,sx*.92);
      const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;this.numericPlane=new THREE.Mesh(new THREE.PlaneGeometry(cols*cellWidth,rows*cellHeight),new THREE.MeshBasicMaterial({map:texture,transparent:true,side:THREE.DoubleSide,depthWrite:false}));this.numericPlane.position.set(-this.tileDimensions.width/2+(c0+c1)*cellWidth/2,this.tileOrigin.y+this.tileDimensions.height/2-(r0+r1)*cellHeight/2,.005);this.cellLabels.add(this.numericPlane);
    }
  }
  private autoDetail(now:number){
    if(this.tween||now-this.lastLodChange<1100)return;const distance=this.camera.position.distanceTo(this.controls.target);
    if((this.lod===0||this.lod===1)&&distance<5.3){
      this.raycaster.setFromCamera(new THREE.Vector2(),this.camera);const hit=this.raycaster.intersectObjects(this.interactives,false)[0],selection=hit?.object.userData.selection as Selection|undefined;
      if(selection?.kind==='layer'&&selection.layer!==undefined){this.lastLodChange=now;this.showLayer(selection.layer);}
      else if(selection?.kind==='tensor'&&selection.tensor){const meta=this.meta(selection.tensor);if(meta){this.lastLodChange=now;this.showTensor(meta);}}
    }else if(this.lod>=2&&distance>26){this.lastLodChange=now;if(this.tensorMeta?.layer!==undefined)this.showLayer(this.tensorMeta.layer);else this.showModel();}
    else if(this.lod===1&&distance>34){this.lastLodChange=now;this.showModel();}
  }
  private streamAdjacentTile(now:number){
    const tile=this.tensorTile,meta=this.tensorMeta,request=this.callbacks.onTileRequest;
    if(!request||!tile||!meta||tile.source!=='real'||this.lod!==3||this.tween||!this.streamArmed)return;
    // Exact sparse token rows retain their own coordinate layout; ordinary tiles use the loader's 256-cell grid.
    if(tile.rows<2||tile.cols<2||tile.row%256!==0||tile.col%256!==0)return;
    this.raycaster.setFromCamera(new THREE.Vector2(),this.camera);const center=new THREE.Vector3();
    if(!this.raycaster.ray.intersectPlane(new THREE.Plane(v(0,0,1),-this.tileOrigin.z),center))return;
    const cw=this.tileDimensions.width/tile.cols,ch=this.tileDimensions.height/tile.rows,localCol=(center.x+this.tileDimensions.width/2)/cw,localRow=(this.tileOrigin.y+this.tileDimensions.height/2-center.y)/ch;
    if(localCol>=0&&localCol<tile.cols&&localRow>=0&&localRow<tile.rows){this.streamFailed.clear();return;}
    if(this.streamPending||now-this.streamRequestedAt<450)return;
    const fullRows=meta.shape[0],fullCols=meta.shape.slice(1).reduce((a,b)=>a*b,1),row=Math.max(0,Math.min(Math.floor((fullRows-1)/256)*256,Math.floor((tile.row+localRow)/256)*256)),col=Math.max(0,Math.min(Math.floor((fullCols-1)/256)*256,Math.floor((tile.col+localCol)/256)*256));
    if(row===tile.row&&col===tile.col)return;
    const key=`${tile.name}:${row}:${col}`;if(this.streamFailed.has(key))return;
    const pending={key,name:tile.name,row,col};this.streamPending=pending;this.streamRequestedAt=now;
    void request(tile.name,row,col).then(()=>{
      // A cancelled fetch may resolve without installing a tile. Treat it as a failed coordinate until re-entry.
      if(this.streamPending===pending&&(this.tensorTile?.row!==row||this.tensorTile?.col!==col))this.streamFailed.add(key);
    }).catch(()=>{if(this.streamPending===pending)this.streamFailed.add(key);}).finally(()=>{if(this.streamPending===pending)this.streamPending=undefined;});
  }
  focus(selection:Selection){if(selection.kind==='model')this.showModel();else if(selection.kind==='layer'&&selection.layer!==undefined)this.showLayer(selection.layer);else if(selection.kind==='cell'&&this.tensorTile?.name===selection.tensor&&selection.row!==undefined&&selection.col!==undefined){this.focusCell(selection.row,selection.col);this.callbacks.onSelect(selection);}else if(selection.tensor){const meta=this.meta(selection.tensor);if(meta)this.showTensor(meta);}}
  back(){const previous=this.history.pop();if(!previous){if(this.selection.kind!=='model')this.showModel();return;}const remaining=[...this.history];this.focus(previous);this.history=remaining;}
  setMicroFamily(family:MicroFamily,config:MicroSceneConfig){this.familyDirty=this.familyDirty||this.microFamily!==family||JSON.stringify(this.microConfig)!==JSON.stringify(config);this.microFamily=family;this.microConfig={...config};this.microLayers=config.layers;if(this.familyDirty){this.followStep=undefined;this.lastRouting=undefined;}}
  setMicro(enabled:boolean,parameters?:{name:string;shape:number[]}[]){if(parameters)this.microParameters=parameters;const changed=this.micro!==enabled||this.familyDirty;this.micro=enabled;if(parameters){const indices=parameters.flatMap(p=>{const match=/layers\.(\d+)/.exec(p.name);return match?[Number(match[1])]:[];});if(indices.length)this.microLayers=Math.max(...indices)+1;}this.familyDirty=false;if(changed){this.history=[];this.selection={kind:'model'};this.showModel();}}
  setFollow(enabled:boolean){this.autoFollow=enabled;}
  setPlayback(playing:boolean,speed=.5){this.playing=playing;this.playbackSpeed=Math.max(.05,Math.min(5,speed));this.flowGroup.visible=playing&&this.micro;}
  private highlightRouting(step:StepLike){
    if(!this.micro||this.microFamily!=='gpt-oss')return;
    const count=this.microConfig.experts??4;
    if(step.name==='Top-k expert routing'&&step.matrix?.shape.at(-1)===count){const row=Math.max(0,(step.matrix.shape[0]??1)-1);this.lastRouting={layer:step.layer??0,weights:step.matrix.values.slice(row*count,(row+1)*count)};}
    for(const [key,meshes] of this.expertMeshes){const [layer,expert]=key.split(':').map(Number),weight=this.lastRouting?.layer===layer&&step.layer===layer?this.lastRouting.weights[expert]:undefined;
      for(const mesh of meshes){const material=mesh.material as THREE.MeshStandardMaterial;if(mesh.userData.baseOpacity===undefined)mesh.userData.baseOpacity=material.opacity;material.opacity=weight===undefined?mesh.userData.baseOpacity:weight>0?Math.max(.65,mesh.userData.baseOpacity):.2;material.emissiveIntensity=weight===undefined?.045:weight>0?.16+Math.min(1,weight)*.55:.012;material.transparent=material.opacity<1;}
    }
    if(this.lastRouting&&this.lastRouting.layer===step.layer&&this.expertMeshes.size){const active=this.lastRouting.weights.flatMap((weight,e)=>weight>0?[`E${e} ${(weight*100).toFixed(1)}%`]:[]),position=this.selection.kind==='model'?v(2.2,(this.layerCenters.get(step.layer??0)?.y??3)+1.05,1.75):v(3.15,.02,.3);this.label(`LAST TOKEN → ${active.join(' + ')} [micro]`,position,{color:'#b9dec7',width:4.9,scale:.72,parent:this.overlay});}
  }
  setActiveStep(step:StepLike){
    const previousLayer=this.followStep?.layer;this.followStep=step;this.clear(this.activeMarker);this.clear(this.overlay);this.clear(this.flowGroup);
    this.highlightRouting(step);
    const layer=step.layer??-1,pos=this.layerCenters.get(layer)??(this.selection.kind==='model'?(layer<0?v(-1.1,.48,0):v(-1.1,10,0)):undefined);
    if(pos){
      const markerWidth=this.micro&&this.microFamily==='gpt-oss'?7.9:this.micro&&this.microFamily==='gemma'?7.2:5.7;
      const ring=new THREE.Mesh(new THREE.BoxGeometry(markerWidth,.12,3.35),new THREE.MeshBasicMaterial({color:step.phase==='backward'?0xe19bb1:step.phase==='update'?0xffcd8b:0x9affdf,transparent:true,opacity:.19,depthWrite:false}));ring.position.copy(pos);this.activeMarker.add(ring);
      const line=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(markerWidth+.02,.14,3.38)),new THREE.LineBasicMaterial({color:step.phase==='backward'?0xe8a4c8:0xaeffea,transparent:true,opacity:.95}));line.position.copy(pos);this.activeMarker.add(line);this.activeMarker.visible=true;
      if(this.autoFollow&&this.micro&&previousLayer!==layer){const drift=THREE.MathUtils.clamp((pos.y-4.35)*.075,-.3,.3);this.fly(v(18.5,13+drift,22.5),v(.1,4.35+drift,0),550);}
      if(this.micro&&step.outputs?.[0]?.values.length){
        const values=step.outputs[0].values.slice(0,16),finite=values.filter(Number.isFinite),limit=Math.max(1e-12,...finite.map(Math.abs)),direction=step.phase==='backward'?-1:1,next=this.layerCenters.get(layer+direction)??pos.clone().add(v(0,.7*direction,0));this.flowFrom.set(2.02,pos.y,1.77);this.flowTo.set(2.02,next.y,1.77);this.flowGroup.position.copy(this.flowFrom);this.flowStart=performance.now();
        for(let i=0;i<values.length;i++){const cube=new THREE.Mesh(new THREE.BoxGeometry(.12,.055,.09),new THREE.MeshBasicMaterial({color:Number.isFinite(values[i])?this.heatColor(values[i],limit):new THREE.Color('#35424a'),toneMapped:false,fog:false}));cube.position.set(0,(i-(values.length-1)/2)*.062,0);this.flowGroup.add(cube);}
        this.label('[micro] schematic flow',v(.18,.78,0),{color:'#8fd7c6',width:2.3,scale:.67,parent:this.flowGroup});this.flowGroup.visible=this.playing;
      }
    }
    if(this.selection.kind==='tensor'||this.selection.kind==='cell')return;
    let matrix=step.matrix;
    if(!matrix&&step.outputs?.[0]){const output=step.outputs[0],cols=output.shape.length>1?output.shape.slice(1).reduce((a,b)=>a*b,1):output.values.length,rows=output.shape.length>1?output.shape[0]:1;matrix={shape:[rows,cols],values:output.values};}
    if(matrix&&matrix.shape.length>=1&&matrix.values.length){
      const realCols=matrix.shape.at(-1)??1,realRows=matrix.shape.length>1?matrix.shape[0]:1,cols=Math.min(realCols,16),rows=Math.min(realRows,16),values:number[]=[],mask:boolean[]=[];
      for(let row=0;row<rows;row++)for(let col=0;col<cols;col++){values.push(matrix.values[row*realCols+col]??NaN);mask.push(matrix.mask?.[row*realCols+col]??false);}
      const width=3*Math.min(1,cols/rows),height=3*Math.min(1,rows/cols),texture=this.heatTexture(values,rows,cols,mask,undefined,true),sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,transparent:true,depthTest:false})),anchor=this.selection.kind==='model'?v(4.8,THREE.MathUtils.clamp(pos?.y??4,4,7.6),2):v(1.2,3.5,2);
      sprite.scale.set(width,height,1);sprite.position.copy(anchor);sprite.renderOrder=8;this.overlay.add(sprite);
      this.label(`[micro] ${step.name??'Computed activation'}`,anchor.clone().add(v(0,height/2+.43,0)),{color:'#c2eee5',width:5.2,scale:.8,parent:this.overlay});
      this.label(`${matrix.shape.join(' × ')}${realRows>rows||realCols>cols?' · top-left window':''} · actual computed values`,anchor.clone().add(v(0,-height/2-.3,0)),{color:'#75a59d',width:4.6,scale:.75,parent:this.overlay});
      if(pos)this.line([pos.clone().add(v(2.85,0,1.6)),anchor.clone().add(v(-width/2,0,0))],step.phase==='backward'?0xdc96b8:0x75dfca,.7,this.overlay);
    }
  }
  private hits(event:MouseEvent){const rect=this.renderer.domElement.getBoundingClientRect();this.pointer.set((event.clientX-rect.left)/rect.width*2-1,-((event.clientY-rect.top)/rect.height*2-1));this.raycaster.setFromCamera(this.pointer,this.camera);return this.raycaster.intersectObjects(this.interactives,false);}
  private hitCell(hit:THREE.Intersection){
    const tile=this.tensorTile;if(!tile)return;
    if(hit.object===this.cellMesh&&hit.instanceId!==undefined)return this.cellMesh.userData.cells?.[hit.instanceId] as {row:number;col:number}|undefined;
    if(hit.object===this.tilePlane&&hit.uv){const col=Math.min(tile.cols-1,Math.floor(hit.uv.x*tile.cols)),row=Math.min(tile.rows-1,Math.floor((1-hit.uv.y)*tile.rows));return {row:tile.row+row,col:tile.col+col};}
  }
  private highlight(row?:number,col?:number){
    const tile=this.tensorTile;if(!tile)return;this.highlightGroup.visible=true;const key=`${row}:${col}`;if(key===this.highlightKey)return;this.highlightKey=key;this.clear(this.highlightGroup);const cw=this.tileDimensions.width/tile.cols,ch=this.tileDimensions.height/tile.rows;
    const band=(width:number,height:number,x:number,y:number)=>{const plane=new THREE.Mesh(new THREE.PlaneGeometry(width,height),new THREE.MeshBasicMaterial({color:0xffdea0,transparent:true,opacity:.19,depthWrite:false,side:THREE.DoubleSide,toneMapped:false,fog:false}));plane.position.set(x,y,.009);this.highlightGroup.add(plane);const points=[v(x-width/2,y-height/2,.011),v(x+width/2,y-height/2,.011),v(x+width/2,y+height/2,.011),v(x-width/2,y+height/2,.011),v(x-width/2,y-height/2,.011)];this.line(points,0xffe7b7,.75,this.highlightGroup);};
    if(row!==undefined)band(this.tileDimensions.width,ch,0,this.tileOrigin.y+this.tileDimensions.height/2-(row-tile.row+.5)*ch);
    if(col!==undefined)band(cw,this.tileDimensions.height,-this.tileDimensions.width/2+(col-tile.col+.5)*cw,this.tileOrigin.y);
  }
  private onPointerMove(event:PointerEvent){
    const hit=this.hits(event)[0],mesh=hit?.object as HitMesh|undefined;
    if(this.hovered!==mesh){if(this.hovered&&this.hovered.material instanceof THREE.MeshStandardMaterial)this.hovered.material.emissiveIntensity=.045;this.hovered=mesh;if(mesh?.material instanceof THREE.MeshStandardMaterial)mesh.material.emissiveIntensity=.22;this.renderer.domElement.style.cursor=mesh?'pointer':'grab';}
    const cell=hit?this.hitCell(hit):undefined,axis=hit?.object.userData.axis as 'row'|'column'|undefined,tile=this.tensorTile;
    if(axis&&hit?.uv&&tile&&this.tensorMeta){if(axis==='row'){const row=tile.row+Math.min(tile.rows-1,Math.floor((1-hit.uv.y)*tile.rows));this.highlight(row);this.hoverElement.textContent=`[${tile.source}] Row ${row}\n${axesMeaning(this.tensorMeta)}`;}else{const col=tile.col+Math.min(tile.cols-1,Math.floor(hit.uv.x*tile.cols));this.highlight(undefined,col);this.hoverElement.textContent=`[${tile.source}] Column ${col}\n${axesMeaning(this.tensorMeta)}`;}}
    else if(cell&&tile&&this.tensorMeta){const value=tile.values[(cell.row-tile.row)*tile.cols+cell.col-tile.col],coordinate=this.tensorMeta.shape.length===1?`w[${cell.row}]`:`w[${cell.row}][${cell.col}]`;this.hoverElement.textContent=`[${tile.source}] ${coordinate} = ${value.toPrecision(8)}\n${cellMeaning(this.tensorMeta,cell.row,cell.col)}`;this.highlight(cell.row,this.tensorMeta.shape.length===1?undefined:cell.col);this.hoveredCell=`${cell.row}:${cell.col}`;}
    else{this.hoverElement.style.display='none';this.highlightGroup.visible=false;this.hoveredCell=undefined;return;}
    const rect=this.container.getBoundingClientRect();this.hoverElement.style.left=`${Math.min(rect.width-345,Math.max(5,event.clientX-rect.left+18))}px`;this.hoverElement.style.top=`${Math.min(rect.height-130,Math.max(5,event.clientY-rect.top+18))}px`;this.hoverElement.style.display='block';
  }
  private focusCell(row:number,col:number){const tile=this.tensorTile;if(!tile)return;const cw=this.tileDimensions.width/tile.cols,ch=this.tileDimensions.height/tile.rows,p=v(-this.tileDimensions.width/2+(col-tile.col+.5)*cw,this.tileOrigin.y+this.tileDimensions.height/2-(row-tile.row+.5)*ch,.002);this.fly(p.clone().add(v(0,0,Math.max(.015,Math.min(cw,ch)*18))),p,800);}
  private pick(event:MouseEvent,double:boolean){
    const hit=this.hits(event)[0];if(!hit)return;const cell=this.hitCell(hit);
    if(cell&&this.tensorMeta){const selection:Selection={kind:'cell',tensor:this.tensorMeta.name,layer:this.tensorMeta.layer,...cell};this.callbacks.onSelect(selection);if(double)this.focusCell(cell.row,cell.col);return;}
    const selection=(hit.object.userData.selection??this.selection) as Selection;
    if(selection.kind==='layer'&&selection.layer!==undefined)this.showLayer(selection.layer);
    else if(selection.tensor){const meta=this.meta(selection.tensor);if(meta){if(this.selection.kind==='tensor'&&!double)this.callbacks.onSelect(selection);else this.showTensor(meta);}}
  }
  private animate=()=>{
    if(this.disposed)return;this.frame=requestAnimationFrame(this.animate);const now=performance.now(),dt=now-this.lastFrame;this.lastFrame=now;
    if(this.tween){const t=Math.min(1,(now-this.tween.start)/this.tween.duration),e=1-Math.pow(1-t,3);this.camera.position.lerpVectors(this.tween.from,this.tween.to,e);this.controls.target.lerpVectors(this.tween.targetFrom,this.tween.targetTo,e);if(t===1)this.tween=undefined;}
    this.controls.mouseButtons.LEFT=this.lod>=2?THREE.MOUSE.PAN:THREE.MOUSE.ROTATE;this.controls.update();
    if(this.playing&&this.flowGroup.children.length){const progress=((now-this.flowStart)/(1400/this.playbackSpeed))%1,e=progress*progress*(3-2*progress);this.flowGroup.position.lerpVectors(this.flowFrom,this.flowTo,e);}
    // Keep adequate depth precision at overview scale while allowing sub-centimeter cell views.
    const near=THREE.MathUtils.clamp(this.camera.position.distanceTo(this.controls.target)*.001,.0001,.04);if(Math.abs(near-this.camera.near)>this.camera.near*.04){this.camera.near=near;this.camera.updateProjectionMatrix();}
    this.updateCellDetail(now);this.autoDetail(now);this.streamAdjacentTile(now);this.renderer.render(this.scene,this.camera);this.measuredFrames++;
    this.slowFor=dt>20?this.slowFor+dt:0;if(this.slowFor>1000&&this.quality>.65){this.quality=.65;this.renderer.setPixelRatio(Math.min(window.devicePixelRatio,1));this.slowFor=0;}
    if(now-this.measuredAt>750){const fps=this.measuredFrames*1000/(now-this.measuredAt);this.callbacks.onStats?.({fps,renderer:'WebGL2',lod:this.lod,instances:this.cellMesh?.count??this.interactives.length});this.measuredAt=now;this.measuredFrames=0;}
  };
  dispose(){this.disposed=true;cancelAnimationFrame(this.frame);this.observer.disconnect();this.controls.dispose();const canvas=this.renderer.domElement;canvas.removeEventListener('pointermove',this.pointerMove);canvas.removeEventListener('pointerleave',this.pointerLeave);canvas.removeEventListener('pointerdown',this.pointerDown);canvas.removeEventListener('pointerup',this.pointerUp);canvas.removeEventListener('dblclick',this.doubleClick);window.removeEventListener('keydown',this.keyDown);this.clear(this.content);this.clear(this.decor);this.clear(this.overlay);this.clear(this.activeMarker);this.clear(this.tileGroup);this.clear(this.previousTileGroup);this.clear(this.cellLabels);this.clear(this.loadingGroup);this.clear(this.highlightGroup);this.clear(this.flowGroup);this.renderer.dispose();canvas.remove();this.hoverElement.remove();}
}
