import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { ModelManifest, TensorMeta, WeightTile } from '../data/hf';

export interface Selection { kind:'model'|'layer'|'tensor'|'cell'; layer?:number; tensor?:string; row?:number; col?:number }
interface Callbacks { onSelect:(selection:Selection)=>void; onStats?:(stats:{fps:number;renderer:string;lod:number;instances:number})=>void }
interface StepLike { name?:string; layer?:number; phase?:string; parameter?:string; matrix?:{shape:number[];values:number[];mask?:boolean[]}; inputs?:{name:string;shape:number[];values:number[]}[]; outputs?:{name:string;shape:number[];values:number[]}[] }
type HitMesh = THREE.Mesh | THREE.InstancedMesh;
const C={delta:0x54dccb,attention:0xa493ff,ffn:0xc5ae83,norm:0xa0b3bf,embedding:0x7fbae2,vision:0x739b91,mtp:0xc1a0e5,head:0xe9cfa1};
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
  private activeMarker=new THREE.Group();
  private followStep?:StepLike;
  private autoFollow=true;
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
  private loadingGroup=new THREE.Group();
  private hoveredCell?:string;
  private hoverElement:HTMLDivElement;
  private cellRangeKey='';
  private numericPlane?:THREE.Mesh;
  private updateCellsAt=0;
  private lastLodChange=0;
  private pointerMove=(event:PointerEvent)=>this.onPointerMove(event);
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
    this.renderer.toneMappingExposure=1.35;
    this.renderer.domElement.setAttribute('aria-label','Interactive three-dimensional Qwen model. Drag to orbit, scroll to zoom, click a layer, double-click a tensor, Escape to return.');
    this.renderer.domElement.style.cssText='display:block;width:100%;height:100%;touch-action:none;outline:none';
    container.appendChild(this.renderer.domElement);
    this.scene.fog=new THREE.FogExp2(0x070d13,.014);
    this.scene.add(new THREE.HemisphereLight(0xc8e8ef,0x12202f,2.0));
    const key=new THREE.DirectionalLight(0xd5f0ff,3.4);key.position.set(6,15,8);this.scene.add(key);
    const rim=new THREE.DirectionalLight(0x758bc8,2.3);rim.position.set(-8,6,-9);this.scene.add(rim);
    const teal=new THREE.PointLight(C.delta,40,22,2);teal.position.set(-5,4,5);this.scene.add(teal);
    this.scene.add(this.decor,this.content,this.overlay,this.activeMarker,this.tileGroup,this.cellLabels,this.loadingGroup);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);
    this.controls.enableDamping=true;this.controls.dampingFactor=.075;this.controls.minDistance=.015;this.controls.maxDistance=65;
    this.controls.maxPolarAngle=Math.PI*.92;this.controls.zoomSpeed=.8;this.controls.panSpeed=.8;this.controls.zoomToCursor=true;
    this.controls.addEventListener('start',()=>{this.tween=undefined;});
    this.createGround();
    this.camera.position.set(16,12.3,19);this.controls.target.set(.4,5.2,0);
    const canvas=this.renderer.domElement;
    canvas.addEventListener('pointermove',this.pointerMove);canvas.addEventListener('pointerdown',this.pointerDown);canvas.addEventListener('pointerup',this.pointerUp);canvas.addEventListener('dblclick',this.doubleClick);
    window.addEventListener('keydown',this.keyDown);
    this.hoverElement=document.createElement('div');this.hoverElement.style.cssText='position:absolute;pointer-events:none;display:none;z-index:8;max-width:330px;padding:10px 13px;border:1px solid #40636e;border-radius:9px;background:rgba(9,21,28,.96);color:#c7e0e4;font:12px/1.55 ui-monospace,monospace;box-shadow:0 8px 30px #0008;white-space:pre-line';container.appendChild(this.hoverElement);
    this.observer=new ResizeObserver(()=>this.resize());this.observer.observe(container);this.resize();
    this.showModel();this.animate();
  }

  resize(){const w=Math.max(1,this.container.clientWidth),h=Math.max(1,this.container.clientHeight);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();this.renderer.setSize(w,h,false);}
  private box(width:number,height:number,depth:number,color:number,position:THREE.Vector3,selection?:Selection,opacity=1){
    const geometry=new THREE.BoxGeometry(width,height,depth);
    const material=new THREE.MeshStandardMaterial({color,metalness:.35,roughness:.36,transparent:opacity<1,opacity,emissive:color,emissiveIntensity:.045});
    const mesh=new THREE.Mesh(geometry,material);mesh.position.copy(position);
    const edges=new THREE.LineSegments(new THREE.EdgesGeometry(geometry),new THREE.LineBasicMaterial({color,transparent:true,opacity:.55}));mesh.add(edges);
    if(selection){mesh.userData.selection=selection;this.interactives.push(mesh);}
    this.content.add(mesh);return mesh;
  }
  private label(text:string,position:THREE.Vector3,options:{color?:string;scale?:number;width?:number;opacity?:number;parent?:THREE.Group}={}){
    const canvas=document.createElement('canvas');canvas.width=1024;canvas.height=128;
    const ctx=canvas.getContext('2d')!;ctx.clearRect(0,0,1024,128);ctx.font='500 44px "SF Pro Display", Inter, -apple-system, sans-serif';ctx.textBaseline='middle';ctx.fillStyle=options.color??'#b2c5cf';ctx.fillText(text,14,64,995);
    const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;
    const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,transparent:true,opacity:options.opacity??.95,depthTest:false}));
    const s=options.scale??1;sprite.scale.set((options.width??3.2)*s,.4*s,1);sprite.position.copy(position);sprite.renderOrder=5;(options.parent??this.content).add(sprite);return sprite;
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
  private resetContent(){this.clear(this.content);this.clear(this.overlay);this.clear(this.tileGroup);this.clear(this.cellLabels);this.clear(this.loadingGroup);this.interactives=[];this.hovered=undefined;this.hoveredCell=undefined;this.hoverElement.style.display='none';this.cellMesh=undefined;this.tilePlane=undefined;this.numericPlane=undefined;this.cellRangeKey='';this.layerCenters.clear();this.tensorMeta=undefined;this.tensorTile=undefined;this.activeMarker.visible=false;this.decor.visible=true;}
  private enter(s:Selection,push=true){if(push&&JSON.stringify(s)!==JSON.stringify(this.selection))this.history.push({...this.selection});this.selection={...s};this.callbacks.onSelect({...s});}
  private fly(position:THREE.Vector3,target:THREE.Vector3,duration=950){this.tween={start:performance.now(),duration,from:this.camera.position.clone(),to:position,targetFrom:this.controls.target.clone(),targetTo:target};}
  private layerData(){
    if(!this.micro)return this.manifest.layers;
    return Array.from({length:this.microLayers},(_,index)=>{const params=this.microParameters.filter(p=>p.name.startsWith(`layers.${index}.`));return {index,kind:index%4===3?'full_attention':'linear_attention',names:params.map(p=>p.name),params:params.reduce((n,p)=>n+p.shape.reduce((a,b)=>a*b,1),0)||1,bytes:0};});
  }
  private meta(name:string):TensorMeta|undefined{
    if(!this.micro)return this.manifest.tensors[name];
    const p=this.microParameters.find(x=>x.name===name);if(!p)return;const params=p.shape.reduce((a,b)=>a*b,1),layer=/layers\.(\d+)/.exec(name);
    return {name,shape:p.shape,dtype:'Float64 [micro]',shard:'micro-worker',dataOffsets:[0,0],absoluteOffsets:[0,0],params,bytes:params*8,component:name.includes('linear_attn')?'deltaNet':name.includes('self_attn')?'attention':name.includes('mlp')?'ffn':'norms',layer:layer?Number(layer[1]):undefined};
  }

  showModel(){
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
    const target=v(.1,this.micro?4.5:5.45,0);this.fly(v(this.micro?13:16,this.micro?10.5:12.3,this.micro?16.5:19),target);
  }

  showLayer(index:number){
    const layer=this.layerData().find(x=>x.index===index);if(!layer)return;
    this.resetContent();this.lod=1;this.enter({kind:'layer',layer:index});this.decor.visible=false;
    const full=layer.kind==='full_attention',mixerColor=full?C.attention:C.delta,groups=[{key:'norm',names:layer.names.filter(n=>n.includes('layernorm')),x:-5.05,title:'NORMALIZE'},{key:'mixer',names:layer.names.filter(n=>n.includes(full?'self_attn':'linear_attn')),x:-.8,title:full?'GATED ATTENTION':'GATED DELTANET'},{key:'ffn',names:layer.names.filter(n=>n.includes('.mlp.')),x:4.9,title:'FEED-FORWARD'}];
    this.label(`LAYER ${String(index).padStart(2,'0')} · ${full?'GATED ATTENTION':'GATED DELTANET'}`,v(0,8.25,0),{color:full?'#bfafff':'#8ae8d9',width:8,scale:1.2});
    this.label(`${count(layer.params)} parameters · ${this.micro?'[micro]':'[config]'} · every block is a named tensor`,v(0,7.7,0),{color:'#73929f',width:8,scale:.9});
    for(const group of groups){
      this.label(group.title,v(group.x,6.9,0),{color:group.key==='ffn'?'#c9b78f':group.key==='mixer'?(full?'#b5a2f0':'#68cdbc'):'#8ca5b1',width:3.2,scale:.9});
      const cols=group.key==='norm'?1:3;
      for(let i=0;i<group.names.length;i++){
        const meta=this.meta(group.names[i]);if(!meta)continue;
        const col=i%cols,row=Math.floor(i/cols),px=group.x+(col-(cols-1)/2)*(group.key==='ffn'?2.1:1.62),py=(group.key==='ffn'?4.55:5.8)-row*1.38,inDim=meta.shape.slice(1).reduce((a,b)=>a*b,1),outDim=meta.shape[0];
        const scale=group.key==='ffn'?2.7/(this.micro?128:17408):1.23/Math.max(inDim,outDim,1),w=Math.max(.2,inDim*scale),h=Math.max(.12,outDim*scale),d=meta.shape.length===1?.13:.24;
        const mesh=this.box(w,h,d,colorFor(meta.name),v(px,py,.12),{kind:'tensor',layer:index,tensor:meta.name});mesh.userData.tensor=meta.name;
        this.label(leaf(meta.name),v(px,py-h/2-.23,.38),{color:'#a5bbc4',width:group.key==='ffn'?3.5:2.7,scale:.7});
        this.label(meta.shape.join(' × '),v(px,py-h/2-.48,.4),{color:'#586f7a',width:2.2,scale:.65});
        if(full&&meta.name.includes('q_proj'))this.headBands(mesh,w,h,this.micro?6:this.manifest.config.text_config.num_attention_heads,true);
        else if(full&&/[kv]_proj/.test(meta.name))this.headBands(mesh,w,h,this.micro?1:this.manifest.config.text_config.num_key_value_heads,false);
      }
      this.line([v(group.x-1.6,.76,-.1),v(group.x+1.6,.76,-.1)],group.key==='mixer'?mixerColor:C.ffn,.25);
    }
    this.arrow(v(-4.25,3.0,.55),v(-3.15,3.0,.55),C.norm);this.arrow(v(1.5,2.35,.55),v(3.15,2.35,.55),mixerColor);
    this.line([v(-5.45,1.2,.3),v(-5.45,.38,.3),v(6.5,.38,.3),v(6.5,1.2,.3)],0x7897a4,.6);this.label('RESIDUAL STREAM  →  preserve input + add learned changes',v(.45,.03,.4),{color:'#76929f',width:8,scale:.86});
    this.label(full?'6 query heads share each key / value head':'State memory is an activation, not a learned weight',v(-.9,1.15,.3),{color:full?'#8f80c0':'#5b9f94',width:5.5,scale:.85});
    if(full){
      const qHeads=this.micro?6:this.manifest.config.text_config.num_attention_heads,kvHeads=this.micro?1:this.manifest.config.text_config.num_key_value_heads,span=3.05,left=-6.15;
      for(let h=0;h<qHeads;h++){const x=left+(h+.5)*span/qHeads,kv=Math.floor(h/(qHeads/kvHeads)),kx=left+(kv+.5)*span/kvHeads;this.line([v(x,2.2,.4),v(kx,1.48,.4)],C.attention,.34);const node=new THREE.Mesh(new THREE.SphereGeometry(.034,8,6),new THREE.MeshBasicMaterial({color:0xcdbdff}));node.position.set(x,2.2,.4);this.content.add(node);}
      for(let h=0;h<kvHeads;h++){const x=left+(h+.5)*span/kvHeads,node=new THREE.Mesh(new THREE.SphereGeometry(.073,10,8),new THREE.MeshBasicMaterial({color:C.attention}));node.position.set(x,1.48,.4);this.content.add(node);}
      this.label(`${qHeads} Q HEADS → ${kvHeads} SHARED KV HEADS`,v(-4.63,1.0,.4),{color:'#8679af',width:3.8,scale:.75});
    }else{
      const p=`layers.${index}.linear_attn`,stateHeads=this.micro?(this.meta(`${p}.A_log`)?.shape[0]??3):this.manifest.config.text_config.linear_num_value_heads,stateDim=this.micro?(this.meta(`${p}.norm.weight`)?.shape[0]??8):this.manifest.config.text_config.linear_value_head_dim;
      this.box(2.45,.84,.15,C.delta,v(-4.85,1.9,.25),undefined,.17);this.label('RECURRENT STATE · ACTIVATION',v(-4.85,2.55,.3),{color:'#75b1a6',width:3.55,scale:.75});this.label(`${stateHeads} heads × ${stateDim} × ${stateDim}`,v(-4.85,1.18,.3),{color:'#548b80',width:3.1,scale:.72});
    }
    this.fly(v(1.5,8.8,18.8),v(.15,3.95,0));
  }
  private headBands(mesh:THREE.Mesh,width:number,height:number,heads:number,queryGate:boolean){
    for(let h=1;h<heads;h++){const y=-height/2+h*height/heads;const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints([v(-width/2,y,.126),v(width/2,y,.126)]),new THREE.LineBasicMaterial({color:0xddd2ff,transparent:true,opacity:.65}));mesh.add(line);}
    if(queryGate)for(let h=0;h<heads;h++){const band=new THREE.Mesh(new THREE.PlaneGeometry(width,height/heads/2),new THREE.MeshBasicMaterial({color:0x291d48,transparent:true,opacity:.33,side:THREE.DoubleSide}));band.position.set(0,-height/2+(h+.25)*height/heads,.125);mesh.add(band);}
  }

  showTensor(meta:TensorMeta,tile?:WeightTile){
    this.resetContent();this.lod=2;this.enter({kind:'tensor',layer:meta.layer,tensor:meta.name});this.tensorMeta=meta;this.tensorTile=tile;this.decor.visible=false;
    this.renderTensor();
    this.fly(v(.2,4.5,12.3),v(0,3.8,0));
  }
  setTile(tile:WeightTile){if(this.tensorMeta&&this.tensorMeta.name===tile.name){this.tensorTile=tile;this.renderTensor();}}
  setLoading(progress:number){
    this.clear(this.loadingGroup);if(progress>=1||progress<0||!this.tensorMeta)return;
    const radius=.34,geometry=new THREE.TorusGeometry(radius,.022,8,64,Math.max(.08,progress)*Math.PI*2),ring=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color:C.delta}));ring.position.copy(this.tileOrigin).add(v(0,-.7,.18));this.loadingGroup.add(ring);
  }
  private renderTensor(){
    const meta=this.tensorMeta;if(!meta)return;this.clear(this.content);this.clear(this.tileGroup);this.clear(this.cellLabels);this.clear(this.loadingGroup);this.interactives=[];this.cellMesh=undefined;this.tilePlane=undefined;this.numericPlane=undefined;this.cellRangeKey='';
    const tile=this.tensorTile,fullRows=meta.shape[0],fullCols=meta.shape.slice(1).reduce((a,b)=>a*b,1),rows=tile?.rows??Math.min(256,fullRows),cols=tile?.cols??Math.min(256,fullCols),width=Math.max(.65,6.6*Math.min(1,cols/rows)),height=Math.max(.65,6.6*Math.min(1,rows/cols));this.tileDimensions={width,height};
    this.box(width+.13,height+.13,.115,0x1d303b,this.tileOrigin.clone().add(v(0,0,-.075)),undefined,.9);
    const texture=tile?this.heatTexture(tile.values,rows,cols,undefined,tile.stats):this.emptyTexture();
    const plane=new THREE.Mesh(new THREE.PlaneGeometry(width,height),new THREE.MeshBasicMaterial({map:texture,side:THREE.DoubleSide}));plane.position.copy(this.tileOrigin);plane.userData.selection={kind:'tensor',layer:meta.layer,tensor:meta.name};this.tilePlane=plane;this.tileGroup.add(plane);this.interactives.push(plane);
    this.label(short(meta.name),v(0,7.95,0),{color:'#d1e2e7',width:9.2});
    this.label(`${meta.shape.join(' × ')} · ${this.micro?'[micro]':'[config]'} shape · ${count(meta.params)} parameters`,v(0,7.53,0),{color:'#8aa4af',width:7.5,scale:.85});
    const startRow=tile?.row??0,startCol=tile?.col??0;
    this.label(tile?`[${tile.source}] TILE · rows ${startRow}–${startRow+rows-1} / ${fullRows} · columns ${startCol}–${startCol+cols-1} / ${fullCols}`:'NOT LOADED — CLICK TO FETCH',v(0,7.13,0),{color:tile?'#75d8c7':'#a5b4b9',width:8.5,scale:.78});
    this.label(`INPUT FEATURES  ${startCol} → ${startCol+cols-1}`,v(0,this.tileOrigin.y-height/2-.34,0),{color:'#87a8b4',width:5.4,scale:.85});
    let rowMeaning='OUTPUT FEATURES';
    if(meta.name.includes('embed'))rowMeaning='TOKEN IDs';else if(meta.name.includes('lm_head'))rowMeaning='VOCABULARY LOGITS';else if(meta.name.includes('q_proj')){const dim=this.micro?16:this.manifest.config.text_config.head_dim,head=Math.floor(startRow/(2*dim)),offset=startRow%(2*dim);rowMeaning=`HEAD ${head} · ${offset>=dim?'GATE':'QUERY'} DIMENSIONS`;}else if(/[kv]_proj/.test(meta.name)){const dim=this.micro?16:this.manifest.config.text_config.head_dim;rowMeaning=`KV HEAD ${Math.floor(startRow/dim)} DIMENSIONS`;}
    this.label(`${rowMeaning}  ${startRow} → ${startRow+rows-1}`,v(0,this.tileOrigin.y-height/2-.67,0),{color:'#70939e',width:7,scale:.78});
    for(const f of [0,.25,.5,.75,1]){
      const row=Math.min(rows-1,Math.floor(f*rows)),col=Math.min(cols-1,Math.floor(f*cols));
      this.label(String(startRow+row),v(-width/2-.36,this.tileOrigin.y+height/2-f*height,0),{color:'#65848e',width:.85,scale:.8});
      this.label(String(startCol+col),v(-width/2+f*width,this.tileOrigin.y+height/2+.18,0),{color:'#65848e',width:.8,scale:.75});
    }
    if(meta.name.includes('.self_attn.')&&/[qkv]_proj/.test(meta.name))this.tensorHeadRuler(meta,tile);
    if(tile){this.colorLegend(tile);this.label('Scroll closer to resolve individual weights · drag to orbit / pan',v(0,-.66,0),{color:'#567680',width:8,scale:.77});}
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
    const isQ=meta.name.includes('q_proj'),dim=this.micro?16:this.manifest.config.text_config.head_dim,heads=this.micro?(isQ?6:1):(isQ?this.manifest.config.text_config.num_attention_heads:this.manifest.config.text_config.num_key_value_heads),band=dim*(isQ?2:1),height=6.6,w=.23,x=this.tileDimensions.width/2+.76;
    for(let h=0;h<heads;h++){const y=3.8+height/2-(h+.5)*height/heads,active=tile&&tile.row<((h+1)*band)&&tile.row+tile.rows>h*band;
      const mesh=this.box(w,height/heads*.94,.025,active?0xd4b8ff:0x5d537f,v(x,y,.02),{kind:'tensor',layer:meta.layer,tensor:meta.name,row:h*band,col:tile?.col??0},active?1:.5);mesh.userData.head=h;
      if(isQ){const gate=new THREE.Mesh(new THREE.PlaneGeometry(w*.45,height/heads*.94),new THREE.MeshBasicMaterial({color:0x30223e}));gate.position.set(w*.25,0,.014);mesh.add(gate);}
      if(heads<=8||h%4===0||active)this.label(`${h}`,v(x+.38,y,.02),{color:active?'#d9c3ff':'#685e83',width:.5,scale:.72});
    }
    this.label(isQ?'Q | gate':'KV',v(x,7.4,.02),{color:'#ac92d6',width:1.35,scale:.75});this.label('HEADS',v(x,7.13,.02),{color:'#695981',width:1.15,scale:.7});
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
    if(!this.cellMesh){this.cellMesh=new THREE.InstancedMesh(new THREE.PlaneGeometry(cellWidth*.975,cellHeight*.975),new THREE.MeshBasicMaterial({side:THREE.DoubleSide}),capacity);this.cellMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);this.cellMesh.userData.selection={kind:'cell',tensor:tile.name,layer:this.tensorMeta.layer};this.cellMesh.frustumCulled=false;this.tileGroup.add(this.cellMesh);this.interactives.unshift(this.cellMesh);}
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
  focus(selection:Selection){if(selection.kind==='model')this.showModel();else if(selection.kind==='layer'&&selection.layer!==undefined)this.showLayer(selection.layer);else if(selection.kind==='cell'&&this.tensorTile?.name===selection.tensor&&selection.row!==undefined&&selection.col!==undefined){this.focusCell(selection.row,selection.col);this.callbacks.onSelect(selection);}else if(selection.tensor){const meta=this.meta(selection.tensor);if(meta)this.showTensor(meta);}}
  back(){const previous=this.history.pop();if(!previous){if(this.selection.kind!=='model')this.showModel();return;}const remaining=[...this.history];this.focus(previous);this.history=remaining;}
  setMicro(enabled:boolean,parameters?:{name:string;shape:number[]}[]){if(parameters)this.microParameters=parameters;const changed=this.micro!==enabled;this.micro=enabled;if(parameters){const indices=parameters.flatMap(p=>{const match=/layers\.(\d+)/.exec(p.name);return match?[Number(match[1])]:[];});if(indices.length)this.microLayers=Math.max(...indices)+1;}if(changed)this.showModel();}
  setFollow(enabled:boolean){this.autoFollow=enabled;}
  setActiveStep(step:StepLike){
    const previousLayer=this.followStep?.layer;this.followStep=step;this.clear(this.activeMarker);this.clear(this.overlay);
    const layer=step.layer??-1,pos=this.layerCenters.get(layer)??(this.selection.kind==='model'?(layer<0?v(-1.1,.48,0):v(-1.1,10,0)):undefined);
    if(pos){
      const ring=new THREE.Mesh(new THREE.BoxGeometry(5.7,.12,3.35),new THREE.MeshBasicMaterial({color:step.phase==='backward'?0xe19bb1:step.phase==='update'?0xffcd8b:0x9affdf,transparent:true,opacity:.19,depthWrite:false}));ring.position.copy(pos);this.activeMarker.add(ring);
      const line=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(5.72,.14,3.38)),new THREE.LineBasicMaterial({color:step.phase==='backward'?0xe8a4c8:0xaeffea,transparent:true,opacity:.95}));line.position.copy(pos);this.activeMarker.add(line);this.activeMarker.visible=true;
      if(this.autoFollow&&this.micro&&previousLayer!==layer)this.fly(v(11,pos.y+5.5,14),v(.2,Math.max(3,pos.y+.2),0),550);
    }
    if(this.selection.kind==='tensor'||this.selection.kind==='cell')return;
    let matrix=step.matrix;
    if(!matrix&&step.outputs?.[0]){const output=step.outputs[0],cols=output.shape.length>1?output.shape.slice(1).reduce((a,b)=>a*b,1):output.values.length,rows=output.shape.length>1?output.shape[0]:1;matrix={shape:[rows,cols],values:output.values};}
    if(matrix&&matrix.shape.length>=1&&matrix.values.length){
      const realCols=matrix.shape.at(-1)??1,realRows=matrix.shape.length>1?matrix.shape[0]:1,cols=Math.min(realCols,16),rows=Math.min(realRows,16),values:number[]=[],mask:boolean[]=[];
      for(let row=0;row<rows;row++)for(let col=0;col<cols;col++){values.push(matrix.values[row*realCols+col]??NaN);mask.push(matrix.mask?.[row*realCols+col]??false);}
      const texture=this.heatTexture(values,rows,cols,mask,undefined,true),sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,transparent:true,depthTest:false})),anchor=this.selection.kind==='model'?v(4.8,Math.max(3,pos?.y??4),2):v(1.2,3,2);
      sprite.scale.set(3.35,3.35*rows/cols,1);sprite.position.copy(anchor);sprite.renderOrder=8;this.overlay.add(sprite);
      this.label(`[micro] ${step.name??'Computed activation'}`,anchor.clone().add(v(0,3.35*rows/cols/2+.43,0)),{color:'#c2eee5',width:5.2,scale:.8,parent:this.overlay});
      this.label(`${matrix.shape.join(' × ')}${realRows>rows||realCols>cols?' · top-left window':''} · actual computed values`,anchor.clone().add(v(0,-3.35*rows/cols/2-.3,0)),{color:'#75a59d',width:4.6,scale:.75,parent:this.overlay});
      if(pos)this.line([pos.clone().add(v(2.85,0,1.6)),anchor.clone().add(v(-1.65,0,0))],step.phase==='backward'?0xdc96b8:0x75dfca,.7,this.overlay);
    }
  }
  private hits(event:MouseEvent){const rect=this.renderer.domElement.getBoundingClientRect();this.pointer.set((event.clientX-rect.left)/rect.width*2-1,-((event.clientY-rect.top)/rect.height*2-1));this.raycaster.setFromCamera(this.pointer,this.camera);return this.raycaster.intersectObjects(this.interactives,false);}
  private hitCell(hit:THREE.Intersection){
    const tile=this.tensorTile;if(!tile)return;
    if(hit.object===this.cellMesh&&hit.instanceId!==undefined)return this.cellMesh.userData.cells?.[hit.instanceId] as {row:number;col:number}|undefined;
    if(hit.object===this.tilePlane&&hit.uv){const col=Math.min(tile.cols-1,Math.floor(hit.uv.x*tile.cols)),row=Math.min(tile.rows-1,Math.floor((1-hit.uv.y)*tile.rows));return {row:tile.row+row,col:tile.col+col};}
  }
  private onPointerMove(event:PointerEvent){
    const hit=this.hits(event)[0],mesh=hit?.object as HitMesh|undefined;
    if(this.hovered!==mesh){if(this.hovered&&this.hovered.material instanceof THREE.MeshStandardMaterial)this.hovered.material.emissiveIntensity=.045;this.hovered=mesh;if(mesh?.material instanceof THREE.MeshStandardMaterial)mesh.material.emissiveIntensity=.22;this.renderer.domElement.style.cursor=mesh?'pointer':'grab';}
    const cell=hit?this.hitCell(hit):undefined;
    if(cell&&this.tensorTile){const tile=this.tensorTile,value=tile.values[(cell.row-tile.row)*tile.cols+cell.col-tile.col];this.hoverElement.textContent=`[${tile.source}] w[${cell.row}][${cell.col}] = ${value.toPrecision(8)}\nMultiplies input feature ${cell.col}\nContributes to output feature ${cell.row}`;const rect=this.container.getBoundingClientRect();this.hoverElement.style.left=`${Math.min(rect.width-345,Math.max(5,event.clientX-rect.left+18))}px`;this.hoverElement.style.top=`${Math.min(rect.height-130,Math.max(5,event.clientY-rect.top+18))}px`;this.hoverElement.style.display='block';this.hoveredCell=`${cell.row}:${cell.col}`;}
    else{this.hoverElement.style.display='none';this.hoveredCell=undefined;}
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
    this.controls.update();this.updateCellDetail(now);this.autoDetail(now);this.renderer.render(this.scene,this.camera);this.measuredFrames++;
    this.slowFor=dt>20?this.slowFor+dt:0;if(this.slowFor>1000&&this.quality>.65){this.quality=.65;this.renderer.setPixelRatio(Math.min(window.devicePixelRatio,1));this.slowFor=0;}
    if(now-this.measuredAt>750){const fps=this.measuredFrames*1000/(now-this.measuredAt);this.callbacks.onStats?.({fps,renderer:'WebGL2',lod:this.lod,instances:this.cellMesh?.count??this.interactives.length});this.measuredAt=now;this.measuredFrames=0;}
  };
  dispose(){this.disposed=true;cancelAnimationFrame(this.frame);this.observer.disconnect();this.controls.dispose();const canvas=this.renderer.domElement;canvas.removeEventListener('pointermove',this.pointerMove);canvas.removeEventListener('pointerdown',this.pointerDown);canvas.removeEventListener('pointerup',this.pointerUp);canvas.removeEventListener('dblclick',this.doubleClick);window.removeEventListener('keydown',this.keyDown);this.clear(this.content);this.clear(this.decor);this.clear(this.overlay);this.clear(this.activeMarker);this.clear(this.tileGroup);this.clear(this.cellLabels);this.clear(this.loadingGroup);this.renderer.dispose();canvas.remove();this.hoverElement.remove();}
}
