import {describeTensor} from './content';
import type {TensorMeta} from '../data/hf';

export type Family='qwen'|'gpt-oss'|'llama'|'gemma';
export const familyOrder:Family[]=['qwen','gpt-oss','llama','gemma'];
export const familyInfo={
  qwen:{name:'Qwen3.8',mini:'Micro-Qwen',organization:'Qwen',short:'Hybrid memory',accent:'#72dbc2',title:'A memory that changes.<br>A model that learns.',summary:'Recurrent DeltaNet memory meets full attention.',features:['3 recurrent layers, then 1 full-attention layer','Gated delta-rule memory and partial rotary positions','A dense gated feed-forward block in every layer']},
  'gpt-oss':{name:'gpt-oss',mini:'Micro-gpt-oss',organization:'OpenAI',short:'Mixture of experts',accent:'#e6b979',title:'Different experts.<br>One next-word decision.',summary:'A learned router selects a few expert networks for each token.',features:['Sparse expert routing replaces the dense feed-forward block','Alternating local and full attention with learned attention sinks','Only selected experts contribute to each token’s result']},
  llama:{name:'Llama 3.2',mini:'Micro-Llama',organization:'Meta',short:'Dense transformer',accent:'#8ebcff',title:'The dense transformer.<br>See every connection.',summary:'A clear baseline: every layer uses full attention and a dense feed-forward network.',features:['Full causal grouped-query attention in every layer','Rotary positions, RMSNorm and SwiGLU feature gates','Shared embedding and language-head weights']},
  gemma:{name:'Gemma 3',mini:'Micro-Gemma',organization:'Google',short:'Local + global',accent:'#d6a2ec',title:'Look nearby.<br>Then see the whole context.',summary:'Frequent local windows keep attention focused; a global layer reconnects the context.',features:['Five local-attention layers, then one global layer','Query/key normalization and normalization after each branch','GELU feature gates and scaled, shared token embeddings']}
} satisfies Record<Family,{name:string;mini:string;organization:string;short:string;accent:string;title:string;summary:string;features:string[]}>;
export interface FamilyReference {name:string;organization:string;repo:string;revision:string;config:Record<string,any>;features:string[];sources:{title:string;url:string}[];limitations:string[];license:string;configStatus?:string;geometry?:Record<string,any>;}
export type FamilyReferences=Partial<Record<Family,FamilyReference>>;

export function layerKind(family:Family,index:number){return family==='qwen'?(index%4===3?'Full attention':'DeltaNet'):family==='llama'?'Full attention':family==='gemma'?(index%6===5?'Global attention':'Local attention'):(index%2===0?'Local attention + MoE':'Full attention + MoE');}

export function familyTensor(meta:TensorMeta,family:Family){
  const d=describeTensor(meta),n=meta.name;if(family==='qwen')return d;
  if(n.endsWith('.bias'))return {title:'Learned bias',what:'Adds a learned offset to a projected feature.',why:'A learned baseline need not depend on the current input.',math:'output = input × Wᵀ + bias.',analogy:'Setting a baseline before reacting to the signal.'};
  if(n.includes('embed_tokens')&&(family==='llama'||family==='gemma'))return {title:'Shared embedding and output head',what:'The same matrix turns input IDs into vectors and turns final vectors into vocabulary scores.',why:'Sharing reduces parameters and lets both input and output learning improve one representation.',math:family==='gemma'?'input = embedding[id] × √hidden; logits = final_hidden × embeddingᵀ.':'input = embedding[id]; logits = final_hidden × embeddingᵀ.',analogy:'One dictionary used for both reading and writing.'};
  if(n.includes('q_proj'))return {title:'Query projection',what:'Creates one query vector per attention head.',why:'Queries determine which earlier token information to retrieve.',math:family==='gpt-oss'?'Q = normalized hidden × Wᵀ + bias. Each head has its own row band.':'Q = normalized hidden × Wᵀ. Each head has its own row band.',analogy:'Several different questions asked about the same context.'};
  if(n.includes('router'))return {title:'Expert router',what:'Scores the experts for each token and selects the highest-scoring few.',why:'Sparse routing gives different tokens different computational paths.',math:'scores = x × W_routerᵀ + bias; select top-k; softmax selected scores.',analogy:'A dispatcher choosing the specialists for each request.'};
  if(n.includes('sinks'))return {title:'Attention sinks',what:'Adds a learned no-value position to attention’s probability denominator.',why:'A head can allocate probability mass without reading an earlier token.',math:'denominator = exp(sink) + Σ exp(allowed attention scores).',analogy:'An abstain option in a vote over past words.'};
  if(n.includes('norm'))return {...d,math:family==='gemma'?'y = x / RMS(x) × (1 + w).':'y = x / RMS(x) × w.'};
  if(n.includes('gate_proj')&&family==='gemma')return {...d,math:'GELU_tanh(x × W_gateᵀ) × (x × W_upᵀ).',what:'Uses a smooth GELU gate to select expanded features.'};
  if(n.includes('gate_proj')&&family==='gpt-oss')return {...d,math:'g = min(gate, 7); u = clip(up, −7, 7); output = g × sigmoid(1.702g) × (u + 1).',what:'Controls the features inside one selected expert.'};

  return d;
}

export function familyTour(family:Family){
  const f=familyInfo[family];return [
    {title:`Inside ${f.name}`,body:`${f.summary} The 3D model is a trained miniature built for inspection; it does not contain ${f.organization}’s pretrained weights.`,action:'model'},
    {title:'The numbers are computed',body:'Each visible micro weight belongs to the working model. Inference multiplies those weights to produce predictions. Nothing in the prediction bars is scripted.',action:'embedding'},
    {title:family==='gpt-oss'?'A router chooses specialists':family==='gemma'?'A short attention window':'Every layer reads the past',body:f.features[0],action:'distinctive'},
    {title:'Inspect a real calculation',body:'Follow the first attention block. Its score matrix shows which earlier positions can contribute. Gray cells are excluded by the mask.',action:'attention'},
    {title:'Different feed-forward paths',body:family==='gpt-oss'?'Follow the selected expert routes and their combined output. The recorded routes come from actual router scores.':family==='gemma'?'The GELU-gated feed-forward block transforms each token’s features. Both branch outputs are normalized before residual addition.':'A dense SwiGLU block expands, gates and compresses features for every token.',action:'ffn'},
    {title:'The next word comes from the weights',body:'The final hidden vector produces vocabulary scores. Softmax turns them into probabilities; temperature controls how a token is selected.',action:'inference'},
    {title:'Watch one training step',body:'The model compares its prediction with the corpus target, calculates gradients, then changes the weights. This is actual worker computation.',action:'training'},
    {title:'Follow the gradient',body:'Inspect the raw batch gradient and the exact before/after update. Reset weights to seed 42 to reproduce learning from scratch.',action:'gradient'},
    {title:'Compare the four families',body:'Switch tabs to compare recurrent memory, sparse expert routing, dense attention and local/global attention. Each family keeps its own trained state during this session.',action:'finish'}
  ];
}
