import type {TensorMeta} from '../data/hf';
export const esc=(s:unknown)=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));
export const count=(n:number)=>n>=1e9?`${(n/1e9).toFixed(2)}B`:n>=1e6?`${(n/1e6).toFixed(2)}M`:n>=1e3?`${(n/1e3).toFixed(1)}K`:String(n);
export const numeric=(n:number)=>!Number.isFinite(n)?(n<0?'−∞':'∞'):Math.abs(n)<.0001&&n!==0?n.toExponential(3):n.toFixed(5);
export const components:Record<string,{label:string;color:string}>={
  ffn:{label:'Feed-forward',color:'#c9b48a'},delta:{label:'DeltaNet',color:'#67d7bd'},attention:{label:'Full attention',color:'#ae9df2'},embedding:{label:'Embedding',color:'#78b9e0'},lm_head:{label:'Language head',color:'#78b9e0'},vision:{label:'Vision encoder',color:'#8c9cad'},mtp:{label:'Multi-token head',color:'#b8a3ca'},norm:{label:'Normalization',color:'#8994a3'},norms:{label:'Normalization',color:'#8994a3'}
};
export function describeTensor(t:TensorMeta) {
  const n=t.name;
  let title='Learned parameter',what='A tensor of learned numbers that controls one part of the model.',why='Training adjusts these numbers to improve predictions.',math='The surrounding operation determines how these numbers transform a signal.',analogy='One precisely shaped part inside a larger machine.';
  if(n.includes('embed_tokens')){title='Token embedding';what='Turns a token ID into a learned vector of features.';why='The model needs numerical features to work with words and word pieces.';math='Read embedding[token ID, :].';analogy='A dictionary whose definitions are coordinates.';}
  else if(n.includes('lm_head')){title='Language-model head';what='Scores every vocabulary token from the final hidden vector.';why='A score for every token lets the model choose what comes next.';math='logits = hidden × Wᵀ; softmax turns scores into probabilities.';analogy='A voting panel with one score for each possible next word.';}
  else if(n.includes('q_proj')){title='Queries + output gates';what='Creates a query and an output gate for each attention head.';why='The query finds relevant earlier tokens; the gate controls the result.';math='Each head band packs query rows followed by gate rows.';analogy='A question and a volume knob for its answer.';}
  else if(n.includes('k_proj')){title='Key projection';what='Creates the features that attention queries compare against.';why='Keys make each earlier token discoverable by its context.';math='K = normalized hidden × Wᵀ.';analogy='Searchable labels attached to earlier words.';}
  else if(n.includes('v_proj')){title='Value projection';what='Creates the information that attention can retrieve.';why='Matching a key is useful only if there is content to read.';math='V = normalized hidden × Wᵀ.';analogy='The content inside a labelled drawer.';}
  else if(n.includes('in_proj_qkv')){title='DeltaNet Q, K and V';what='Projects queries, keys and values into a compact memory system.';why='The model can update a fixed-size state instead of comparing every pair of tokens.';math='Project → causal convolution → SiLU → split Q / K / V.';analogy='A question, an address, and a note to store there.';}
  else if(n.includes('in_proj_z')){title='DeltaNet output gate';what='Controls which parts of a memory read reach the residual stream.';why='Useful memory should pass through selectively.';math='normalized readout × SiLU(z).';analogy='A set of dimmer switches on remembered features.';}
  else if(n.includes('in_proj_a')||n.includes('A_log')||n.includes('dt_bias')){title='Memory decay';what='Controls how quickly DeltaNet forgets older state.';why='Different information needs different memory lifetimes.';math='g = −exp(A_log) × softplus(a + dt_bias); state decay = exp(g).';analogy='An adjustable fading rate for each memory head.';}
  else if(n.includes('in_proj_b')){title='Memory write gate';what='Controls how strongly the delta correction updates memory.';why='The model should not overwrite every memory with equal force.';math='β = sigmoid(b); correction = β × (value − prediction).';analogy='The pressure of a pencil when correcting a note.';}
  else if(n.includes('conv1d')){title='Short causal convolution';what='Mixes each feature with its recent token history.';why='Local word order matters before writing to recurrent memory.';math='A learned depthwise kernel reads only the current and earlier positions.';analogy='A small sliding window over the last few words.';}
  else if(n.includes('gate_proj')){title='Feed-forward gate';what='Selects which expanded features should pass through.';why='A nonlinear gate lets the model represent more than a simple weighted sum.';math='SiLU(x × W_gateᵀ) × (x × W_upᵀ).';analogy='A switchboard choosing which feature detectors are active.';}
  else if(n.includes('up_proj')){title='Feed-forward expansion';what='Expands the residual vector into a wider feature space.';why='More intermediate features give the model room to express learned patterns.';math='up = x × W_upᵀ.';analogy='Spreading parts over a larger workbench.';}
  else if(n.includes('down_proj')){title='Feed-forward compression';what='Projects the gated features back to residual width.';why='The branch must fit the residual stream before the two are added.';math='output = gated features × W_downᵀ.';analogy='Assembling the useful parts back into one package.';}
  else if(n.includes('o_proj')||n.includes('out_proj')){title='Mixer output projection';what='Combines the heads into a residual-width vector.';why='Each head contributes a different view to a shared representation.';math='output = combined heads × Wᵀ.';analogy='An editor combining several reporters’ findings.';}
  else if(n.includes('norm')){title='Learned normalization scale';what='Keeps signal magnitude controlled and learns a scale for each feature.';why='Stable magnitudes make deep stacks easier to train.';math=n.includes('linear_attn.norm')?'y = x / RMS(x) × w; then apply the SiLU output gate.':'y = x / RMS(x) × (1 + w).';analogy='Level controls that keep a long audio chain balanced.';}
  else if(n.includes('visual')){title='Vision-encoder parameter';what='Transforms image patches into features for the language model.';why='Images need a learned representation before text and visual information can interact.';math='The exact operation depends on this tensor’s module; the stored shape is shown below.';analogy='A translator from image patches to the model’s internal language.';}
  else if(n.includes('mtp')){title='Multi-token prediction parameter';what='Belongs to the checkpoint’s auxiliary future-token prediction module.';why='It provides an additional learning signal about future tokens.';math='This real tensor is inspectable; Micro-Qwen omits the MTP module.';analogy='Practising a few steps ahead while learning a sequence.';}
  return {title,what,why,math,analogy};
}
export const glossary:[string,string][]=[
 ['Parameter','A learned number, such as a connection weight. Training changes it; ordinary inference does not.'],
 ['Token','A piece of text represented by an ID. Real Qwen uses subword pieces; Micro-Qwen uses whole words.'],
 ['Embedding','A learned vector read from the row for a token ID. Its coordinates describe features useful for prediction.'],
 ['Logit','An unnormalized score for a possible next token. A larger score usually becomes a larger probability.'],
 ['Softmax','Turns a list of scores into positive probabilities that sum to one. It gives more probability to larger scores.'],
 ['Attention head','A learned way to compare the current token with earlier tokens and read their values. Different heads can retrieve different context.'],
 ['KV cache','Stored keys and values from earlier tokens. It avoids recomputing those projections during full-attention generation.'],
 ['Residual','The running hidden representation plus a branch’s contribution. Addition gives information and gradients a direct path through a deep model.'],
 ['RMSNorm','Divides a vector by its root-mean-square magnitude, then applies learned feature scales. This keeps signal size controlled.'],
 ['Gradient','The local sensitivity of the loss to a number. Its sign and size tell training which small change could reduce the loss.'],
 ['Learning rate','The size of a training update. A smaller value takes more cautious steps; an excessive value can destabilize learning.'],
 ['Loss','A numerical penalty for a prediction. Here it is the negative log probability of the correct next token, averaged over targets.'],
 ['GQA','Grouped-query attention: several query heads share one key/value head. This reduces the key/value state that must be stored.'],
 ['RoPE','Rotary position embedding rotates part of each query and key according to token position. Attention can then take word order into account.'],
 ['Linear attention','Attention that maintains a compact state instead of a growing matrix of token pairs. DeltaNet uses a recurrent update of that state.'],
 ['Delta rule','Correct a memory using the difference between a new value and what its key already predicts. The write gate controls the strength of that correction.'],
 ['SiLU','A smooth activation x × sigmoid(x), also called swish. It is used by the feed-forward and DeltaNet output gates.'],
 ['Fine-tuning','Continue training from existing learned weights on a selected dataset. It changes some or all parameters, while inference only uses them.']
];
export const tourStops=[
 {title:'A model is a collection of learned numbers',body:'The tower is the actual checkpoint’s structure. Every slab contains tensors: arrays of parameters. The colours tell you what each part does.',action:'model'},
 {title:'Start with a token’s embedding',body:'Text is split into token IDs. An embedding lookup reads one learned row per ID. Open the real tokenizer below to compare subwords with Micro-Qwen’s words.',action:'embedding'},
 {title:'Memory without an ever-growing score matrix',body:'Three of every four layers use DeltaNet. Its state decays, predicts a value, and writes the prediction error back into memory.',action:'delta'},
 {title:'Six queries share one set of keys and values',body:'Every fourth layer uses full attention. Queries compare with earlier keys and retrieve weighted values. Query bands also contain output gates.',action:'attention'},
 {title:'Most parameters live in the feed-forward blocks',body:'The feed-forward branch expands a token’s features, gates them with SiLU, and projects back down. The wide matrices account for most of this checkpoint.',action:'ffn'},
 {title:'The residual stream carries the working representation',body:'Each branch adds a contribution to the same running vector. Normalization controls its magnitude before each branch.',action:'residual'},
 {title:'Turn features into a next-word distribution',body:'Switch to the actual Micro-Qwen simulation. The language head produces scores for every micro word; softmax converts them into probabilities.',action:'inference'},
 {title:'Sampling chooses one possible continuation',body:'Temperature zero picks the highest probability. A positive temperature makes a seeded random draw. The displayed probabilities come from the working model.',action:'sampling'},
 {title:'Learning starts with an error signal',body:'One real Micro-Qwen training step compares predictions with targets. Negative log probability becomes the loss. Playback then traces gradients backward.',action:'training'},
 {title:'A gradient explains a small change',body:'The chain rule traces the loss through each dependency. The update subtracts the learning rate times the clipped gradient from every weight.',action:'gradient'},
 {title:'Billions of numbers need billions of bytes',body:'BF16 uses two bytes per parameter. The checkpoint is about 55.56 GB. This explorer downloads selected tiles within a 150 MB budget.',action:'memory'},
 {title:'Fine-tuning changes weights; inference changes state',body:'Training changes learned parameters. Inference leaves those weights fixed while activations and DeltaNet memory change. Reset the micro weights to watch the difference yourself.',action:'finish'}
];
