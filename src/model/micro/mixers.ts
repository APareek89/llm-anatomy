import {Tensor,Tape,fmt,view,type RecordedStep} from './tensor';
export function causalConv(t:Tape,x:Tensor,w:Tensor,kernel:number) {
  const [rows,dim]=x.shape,out=new Tensor(x.shape,undefined,'Causal depthwise convolution');
  for(let r=0;r<rows;r++)for(let d=0;d<dim;d++)for(let lag=0;lag<kernel&&lag<=r;lag++)out.data[r*dim+d]+=x.data[(r-lag)*dim+d]*w.data[d*kernel+kernel-1-lag];
  return t.add(out,[x,w],()=>{for(let r=0;r<rows;r++)for(let d=0;d<dim;d++)for(let lag=0;lag<kernel&&lag<=r;lag++){const i=(r-lag)*dim+d,j=d*kernel+kernel-1-lag;x.grad[i]+=out.grad[r*dim+d]*w.data[j];w.grad[j]+=out.grad[r*dim+d]*x.data[i];}},out.name,'y[t,d] = Σₗ x[t−l,d] conv[d,K−1−l]','Mix each feature with its recent history without reading future words.',`${fmt(x.data[0])} × ${fmt(w.data[kernel-1])} = ${fmt(out.data[0])}; earlier positions are zero-padded`);
}
export function l2norm(t:Tape,x:Tensor,dim:number,name:string) {
  const out=new Tensor(x.shape,undefined,name), inv=new Float64Array(x.length/dim);
  for(let r=0;r<inv.length;r++){let sq=0;for(let d=0;d<dim;d++)sq+=x.data[r*dim+d]**2;inv[r]=1/Math.sqrt(sq+1e-6);for(let d=0;d<dim;d++)out.data[r*dim+d]=x.data[r*dim+d]*inv[r];}
  return t.add(out,[x],()=>{for(let r=0;r<inv.length;r++){let dot=0;for(let d=0;d<dim;d++)dot+=out.grad[r*dim+d]*x.data[r*dim+d];for(let d=0;d<dim;d++){const i=r*dim+d;x.grad[i]+=out.grad[i]*inv[r]-x.data[i]*inv[r]**3*dot;}}},name,'q̂ = q / √(Σ q² + 10⁻⁶)','Normalize the query/key length so the recurrent memory update stays controlled.',`${fmt(x.data[0])} × ${fmt(inv[0])} = ${fmt(out.data[0])}`);
}
export function rope(t:Tape,x:Tensor,heads:number,dim:number,fraction:number,theta:number) {
  const rows=x.shape[0],rot=Math.floor(dim*fraction/2)*2,out=new Tensor(x.shape,x.data,'Partial rotary position embedding');
  for(let r=0;r<rows;r++)for(let h=0;h<heads;h++)for(let d=0;d<rot/2;d++){const a=r*heads*dim+h*dim+d,b=a+rot/2,angle=r/Math.pow(theta,2*d/Math.max(1,rot)),c=Math.cos(angle),s=Math.sin(angle);out.data[a]=x.data[a]*c-x.data[b]*s;out.data[b]=x.data[b]*c+x.data[a]*s;}
  return t.add(out,[x],()=>{for(let r=0;r<rows;r++)for(let h=0;h<heads;h++){const base=r*heads*dim+h*dim;for(let d=0;d<rot/2;d++){const a=base+d,b=a+rot/2,angle=r/Math.pow(theta,2*d/Math.max(1,rot)),c=Math.cos(angle),s=Math.sin(angle);x.grad[a]+=out.grad[a]*c+out.grad[b]*s;x.grad[b]+=out.grad[b]*c-out.grad[a]*s;}for(let d=rot;d<dim;d++)x.grad[base+d]+=out.grad[base+d];}},out.name,'[a,b] → [a cos θ − b sin θ, b cos θ + a sin θ]','Rotate part of each head by token position; attention can then distinguish order.',rot?`At token 0, angle = 0: ${fmt(x.data[0])} × 1 − ${fmt(x.data[rot/2])} × 0 = ${fmt(out.data[0])}`:`Rotary dimension = ${rot}; no coordinates rotated in this tiny test configuration`);
}
export function attention(t:Tape,q:Tensor,k:Tensor,v:Tensor,qheads:number,kvheads:number,dim:number) {
  let scoreRecord:RecordedStep|undefined, probabilityRecord:RecordedStep|undefined;
  const n=q.shape[0],out=new Tensor([n,qheads*dim],undefined,'Grouped-query attention output'), probs=new Float64Array(qheads*n*n), scores=new Float64Array(qheads*n*n),scale=1/Math.sqrt(dim),ratio=qheads/kvheads;
  for(let h=0;h<qheads;h++)for(let r=0;r<n;r++){
    const kh=Math.floor(h/ratio),qb=r*qheads*dim+h*dim,base=(h*n+r)*n;let max=-Infinity;
    for(let s=0;s<=r;s++){let score=0;for(let d=0;d<dim;d++)score+=q.data[qb+d]*k.data[s*kvheads*dim+kh*dim+d];score*=scale;scores[base+s]=score;max=Math.max(max,score);}
    for(let s=r+1;s<n;s++)scores[base+s]=-Infinity;
    let total=0;for(let s=0;s<=r;s++){probs[base+s]=Math.exp(scores[base+s]-max);total+=probs[base+s];}
    for(let s=0;s<=r;s++){probs[base+s]/=total;for(let d=0;d<dim;d++)out.data[qb+d]+=probs[base+s]*v.data[s*kvheads*dim+kh*dim+d];}
  }
  if(t.record){const row=Math.min(1,n-1),sc=scores.slice(0,n*n),pp=probs.slice(0,n*n),mask=Array.from({length:n*n},(_,i)=>i%n>Math.floor(i/n));
    t.steps.push({id:t.steps.length,name:'Q · Kᵀ, scaling and causal mask',layer:t.layer,phase:'forward',inputs:[view(q),view(k)],outputs:[{name:'Head 0 scaled causal scores',shape:[n,n],values:Array.from(sc)}],formula:'scores[t,s] = q[t] · k[s] / √d; s > t → −∞',explanation:'Compare each query with earlier keys. Future words are hidden by the causal mask.',arithmetic:`(${Array.from({length:dim},(_,d)=>`${fmt(q.data[row*qheads*dim+d])} × ${fmt(k.data[d])}`).join(' + ')}) / √${dim} = ${fmt(scores[row*n])}`,matrix:{shape:[n,n],values:Array.from(sc),mask}});
    const sum=Array.from(sc.slice(row*n,row*n+row+1)).map(x=>Math.exp(x-Math.max(...sc.slice(row*n,row*n+row+1)))).reduce((a,b)=>a+b,0);
    t.steps.push({id:t.steps.length,name:'Causal attention softmax',layer:t.layer,phase:'forward',inputs:[{name:'Head 0 scores',shape:[n,n],values:Array.from(sc)}],outputs:[{name:'Head 0 probabilities',shape:[n,n],values:Array.from(pp)}],formula:'p[t,s] = exp(score[t,s] − max) / Σ exp(score[t,j] − max)',explanation:'Normalize allowed scores into probabilities that sum to one. Masked positions receive exactly zero.',arithmetic:`exp(${fmt(sc[row*n])} − ${fmt(Math.max(...sc.slice(row*n,row*n+row+1)))}) / ${fmt(sum)} = ${fmt(pp[row*n])}`,matrix:{shape:[n,n],values:Array.from(pp),mask}});
    scoreRecord=t.steps[t.steps.length-2];probabilityRecord=t.steps[t.steps.length-1];
  }
  return t.add(out,[q,k,v],()=>{const dp=new Float64Array(n),gs=t.record?new Float64Array(n*n):undefined,gp=t.record?new Float64Array(n*n):undefined;for(let h=0;h<qheads;h++)for(let r=0;r<n;r++){const kh=Math.floor(h/ratio),qb=r*qheads*dim+h*dim,base=(h*n+r)*n;let dot=0;
    for(let s=0;s<=r;s++){let g=0;for(let d=0;d<dim;d++){const vi=s*kvheads*dim+kh*dim+d;g+=out.grad[qb+d]*v.data[vi];v.grad[vi]+=probs[base+s]*out.grad[qb+d];}dp[s]=g;if(h===0&&gp)gp[r*n+s]=g;dot+=g*probs[base+s];}
    for(let s=0;s<=r;s++){const ds=probs[base+s]*(dp[s]-dot)*scale;if(h===0&&gs)gs[r*n+s]=ds/scale;for(let d=0;d<dim;d++){const ki=s*kvheads*dim+kh*dim+d;q.grad[qb+d]+=ds*k.data[ki];k.grad[ki]+=ds*q.data[qb+d];}}
  }if(scoreRecord&&probabilityRecord&&gs&&gp){scoreRecord.outputs[0].gradient=Array.from(gs);scoreRecord.inputs[0].gradient=Array.from(q.grad);scoreRecord.inputs[1].gradient=Array.from(k.grad);probabilityRecord.outputs[0].gradient=Array.from(gp);probabilityRecord.inputs[0].gradient=Array.from(gs);}},'Attention weighted sum','O[t,h] = Σₛ softmax(QKᵀ/√d)[t,s] V[s,kv(h)]','Each query head reads a weighted mixture of values; several query heads share one key/value head.',`${Array.from({length:n},(_,s)=>`${fmt(probs[(n-1)*n+s])} × ${fmt(v.data[s*kvheads*dim])}`).join(' + ')} = ${fmt(out.data[(n-1)*qheads*dim])}`,t.record?{shape:[n,n],values:Array.from(probs.slice(0,n*n))}:undefined);
}
const softplus=(x:number)=>x>20?x:Math.log1p(Math.exp(x));
export function deltaRule(t:Tape,q:Tensor,k:Tensor,v:Tensor,a:Tensor,b:Tensor,Alog:Tensor,dt:Tensor,keyheads:number,valueheads:number,dk:number,dv:number) {
  const stateRecords:RecordedStep[]=[];
  const n=q.shape[0],size=dk*dv,heads=valueheads,ratio=heads/keyheads,scale=1/Math.sqrt(dk),out=new Tensor([n,heads*dv],undefined,'DeltaNet recurrent read');
  const states=new Float64Array((n+1)*heads*size),decays=new Float64Array(n*heads),betas=new Float64Array(n*heads),deltas=new Float64Array(n*heads*dv),predictions=new Float64Array(n*heads*dv);
  for(let r=0;r<n;r++)for(let h=0;h<heads;h++){
    const kh=Math.floor(h/ratio),qi=r*keyheads*dk+kh*dk,vi=r*heads*dv+h*dv,prev=(r*heads+h)*size,next=((r+1)*heads+h)*size,idx=r*heads+h;
    const g=-Math.exp(Alog.data[h])*softplus(a.data[idx]+dt.data[h]);const decay=Math.exp(g),beta=1/(1+Math.exp(-b.data[idx]));decays[idx]=decay;betas[idx]=beta;
    for(let j=0;j<dv;j++){let pred=0;for(let i=0;i<dk;i++)pred+=k.data[qi+i]*states[prev+i*dv+j]*decay;predictions[vi+j]=pred;const delta=beta*(v.data[vi+j]-pred);deltas[vi+j]=delta;for(let i=0;i<dk;i++){states[next+i*dv+j]=states[prev+i*dv+j]*decay+k.data[qi+i]*delta;out.data[vi+j]+=q.data[qi+i]*scale*states[next+i*dv+j];}}
    if(t.record&&h===0){const before=Array.from(states.slice(prev,prev+size)),after=Array.from(states.slice(next,next+size));t.steps.push({id:t.steps.length,name:`Delta rule state update · token ${r+1}`,layer:t.layer,phase:'forward',token:r,inputs:[{name:'Previous state S',shape:[dk,dv],values:before},{name:'Key k',shape:[dk],values:Array.from(k.data.slice(qi,qi+dk))},{name:'Value v',shape:[dv],values:Array.from(v.data.slice(vi,vi+dv))},{name:'Decay and write gate',shape:[2],values:[decay,beta]}],outputs:[{name:'Updated state S',shape:[dk,dv],values:after}],formula:'S̄ = exp(g) S; δ = β(v − kᵀS̄); S′ = S̄ + k ⊗ δ; y = qᵀS′ / √dₖ',explanation:'Decay old memory, predict the value from its key, then write only the prediction error into memory.',arithmetic:`S′[0,0] = ${fmt(decay)} × ${fmt(before[0])} + ${fmt(k.data[qi])} × ${fmt(beta)} × (${fmt(v.data[vi])} − ${fmt(predictions[vi])}) = ${fmt(after[0])}`,matrix:{shape:[dk,dv],values:after}});stateRecords[r]=t.steps[t.steps.length-1];}
  }
  return t.add(out,[q,k,v,a,b,Alog,dt],()=>{
    const ds=new Float64Array(heads*size),ddelta=new Float64Array(dv),dbar=new Float64Array(size);
    for(let r=n-1;r>=0;r--)for(let h=0;h<heads;h++){
      const kh=Math.floor(h/ratio),qi=r*keyheads*dk+kh*dk,vi=r*heads*dv+h*dv,prev=(r*heads+h)*size,next=((r+1)*heads+h)*size,idx=r*heads+h,dsbase=h*size,decay=decays[idx],beta=betas[idx];
      for(let i=0;i<dk;i++)for(let j=0;j<dv;j++){q.grad[qi+i]+=out.grad[vi+j]*states[next+i*dv+j]*scale;ds[dsbase+i*dv+j]+=out.grad[vi+j]*q.data[qi+i]*scale;}
      if(t.record&&h===0)stateRecords[r].outputs[0].gradient=Array.from(ds.slice(dsbase,dsbase+size));
      ddelta.fill(0);for(let i=0;i<dk;i++)for(let j=0;j<dv;j++){const g=ds[dsbase+i*dv+j];ddelta[j]+=k.data[qi+i]*g;k.grad[qi+i]+=g*deltas[vi+j];dbar[i*dv+j]=g;}
      let dbeta=0;for(let j=0;j<dv;j++){dbeta+=ddelta[j]*(v.data[vi+j]-predictions[vi+j]);v.grad[vi+j]+=ddelta[j]*beta;const dp=-ddelta[j]*beta;for(let i=0;i<dk;i++){k.grad[qi+i]+=dp*states[prev+i*dv+j]*decay;dbar[i*dv+j]+=dp*k.data[qi+i];}}
      let ddecay=0;for(let i=0;i<size;i++){ddecay+=dbar[i]*states[prev+i];ds[dsbase+i]=dbar[i]*decay;}
      if(t.record&&h===0){stateRecords[r].inputs[0].gradient=Array.from(ds.slice(dsbase,dsbase+size));stateRecords[r].inputs[3].gradient=[ddecay,dbeta];}
      b.grad[idx]+=dbeta*beta*(1-beta);const dg=ddecay*decay,expA=Math.exp(Alog.data[h]),z=a.data[idx]+dt.data[h],da=-dg*expA/(1+Math.exp(-z));a.grad[idx]+=da;dt.grad[h]+=da;Alog.grad[h]+=-dg*expA*softplus(z);
    }
    if(t.record)for(let r=0;r<n;r++){stateRecords[r].inputs[1].gradient=Array.from(k.grad.slice(r*keyheads*dk,r*keyheads*dk+dk));stateRecords[r].inputs[2].gradient=Array.from(v.grad.slice(r*heads*dv,r*heads*dv+dv));}
  },'DeltaNet state read','yₜ = q̂ₜᵀ Sₜ / √dₖ','Read the updated state with the normalized query. Each value head maintains its own learned associative memory.',`${Array.from({length:dk},(_,i)=>`${fmt(q.data[i])} × ${fmt(states[heads*size+i*dv])}`).join(' + ')}; divide by √${dk} → ${fmt(out.data[0])}`,t.record?{shape:[dk,dv],values:Array.from(states.slice(n*heads*size,n*heads*size+size))}:undefined);
}
