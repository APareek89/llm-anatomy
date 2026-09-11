export class ModelWorker {
  private worker=new Worker(new URL('../model/micro/worker.ts',import.meta.url),{type:'module'});
  private id=0;
  private requests=new Map<number,{resolve:(value:any)=>void;reject:(error:Error)=>void;progress?:(message:any)=>void}>();
  constructor(){
    this.worker.onmessage=event=>{
      const m=event.data,request=this.requests.get(m.id);if(!request)return;
      if(m.type==='progress'){request.progress?.(m);return;}
      this.requests.delete(m.id);
      if(m.type==='error')request.reject(new Error(m.error));else request.resolve(m);
    };
    this.worker.onerror=event=>{for(const r of this.requests.values())r.reject(new Error(event.message||'Model worker failed'));this.requests.clear();};
  }
  call(type:string,options:Record<string,unknown>={},progress?:(message:any)=>void):Promise<any>{
    const id=++this.id;
    return new Promise((resolve,reject)=>{this.requests.set(id,{resolve,reject,progress});this.worker.postMessage({id,type,...options});});
  }
}
