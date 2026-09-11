export const CORPUS = [
  'the cat sat on the mat', 'the dog sat on the rug', 'the bird sat in the tree',
  'the cat sleeps on the mat', 'the dog sleeps on the rug', 'the bird sleeps in the tree',
  'a cat sat on the mat', 'a dog sat on the rug', 'a bird sat in the tree',
  'the small cat sat on the mat', 'the small dog sat on the rug', 'the small bird sat in the tree',
  'the happy cat sat on the mat', 'the happy dog sat on the rug', 'the happy bird sat in the tree',
  'the cat rests on the mat', 'the dog rests on the rug', 'the bird rests in the tree',
  'a small cat sleeps on the mat', 'a small dog sleeps on the rug', 'a small bird sleeps in the tree',
  'the cat is on the mat', 'the dog is on the rug', 'the bird is in the tree',
  'the cat likes the soft mat', 'the dog likes the soft rug', 'the bird likes the tall tree',
  'the cat sat on the mat today', 'the dog sat on the rug today', 'the bird sat in the tree today',
];
export class WordTokenizer {
  vocab:string[]; lookup:Map<string,number>;
  constructor(corpus=CORPUS){this.vocab=['<pad>','<unk>','<bos>','<eos>',...Array.from(new Set(corpus.flatMap(s=>s.split(' ')))).sort()];this.lookup=new Map(this.vocab.map((s,i)=>[s,i]));}
  encode(text:string){return text.toLowerCase().match(/[a-z]+|[0-9]+|[^\s\w]/g)?.map(x=>this.lookup.get(x)??1)??[];}
  decode(ids:number[]){return ids.map(i=>this.vocab[i]??'<unk>').join(' ');}
}
export class Random {
  state:number;
  constructor(seed:number){this.state=seed>>>0;}
  next(){let t=this.state+=0x6D2B79F5;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;}
  normal(){return Math.sqrt(-2*Math.log(Math.max(1e-12,this.next())))*Math.cos(2*Math.PI*this.next());}
}
