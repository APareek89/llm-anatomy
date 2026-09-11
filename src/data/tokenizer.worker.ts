import { loadTokenizer } from './tokenizer';

// Parsing the real tokenizer and running BPE stays off the rendering thread.
self.onmessage = async (event: MessageEvent<{id:number; text:string}>) => {
  const {id,text} = event.data;
  try {
    const tokenizer = await loadTokenizer();
    const pieces = tokenizer.tokenPieces(text);
    self.postMessage({id,pieces,ids:pieces.map(p=>p.id),decoded:tokenizer.decode(pieces.map(p=>p.id)),source:'real'});
  } catch (error) {
    self.postMessage({id,error:error instanceof Error ? error.message : String(error)});
  }
};
