import type {RecordedStep} from '../model/micro/tensor';

/** Recorded snapshots make backwards stepping and scrubbing exact and reversible. */
export class Playback {
  steps:RecordedStep[]=[];
  index=0;
  speed=.5;
  playing=false;
  private timer?:ReturnType<typeof setTimeout>;
  constructor(private change:(step:RecordedStep|undefined,index:number,total:number,playing:boolean)=>void,private ended?:()=>void){}
  load(steps:RecordedStep[]){this.pause();this.steps=steps;this.index=0;this.emit();}
  seek(index:number){this.index=Math.max(0,Math.min(this.steps.length-1,index));this.emit();}
  step(delta:number){this.pause();this.seek(this.index+delta);}
  play(){if(!this.steps.length)return;if(this.index>=this.steps.length-1)this.index=0;this.playing=true;this.emit();this.schedule();}
  pause(){this.playing=false;clearTimeout(this.timer);this.emit();}
  toggle(){this.playing?this.pause():this.play();}
  setSpeed(speed:number){this.speed=Math.max(.05,Math.min(5,speed));if(this.playing){clearTimeout(this.timer);this.schedule();}}
  private schedule(){this.timer=setTimeout(()=>{if(!this.playing)return;if(this.index>=this.steps.length-1){this.pause();this.ended?.();return;}this.index++;this.emit();this.schedule();},500/this.speed);}
  private emit(){this.change(this.steps[this.index],this.index,this.steps.length,this.playing);}
}
