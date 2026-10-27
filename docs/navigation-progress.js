// This marker previews a navigation instruction; it does not measure device location.
let navigationProgressFrame=0,navigationProgressRun=0;
function instructionSegments(step){
 if(step.kind==='transfer')return [{floor:step.floor,points:[displayPoint(step.to)]}];
 const out=[];
 for(let i=1;i<(step.nodes||[]).length;i++){
  const points=routeGeometry(mapUI.graph,step.nodes[i-1],step.nodes[i]);
  if(points.length)out.push({floor:step.floor,points:squareRoute(points)});
 }
 if(!out.length&&step.nodes?.length){const p=displayPoint(step.nodes[0]);if(p)out.push({floor:step.floor,points:[p]});}
 return out;
}
function sampleInstruction(segments,fraction){
 const edges=[];let length=0;
 for(const segment of segments)for(let i=1;i<segment.points.length;i++){
  const a=segment.points[i-1],b=segment.points[i],size=Math.hypot(b[0]-a[0],b[1]-a[1]);
  if(size){edges.push({floor:segment.floor,a,b,start:length,length:size});length+=size;}
 }
 if(!edges.length){const s=segments.at(-1);return s?.points.length?{floor:s.floor,point:s.points.at(-1)}:null;}
 const distance=Math.max(0,Math.min(1,fraction))*length;
 const e=edges.find(e=>distance<=e.start+e.length)||edges.at(-1),t=Math.max(0,Math.min(1,(distance-e.start)/e.length));
 return {floor:e.floor,point:e.a.map((v,j)=>v+(e.b[j]-v)*t)};
}
function drawNavigationPosition(){
 document.getElementById('navigationPosition')?.remove();
 const p=navigationState.active&&navigationState.position;if(!p||p.floor!==mapUI.floor)return;
 const scale=mapUI.base*mapUI.zoom,g=svgNode('g',{id:'navigationPosition',role:'img','aria-label':'案内位置（手順連動・自動測位なし）','pointer-events':'none'});
 g.append(svgNode('circle',{cx:0,cy:0,r:17/scale,fill:'#2580c5','fill-opacity':.16}));
 g.append(svgNode('circle',{cx:0,cy:0,r:8/scale,fill:'#2580c5',stroke:'white','stroke-width':3/scale}));
 g.setAttribute('transform',`translate(${p.point.join(' ')})`);ui('unifiedOverlay').append(g);
}
function publishNavigationPosition(position){
 navigationState.position=position;
 const dot=document.getElementById('navigationPosition');
 if(dot&&position?.floor===mapUI.floor)dot.setAttribute('transform',`translate(${position.point.join(' ')})`);else drawNavigationPosition();
 window.dispatchEvent(new CustomEvent('maji-navigation-position',{detail:position}));
}
function cancelNavigationProgress(){navigationProgressRun++;cancelAnimationFrame(navigationProgressFrame);publishNavigationPosition(null);}
function beginNavigationProgress(step){
 cancelAnimationFrame(navigationProgressFrame);const run=++navigationProgressRun,segments=instructionSegments(step);
 publishNavigationPosition(sampleInstruction(segments,0));
 let began=null;const duration=6500;
 function frame(time){if(run!==navigationProgressRun||!navigationState.active)return;if(began===null)began=time;const progress=Math.min(1,(time-began)/duration);publishNavigationPosition(sampleInstruction(segments,progress));if(progress<1)navigationProgressFrame=requestAnimationFrame(frame);}
 navigationProgressFrame=requestAnimationFrame(frame);
}
function moveNavigationStep(delta){
 if(!navigationState.active)return;
 const next=navigationState.index+delta;if(next<0)return;
 if(next>=navigationState.steps.length){stopNavigation();ui('mapMessage').textContent='案内を終了しました。';return;}
 navigationState.index=next;renderNavigation();
}
function initNavigationGestures(){
 const panel=ui('navigationPanel');let down=null;
 panel.addEventListener('pointerdown',e=>{if(e.target.closest('button'))return;down={x:e.clientX,y:e.clientY,id:e.pointerId};panel.setPointerCapture(e.pointerId);});
 panel.addEventListener('pointerup',e=>{if(!down||down.id!==e.pointerId)return;const dx=e.clientX-down.x,dy=e.clientY-down.y;down=null;if(Math.abs(dx)>55&&Math.abs(dx)>Math.abs(dy)*1.5)moveNavigationStep(dx<0?1:-1);});
 panel.addEventListener('pointercancel',()=>{down=null;});
}
