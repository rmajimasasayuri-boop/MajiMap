// Independent heap-based check of every planned route, including alternate entrances.
// Uses the same patient access policy; never adds corridors or makes access assumptions.
function auditedRouteTree(graph,start,via3F=false){
 const labels=new Map(),previous=new Map(),heap=[];
 const better=(a,b)=>!b||a.rides<b.rides||a.rides===b.rides&&a.cost<b.cost-1e-7;
 function push(v){heap.push(v);let i=heap.length-1;while(i){const p=(i-1)>>1;if(!better(v,heap[p]))break;heap[i]=heap[p];i=p;}heap[i]=v;}
 function pop(){const first=heap[0],last=heap.pop();if(heap.length){let i=0;while(i*2+1<heap.length){let child=i*2+1;if(child+1<heap.length&&better(heap[child+1],heap[child]))child++;if(!better(heap[child],last))break;heap[i]=heap[child];i=child;}heap[i]=last;}return first;}
 if(!graph.nodes[start])return {labels,previous};
 const initial={node:start,rides:0,cost:0};labels.set(start,initial);push(initial);
 while(heap.length){const at=pop();if(labels.get(at.node)!==at)continue;for(const e of graph.adj[at.node]||[]){if(e.access!=='patient'||!Number.isFinite(e.cost)||e.cost<0||!routeEdgeAllowed(graph,at.node,e,via3F))continue;const next={node:e.to,rides:at.rides+(['elevator','escalator'].includes(e.kind)?1:0),cost:at.cost+e.cost};if(better(next,labels.get(e.to))){labels.set(e.to,next);previous.set(e.to,at.node);push(next);}}}
 return {labels,previous};
}
function auditRouteSelection(graph,starts,ends,path){
 let best=null;const trees=new Map();
 for(const start of starts)for(const end of ends){const via3F=routeNeeds3F(graph,start,end),key=start+'|'+via3F;let tree=trees.get(key);if(!tree){tree=auditedRouteTree(graph,start,via3F);trees.set(key,tree);}const score=tree.labels.get(end);if(!score)continue;if(!best||score.rides<best.rides||score.rides===best.rides&&score.cost<best.cost-1e-7)best={...score,start,end,tree};}
 if(!best)return {valid:false,path:[],reason:'unreachable'};
 let cost=0,rides=0,valid=starts.includes(path[0])&&ends.includes(path.at(-1));const via3F=routeNeeds3F(graph,path[0]||'',path.at(-1)||'');
 for(let i=1;i<path.length;i++){const edges=(graph.adj[path[i-1]]||[]).filter(e=>e.to===path[i]&&e.access==='patient'&&routeEdgeAllowed(graph,path[i-1],e,via3F));const edge=edges.reduce((a,b)=>!a||b.cost<a.cost?b:a,null);if(!edge){valid=false;break;}cost+=edge.cost;rides+=['elevator','escalator'].includes(edge.kind)?1:0;}
 valid=valid&&rides===best.rides&&Math.abs(cost-best.cost)<1e-5;
 const optimal=[best.end];while(optimal[0]!==best.start){const prev=best.tree.previous.get(optimal[0]);if(!prev)return {valid:false,path:[],reason:'invalid-predecessor'};optimal.unshift(prev);}
 return {valid,path:valid?path:optimal,start:valid?path[0]:best.start,end:valid?path.at(-1):best.end,rides:best.rides,cost:best.cost,corrected:!valid};
}
function verifyNavigationRoute(graph,from,target,path){const options=typeof additionalNavigation!=='undefined'?additionalNavigation.arrivalOptions||{}:{};return auditRouteSelection(graph,options[from]||[from],options[target.ref]||[navigationTargetReference(target)],path);}

// Compare the full journey through one public waypoint; the same entrance is used
// on arrival and departure, so a shop never becomes an invented shortcut.
function verifyWaypointNavigationRoute(graph,from,target,waypoint){
 const options=typeof additionalNavigation!=='undefined'?additionalNavigation.arrivalOptions||{}:{},starts=options[from]||[from],ends=options[target.ref]||[navigationTargetReference(target)],vias=options[waypoint.ref]||[navigationTargetReference(waypoint)];let best=null;
 for(const via of vias){let leftPrimary=[],rightPrimary=[];for(const start of starts){const p=shortest(graph,start,via);if(p.length){leftPrimary=p;break;}}for(const end of ends){const p=shortest(graph,via,end);if(p.length){rightPrimary=p;break;}}
 const left=auditRouteSelection(graph,starts,[via],leftPrimary),right=auditRouteSelection(graph,[via],ends,rightPrimary);if(!left.path.length||!right.path.length)continue;
 const value={valid:left.valid&&right.valid,path:[...left.path,...right.path.slice(1)],start:left.start,end:right.end,via,viaIndex:left.path.length-1,rides:left.rides+right.rides,cost:left.cost+right.cost,corrected:left.corrected||right.corrected};if(!best||value.rides<best.rides||value.rides===best.rides&&value.cost<best.cost)best=value;
 }return best||{valid:false,path:[],reason:'waypoint-unreachable'};
}
