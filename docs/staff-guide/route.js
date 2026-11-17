'use strict';

function staffRoute(records, plans, start, goal, mode, includeUnconfirmed, requiredLift) {
  const nodes=[], edges=[], segments=[], facilities=[];
  const scale=(floor,p)=>p.map((v,i)=>v*[plans[floor].width,plans[floor].height][i]/1000);
  const distance=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
  const cross=(a,b)=>a[0]*b[1]-a[1]*b[0];
  const sub=(a,b)=>a.map((v,i)=>v-b[i]);
  const project=(p,a,b)=>{
    const d=sub(b,a);
    const t=Math.max(0,Math.min(1,sub(p,a).reduce((s,v,i)=>s+v*d[i],0)/(distance(a,b)**2||1)));
    const point=a.map((v,i)=>v+t*d[i]);
    return {t,point,error:distance(p,point)};
  };
  const allowed=r=>r.modes[mode]==='yes'||(includeUnconfirmed&&r.modes[mode]!=='no');
  const addNode=(floor,p)=>{ nodes.push({floor,p}); edges.push([]); return nodes.length-1; };
  function connect(a,b,cost,uncertain=false,transfer='') {
    edges[a].push({to:b,cost,uncertain,transfer});
    edges[b].push({to:a,cost,uncertain,transfer});
  }
  for (const r of records) {
    if (r.kind!=='corridor'||!allowed(r)) continue;
    const points=r.points.map(p=>scale(r.floor,p));
    let previous;
    for (const p of points) {
      const node=addNode(r.floor,p);
      if (previous!==undefined && distance(nodes[previous].p,p)>0.001) {
        segments.push({recordId:r.id,floor:r.floor,a:nodes[previous].p,b:p,cuts:[[0,previous],[1,node]],uncertain:r.modes[mode]!=='yes'});
      } else if (previous!==undefined) connect(previous,node,0);
      previous=node;
    }
  }
  for (const s of segments) for (const [t,node] of s.cuts.slice(0,2)) {
    for (const other of segments) {
      if (s===other||s.floor!==other.floor) continue;
      const hit=project(nodes[node].p,other.a,other.b);
      if (hit.error>.05 && (!includeUnconfirmed||hit.error>5)) continue;
      const join=addNode(s.floor,hit.point);
      other.cuts.push([hit.t,join]);
      connect(node,join,hit.error,hit.error>.05);
    }
  }
  for (let i=0;i<segments.length;i++) for (let j=i+1;j<segments.length;j++) {
    const s=segments[i],o=segments[j];
    if(s.floor!==o.floor)continue;
    const a=sub(s.b,s.a),b=sub(o.b,o.a),delta=sub(o.a,s.a),den=cross(a,b);
    if(Math.abs(den)<1e-8)continue;
    const t=cross(delta,b)/den,u=cross(delta,a)/den;
    if(t<=0||t>=1||u<=0||u>=1)continue;
    const join=addNode(s.floor,s.a.map((v,k)=>v+t*a[k]));
    s.cuts.push([t,join]);o.cuts.push([u,join]);
  }
  function attachmentPoints(location,max) {
    const p=scale(location.floor,location.point);
    const hits=segments.filter(s=>s.floor===location.floor&&(!location.patientCorridorId||s.recordId===location.patientCorridorId)).map(s=>({...project(p,s.a,s.b),s}));
    return hits.filter(hit=>hit.error<=max).sort((a,b)=>a.error-b.error);
  }
  function cutSegment(floor, hit) {
    const join=addNode(floor,hit.point);
    hit.s.cuts.push([hit.t,join]);
    return join;
  }
  function attach(location,max,uncertain=false) {
    const [best]=attachmentPoints(location,max);
    if(!best)return null;
    const join=cutSegment(location.floor,best);
    const id=addNode(location.floor,scale(location.floor,location.point));
    connect(id,join,best.error,uncertain||best.error>5);
    return id;
  }
  function attachEndpoint(location, isStart) {
    const {width,height}=plans[location.floor];
    const range = location.referenceOnly && !location.approach ? Math.hypot(width,height)*0.08
      : location.referenceOnly ? 12 : includeUnconfirmed ? 34 : 5;
    const hits=attachmentPoints(location,range);
    if (!hits.length) return null;
    const point=location.referenceOnly ? null : scale(location.floor,location.point);
    const outside=addNode(location.floor,point);
    // Try nearby reachable corridor points. The virtual edges only rank candidates
    // and are excluded from both the displayed route and the doorway checks.
    for (const hit of hits) {
      const corridor=cutSegment(location.floor,hit);
      const [from,to]=isStart?[outside,corridor]:[corridor,outside];
      edges[from].push({to,cost:hit.error*20,virtual:location.referenceOnly,
        uncertain:!location.referenceOnly&&hit.error>5});
    }
    return outside;
  }
  const from=attachEndpoint(start,true),to=attachEndpoint(goal,false);
  if(from===null||to===null)return {error:'出発地または目的地を、選択した移動方法で使える通路に接続できませんでした。'};
  function parseFloors(value) {
    const text=value.normalize('NFKC');
    const floors=new Set(text.match(/\d+/g)?.map(Number)||[]);
    for (const [,first,last] of text.matchAll(/(\d+)階?\s*[〜~～–-]\s*(\d+)/g)) {
      for(let f=Math.max(1,+first);f<=Math.min(10,+last);f++) floors.add(f);
    }
    return floors;
  }
  for(const r of records){
    const isStair=mode==='walk';
    if(r.kind!==(isStair?'stairs':'elevator')||!allowed(r))continue;
    if(mode==='stretcher'&&requiredLift&&r.name!==requiredLift)continue;
    const id=attach({floor:r.floor,point:r.points[0]},includeUnconfirmed?45:5,r.modes[mode]!=='yes');
    if(id!==null)facilities.push({id,r,stops:parseFloors(r.stops)});
  }
  for(let i=0;i<facilities.length;i++)for(let j=i+1;j<facilities.length;j++){
    const a=facilities[i],b=facilities[j];
    if(a.r.name!==b.r.name||a.r.floor===b.r.floor)continue;
    const explicit=a.stops.has(b.r.floor)&&b.stops.has(a.r.floor);
    const compatible=(!a.stops.size||a.stops.has(b.r.floor))&&(!b.stops.size||b.stops.has(a.r.floor));
    // Missing stop metadata is common in the stair annotations. A repeated stair
    // identifier can be tried across adjacent annotated floors only.
    const trialConnection=includeUnconfirmed&&compatible&&
      (mode!=='walk'||Math.abs(a.r.floor-b.r.floor)===1);
    if(!explicit&&!trialConnection)continue;
    connect(a.id,b.id,100+65*Math.abs(a.r.floor-b.r.floor),!explicit||a.r.modes[mode]!=='yes'||b.r.modes[mode]!=='yes',a.r.name);
  }
  for(const s of segments){
    s.cuts.sort((a,b)=>a[0]-b[0]);
    for(let i=1;i<s.cuts.length;i++){
      const a=s.cuts[i-1][1],b=s.cuts[i][1];
      connect(a,b,distance(nodes[a].p,nodes[b].p),s.uncertain);
    }
  }
  // A recorded closed or unconfirmed opening blocks paths that cross it.
  const barriers=records.filter(r=>r.kind==='opening'&&!allowed(r)).map(r=>({floor:r.floor,points:r.points.map(p=>scale(r.floor,p))}));
  for(let i=0;i<edges.length;i++)edges[i]=edges[i].filter(edge=>{
    if(edge.virtual)return true;
    const a=nodes[i],b=nodes[edge.to];
    if(a.floor!==b.floor)return true;
    return !barriers.some(({floor,points:[c,d]})=>{
      if(floor!==a.floor)return false;
      const ab=sub(b.p,a.p),cd=sub(d,c),offset=sub(c,a.p),den=cross(ab,cd);
      if(Math.abs(den)<1e-8)return false;
      const t=cross(offset,cd)/den,u=cross(offset,ab)/den;
      return t>=-1e-8&&t<=1+1e-8&&u>=0&&u<=1;
    });
  });
  const costs=Array(nodes.length).fill(Infinity), previous=[], queue=[from], queued=new Set(queue);
  costs[from]=0;
  while(queue.length){
    queue.sort((a,b)=>costs[b]-costs[a]);
    const id=queue.pop();
    queued.delete(id);
    if(id===to)break;
    for(const edge of edges[id]){
      const cost=costs[id]+edge.cost+(edge.uncertain?12:0);
      if(cost>=costs[edge.to])continue;
      costs[edge.to]=cost;
      previous[edge.to]={id,edge};
      if(!queued.has(edge.to)){queue.push(edge.to);queued.add(edge.to);}
    }
  }
  if(!Number.isFinite(costs[to]))return {error:mode==='walk'&&start.floor!==goal.floor
    ?'徒歩経路をつなげませんでした。階段の前後の通路接続を確認してください。'
    :'登録済みの通路をつなげませんでした。出発地・目的地の入口や、使うEVをご確認ください。'};
  const path=[nodes[to]],transfers=[];
  let uncertain=false;
  for(let id=to;id!==from;){
    const step=previous[id];
    if(step.edge.transfer)transfers.unshift({name:step.edge.transfer,from:nodes[step.id].floor,to:nodes[id].floor});
    uncertain ||= step.edge.uncertain;
    id=step.id;
    path.unshift(nodes[id]);
  }
  return {path:path.filter(node=>node.p),transfers,uncertain};
}
