// Current indoor position is user-confirmed. No Wi-Fi/GPS floor is guessed.
function ensureIndoorStart(prompt=true){
 if(mapUI.startConfirmed||reviewEmbedded)return true;
 ui('routeSummary').hidden=true;ui('beginNavigation').disabled=true;navigationState.steps=[];mapUI.path=[];
 if(prompt&&!ui('indoorLocationDialog').open)ui('indoorLocationDialog').showModal();
 return false;
}
function fitNavigationBounds(points,area){
 if(!points.length)return null;const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
 const x=Math.min(...xs),y=Math.min(...ys),w=Math.max(...xs)-x,h=Math.max(...ys)-y;
 return {center:[x+w/2,y+h/2],scale:Math.min(area.width/Math.max(w+100,220),area.height/Math.max(h+100,220))};
}
let navigationFollow=true;
function focusNavigationStep(){
 if(!navigationState.active)return;
 const step=navigationState.steps[navigationState.index];if(!step)return;
 ui('returnToPosition').hidden=false;
 const transfer=step.kind==='transfer',edge=transfer&&mapUI.graph.adj[step.from]?.find(e=>e.to===step.to),escalator=edge?.kind==='escalator';
 const badge=ui('navigationModeBadge');badge.textContent=transfer?(escalator?'エスカレーター · 動く階段':'エレベーター · EV'):'徒歩 · 青線に沿って進む';badge.dataset.kind=transfer?(escalator?'escalator':'elevator'):'walk';
 if(!navigationFollow)return;
 if(document.body.classList.contains('view-3d')){window.dispatchEvent(new Event('maji-navigation-focus'));return;}
 const points=transfer?[displayPoint(step.from)]:instructionSegments(step).flatMap(s=>s.points);if(!points.length)return;
 const v=ui('mapViewport'),panel=ui('navigationPanel'),bottom=panel.getBoundingClientRect().top-v.getBoundingClientRect().top;
 const area={width:v.clientWidth-40,height:Math.max(130,bottom-85)},fit=fitNavigationBounds(points,area),[w,h]=floorInfo(mapUI.floor).size;
 mapUI.zoom=Math.max(.7,Math.min(8,fit.scale/mapUI.base));const scale=mapUI.base*mapUI.zoom;
 mapUI.pan=[(w/2-fit.center[0])*scale,(h/2-fit.center[1])*scale+(65+area.height/2)-v.clientHeight/2];applyMapTransform();
}
function officialIconSVG(f,box){return '<svg class="facility-pict" viewBox="'+box.join(' ')+'" aria-hidden="true"><image href="assets/visitor-floor-'+f+'.svg" width="'+unifiedMapModel.floors[f].size[0]+'" height="'+unifiedMapModel.floors[f].size[1]+'"/></svg>';}
function initVisitorExperience(){
 if(reviewEmbedded)return;
 document.body.classList.add('visitor-experience');
 const params=new URLSearchParams(location.search),at=params.get('at'),from=params.get('from');
 mapUI.startConfirmed=Boolean(from);
 if(at){const d=destinationList.find(d=>d.ref===at);if(d&&!isGuideOnly(d))selectStartPlace(d,true);}
 if(!mapUI.startConfirmed){clearRouteSelection();updateCurrentLabel();if(mapUI.goal)updateRoutePlanner();}else if(mapUI.goal)routePlan();
 ui('closeIndoorLocation').onclick=()=>ui('indoorLocationDialog').close();
 ui('locateBySearch').onclick=()=>{ui('indoorLocationDialog').close();setSelectionMode('start');ui('findPlace').value='';mapUI.resultFloor=null;renderResults();togglePlacePanel(true,true);};
 ui('locateByMap').onclick=()=>{ui('indoorLocationDialog').close();setSelectionMode('start');ui('facilityDetails').hidden=true;togglePlacePanel(false);};
 ui('returnToPosition').onclick=()=>{navigationFollow=true;const p=navigationState.position;if(p&&mapUI.floor!==p.floor)showFloor(p.floor,false);focusNavigationStep();};
 ui('mapViewport').addEventListener('pointerdown',e=>{if(navigationState.active&&!e.target.closest('button,a'))navigationFollow=false;},{capture:true});
 window.addEventListener('maji-navigation-position',()=>{if(!navigationState.active)ui('returnToPosition').hidden=true;});
 ui('beginNavigation').addEventListener('click',()=>{navigationFollow=true;});
 const filter=document.createElement('label');filter.className='result-floor-filter';filter.textContent='階で絞る ';
 const select=document.createElement('select');select.id='resultFloor';select.setAttribute('aria-label','候補の階');select.innerHTML='<option value="">すべての階</option>'+Array.from({length:10},(_,i)=>'<option value="'+(i+1)+'">'+(i+1)+'階</option>').join('');filter.append(select);ui('placeResults').before(filter);select.onchange=()=>{mapUI.resultFloor=Number(select.value)||null;renderResults();};
 ui('findPlace').addEventListener('input',()=>{mapUI.resultFloor=null;select.value='';});
 for(const b of document.querySelectorAll('[data-filter]'))b.addEventListener('click',()=>{mapUI.resultFloor=null;select.value='';renderResults();});
 for(const b of ui('nearbyEquipmentBar').querySelectorAll('button')){
  const category=b.dataset.nearbyCategory;let icon;
  if(b.id==='nearestToilet'){b.textContent='トイレ';icon=officialIconSVG(5,[1200,474,42,42]);}
  else{const d=mobilityData.equipment.find(d=>d.category===category);if(d)icon=officialIconSVG(d.floor,[d.iconPoint[0]-21,d.iconPoint[1]-21,42,42]);}
  if(icon)b.insertAdjacentHTML('afterbegin',icon);
 }
 for(const b of ui('nearbyEquipmentBar').querySelectorAll('button'))b.addEventListener('click',e=>{if(!ensureIndoorStart()){e.preventDefault();e.stopImmediatePropagation();}},true);
 ui('campusMapButton').onclick=()=>ui('campusMapDialog').showModal();ui('closeCampusMap').onclick=()=>ui('campusMapDialog').close();
 // A default mobile view fills the map height; panning reveals the rest of the floor.
 requestAnimationFrame(()=>{focusReadableMap();updateRoutePlanner();});
}
initVisitorExperience();
