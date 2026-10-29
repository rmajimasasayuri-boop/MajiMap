// Keep place discovery, route planning and active navigation as distinct screens.
function syncSearchControls(){
 if(reviewEmbedded)return;const searching=!document.querySelector('main').classList.contains('settings-hidden'),button=ui('openPlaceSearch');
 button.textContent=searching?'‹':'⌕';button.setAttribute('aria-label',searching?'マップへ戻る':'検索候補を開く');button.classList.toggle('search-back',searching);
 ui('clearPlaceSearch').hidden=!ui('findPlace').value;
}
function swapRoutePlaces(){
 if(!mapUI.goal||!mapUI.startConfirmed)return;
 const goal=mapUI.goal,oldManual=mapUI.manual,oldManualGoal=mapUI.manualGoal;
 const start=oldManual?{ref:'hospital:'+oldManual.floor+'::__manualGoal',routeReference:'hospital:'+oldManual.floor+'::__manualGoal',floor:oldManual.floor,name:'地図で指定した目的地',code:'',type:'manual',point:oldManual.point}:destinationList.find(d=>d.ref===mapUI.from||navigationTargetReference(d)===mapUI.from)||{ref:mapUI.from,routeReference:mapUI.from,floor:referenceFloor(mapUI.from),point:routeStartPoint(),name:ui('currentName').textContent,type:'point',code:''};
 mapUI.manual=oldManualGoal;mapUI.manualGoal=oldManual;mapUI.from=oldManualGoal?'hospital:'+goal.floor+'::__manualStart':navigationTargetReference(goal);mapUI.startName=oldManualGoal?null:destinationDisplayName(goal);mapUI.startConfirmed=true;
 choosePlace(start,true);updateCurrentLabel();routePlan();updateRoutePlanner();
}
function enterRoutePlanning(){
 if(!mapUI.goal)return;
 document.body.classList.add('route-planning');togglePlacePanel(false);ui('facilityDetails').hidden=true;updateRoutePlanner();
 if(!ensureIndoorStart())return;
 routePlan();if(navigationState.steps.length){const floor=referenceFloor(mapUI.from)||mapUI.manual?.floor;showFloor(floor,false);if(!document.body.classList.contains('view-3d')){const points=floorRouteLines(mapUI.graph,mapUI.path,floor).flat();const v=ui('mapViewport'),fit=fitNavigationBounds(points,{width:v.clientWidth-40,height:Math.max(160,v.clientHeight-300)});if(fit){const [w,h]=floorInfo(floor).size;mapUI.zoom=Math.max(.7,Math.min(8,fit.scale/mapUI.base));const scale=mapUI.base*mapUI.zoom;mapUI.pan=[(w/2-fit.center[0])*scale,(h/2-fit.center[1])*scale+30];applyMapTransform();}}}
 syncRoutePanels();
}
function syncRoutePanels(){
 if(reviewEmbedded)return;
 document.body.classList.toggle('place-selected',Boolean(mapUI.goal));
 const planning=document.body.classList.contains('route-planning');ui('routeOptionsDock').hidden=!planning||navigationState.active||!mapUI.goal;
 ui('swapRoutePlaces').disabled=!mapUI.startConfirmed||!mapUI.goal;
 syncSearchControls();
}
function initRouteExperience(){
 if(reviewEmbedded)return;
 const detailBody=ui('facilityDetails').querySelector('.facility-detail-body'),heading=document.createElement('div');heading.className='facility-heading';heading.append(ui('facilityDetailFloor'),ui('facilityDetailName'));detailBody.prepend(heading);detailBody.prepend(ui('destinationPhotoPreview'));
 const planner=ui('routePlanner'),header=document.createElement('div');header.className='route-planning-header';header.innerHTML='<button id="closeRoutePlanning" aria-label="施設の案内に戻る">‹</button><span>経路</span>';planner.prepend(header);
 const swap=document.createElement('button');swap.id='swapRoutePlaces';swap.className='swap-route-places';swap.textContent='↕';swap.setAttribute('aria-label','出発地と目的地を入れ替える');planner.append(swap);swap.onclick=swapRoutePlaces;ui('mapViewport').append(planner);
 const dock=document.createElement('section');dock.id='routeOptionsDock';dock.className='route-options-dock';dock.setAttribute('aria-label','ナビ開始');dock.hidden=true;header.after(document.querySelector('.route-travel-tabs'));const start=document.createElement('button');start.id='routeDockNavigate';start.className='primary';start.textContent='ナビを開始';start.onclick=()=>{if(!ensureIndoorStart())return;routePlan();startNavigation();syncRoutePanels();};dock.append(start);ui('mapViewport').append(dock);
 ui('closeRoutePlanning').onclick=()=>{stopNavigation();document.body.classList.remove('route-planning');dock.hidden=true;planner.hidden=true;if(mapUI.goal)updateDestinationPhoto(mapUI.goal);ui('routeSummary').hidden=true;syncRoutePanels();};
 ui('detailRoute').onclick=enterRoutePlanning;
 ui('openPlaceSearch').onclick=()=>{const open=document.querySelector('main').classList.contains('settings-hidden');togglePlacePanel(open,open);syncSearchControls();};
 ui('findPlace').addEventListener('input',syncSearchControls);ui('clearPlaceSearch').addEventListener('click',syncSearchControls);
 window.addEventListener('maji-map-update',syncRoutePanels);window.addEventListener('maji-navigation-position',syncRoutePanels);
 syncRoutePanels();
}
initRouteExperience();
