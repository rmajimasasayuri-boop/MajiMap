// Keep place discovery, route planning and active navigation as distinct screens.
function syncSearchControls(){
 if(reviewEmbedded)return;const searching=!document.querySelector('main').classList.contains('settings-hidden'),button=ui('openPlaceSearch');
 button.textContent=searching?'‹':'⌕';button.setAttribute('aria-label',searching?'マップへ戻る':'検索候補を開く');button.classList.toggle('search-back',searching);
 ui('clearPlaceSearch').hidden=!ui('findPlace').value;if(typeof syncDiscoveryFloors==='function')syncDiscoveryFloors();if(typeof scheduleCandidateIndex==='function')scheduleCandidateIndex();
}
function swapRoutePlaces(){
 if(!mapUI.goal||!mapUI.startConfirmed||mapUI.goal.usage==='emergency-only'||entranceAvailability(mapUI.goal.ref).closed)return;
 const goal=mapUI.goal,oldManual=mapUI.manual,oldManualGoal=mapUI.manualGoal;
 const start=oldManual?{ref:'hospital:'+oldManual.floor+'::__manualGoal',routeReference:'hospital:'+oldManual.floor+'::__manualGoal',floor:oldManual.floor,name:'地図で指定した目的地',code:'',type:'manual',point:oldManual.point}:destinationList.find(d=>d.ref===mapUI.from||navigationTargetReference(d)===mapUI.from)||{ref:mapUI.from,routeReference:mapUI.from,floor:referenceFloor(mapUI.from),point:routeStartPoint(),name:ui('currentName').textContent,type:'point',code:''};
 mapUI.startChosenOnMap=Boolean(oldManualGoal);mapUI.manual=oldManualGoal;mapUI.manualGoal=oldManual;mapUI.from=oldManualGoal?'hospital:'+goal.floor+'::__manualStart':(southWardToiletRefs.includes(goal.ref)?mapUI.routeArrival?.anchor:null)||navigationTargetReference(goal);mapUI.startName=oldManualGoal?null:destinationDisplayName(goal);mapUI.startConfirmed=true;
 choosePlace(start,true);updateCurrentLabel();routePlan();updateRoutePlanner();
}
function enterRoutePlanning(){
 document.body.classList.add('route-planning');togglePlacePanel(false);ui('facilityDetails').hidden=true;updateRoutePlanner();
 if(!mapUI.goal||!ensureIndoorStart(false)){syncRoutePanels();return;}
 if(routeEntranceNotice(mapUI.from,mapUI.goal.ref,null,[])?.blocked){routePlan();return;}
 routePlan();if(navigationState.steps.length){const floor=referenceFloor(mapUI.from)||mapUI.manual?.floor;showFloor(floor,false);if(!document.body.classList.contains('view-3d')){const points=floorRouteLines(mapUI.graph,mapUI.path,floor).flat();const v=ui('mapViewport'),fit=fitNavigationBounds(points,{width:v.clientWidth-40,height:Math.max(160,v.clientHeight-300)});if(fit){const [w,h]=floorInfo(floor).size;mapUI.zoom=Math.max(.7,Math.min(8,fit.scale/mapUI.base));const scale=mapUI.base*mapUI.zoom;mapUI.pan=[(w/2-fit.center[0])*scale,(h/2-fit.center[1])*scale+30];applyMapTransform();}}}
 syncRoutePanels();
}
const entranceNoticeState={key:'',surface:'',dismissed:false};
let startingNavigationFromPreview=false;
let entranceFadeTimer, entranceHideTimer, entranceFading=false;
function syncEntranceNotice(notice,planning){
 const active=navigationState.active;
 if(notice){
  const key=JSON.stringify([mapUI.from,mapUI.goal.ref,notice.text]);
  if(key!==entranceNoticeState.key){
   clearTimeout(entranceFadeTimer);clearTimeout(entranceHideTimer);
   entranceFadeTimer=null;entranceFading=false;
   Object.assign(entranceNoticeState,{key,surface:'',dismissed:false});
  }
  if(!entranceNoticeState.surface)entranceNoticeState.surface=active?'navigation':planning?'planning':'';
  if(active&&(navigationState.index>0||entranceNoticeState.surface==='planning'))entranceNoticeState.dismissed=true;
 }
 for(const [id,surface,shown] of [
  ['routeEntranceNotice','planning',planning&&!active],
  ['navigationEntranceNotice','navigation',active&&navigationState.index===0],
 ]){
  const element=ui(id);
  element.hidden=!notice||!shown||entranceNoticeState.dismissed||entranceNoticeState.surface!==surface;
  element.classList.toggle('entrance-notice-fading',entranceFading);
  if(!element.hidden&&notice.autoDismiss&&!entranceFadeTimer){
   entranceFadeTimer=setTimeout(()=>{
    entranceFading=true;
    syncRoutePanels();
    entranceHideTimer=setTimeout(()=>{entranceNoticeState.dismissed=true;syncRoutePanels();},800);
   },3000);
  }
  const message=element.querySelector('[role=status]');
  const text=notice?.text||'';
  if(message.textContent!==text)message.textContent=text;
  element.dataset.blocked=String(Boolean(notice?.blocked));
 }
}
function syncRoutePanels(){
 if(reviewEmbedded)return;
 document.body.classList.toggle('place-selected',Boolean(mapUI.goal));
 const planning=document.body.classList.contains('route-planning');ui('routeOptionsDock').hidden=!planning||navigationState.active||!mapUI.goal||!mapUI.startConfirmed;
 ui('swapRoutePlaces').disabled=!mapUI.startConfirmed||!mapUI.goal||mapUI.goal.usage==='emergency-only'||entranceAvailability(mapUI.goal.ref).closed;
 const notice=mapUI.goal?routeEntranceNotice(mapUI.from,mapUI.goal.ref,mapUI.graph,mapUI.path):null;
 syncEntranceNotice(notice,!ui('routeOptionsDock').hidden&&!startingNavigationFromPreview);
 ui('routeDockNavigate').disabled=Boolean(notice?.blocked);
 syncSearchControls();
}
function initRouteExperience(){
 if(reviewEmbedded)return;
 const detailBody=ui('facilityDetails').querySelector('.facility-detail-body'),heading=document.createElement('div');heading.className='facility-heading';heading.append(ui('facilityDetailFloor'),ui('facilityDetailName'));detailBody.prepend(heading);detailBody.prepend(ui('destinationPhotoPreview'));
 const planner=ui('routePlanner'),header=document.createElement('div');header.className='route-planning-header';header.innerHTML='<button id="closeRoutePlanning" aria-label="施設の案内に戻る">‹</button>';planner.prepend(header);
 const swap=document.createElement('button');swap.id='swapRoutePlaces';swap.className='swap-route-places';swap.innerHTML='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 20V4m-4 4 4-4 4 4M17 4v16m-4-4 4 4 4-4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';swap.setAttribute('aria-label','出発地と目的地を入れ替える');planner.append(swap);swap.onclick=swapRoutePlaces;ui('mapViewport').append(planner);
 const dock=document.createElement('section');dock.id='routeOptionsDock';dock.className='route-options-dock';dock.setAttribute('aria-label','ナビ開始');dock.hidden=true;header.append(document.querySelector('.route-travel-tabs'));const start=document.createElement('button');start.id='routeDockNavigate';start.className='primary';start.textContent='ナビ開始';start.onclick=()=>{if(!ensureIndoorStart())return;routePlan();startNavigation();syncRoutePanels();};dock.append(start);ui('mapViewport').append(dock);
 for(const id of ['routeEntranceNotice','navigationEntranceNotice']){
  const note=document.createElement('div'),message=document.createElement('span'),close=document.createElement('button');
  note.id=id;note.className='entrance-route-notice';note.hidden=true;
  message.setAttribute('role','status');close.type='button';close.textContent='×';close.setAttribute('aria-label','閉門案内を閉じる');
  close.onclick=()=>{entranceNoticeState.dismissed=true;syncRoutePanels();};
  note.append(message,close);
  if(id==='routeEntranceNotice')dock.prepend(note);else ui('navigationInstruction').after(note);
 }
 for(const icon of document.querySelectorAll('.route-travel-tabs .official-wheelchair-icon'))icon.innerHTML=officialEquipmentIcon('wheelchair');
 ui('closeRoutePlanning').onclick=()=>{stopNavigation();document.body.classList.remove('route-planning');dock.hidden=true;planner.hidden=true;if(mapUI.goal)updateDestinationPhoto(mapUI.goal);ui('routeSummary').hidden=true;syncRoutePanels();};
 ui('detailRoute').onclick=enterRoutePlanning;
 ui('openPlaceSearch').onclick=()=>{const open=document.querySelector('main').classList.contains('settings-hidden');if(open&&mapUI.goal&&ui('findPlace').value===destinationDisplayName(mapUI.goal)){ui('findPlace').value='';mapUI.filter='all';document.querySelectorAll('[data-filter]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.filter==='all')));renderResults();}togglePlacePanel(open,open);syncSearchControls();};
 ui('findPlace').addEventListener('input',syncSearchControls);ui('clearPlaceSearch').addEventListener('click',syncSearchControls);
 window.addEventListener('maji-map-update',syncRoutePanels);window.addEventListener('maji-navigation-position',syncRoutePanels);
 syncRoutePanels();
}
initRouteExperience();
