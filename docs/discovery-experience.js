function resultCategoryMatches(d){if(mapUI.filter==='amenity')return ['services::restaurant','services::cafe','services::seven','services::green','services::cosmetics','services::glasses','services::hair'].includes(d.ref);if(mapUI.filter==='equipment'&&d.ref==='services::laundry')return true;if(mapUI.filter==='counter')return d.type==='reception'||['services::information','services::medicine','services::admission','services::lost'].includes(d.ref);if(mapUI.filter==='food')return ['services::restaurant','services::cafe'].includes(d.ref);if(mapUI.filter==='shopping')return ['services::seven','services::green','services::cosmetics','services::glasses','services::hair'].includes(d.ref);return mapUI.filter==='all'||d.type===mapUI.filter||mapUI.filter==='equipment'&&d.type==='toilet'||mapUI.filter==='service'&&['reception','entrance'].includes(d.type)||mapUI.filter==='ward'&&d.type==='rehab';}
function routeProximityFor(matches){
 const scores=new Map();if(!mapUI.startConfirmed||mapUI.pickMode==='start'||!matches.length||!(mapUI.filter==='equipment'||mapUI.filter==='entrance'||matches.every(d=>placeNearestKind(d))))return scores;
 travelMode=ui('travelChoice').value;const graph=attachManualSelections(buildNetwork()),from=mapUI.manual?'hospital:'+mapUI.manual.floor+'::__manualStart':entryReference(mapUI.from),starts=typeof additionalNavigation!=='undefined'&&additionalNavigation.arrivalOptions?.[from]||[from],trees=new Map();
 for(const d of matches){let best=Infinity;if(d.nearestEligible!==false&&!isGuideOnly(d)&&!entranceAvailability(d.ref).closed)for(const start of starts){const end=navigationTargetReference(d),via3F=routeNeeds3F(graph,start,end),key=start+'|'+via3F;let tree=trees.get(key);if(!tree){tree=auditedRouteTree(graph,start,via3F);trees.set(key,tree);}best=Math.min(best,tree.labels.get(end)?.cost??Infinity);}scores.set(d.ref,best);}return scores;
}
let searchFocusLock=false;
let placeSearchView='home',preferredCandidateFloor=null;
function setPlaceSearchView(view){placeSearchView=view;document.body.dataset.searchView=view;fitFloorRail();}
function resetCandidateScroll(){const list=ui('placeResults');list.style.paddingBottom='';list.scrollTop=0;mapUI.resultFloor=null;for(const id of ['searchChoices','recentPlaces'])if(ui(id))ui(id).scrollTop=0;}
function beginPlaceSearch(){resetCandidateScroll();mapUI.selectingStartOnMap=false;preferredCandidateFloor=null;ui('findPlace').value='';mapUI.filter='all';mapUI.resultFloor=null;document.querySelectorAll('[data-filter]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.filter==='all')));setPlaceSearchView('home');renderResults();updateSearchChoices();}

let candidateIndexOpen=false,candidateIndexFrame=0,candidateIndexSignature='';
function jumpToCandidateFloor(floor){
 const list=ui('placeResults'),heading=list.querySelector('.result-floor-heading[data-floor="'+floor+'"]'),row=heading||[...list.querySelectorAll('.place-row')].find(b=>b.dataset.floor===String(floor));
 if(!row)return;list.style.paddingBottom=Math.max(12,list.clientHeight-40)+'px';mapUI.resultFloor=floor;list.scrollTop+=row.getBoundingClientRect().top-list.getBoundingClientRect().top;syncDiscoveryFloors();
}
function scheduleCandidateIndex(){
 if(reviewEmbedded)return;cancelAnimationFrame(candidateIndexFrame);candidateIndexFrame=requestAnimationFrame(()=>{const open=!document.querySelector('main').classList.contains('settings-hidden'),query=ui('findPlace').value.trim(),signature=query+'|'+mapUI.filter;if(searchFocusLock)return;
 if(open&&(!candidateIndexOpen||signature!==candidateIndexSignature))resetCandidateScroll();
 candidateIndexOpen=open;candidateIndexSignature=signature;});
}
function syncDiscoveryFloors(){
 if(reviewEmbedded)return;const searching=!document.querySelector('main').classList.contains('settings-hidden'),query=ui('findPlace').value,available=new Set(destinationList.filter(d=>resultCategoryMatches(d)&&matchesPlace(d,query)&&(!['toilet','equipment'].includes(d.type)||query.trim()||mapUI.filter===d.type||mapUI.filter==='equipment'&&d.type==='toilet')).map(d=>d.floor));
 for(const b of ui('allFloors').querySelectorAll('button')){const f=Number(b.dataset.floor);b.disabled=searching&&!available.has(f);b.setAttribute('aria-pressed',String(searching?mapUI.resultFloor===f:mapUI.floor===f));b.setAttribute('aria-label',searching?f+'階の候補へ移動':f+'階の地図を表示');}
}
const searchHistoryKey='maji-place-history-v1';
function readPlaceHistory(){try{const value=JSON.parse(localStorage.getItem(searchHistoryKey)||'[]');return Array.isArray(value)?value.filter(ref=>typeof ref==='string'&&destinationList.some(d=>d.ref===ref)).slice(0,10):[];}catch{return [];}}
function rememberPlace(d){if(!d||d.type==='manual'||!destinationList.some(p=>p.ref===d.ref))return;try{localStorage.setItem(searchHistoryKey,JSON.stringify([d.ref,...readPlaceHistory().filter(ref=>ref!==d.ref)].slice(0,10)));}catch{}updateSearchChoices();}
function updateSearchChoices(){
 if(!ui('searchChoices'))return;const current=mapUI.startConfirmed?(mapUI.manual?'地図で指定した場所':ui('currentName').textContent):'未設定';ui('searchCurrentPlace').title='現在地：'+current;ui('searchCurrentPlace').setAttribute('aria-label','現在地：'+current);
 const list=ui('recentPlaces');list.replaceChildren();for(const ref of readPlaceHistory().slice(0,10)){const d=destinationList.find(p=>p.ref===ref),b=document.createElement('button');b.dataset.ref=ref;b.textContent=destinationDisplayName(d);b.onclick=()=>selectPlace(d);list.append(b);}ui('recentEmpty').hidden=Boolean(list.children.length);
}
function chooseSearchCurrentPlace(){
 if(!mapUI.startConfirmed){ui('indoorLocationDialog').showModal();return;}
 if(mapUI.pickMode==='start'){setSelectionMode('goal');togglePlacePanel(false);showFloor(mapUI.manual?.floor||referenceFloor(mapUI.from));if(mapUI.goal)enterRoutePlanning();return;}
 if(mapUI.manual){const p=mapUI.manual;mapUI.manualGoal={...p,point:[...p.point]};choosePlace({ref:'hospital:'+p.floor+'::__manualGoal',routeReference:'hospital:'+p.floor+'::__manualGoal',name:'現在地',floor:p.floor,point:p.point,type:'manual',code:''},true);routePlan();}
 else{const d=destinationList.find(p=>navigationTargetReference(p)===mapUI.from||p.ref===mapUI.from);if(d)selectPlace(d);else{togglePlacePanel(false);showFloor(referenceFloor(mapUI.from));}}
}
function initSearchChoices(){
 const box=document.createElement('section');box.id='searchChoices';box.className='search-choices';box.setAttribute('aria-label','場所の選び方');box.innerHTML='<div class="search-choice-buttons"><button id="searchCurrentPlace">◎ 現在地</button><button id="searchMapPoint">地図上で選択</button><button id="searchCandidateList">候補から選択</button></div><div class="recent-heading">履歴</div><div id="recentPlaces" class="recent-places"></div><small id="recentEmpty">まだ履歴はありません</small>';ui('settings').querySelector('.destination-card').prepend(box);
 ui('searchCurrentPlace').onclick=chooseSearchCurrentPlace;
 ui('searchMapPoint').onclick=()=>{if(mapUI.pickMode==='start')mapUI.selectingStartOnMap=true;ui('findPlace').blur();togglePlacePanel(false);ui('facilityDetails').hidden=true;ui('mapMessage').dataset.important='true';ui('mapMessage').textContent=(mapUI.addingWaypoint?'経由地':mapUI.pickMode==='start'?'出発地':'目的地')+'にする施設や通路を地図上で選んでください。';};
 ui('searchCandidateList').onclick=()=>{setPlaceSearchView('browse');ui('findPlace').blur();ui('findPlace').value='';mapUI.filter='all';document.querySelectorAll('[data-filter]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.filter==='all')));renderResults();resetCandidateScroll();};
 const select=selectPlace,start=selectStartPlace;selectPlace=function(d,...args){rememberPlace(d);return select(d,...args);};selectStartPlace=function(d,...args){if(!isGuideOnly(d))rememberPlace(d);return start(d,...args);};
 const input=ui('findPlace');let saved=0,lockingFrame=0;
 input.addEventListener('pointerdown',()=>{saved=document.querySelector('main').classList.contains('settings-hidden')?0:ui('placeResults').scrollTop;},{capture:true});
 input.addEventListener('focus',()=>{document.body.classList.add('search-input-active');searchFocusLock=true;cancelAnimationFrame(candidateIndexFrame);cancelAnimationFrame(lockingFrame);lockingFrame=requestAnimationFrame(()=>{ui('placeResults').scrollTop=saved;window.scrollTo(0,0);lockingFrame=requestAnimationFrame(()=>{searchFocusLock=false;candidateIndexOpen=true;candidateIndexSignature=input.value.trim()+'|'+mapUI.filter;});});});
 input.addEventListener('blur',()=>document.body.classList.remove('search-input-active'));
 input.setAttribute('enterkeyhint','search');input.setAttribute('autocomplete','off');
 input.addEventListener('keydown',e=>{if(e.key!=='Enter'||e.isComposing||e.keyCode===229)return;e.preventDefault();setPlaceSearchView(input.value.trim()?'results':'home');input.blur();});
 input.addEventListener('input',()=>{searchFocusLock=false;setPlaceSearchView(input.value.trim()?'results':'home');});updateSearchChoices();
}
function activateIndexedFloor(floor){if(!document.querySelector('main').classList.contains('settings-hidden')){preferredCandidateFloor=floor;jumpToCandidateFloor(floor);}else if(mapUI.floor!==floor)showFloor(floor);}
function fitFloorRail(){if(reviewEmbedded)return;document.documentElement.style.setProperty('--floor-rail-height','320px');}
function initFloorSlide(){
 const floors=ui('allFloors');let pointer=null,last=null;
 const move=e=>{for(const b of floors.querySelectorAll('[data-floor]')){const r=b.getBoundingClientRect();if(e.clientX>=r.left&&e.clientX<=r.right&&e.clientY>=r.top&&e.clientY<=r.bottom&&!b.disabled){const f=Number(b.dataset.floor);if(f!==last){last=f;activateIndexedFloor(f);}return;}}};
 floors.addEventListener('pointerdown',e=>{if(e.button!==0||floors.scrollHeight>floors.clientHeight+1)return;e.stopPropagation();e.preventDefault();pointer=e.pointerId;last=null;floors.setPointerCapture(e.pointerId);move(e);},{capture:true});
 floors.addEventListener('pointermove',e=>{if(e.pointerId!==pointer)return;e.stopPropagation();e.preventDefault();move(e);});
 for(const name of ['pointerup','pointercancel'])floors.addEventListener(name,e=>{if(e.pointerId!==pointer)return;e.stopPropagation();if(floors.hasPointerCapture(pointer))floors.releasePointerCapture(pointer);pointer=null;last=null;});
 floors.addEventListener('click',e=>{const b=e.target.closest('[data-floor]');if(!b)return;e.stopImmediatePropagation();if(!b.disabled)activateIndexedFloor(Number(b.dataset.floor));},{capture:true});
}
function initDiscoveryExperience(){
 if(reviewEmbedded)return;
 const rail=document.createElement('nav');rail.className='map-navigation-rail';rail.setAttribute('aria-label','地図の表示と階選択');rail.append(document.querySelector('.view-switch'),document.querySelector('.floor-toolbar'));ui('mapViewport').append(rail);
 initFloorSlide();initSearchChoices();setPlaceSearchView('home');
 const equipment=ui('nearbyEquipmentBar');equipment.className='candidate-equipment';for(const b of [...equipment.querySelectorAll('button')]){const copy=b.cloneNode(true);b.replaceWith(copy);copy.onclick=()=>{setPlaceSearchView('results');mapUI.filter='all';mapUI.resultFloor=null;ui('findPlace').value=({payment:'自動支払機',parking:'駐車場精算機',vending:'自動販売機',aed:'AED',phone:'公衆電話',baby:'ベビールーム',scale:'身長 体重'})[copy.dataset.nearbyCategory]||'トイレ';document.querySelectorAll('[data-filter]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.filter==='all')));renderResults();syncSearchControls();};}ui('placeResults').before(equipment);
 const hours=document.createElement('div');hours.id='facilityHours';hours.className='facility-hours';hours.hidden=true;ui('facilityDetails').querySelector('.facility-detail-body').append(hours);
 for(const b of document.querySelectorAll('[data-filter]'))b.addEventListener('click',syncDiscoveryFloors);
 window.addEventListener('maji-map-update',()=>{scheduleCandidateIndex();syncDiscoveryFloors();updateFacilityHours(destinationPhotoState.place);updateSearchChoices();fitFloorRail();});
 ui('placeResults').addEventListener('scroll',()=>{if(document.querySelector('main').classList.contains('settings-hidden'))return;const list=ui('placeResults'),top=list.getBoundingClientRect().top;let active=null;for(const h of list.querySelectorAll('.result-floor-heading')){if(h.getBoundingClientRect().top<=top+8)active=Number(h.dataset.floor);else break;}if(active&&active!==mapUI.resultFloor){mapUI.resultFloor=active;syncDiscoveryFloors();}},{passive:true});
 fitFloorRail();
 setInterval(()=>updateFacilityHours(destinationPhotoState.place),60000);updateFacilityHours(destinationPhotoState.place);syncDiscoveryFloors();
}
initDiscoveryExperience();
