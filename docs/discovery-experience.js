function resultCategoryMatches(d){return mapUI.filter==='all'||d.type===mapUI.filter||mapUI.filter==='equipment'&&d.type==='toilet'||mapUI.filter==='service'&&['reception','entrance'].includes(d.type)||mapUI.filter==='ward'&&d.type==='rehab';}
function routeProximityFor(matches){
 const scores=new Map();if(!mapUI.startConfirmed||mapUI.pickMode==='start'||!matches.length||!matches.every(d=>['toilet','equipment'].includes(d.type)))return scores;
 const categories=new Set(matches.map(d=>d.type==='toilet'?'toilet':d.category));if(categories.size!==1)return scores;
 travelMode=ui('travelChoice').value;const graph=attachManualSelections(buildNetwork()),from=mapUI.manual?'hospital:'+mapUI.manual.floor+'::__manualStart':entryReference(mapUI.from),starts=typeof additionalNavigation!=='undefined'&&additionalNavigation.arrivalOptions?.[from]||[from],trees=new Map();
 for(const d of matches){let best=Infinity;if(d.nearestEligible!==false&&!isGuideOnly(d))for(const start of starts){const end=navigationTargetReference(d),via3F=routeNeeds3F(graph,start,end),key=start+'|'+via3F;let tree=trees.get(key);if(!tree){tree=auditedRouteTree(graph,start,via3F);trees.set(key,tree);}best=Math.min(best,tree.labels.get(end)?.cost??Infinity);}scores.set(d.ref,best);}return scores;
}
function syncDiscoveryFloors(){
 if(reviewEmbedded||!ui('allSearchFloors'))return;const searching=!document.querySelector('main').classList.contains('settings-hidden'),query=ui('findPlace').value,available=new Set(destinationList.filter(d=>resultCategoryMatches(d)&&matchesPlace(d,query)&&(!['toilet','equipment'].includes(d.type)||query.trim()||mapUI.filter===d.type||mapUI.filter==='equipment'&&d.type==='toilet')).map(d=>d.floor));
 ui('allSearchFloors').hidden=!searching;ui('allSearchFloors').setAttribute('aria-pressed',String(!mapUI.resultFloor));for(const b of ui('allFloors').querySelectorAll('button')){const f=Number(b.dataset.floor);b.disabled=searching&&!available.has(f);b.setAttribute('aria-pressed',String(searching?mapUI.resultFloor===f:mapUI.floor===f));b.setAttribute('aria-label',searching?f+'階の候補に絞る':f+'階の地図を表示');}
}
function initDiscoveryExperience(){
 if(reviewEmbedded)return;
 const rail=document.createElement('nav');rail.className='map-navigation-rail';rail.setAttribute('aria-label','地図の表示と階選択');rail.append(document.querySelector('.view-switch'),document.querySelector('.floor-toolbar'));ui('mapViewport').append(rail);
 const all=document.createElement('button');all.id='allSearchFloors';all.textContent='全階';all.hidden=true;all.onclick=()=>{mapUI.resultFloor=null;renderResults();};ui('allFloors').before(all);
 ui('allFloors').addEventListener('click',e=>{const b=e.target.closest('[data-floor]');if(!b||document.querySelector('main').classList.contains('settings-hidden'))return;e.stopImmediatePropagation();if(!b.disabled){mapUI.resultFloor=Number(b.dataset.floor);renderResults();}},{capture:true});
 const equipment=ui('nearbyEquipmentBar');equipment.className='candidate-equipment';for(const b of [...equipment.querySelectorAll('button')]){const copy=b.cloneNode(true);b.replaceWith(copy);copy.onclick=()=>{mapUI.filter='all';mapUI.resultFloor=null;ui('findPlace').value=({payment:'自動支払機',parking:'駐車場精算機',vending:'自動販売機',aed:'AED',phone:'公衆電話',baby:'ベビールーム',scale:'身長 体重'})[copy.dataset.nearbyCategory]||'トイレ';document.querySelectorAll('[data-filter]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.filter==='all')));renderResults();syncSearchControls();};}ui('placeResults').before(equipment);
 const hours=document.createElement('div');hours.id='facilityHours';hours.className='facility-hours';hours.hidden=true;ui('facilityDetails').querySelector('.facility-detail-body').append(hours);
 for(const b of document.querySelectorAll('[data-filter]'))b.addEventListener('click',syncDiscoveryFloors);
 window.addEventListener('maji-map-update',()=>{syncDiscoveryFloors();updateFacilityHours(destinationPhotoState.place);});
 setInterval(()=>updateFacilityHours(destinationPhotoState.place),60000);updateFacilityHours(destinationPhotoState.place);syncDiscoveryFloors();
}
initDiscoveryExperience();
