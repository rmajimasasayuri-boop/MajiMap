// 2026-10-30: explicit screen transitions and user-confirmed visitor rules.
function searchHome(focus=false){closeVisitorPreview();ui('findPlace').value='';mapUI.filter='all';mapUI.resultFloor=null;togglePlacePanel(true,false);beginPlaceSearch();ui('findPlace').blur();if(focus)ui('findPlace').focus({preventScroll:true});}
function cancelDestination(){mapUI.goal=null;const url=new URL(location.href);url.searchParams.delete('goal');history.replaceState(null,'',url);mapUI.waypoint=null;mapUI.manualGoal=null;mapUI.manualVia=null;mapUI.addingWaypoint=false;clearRouteSelection();closeVisitorPreview();document.body.classList.remove('route-planning','place-selected');ui('routePlanner').hidden=true;ui('routeOptionsDock').hidden=true;ui('findPlace').value='';ui('facilityDetails').hidden=true;searchHome(false);updateRoutePlanner();}
const toiletIconTiles={
 'outpatient:1::publicToilet-central':[[547,512],[547,556]],
 'outpatient:2::publicToilet-central':[[548,470],[548,514],[548,558]],
 'outpatient:3::publicToilet-central':[[544,356],[544,400]],
 'outpatient:3::publicToilet-west':[[142,487],[202,487],[202,531]],
 'outpatient:3::publicToilet-south':[[544,667],[544,711]],
 'outpatient:3::publicToilet-medicine':[[1223,636],[1223,702],[1169,548]],
 'outpatient:4::publicToilet-central':[[540,508],[540,552]],
 'outpatient:4::publicToilet-south':[[540,819],[540,863]],
 'outpatient:5::publicToilet-wardEast':[[1440,378],[1440,422]],
 'outpatient:5::publicToilet-central':[[539,508],[539,552]],
 'outpatient:5::publicToilet-south':[[539,820],[539,864]],
 'outpatient:6::publicToilet-central':[[546,494],[546,538],[546,582]],
 'outpatient:6::publicToilet-south':[[548,866],[548,910]]};
function placeOfficialIcons(d){
 if(d.type==='equipment'){const id={payment:'33',parking:'34',vending:'36',aed:'23',phone:'26',baby:'03',scale:'29'}[d.category];return id?['<img class="facility-pict" alt="" referrerpolicy="no-referrer" src="https://www.med.kindai.ac.jp/img/common/map/icon-'+id+'.png">']:[];}
 if(d.type==='toilet')return (toiletIconTiles[d.ref]?.map(p=>[...p,44,44])||toiletHitBounds[d.ref]||[]).map(box=>officialIconSVG(d.floor,box));return [];
}
function refreshResultDetails(){
 const rows=[...ui('placeResults').querySelectorAll('.place-row')];for(const row of rows){const d=destinationList.find(d=>row.dataset.placeRef===d.ref);if(!d)continue;
 row.querySelector('.code')?.remove();row.querySelector('.place-row-copy small')?.remove();const icons=placeOfficialIcons(d);if(icons.length){const box=document.createElement('span');box.className='official-result-icons';box.innerHTML=icons.join('');row.prepend(box);}
 const state=entranceAvailability(d.ref);if(state.label){const badge=document.createElement('span');badge.className='entrance-status';badge.dataset.state=state.state;badge.textContent=state.label;row.querySelector('.place-row-copy').append(badge);}
 }
}
function appendHistoryStars(){for(const b of [...ui('recentPlaces').querySelectorAll('button')]){if(b.closest('.history-place-row'))continue;const ref=b.dataset.ref,d=destinationList.find(d=>d.ref===ref);if(!d)continue;const row=document.createElement('div');row.className='history-place-row';b.before(row);row.append(b);const star=document.createElement('button');star.className='history-star';const saved=readPlaceFavorites().includes(ref);star.textContent=saved?'★':'☆';star.setAttribute('aria-label',destinationDisplayName(d)+(saved?'のお気に入りを解除':'をお気に入りに登録'));star.onclick=()=>{toggleVisitorFavorite(d);updateSearchChoices();};row.append(star);}}
function previewManualDestination(point){const p=manualPointAt(point,mapUI.floor,mapUI.base*mapUI.zoom);if(!p){ui('mapMessage').dataset.important='true';ui('mapMessage').textContent='通路に接続できる場所を選んでください。';return;}const d={ref:'hospital:'+p.floor+'::__manualGoal',routeReference:'hospital:'+p.floor+'::__manualGoal',name:'地図で指定した場所',floor:p.floor,point:p.point,type:'manual',manualSelection:p,code:''};showVisitorPreview(d);}
function initVisitorUpdates(){
 if(reviewEmbedded)return;
 const baseOverlay=renderOverlay;renderOverlay=function(){baseOverlay();const d=mapUI.previewPlace;if(d?.manualSelection&&d.floor===mapUI.floor){const marker=svgNode('circle',{cx:d.point[0],cy:d.point[1],r:12,fill:'#d83a40',stroke:'white','stroke-width':3,'pointer-events':'none'});ui('unifiedOverlay').append(marker);}};
 const baseChoose=choosePlace;choosePlace=function(d,...args){mapUI.manualVia=null;return baseChoose(d,...args);};
 const baseResults=renderResults;renderResults=function(){baseResults();refreshResultDetails();};
 const baseChoices=updateSearchChoices;updateSearchChoices=function(){baseChoices();appendHistoryStars();};
 const baseFavorites=updateVisitorFavorites;updateVisitorFavorites=function(){baseFavorites();appendHistoryStars();};
 const basePanel=togglePlacePanel;togglePlacePanel=function(open,focus=false){if(open)closeVisitorPreview();basePanel(open,focus);};
 const baseSelect=selectPlace;selectPlace=function(d,...args){const via=mapUI.addingWaypoint;const result=baseSelect(d,...args);if(via&&mapUI.waypoint){togglePlacePanel(false);ui('findPlace').blur();closeVisitorPreview();syncRoutePanels();}return result;};
 mapUI.waypoint=null;mapUI.manualVia=null;mapUI.addingWaypoint=false;previewOffersWaypoint=()=>false;beginWaypointSelection=()=>{};setVisitorWaypoint=()=>{};

 const basePreviewActions=updatePreviewActions;updatePreviewActions=function(){basePreviewActions();const d=destinationPhotoState.place;if(d?.type==='manual'){ui('detailRoute').textContent='経路';ui('detailNavigate').disabled=false;ui('detailRoute').disabled=false;}if(d?.type!=='manual')ui('detailRoute').textContent='経路';if(d&&entranceAvailability(d.ref).closed){ui('detailNavigate').disabled=true;ui('detailNavigate').textContent='閉鎖中';ui('detailRoute').disabled=true;}};
 const baseNavigate=navigateVisitorPreview;navigateVisitorPreview=function(){const d=destinationPhotoState.place;if(d?.manualSelection)mapUI.manualGoal=d.manualSelection;if(d?.type!=='manual')ui('detailRoute').textContent='経路';if(d&&entranceAvailability(d.ref).closed)return;return baseNavigate();};
 ui('detailNavigate').onclick=()=>navigateVisitorPreview();
 const baseManual=selectMapPoint;selectMapPoint=function(point){if(mapUI.pickMode==='start')return baseManual(point);previewManualDestination(point);};
 ui('detailRoute').onclick=()=>{const d=destinationPhotoState.place;if(!d||entranceAvailability(d.ref).closed)return;if(d.manualSelection)mapUI.manualGoal=d.manualSelection;closeVisitorPreview();choosePlace(d,true);rememberPlace(d);enterRoutePlanning();};
 ui('closeFacilityDetails').onclick=()=>{const other=mapUI.previewPlace&&mapUI.goal&&mapUI.previewPlace.ref!==mapUI.goal.ref;closeVisitorPreview();if(other||mapUI.goal&&document.body.classList.contains('route-planning')){enterRoutePlanning();return;}cancelDestination();};
 ui('closeRoutePlanning').textContent='×';ui('closeRoutePlanning').setAttribute('aria-label','目的地を取り消して検索へ戻る');ui('closeRoutePlanning').onclick=cancelDestination;
 ui('clearPlaceSearch').onclick=()=>{ui('findPlace').blur();if(mapUI.goal)cancelDestination();else searchHome(false);};
 ui('openPlaceSearch').onclick=()=>{if(document.body.classList.contains('place-searching')){if(placeSearchView!=='home'){searchHome(false);return;}togglePlacePanel(false);ui('findPlace').blur();if(mapUI.goal)enterRoutePlanning();}else searchHome(true);};
 const travel=ui('travelChoice');let saved;try{saved=localStorage.getItem('maji-visitor-travel');}catch{}travel.value=saved==='elevator'?'elevator':'auto';travelMode=travel.value;const changed=travel.onchange;travel.onchange=()=>{try{localStorage.setItem('maji-visitor-travel',travel.value);}catch{}changed();};updateRoutePlanner();
 const oldSync=syncVisitorRefinements;syncVisitorRefinements=function(){mapUI.waypoint=null;mapUI.manualVia=null;mapUI.addingWaypoint=false;oldSync();ui('routeAddWaypoint').hidden=true;ui('routeViaRow').hidden=true;};

 const oldClose=closeVisitorPreview;closeVisitorPreview=function(){oldClose();mapUI.previewPlace=null;};
 const oldPhoto=renderPlacePhoto;renderPlacePhoto=function(d){oldPhoto(d);if(d?.type==='entrance'){const host=ui('facilityHours'),state=entranceAvailability(d.ref);host.hidden=false;host.textContent=state.label;host.className='facility-hours entrance-status';host.dataset.state=state.state;}updatePreviewActions();};
 const oldPlan=routePlan;routePlan=function(){if(mapUI.goal&&entranceAvailability(mapUI.goal.ref).closed){clearRouteSelection();ui('mapMessage').dataset.important='true';ui('mapMessage').textContent='この出入口は閉鎖中です。時間外出入口を選択してください。';return;}return oldPlan();};
 const nearest=document.createElement('button');nearest.id='nearestEntrance';nearest.textContent='最寄りの出入口';nearest.className='secondary';nearest.onclick=()=>{if(!ensureIndoorStart())return;mapUI.filter='entrance';ui('findPlace').value='';togglePlacePanel(true);setPlaceSearchView('browse');renderResults();};ui('searchChoices').querySelector('.search-choice-buttons').append(nearest);
 updateSearchChoices();renderResults();syncVisitorRefinements();
 let last=regularEntranceRefs.map(r=>entranceAvailability(r).state).join();setInterval(()=>{const next=regularEntranceRefs.map(r=>entranceAvailability(r).state).join();if(next!==last){last=next;renderResults();if(mapUI.goal){stopNavigation();routePlan();}}},15000);
}
initVisitorUpdates();
