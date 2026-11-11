// 2026-10-30: explicit screen transitions and user-confirmed visitor rules.
function searchHome(focus=false){closeVisitorPreview();ui('findPlace').value='';mapUI.filter='all';mapUI.resultFloor=null;togglePlacePanel(true,false);beginPlaceSearch();ui('findPlace').blur();if(focus)ui('findPlace').focus({preventScroll:true});}
function cancelDestination(){mapUI.selectingEndpointOnMap=null;mapUI.goal=null;const url=new URL(location.href);url.searchParams.delete('goal');history.replaceState(null,'',url);mapUI.waypoint=null;mapUI.manualGoal=null;mapUI.manualVia=null;mapUI.addingWaypoint=false;clearRouteSelection();closeVisitorPreview();document.body.classList.remove('route-planning','place-selected');ui('routePlanner').hidden=true;ui('routeOptionsDock').hidden=true;ui('findPlace').value='';ui('facilityDetails').hidden=true;searchHome(false);updateRoutePlanner();}
const toiletIconTiles={
 'outpatient:2::publicToilet-south':[[1647,773],[1698,773],[1668,719]],
 'outpatient:2::publicToilet-gastro':[[1048,565],[1048,628],[1048,688]],
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
 if(d.type==='equipment'){
  const icon=officialEquipmentIcon(d.category);
  return icon?[icon]:[];
 }
 if(d.type==='toilet'){
 const names={'01':'男性トイレ','02':'女性トイレ','05':'車椅子対応','24':'オストメイト対応','35':'ストレッチャー対応',unisex:'男女共用トイレ'};
 return (d.detail?.toiletIcons||['01','02']).map(id=>'<img class="facility-pict" alt="'+names[id]+'" title="'+names[id]+'" src="assets/legend-toilet-'+id+'.svg?v=20261110l'+'">');
 }return [];
}
const candidateSignColors={green:'#b0c548',yellow:'#e8b52f',blue:'#3989c6'};
function orderWardCandidates(places){const ordered=[...places],floors=new Set(places.filter(d=>d.type==='ward').map(d=>d.floor)),rank={westA:0,westB:1,eastA:2,eastB:3};for(const floor of floors){const indices=ordered.flatMap((d,i)=>d.type==='ward'&&d.floor===floor?[i]:[]),wards=indices.map(i=>ordered[i]).sort((a,b)=>(rank[a.ref.split('::').pop()]??99)-(rank[b.ref.split('::').pop()]??99));indices.forEach((index,i)=>ordered[index]=wards[i]);}return ordered;}
function styleCandidateCode(code,d){code.hidden=!(d.type==='clinic'||d.type==='ward'||d.type==='rehab');code.textContent=d.code||'';const ward=d.ref.split('::').pop(),colors={eastA:candidateSignColors.blue,eastB:candidateSignColors.yellow,westA:candidateSignColors.green,westB:'#eb808b'};const signColor=/^[1-6][ABJK]$/.test(d.code||'')?candidateSignColors.green:/^[1-6][CE]$/.test(d.code||'')?candidateSignColors.yellow:candidateSignColors.blue;const color=d.type==='rehab'?candidateSignColors.green:d.type==='ward'?(colors[ward]||candidateSignColors.blue):signColor;const unit=d.floor===5&&/^(ACU|HCU|ICU|CCMC|SCU)$/.test(d.code||'');code.dataset.kind=unit?'unit':d.type;code.style.setProperty('--sign-color',color);code.style.background=d.type==='ward'&&!unit?'#fff':color;
 if(d.code==='CCMC'){const label=document.createElement('span');label.className='code-condensed';label.textContent=d.code;code.replaceChildren(label);}
 else if(d.type==='ward'&&/^10[東西][AB]$/.test(d.code||'')){const ten=document.createElement('span'),digits=document.createElement('span'),rest=document.createElement('span');ten.className='code-ten';digits.textContent='10';ten.append(digits);rest.className='code-remainder';rest.textContent=d.code.slice(2);code.replaceChildren(ten,rest);}
}
function refreshResultDetails(){
 const rows=[...ui('placeResults').querySelectorAll('.place-row')];for(const row of rows){const d=destinationList.find(d=>row.dataset.placeRef===d.ref);if(!d)continue;
 const code=row.querySelector('.code');if(code)styleCandidateCode(code,d);row.querySelector('.place-row-copy small')?.remove();const icons=placeOfficialIcons(d);if(icons.length){const box=document.createElement('span');box.className='official-result-icons';box.innerHTML=icons.join('');row.prepend(box);}
 const state=entranceAvailability(d.ref);if(state.label){const badge=document.createElement('span');badge.className='entrance-status';badge.dataset.state=state.state;badge.textContent=state.label;row.querySelector('.place-row-copy').append(badge);}
 }
}
function renderFavoriteStar(button){button.innerHTML='<svg class="favorite-star-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M12 2.8 14.8 8.6 21.2 9.5 16.6 14 17.7 20.4 12 17.4 6.3 20.4 7.4 14 2.8 9.5 9.2 8.6Z"/></svg>';}
function placeListText(d){const name=destinationDisplayName(d);return /^(?:[1-9]|10)(?:[A-Z]|[東西][AB])$/.test(d.code||'')&&!name.startsWith(d.code)?d.code+' '+name:name;}
function appendHistoryStars(){for(const b of [...ui('recentPlaces').querySelectorAll('button')]){if(b.classList.contains('history-star'))continue;const ref=b.dataset.ref,d=destinationList.find(d=>d.ref===ref);if(!d)continue;b.textContent=placeListText(d);if(b.closest('.history-place-row'))continue;const row=document.createElement('div');row.className='history-place-row';b.before(row);row.append(b);const star=document.createElement('button');star.className='history-star';const saved=readPlaceFavorites().includes(ref);renderFavoriteStar(star);star.dataset.saved=String(saved);star.setAttribute('aria-label',destinationDisplayName(d)+(saved?'のお気に入りを解除':'をお気に入りに登録'));star.onclick=()=>{toggleVisitorFavorite(d);updateSearchChoices();};row.append(star);}}
function previewManualDestination(point){return; /* Registered facilities only. */}

let facilityNameObserver=null;
function updateCardHeading(d){
 if(!d)return;facilityNameObserver?.disconnect();const title=ui('facilityDetailName');title.classList.remove('inline-code-heading');title.style.removeProperty('--heading-size');title.replaceChildren();const code=document.createElement('small');code.className='facility-arrival-code';code.textContent=d.detail?.mapCaption|| (d.ref==='outpatient:3::patientParking'?'屋外・駐車場':d.ref==='outpatient:3::building7Entrance'?'屋外・7号館':/^(?:[1-9]|10)(?:[A-Z]|[東西][AB])$/.test(d.code||'')?d.code:d.floor+'階');const track=document.createElement('span');track.className='facility-name-track';const name=document.createElement('span');name.textContent=['equipment','toilet'].includes(d.type)?destinationDisplayName(d).replace(/^\d+階\s+/,''):destinationDisplayName(d);track.append(name);title.append(code,track);
 const measure=()=>{if(!track.isConnected||track.clientWidth===0)return;title.classList.remove('inline-code-heading');track.className='facility-name-track two-line';title.style.removeProperty('--heading-size');track.style.maxHeight='none';name.style.fontSize='';track.replaceChildren(name);const base=Math.max(16,parseFloat(getComputedStyle(title).fontSize)||16);const fits=(lines)=>name.getBoundingClientRect().height<=parseFloat(getComputedStyle(name).fontSize)*1.4*lines+2;let done=false;for(let size=base;size>=16;size-=.5){name.style.fontSize=size+'px';if(fits(3)){done=true;break;}}if(!done){title.classList.add('inline-code-heading');track.classList.remove('two-line');for(let size=base;size>=12;size-=.5){title.style.setProperty('--heading-size',size+'px');name.style.fontSize=size+'px';if(fits(3))break;}}track.title=destinationDisplayName(d);};
 facilityNameObserver=new ResizeObserver(measure);facilityNameObserver.observe(track);requestAnimationFrame(measure);document.fonts?.ready.then(measure);
}
function initVisitorUpdates(){
 if(reviewEmbedded)return;
 const baseOverlay=renderOverlay;renderOverlay=function(){baseOverlay();syncPlaceHighlights();const d=mapUI.previewPlace;if(d?.manualSelection&&d.floor===mapUI.floor){const marker=svgNode('circle',{cx:d.point[0],cy:d.point[1],r:12,fill:mapUI.pickMode==='start'?'#2580c5':'#d83a40',stroke:'white','stroke-width':3,'pointer-events':'none'});ui('unifiedOverlay').append(marker);}};
 const baseChoose=choosePlace;choosePlace=function(d,...args){mapUI.manualVia=null;const result=baseChoose(d,...args);if(d)ui('findPlace').value=placeListText(d);return result;};
 const baseResults=renderResults;renderResults=function(){baseResults();refreshResultDetails();};
 const baseChoices=updateSearchChoices;updateSearchChoices=function(){baseChoices();appendHistoryStars();};
 const baseFavorites=updateVisitorFavorites;updateVisitorFavorites=function(){baseFavorites();renderFavoriteStar(ui('favoritePlaceButton'));for(const [i,row] of [...ui('favoritePlaces').querySelectorAll('.favorite-place-row')].entries()){const d=destinationList.find(d=>d.ref===readPlaceFavorites()[i]);if(d)row.firstElementChild.textContent=placeListText(d);renderFavoriteStar(row.lastElementChild);}appendHistoryStars();};
 const favoriteDialog=document.createElement('dialog');favoriteDialog.id='favoriteRemovalDialog';favoriteDialog.className='map-dialog';favoriteDialog.innerHTML='<h2>お気に入りから削除しますか？</h2><p id="favoriteRemovalName"></p><div class="actions"><button id="cancelFavoriteRemoval">キャンセル</button><button id="confirmFavoriteRemoval" class="primary">削除</button></div>';document.body.append(favoriteDialog);let pendingFavorite=null;const baseFavorite=toggleVisitorFavorite;toggleVisitorFavorite=function(d=destinationPhotoState.place){if(d&&readPlaceFavorites().includes(destinationIdentity(d.ref))){pendingFavorite=d;ui('favoriteRemovalName').textContent=destinationDisplayName(d);favoriteDialog.showModal();return;}return baseFavorite(d);};ui('favoritePlaceButton').onclick=()=>{baseFavorite();updateSearchChoices();};ui('cancelFavoriteRemoval').onclick=()=>{pendingFavorite=null;favoriteDialog.close();};ui('confirmFavoriteRemoval').onclick=()=>{const d=pendingFavorite;pendingFavorite=null;favoriteDialog.close();if(d){baseFavorite(d);updateSearchChoices();}};
 const basePanel=togglePlacePanel;togglePlacePanel=function(open,focus=false){if(open)closeVisitorPreview();basePanel(open,focus);if(open)resetCandidateScroll();};
 const baseSelect=selectPlace;selectPlace=function(d,...args){const via=mapUI.addingWaypoint;if(mapUI.pickMode!=='start')mapUI.destinationSearchQuery=ui('findPlace').value;const result=baseSelect(d,...args);if(via&&mapUI.waypoint){togglePlacePanel(false);ui('findPlace').blur();closeVisitorPreview();syncRoutePanels();}return result;};

 const basePreviewActions=updatePreviewActions;
 updatePreviewActions=function(){
  basePreviewActions();
  const d=destinationPhotoState.place,route=ui('detailRoute'),navigate=ui('detailNavigate');
  const parking=d?.ref==='outpatient:3::patientParking'&&!unifiedMapModel.floors[3].outdoorExtension.completed;
  route.textContent='出発地';
  route.disabled=!d||d.usage==='emergency-only'||isGuideOnly(d);
  const mapChoice=mapUI.selectingEndpointOnMap;
  route.hidden=parking||mapChoice==='goal';
  navigate.textContent='目的地';
  navigate.disabled=!d||isGuideOnly(d);
  navigate.hidden=parking||mapChoice==='start';
  const website=ui('detailWebsite');
  if(mapChoice&&d&&website.hidden){
   website.href='https://www.med.kindai.ac.jp/floormap_facility.html';
   website.onclick=null;
   website.hidden=false;
  }
  ui('facilityDetails').dataset.previewAction='endpoints';
 };
 ui('detailRoute').onclick=()=>{
  const d=destinationPhotoState.place;
  if(!d||d.usage==='emergency-only'||isGuideOnly(d))return;
  closeVisitorPreview();
  mapUI.selectingEndpointOnMap=null;
  selectStartPlace(d,true);
  if(mapUI.goal)ui('findPlace').value=placeListText(mapUI.goal);
  enterRoutePlanning();
  updateRoutePlanner();
  renderOverlay();
 };
 previewPlacePhoto=function(d,anchor,pinned=false){if(pinned&&d)showVisitorPreview(d);};
 ui('detailNavigate').onclick=()=>{
  const d=destinationPhotoState.place;
  if(!d||isGuideOnly(d))return;
  mapUI.selectingEndpointOnMap=null;
  setSelectionMode('goal');
  closeVisitorPreview();
  choosePlace(d,true);
  rememberPlace(d);
  enterRoutePlanning();
 };
 ui('closeFacilityDetails').onclick=()=>{if(mapUI.previewPlace){closeVisitorPreview();return;}const other=mapUI.previewPlace&&mapUI.goal&&mapUI.previewPlace.ref!==mapUI.goal.ref;closeVisitorPreview();if(other||mapUI.goal&&document.body.classList.contains('route-planning')){enterRoutePlanning();return;}cancelDestination();};
 ui('closeRoutePlanning').textContent='‹';ui('closeRoutePlanning').setAttribute('aria-label','目的地のマップと写真カードに戻る');ui('closeRoutePlanning').onclick=()=>{const d=mapUI.goal;if(!d){document.body.classList.remove('route-planning');updateRoutePlanner();searchHome(false);return;}clearRouteSelection();closeVisitorPreview();document.body.classList.remove('route-planning');ui('routePlanner').hidden=true;ui('routeOptionsDock').hidden=true;togglePlacePanel(false,false);ui('findPlace').value=placeListText(d);ui('findPlace').blur();renderPlacePhoto(d);updateRoutePlanner();showFloor(d.floor,true);syncRoutePanels();requestAnimationFrame(focusReadableMap);};
 ui('clearPlaceSearch').onclick=()=>{ui('findPlace').blur();cancelDestination();togglePlacePanel(false,false);mapUI.plot=false;setSelectionMode('goal');showFloor(mapUI.manual?.floor||referenceFloor(mapUI.from)||3,true);syncSearchControls();};
 ui('openPlaceSearch').onclick=()=>{if(document.body.classList.contains('place-searching')){if(placeSearchView!=='home'){searchHome(false);return;}togglePlacePanel(false);ui('findPlace').blur();if(mapUI.goal)enterRoutePlanning();}else searchHome(true);};
 const travel=ui('travelChoice');let saved;try{saved=localStorage.getItem('maji-visitor-travel');}catch{}travel.value=saved==='elevator'?'elevator':'auto';travelMode=travel.value;const changed=travel.onchange;travel.onchange=()=>{try{localStorage.setItem('maji-visitor-travel',travel.value);}catch{}changed();};updateRoutePlanner();

 const oldClose=closeVisitorPreview;closeVisitorPreview=function(){oldClose();mapUI.previewPlace=null;};
 const information=destinationList.find(d=>d.ref==='services::information');if(information)information.name='インフォメーション・情報コーナー';
 const oldPhoto=renderPlacePhoto;renderPlacePhoto=function(d){oldPhoto(d);updateCardHeading(d);updateFacilityHours(d);updatePreviewActions();};
 const oldPlan=routePlan;routePlan=function(){const notice=mapUI.goal&&routeEntranceNotice(mapUI.from,mapUI.goal.ref,null,[]);if(notice?.blocked){clearRouteSelection();if(!reviewEmbedded)syncRoutePanels();return;}return oldPlan();};
 updateSearchChoices();updateVisitorFavorites();renderResults();syncVisitorRefinements();updateCardHeading(destinationPhotoState.place);if(mapUI.goal&&!document.body.classList.contains('place-searching'))ui('findPlace').value=placeListText(mapUI.goal);

 let last=regularEntranceRefs.map(r=>entranceAvailability(r).state).join();setInterval(()=>{const next=regularEntranceRefs.map(r=>entranceAvailability(r).state).join();if(next!==last){last=next;renderResults();updateFacilityHours(destinationPhotoState.place);updatePreviewActions();if(mapUI.goal){stopNavigation();routePlan();}}},15000);
}
initVisitorUpdates();

if(!reviewEmbedded)requestAnimationFrame(()=>{
 if(useMobileWidthFit(ui('mapViewport'))){mapUI.zoom=1;mapUI.pan=[0,0];fitMap();}
 else focusReadableMap();
 updateRoutePlanner();
});
