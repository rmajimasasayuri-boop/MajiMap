function updateRoutePlanner(){
 const panel=document.getElementById('routePlanner');if(!panel)return;
 panel.hidden=!document.body.classList.contains('route-planning');
 const display=d=>typeof placeListText==='function'?placeListText(d):destinationDisplayName(d);const origin=destinationList.find(d=>d.ref===ui('fromSelect').value&&navigationTargetReference(d)===mapUI.from)||destinationList.find(d=>navigationTargetReference(d)===mapUI.from);
 const start=!mapUI.startConfirmed?'出発地を設定してください':mapUI.manual?'地図で指定した現在地':origin?display(origin):ui('currentName').textContent;
 ui('routeStartLabel').textContent=start;ui('routeGoalLabel').textContent=mapUI.goal?display(mapUI.goal):'目的地を設定してください';
 ui('routeStartField').setAttribute('aria-label','出発地を変更：'+start);
 ui('routeGoalField').setAttribute('aria-label','目的地を変更：'+(mapUI.goal?display(mapUI.goal):'未設定'));
 for(const b of document.querySelectorAll('[data-travel-choice]'))b.setAttribute('aria-pressed',String(b.dataset.travelChoice===ui('travelChoice').value));
}
function initRoutePlanner(){
 for(const b of document.querySelectorAll('[data-travel-choice]'))b.onclick=()=>{ui('travelChoice').value=b.dataset.travelChoice;ui('travelChoice').onchange();updateRoutePlanner();};
 ui('routeStartField').onclick=()=>{setSelectionMode('start');ui('findPlace').value='';renderResults();togglePlacePanel(true,true);};
 ui('routeGoalField').onclick=()=>{setSelectionMode('goal');togglePlacePanel(true,true);};
 window.addEventListener('maji-map-update',updateRoutePlanner);
}
