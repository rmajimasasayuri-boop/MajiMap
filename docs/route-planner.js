function updateRoutePlanner(){
 const panel=document.getElementById('routePlanner');if(!panel)return;
 panel.hidden=!mapUI.goal||!document.body.classList.contains('route-planning');
 const start=!mapUI.startConfirmed?'現在地を設定してください':mapUI.manual?'地図で指定した現在地':ui('currentName').textContent;
 ui('routeStartLabel').textContent=start;ui('routeGoalLabel').textContent=mapUI.goal?destinationDisplayName(mapUI.goal):'目的地を選択';
 ui('routeStartField').setAttribute('aria-label','出発地を変更：'+start);
 ui('routeGoalField').setAttribute('aria-label','目的地を変更：'+(mapUI.goal?destinationDisplayName(mapUI.goal):'未設定'));
 for(const b of document.querySelectorAll('[data-travel-choice]'))b.setAttribute('aria-pressed',String(b.dataset.travelChoice===ui('travelChoice').value));
}
function initRoutePlanner(){
 for(const b of document.querySelectorAll('[data-travel-choice]'))b.onclick=()=>{ui('travelChoice').value=b.dataset.travelChoice;ui('travelChoice').onchange();updateRoutePlanner();};
 ui('routeStartField').onclick=()=>{setSelectionMode('start');ui('findPlace').value='';renderResults();togglePlacePanel(true,true);};
 ui('routeGoalField').onclick=()=>{setSelectionMode('goal');togglePlacePanel(true,true);};
 ui('chooseCurrentMapPoint').onclick=()=>{setSelectionMode('start');togglePlacePanel(false);ui('facilityDetails').hidden=true;ui('mapMessage').dataset.important='true';ui('mapMessage').textContent='現在いる場所に近い、登録済み通路または施設名を押してください。青い点が出発地になります。';};
 window.addEventListener('maji-map-update',updateRoutePlanner);
}
