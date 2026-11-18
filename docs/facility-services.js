// Official facility page and Cabinet Office holiday calendar, checked 2026-10-29.
const facilityPage='https://www.med.kindai.ac.jp/floormap_facility.html';
const facilityFragment=text=>facilityPage+'#:~:text='+encodeURIComponent(text);
const facilityInformation={
 'services::seven':{url:facilityPage+'#anc_store01',hours:'7:00–22:00（年中無休）',rule:'daily',weekday:[420,1320]},
 'services::green':{url:facilityPage+'#anc_store02',hours:'平日8:00–19:00／土日祝9:00–17:00',rule:'green',weekday:[480,1140],weekend:[540,1020],anniversary:true,yearEnd:true,extra:'医療材料・雑貨：平日9:00–17:00（土日祝・11/5休）'},
 'services::cosmetics':{url:facilityFragment('化粧品コーナー'),hours:'平日9:00–17:00',rule:'weekday',weekday:[540,1020]},
 'services::glasses':{url:facilityFragment('西村眼鏡店'),hours:'平日9:30–17:30',rule:'weekday',weekday:[570,1050]},
 'services::hair':{url:facilityFragment('スヴェンソン近大ヘアケアステーション'),hours:'平日・第2/第4土曜9:30–18:00',rule:'hair',weekday:[570,1080]},
 'services::restaurant':{url:facilityFragment('ffields（エフフィールド）'),hours:'月–土10:00–18:00（日祝休）',rule:'restaurant',weekday:[600,1080]},
 'services::cafe':{url:facilityFragment('CAFE CORE（カフェ·コア）'),hours:'平日8:00–18:00（土日祝休）',rule:'weekday',weekday:[480,1080]},
 'outpatient:3::nankaiBusStop':{url:'https://transfer-cloud.navitime.biz/nankaibus/courses?busstop=00320794'},
 'outpatient:3::rotaryBusStop':{url:'https://www.city.osakasayama.osaka.jp/kurashi_anshin/doro_kotsu/2/1/index.html'},
 'services::lost':{url:facilityFragment('落とし物案内'),hours:'平日9:00–17:00／土曜9:00–14:00',rule:'lost',weekday:[540,1020],saturday:[540,840]},
 'services::information':{url:facilityPage+'#cell_f3info04'},
 'services::medicine':{url:facilityPage+'#cell_f3info02'},
 'services::admission':{url:facilityPage+'#cell_f3info05'},
 'hospital:3::generalReception':{url:facilityPage+'#cell_f3info01'}
};
const japanHolidayDates={2026:['01-01','01-12','02-11','02-23','03-20','04-29','05-03','05-04','05-05','05-06','07-20','08-11','09-21','09-22','09-23','10-12','11-03','11-23'],2027:['01-01','01-11','02-11','02-23','03-21','03-22','04-29','05-03','05-04','05-05','07-19','08-11','09-20','09-23','10-11','11-03','11-23']};
function japanClock(date=new Date()){const p=Object.fromEntries(new Intl.DateTimeFormat('en-GB',{timeZone:'Asia/Tokyo',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hourCycle:'h23'}).formatToParts(date).map(p=>[p.type,p.value]));return {year:Number(p.year),month:Number(p.month),day:Number(p.day),key:p.month+'-'+p.day,weekday:new Date(Date.UTC(Number(p.year),Number(p.month)-1,Number(p.day))).getUTCDay(),minute:Number(p.hour)*60+Number(p.minute)};}
function facilityBusinessState(info,date=new Date()){
 if(!info?.rule)return null;const t=japanClock(date),holidays=japanHolidayDates[t.year];
 if(info.rule!=='daily'&&!holidays)return {state:'unknown',label:'営業日を要確認',hours:info.hours};
 if(info.anniversary&&t.key==='11-05')return {state:'closed',label:'営業時間外（創立記念日）',hours:info.hours};
 if(info.yearEnd&&(t.month===12&&t.day>=25||t.month===1&&t.day<=7))return {state:'unknown',label:'年末年始の営業は要確認',hours:info.hours};
 const holiday=holidays?.includes(t.key),weekend=t.weekday===0||t.weekday===6;let interval=info.weekday;
 if(info.rule==='green'&&(weekend||holiday))interval=info.weekend;
 if(info.rule==='weekday'&&(weekend||holiday)||info.rule==='restaurant'&&(t.weekday===0||holiday)||info.rule==='lost'&&(t.weekday===0||holiday)||info.rule==='hair'&&(t.weekday===0||holiday||t.weekday===6&&![2,4].includes(Math.ceil(t.day/7))))interval=null;
 if(info.rule==='lost'&&t.weekday===6&&!holiday)interval=info.saturday;
 if(!interval)return {state:'closed',label:'営業時間外（休業日）',hours:info.hours};
 const [open,close]=interval,fmt=n=>Math.floor(n/60)+':'+String(n%60).padStart(2,'0');
 const state=t.minute<open&&open-t.minute<=30?'soon-open':t.minute>=open&&t.minute<close?(close-t.minute<=30?'soon-close':'open'):'closed';
 return {state,label:{'soon-open':'まもなく営業','soon-close':'まもなく終了',open:'営業中',closed:'営業時間外'}[state],today:fmt(open)+'–'+fmt(close),hours:info.hours};
}
function facilityLinksFor(d){if(d?.type==='entrance')return [{name:'出入口・開門時間（病院公式）',url:d.ref==='hospital:3::afterHoursEntrance'?'https://www.med.kindai.ac.jp/notice/2026_0114_6619.html':'https://www.med.kindai.ac.jp/notice/2026_0917_6883.html'}];if(d?.ref==='outpatient:3::patientParking')return [{name:'料金・利用案内（病院公式）',url:'https://www.med.kindai.ac.jp/parking_price.html'},{name:'患者さま駐車場 混雑状況',url:'https://kdu-fp.co.jp/parking'}];if(!d||['hospital:3::generalReception','services::information','services::medicine','services::admission'].includes(d.ref))return [];if(d.ref==='hospital:7::rehab')return [{name:'リハビリテーション部',url:'https://www.med.kindai.ac.jp/departments_and_centers/departments/05rehabilitation/'}];if(d.type==='clinic'){const names=clinicWebsiteNames[d.code]||[],seen=new Set(),links=[];for(const name of names){const url=departmentWebsites[name];if(url&&!seen.has(url)){seen.add(url);links.push({name,url});}}return links;}const info=facilityInformation[d.ref];if(info)return [{name:d.name,url:info.url}];return [];}
function facilityHoursModel(d,date=new Date()){
 if(d?.type==='entrance'){
  const status=entranceAvailability(d.ref,date),regular=regularEntranceRefs.includes(d.ref),open=d.ref==='hospital:3::receptionEntrance'?'7:50':'8:00',closedDay=Boolean(status.closedToday);
  return {state:status.state,label:regular?{open:'利用可能',closed:'閉鎖中','soon-close':'まもなく閉鎖',unknown:'要確認'}[status.state]:'利用可能',today:regular?(closedDay?'閉門':status.hours||'要確認'):'24時間',closedToday:regular&&closedDay,regular:regular?'平日 '+open+'–18:00 ／ 土曜 '+open+'–13:00（日曜・祝日閉門）':'毎日24時間',note:'病院公式の1月時間表・2026年10月1日変更を反映。日本時間。臨時変更は公式案内をご確認ください。'};
 }
 const info=facilityInformation[d?.ref],status=facilityBusinessState(info,date);if(!status)return null;
 return {...status,today:status.today||(status.state==='unknown'?'要確認':'休業'),closedToday:status.state==='closed'&&!status.today,regular:info.hours,extra:info.extra,note:'日本時間・通常の時間による目安。臨時変更は公式案内でご確認ください。'};
}
const sayarinStopRef='outpatient:3::rotaryBusStop';
// Osaka-Sayama City's Express B timetable, Kindai University Hospital departure row.
// https://www.city.osakasayama.osaka.jp/material/files/group/29/kyukou.pdf
const sayarinHospitalDepartures=['9:50','10:50','11:50','12:50','13:50','14:50','15:50','16:50','17:50'];
function departureMinute(time){
 const [hour,minute]=time.split(':').map(Number);
 return hour*60+minute;
}
function isHourlySchedule(times){
 return times.length>1&&times.slice(1).every((time,index)=>departureMinute(time)-departureMinute(times[index])===60);
}
function sayarinDepartures(date=new Date()){
 const {minute}=japanClock(date);
 const [next,...later]=sayarinHospitalDepartures.filter(time=>departureMinute(time)>=minute);
 return {next:next||'翌日 '+sayarinHospitalDepartures[0],later,final:sayarinHospitalDepartures.at(-1)};
}
function renderSayarinDepartures(host){
 const {next,later,final}=sayarinDepartures();
 const title='急行Bコース｜近畿大学病院発';
 const first=sayarinHospitalDepartures[0];
 const nextLabel=next===final?'最終':next===first||next==='翌日 '+first?'始発':'';
 const scheduleKey=[next,nextLabel,...later].join('|');
 if(host.dataset.reference===sayarinStopRef&&host.dataset.departures===scheduleKey)return;
 host.dataset.reference=sayarinStopRef;
 host.dataset.departures=scheduleKey;
 const heading=document.createElement('span');
 heading.className='bus-route-heading';
 heading.textContent=title;
 const nextRow=document.createElement('div');
 nextRow.className='bus-next-departure';
 const laterRow=document.createElement('div');
 laterRow.className='bus-later-departures';
 const departure=(time,label)=>{
  const item=document.createElement('span');
  item.className='bus-departure';
  item.append(time);
  if(label){
   const tag=document.createElement('span');
   tag.className=label==='始発'?'bus-first-label':'bus-final-label';
   tag.textContent=label;
   item.append(tag);
  }
  return item;
 };
 nextRow.append(departure(next,nextLabel),heading);
 if(isHourlySchedule(sayarinHospitalDepartures)){
  const frequency=document.createElement('span');
  frequency.className='bus-departure-frequency';
  frequency.textContent='毎時'+departureMinute(first)%60+'分発';
  laterRow.append(frequency,'　',departure(final,'最終'));
 }else{
  later.forEach((time,index)=>{
   if(index)laterRow.append('、');
   laterRow.append(departure(time,time===final?'最終':''));
  });
 }
 host.replaceChildren(nextRow);
 if(later.length)host.append(laterRow);
 host.title='大阪狭山市の公式時刻表による予定時刻です。臨時変更は公式サイトをご確認ください。';
}
const nankaiStopRef='outpatient:3::nankaiBusStop';
// Official stop timetable: https://static-transfer-cloud.navitime.biz/nankaibus/ja/busstop-timetable/20260518/0700200_01.pdf
const nankaiTimetables=[
 {label:'213 三原台回り（泉ヶ丘駅経由）',
  weekday:["6:15","6:32","7:01","7:11","7:34","7:52","8:11","8:28","9:01","9:25","9:59","10:27","10:57","11:22","11:52","12:22","12:52","13:22","13:52","14:22","14:52","15:22","15:52","16:22","16:53","17:23","17:54","18:06","18:30","18:51","19:03","19:25","19:45","20:18","20:56","21:17","21:53","22:18","22:43","23:04"],
  weekend:["6:22","6:47","7:17","7:47","8:17","8:47","9:17","9:47","10:17","10:47","11:17","11:47","12:17","12:47","13:17","13:47","14:17","14:47","15:17","15:47","16:17","16:52","17:18","17:53","18:19","18:54","19:30","19:53","20:17","20:57","21:18","21:43","22:28","22:57"]},
 {label:'270 金剛駅前行き（急行）',
  weekday:["7:23","7:45","8:30","8:42","9:32","10:02","10:32","11:02","11:32","12:02","12:32","13:02","13:32","14:02","14:32","15:02","15:32","16:02","16:32","17:02","17:45","18:16","18:40","19:06"],
  weekend:["7:27","7:59","8:27","8:42","9:37","10:07","10:37","11:07","11:37","12:07","12:37","13:07","13:32","14:07","14:37","15:07","15:37","16:07","16:42","17:09","17:48","18:14","18:37","19:03"]}
];

function nankaiDepartureRows(date=new Date()){
 const t=japanClock(date),holidays=japanHolidayDates[t.year];
 const day=t.weekday===0||t.weekday===6||holidays?.includes(t.key)?'weekend':holidays?'weekday':null;
 return nankaiTimetables.map(({label,[day]:times})=>{
  if(!times)return {label,time:'公式時刻表で確認',final:''};
  const next=times.find(time=>departureMinute(time)>=t.minute);
  return {label,time:next||'本日の運行終了',final:times.at(-1)};
 });
}
function renderNankaiDepartures(host){
 const rows=nankaiDepartureRows();
 const key=JSON.stringify(rows);
 if(host.dataset.reference===nankaiStopRef&&host.dataset.departures===key)return;
 host.dataset.reference=nankaiStopRef;
 host.dataset.departures=key;
 host.replaceChildren(...rows.map(({label,time,final})=>{
  const row=document.createElement('div');
  row.className='nankai-departure-row';
  const route=document.createElement('span');
  route.textContent=label;
  const times=document.createElement('span');
  const next=document.createElement('strong');
  next.textContent=time;
  times.append(next);
  if(final&&final!==time)times.append('　'+final+' 最終');
  if(final===time)times.append(' 最終');
  row.append(route,times);
  return row;
 }));
 host.title='南海バスの公式時刻表による予定時刻です。臨時変更は公式サイトをご確認ください。';
}

function updateFacilityHours(d){
 const host=document.getElementById('facilityHours');if(!host)return;
 if(d?.ref===sayarinStopRef){host.hidden=false;renderSayarinDepartures(host);return;}
 if(d?.ref===nankaiStopRef){host.hidden=false;renderNankaiDepartures(host);return;}
 const model=facilityHoursModel(d);host.hidden=!model;if(!model)return;const expanded=host.dataset.reference===d.ref&&Boolean(host.querySelector('details')?.open);host.dataset.reference=d.ref;host.replaceChildren();
 const badge=document.createElement('strong');badge.className='business-state';badge.dataset.state=model.state;badge.textContent=model.label;
 const today=document.createElement('span');today.className='facility-hours-today';today.textContent='本日 '+model.today;if(model.closedToday)today.dataset.state='closed';
 const regular=document.createElement('details');regular.className='facility-hours-regular';const toggle=document.createElement('summary');toggle.textContent='通常の時間';const hours=document.createElement('span');hours.textContent=model.regular;regular.append(toggle,hours);regular.open=expanded;toggle.onpointerdown=e=>e.stopPropagation();toggle.onclick=e=>{e.preventDefault();e.stopPropagation();regular.open=!regular.open;};
 if(model.extra){const extra=document.createElement('small');extra.textContent=model.extra;regular.append(extra);}
 host.append(badge,today,regular);host.title=model.note;
}
