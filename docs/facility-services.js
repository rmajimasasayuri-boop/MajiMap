// Official facility page and Cabinet Office holiday calendar, checked 2026-10-29.
const facilityPage='https://www.med.kindai.ac.jp/floormap_facility.html';
const facilityFragment=text=>facilityPage+'#:~:text='+encodeURIComponent(text);
const facilityInformation={
 'services::seven':{url:facilityPage+'#anc_store01',hours:'7:00–22:00（年中無休）',rule:'daily',weekday:[420,1320]},
 'services::green':{url:facilityPage+'#anc_store02',hours:'平日8:00–19:00／土日祝9:00–17:00',rule:'green',weekday:[480,1140],weekend:[540,1020],anniversary:true,yearEnd:true,extra:'医療材料・雑貨：平日9:00–17:00（土日祝・11/5休）'},
 'services::cosmetics':{url:facilityFragment('化粧品コーナー'),hours:'平日9:00–17:00',rule:'weekday',weekday:[540,1020]},
 'services::glasses':{url:facilityFragment('西村眼鏡店'),hours:'平日9:30–17:30',rule:'weekday',weekday:[570,1050]},
 'services::hair':{url:facilityFragment('スヴェンソン近大ヘアケアステーション'),hours:'平日・第2/第4土曜9:30–18:00',rule:'hair',weekday:[570,1080]},
 'services::restaurant':{url:facilityPage+'#anc_cafe01',hours:'月–土10:00–18:00（日祝休）',rule:'restaurant',weekday:[600,1080]},
 'services::cafe':{url:facilityPage+'#anc_cafe02',hours:'平日8:00–18:00（土日祝休）',rule:'weekday',weekday:[480,1080]},
 'services::laundry':{url:facilityFragment('コインランドリー')},
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
function facilityLinksFor(d){if(!d)return [];if(d.ref==='hospital:7::rehab')return [{name:'リハビリテーション部',url:'https://www.med.kindai.ac.jp/departments_and_centers/departments/05rehabilitation/'}];if(d.type==='clinic'){const names=clinicWebsiteNames[d.code]||[],seen=new Set(),links=[];for(const name of names){const url=departmentWebsites[name];if(url&&!seen.has(url)){seen.add(url);links.push({name,url});}}return links.length?links:[{name:d.name,url:facilityPage+'#cell_f'+d.code}];}const info=facilityInformation[d.ref];if(info)return [{name:d.name,url:info.url}];if(d.type==='equipment'&&d.category==='phone')return [{name:'公衆電話の設置場所',url:facilityFragment('1号館（外来棟）2階コンビニ付近')}];return [];}
function updateFacilityHours(d){const host=document.getElementById('facilityHours');if(!host)return;const info=facilityInformation[d?.ref],status=facilityBusinessState(info);host.hidden=!status;if(!status)return;host.replaceChildren();const badge=document.createElement('strong');badge.className='business-state';badge.dataset.state=status.state;badge.textContent=status.label;const hours=document.createElement('span');hours.textContent=status.today?'本日 '+status.today:status.hours;host.append(badge,hours);host.title=info.hours+'。日本時間・通常営業時間による目安。臨時変更は公式案内でご確認ください。確認日：2026-10-29';if(info.extra){const extra=document.createElement('small');const medical=facilityBusinessState({...info,rule:'weekday',weekday:[540,1020],hours:'平日9:00–17:00'});extra.textContent='医療材料・雑貨：';const status=document.createElement('span');status.className='business-state';status.dataset.state=medical.state;status.textContent=medical.label;extra.append(status,document.createTextNode('（平日9:00–17:00）'));host.append(extra);}}
