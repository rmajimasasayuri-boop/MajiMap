// Search vocabulary changes labels only; route destinations and arrival points stay unchanged.
const placeSearchAliases={
 'outpatient:1::1A':['ほうしゃせんちりょう','放射線治療'],
 'outpatient:1::1E':['あいそとーぷ','ぺっと','PET検査'],
 'outpatient:2::2S':['レントゲン','れんとげん','X線','エックス線','x-ray','CT','シーティー','MRI','エムアールアイ','画像検査','放射線検査'],
 'outpatient:2::2R':['ER','救急','きゅうきゅうがいらい','救急センター'],
 'outpatient:2::2E':['つういんちりょうせんたー'],
 'outpatient:2::2L':['ないしきょう','胃カメラ','大腸カメラ'],
 'outpatient:3::3L':['さいけつ','血液検査','けつえきけんさ','せいりけんさ'],
 'outpatient:3::3P':['にゅういんてつづき','入院受付','かんじゃしえんせんたー','術前サポート'],
 'hospital:3::generalReception':['そうごううけつけ','そうごう','受付','総合','総合案内','初診受付','初めての受付','はじめてのうけつけ','初診','しょしん','しんさつうけつけ'],
 'services::medicine':['お薬','くすり','薬受取','おくすり','薬の受け取り'],
 'services::information':['いんふぉめーしょん','案内所'],
 'services::admission':['にゅういんあんない'],
 'services::seven':['せぶん','セブン','7イレブン','こんびに','コンビニ','こんびにえんすすとあ','せぶんいれぶん','売店','ばいてん','おにぎり','お弁当','おべんとう','買い物','かいもの'],
 'services::green':['ぐりーんりーぶす','医療用品'],
 'services::restaurant':['食堂','しょくどう','レストラン','れすとらん','食事','しょくじ','ご飯','ごはん','昼食','ちゅうしょく','ランチ','らんち','飲食店','いんしょくてん','エフフィールド','えふふぃーるど','ffields'],
 'services::cafe':['かふぇ','喫茶','きっさ','コーヒー','こーひー','珈琲','カフェコア','かふぇこあ','coffee'],
 'services::laundry':['洗濯','せんたく','らんどりー'],
 'services::lost':['落とし物','わすれもの','忘れ物'],
 'hospital:7::rehab':['りはびり','リハビリ']
};
const placeNameReadings={
 '外科':['げか'],'内科':['ないか'],'整形外科':['せいけいげか'],'脳神経外科':['のうしんけいげか'],'脳神経内科':['のうしんけいないか'],
 '消化器内科':['しょうかきないか'],'循環器内科':['じゅんかんきないか'],'心臓血管外科':['しんぞうけっかんげか'],'心臓外科':['しんぞうげか'],
 '血液内科':['けつえきないか'],'呼吸器':['こきゅうき'],'膠原病':['こうげんびょう'],'腫瘍内科':['しゅようないか'],
 '小児科':['しょうにか','こども'],'思春期':['ししゅんき'],'耳鼻':['じびか','耳鼻科','耳鼻咽喉科'],'眼科':['がんか','目医者','眼医者'],
 '皮膚科':['ひふか'],'形成外科':['けいせいげか'],'麻酔科':['ますいか'],'歯科':['しか','歯医者','はいしゃ'],
 '泌尿器':['ひにょうき'],'腎臓':['じんぞう'],'糖尿病':['とうにょうびょう'],'内分泌':['ないぶんぴつ'],'代謝':['たいしゃ'],
 '婦人科':['ふじんか'],'産科':['さんか'],'産婦人科':['さんふじんか'],'メンタルヘルス':['精神科','せいしんか','めんたる'],
 '漢方':['かんぽう'],'遺伝子':['いでんし'],'血液浄化':['けつえきじょうか'],'救急':['きゅうきゅう'],'中央処置':['ちゅうおうしょち'],
 '受付':['うけつけ'],'病棟':['びょうとう'],'出入口':['でいりぐち','入口','出口'],'時間外':['じかんがい','夜間','やかん'],
 'トイレ':['といれ','お手洗い','おてあらい','化粧室','WC'],
 '自動支払機':['会計','かいけい','精算','せいさん','支払い','しはらい','支払機'],
 '精算機':['せいさんき'],'駐車場':['ちゅうしゃじょう','駐車券','駐車料金'],
 '自動販売機':['じどうはんばいき','自販機','じはんき','飲み物','のみもの'],
 '公衆電話':['こうしゅうでんわ','電話','でんわ'],'ベビールーム':['べびーるーむ','授乳','じゅにゅう','おむつ','赤ちゃん'],
 '身長':['しんちょう'],'体重':['たいじゅう'],'コンビニ':['こんびに','売店','ばいてん'],'眼鏡':['めがね','メガネ'],
 'ヘアケア':['へあけあ'],'化粧品':['けしょうひん'],'AED':['エーイーディー','えーいーでぃー']
};
function normalizePlaceSearch(s){return String(s||'').normalize('NFKC').toLowerCase().replace(/[ァ-ヶ]/g,c=>String.fromCharCode(c.charCodeAt(0)-0x60)).replace(/[\s・、（）()「」『』／/\-ー]/g,'');}
function placeSearchFields(d){
 const name=[d.name,destinationDisplayName(d),...(d.departments||[])].join(' ');
 const aliases=[...(placeSearchAliases[d.ref]||[]),d.detail?.aliases||''];
 for(const [term,words]of Object.entries(placeNameReadings))if(name.includes(term))aliases.push(...words);
 return {name:normalizePlaceSearch(name),code:normalizePlaceSearch(d.code),aliases:aliases.map(normalizePlaceSearch),all:normalizePlaceSearch(name+' '+d.code+' '+aliases.join(' ')+' '+d.floor+'階 '+d.floor+'F')};
}
function placeSearchScore(d,query){
 const tokens=String(query||'').normalize('NFKC').split(/[\s　、,／/]+/).map(normalizePlaceSearch).filter(Boolean);
 if(!tokens.length)return 0;const fields=placeSearchFields(d);
 if(!tokens.every(token=>fields.all.includes(token)))return -1;
 return tokens.reduce((score,token)=>score+(fields.code===token?1000:normalizePlaceSearch(destinationDisplayName(d))===token?900:fields.aliases.includes(token)?850:fields.name.includes(token)?700:500),0);
}
