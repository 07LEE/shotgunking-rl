(async()=>{
 const checks=[];
 function check(ok,name){if(!ok)throw Error(name);checks.push(name)}
 async function idle(){for(let i=0;i<300;i++){if(data&&!loading&&loadedId===currentId)return;await new Promise(r=>setTimeout(r,20))}throw Error('load timeout')}
 async function pick(id){$('items').value=id;$('items').onchange();await idle()}
 window.confirm=()=>true;
 try {
  await idle();
  check(currentId.endsWith('b-partial.png.json'),'starts at unfinished screenshot');
  check($('items').options.length===2&&!Array.from($('items').options).some(x=>x.value.includes('a-complete')),'completed screenshot hidden');
  switchReviewTab('screen');await idle();
  check($('ammo_loaded').disabled&&$('ammoReviewed').disabled,'reviewed ammunition locked');
  check(!$('attack').disabled&&!$('combatReviewed').disabled,'unfinished combat editable');
  check($('screenKind').disabled,'reviewed screen kind locked');
  const originalApi=api;
  api=async(url,...args)=>url==='/api/predict'?{floor:null,screen_state:{screen_kind:null,ammo:{loaded:9,reserve:9},combat_stats:{...Object.fromEntries(combatFields.map(key=>[key,null])),range_min:3},souls:null}}:originalApi(url,...args);
  await $('predictState').onclick();api=originalApi;
  check($('ammo_loaded').value==='2'&&$('ammoReviewed').checked&&$('range_min').value==='3'&&$('attack').value==='4','auto-fill leaves reviewed values and approvals intact');
  $('showCompleted').checked=true;await $('showCompleted').onchange();await idle();
  check($('items').options.length===3,'explicit completed view');
  await pick('sessions/play/annotations/a-complete.png.json');
  check($('ammo_loaded').disabled&&$('attack').disabled&&$('addSoul').disabled,'completed page sections locked');
  check($('draft').disabled&&$('approve').disabled,'completed page save actions locked');
  const oldRevision=revision;await save(false);check(revision===oldRevision,'completed page cannot be saved accidentally');
  switchReviewTab('pieces');await idle();
  check($('label').disabled&&Array.from($('grid').children).every(x=>x.disabled),'approved board locked');
  switchReviewTab('screen');await idle();
  document.querySelector('[aria-label="Edit reviewed ammunition"]').click();
  check(!$('ammo_loaded').disabled&&!$('ammoReviewed').checked&&dirty,'explicit edit reopens only ammo review');
  check($('attack').disabled&&$('screenKind').disabled,'other reviewed sections stay locked');
  $('ammo_loaded').value='1';$('ammo_loaded').oninput();await save(false);
  let stored=await api('/api/item?id='+encodeURIComponent(currentId));
  check(stored.data.confirmed&&stored.data.screen_state.reviews.screen_kind&&!stored.data.screen_state.reviews.ammo&&stored.data.screen_state.ammo.loaded===1,'editing ammo preserves board and unrelated approvals');
  $('ammoReviewed').checked=true;$('ammoReviewed').oninput();await save(false);
  check($('ammo_loaded').disabled&&$('draft').disabled,'review completion relocks edited section');
  $('showCompleted').checked=false;await $('showCompleted').onchange();await idle();
  check(!Array.from($('items').options).some(x=>x.value.includes('a-complete')),'finished screenshot leaves pending queue');
  $('sessions').value='finished';await $('sessions').onchange();
  check(document.querySelector('main').hidden&&$('items').disabled,'completed session shows no editable screenshot');
  check($('reviewSummaryText').textContent.includes('No pending'),'completed session explains empty queue');
  $('showCompleted').checked=true;await $('showCompleted').onchange();await idle();
  check(!document.querySelector('main').hidden&&$('ammo_loaded').disabled,'completed session can be inspected read-only');
  $('sessions').value='play';await $('sessions').onchange();await idle();
  switchReviewTab('pieces');await idle();await pick('sessions/play/annotations/c-pending.png.json');
  await save(true);await idle();
  check(currentId.endsWith('b-partial.png.json'),'next pending wraps to earlier unfinished image');
  $('showCompleted').checked=false;await $('showCompleted').onchange();await idle();
  $('sessions').value='missing-floor';await $('sessions').onchange();await idle();
  switchReviewTab('pieces');await idle();await save(true);
  check(currentReviewTab==='screen'&&!$('floor').disabled&&!$('approve').disabled,'missing floor can be supplied despite reviewed page sections');
  $('floor').value='2';$('floor').oninput();await save(true);
  const floorSaved=await api('/api/item?id='+encodeURIComponent('sessions/missing-floor/annotations/no-floor.png.json'));
  check(floorSaved.data.floor===2&&floorSaved.data.confirmed,'floor hand-off completes piece approval');
  $('showCompleted').checked=true;await $('showCompleted').onchange();await idle();
  switchReviewTab('screen');await idle();$('sessions').value='cards';await $('sessions').onchange();await idle();
  check($('screenKind').disabled&&$('draft').disabled&&$('approve').disabled,'reviewed non-gameplay screenshot is read-only');
  check(!$('piecesTab').disabled,'completed non-gameplay screen does not trap category navigation');
  switchReviewTab('pieces');await navigate(1);await idle();
  check(currentId.endsWith('b-partial.png.json'),'category navigation reaches another session with unfinished pieces');
  document.body.innerHTML='<pre id="browser-test-result">'+JSON.stringify({ok:true,checks})+'</pre>';
 }catch(e){document.body.innerHTML='<pre id="browser-test-result">'+JSON.stringify({ok:false,checks,error:e.stack})+'</pre>'}
})();
